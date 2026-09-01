"""Lambda watchdog for the minimal-llm training run.

Runs on an EventBridge schedule and keeps exactly one training instance alive:

* nothing running    -> launch spot, falling back to on-demand
* spot running       -> nothing to do
* on-demand running  -> migrate back to spot once spot capacity returns

Living inside AWS rather than in a chat session matters: the previous watchdog
depended on an assistant re-arming a timer each cycle, and when that chain broke
the run sat dead for 22 hours. EventBridge fires whether or not anyone is
watching.

The boot script is read from S3 (not baked into this package) so it can be
updated without redeploying the function.
"""

from __future__ import annotations

import datetime
import json
import os

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("REGION", "us-east-2")
BUCKET = os.environ.get("BUCKET", "minimal-llm-fabimur")
RUN_NAME = os.environ.get("RUN_NAME", "aws-g5-run1")
TAG_NAME = os.environ.get("TAG_NAME", "minimal-llm-train")
KEY_NAME = os.environ.get("KEY_NAME", "minimal-llm-train")
IAM_PROFILE = os.environ.get("IAM_PROFILE", "minimal-llm-training-profile")
SECURITY_GROUP = os.environ.get("SECURITY_GROUP", "sg-0f131240a8a6b9350")
USER_DATA_KEY = os.environ.get("USER_DATA_KEY", "infra/user-data.sh")
ROOT_VOLUME_GB = int(os.environ.get("ROOT_VOLUME_GB", "100"))

# Only launch when spot capacity looks genuinely available. Scores run 1-10;
# migrating on a low score risks killing a healthy on-demand instance for a
# spot request that then fails.
MIGRATION_SCORE_THRESHOLD = int(os.environ.get("MIGRATION_SCORE_THRESHOLD", "8"))
MIGRATION_COOLDOWN_HOURS = int(os.environ.get("MIGRATION_COOLDOWN_HOURS", "2"))

SUBNETS = [
    ("us-east-2c", "subnet-005ec8d61a5865041"),
    ("us-east-2b", "subnet-0133b5c88737bd22e"),
    ("us-east-2a", "subnet-0cab4cb88f1bf13a6"),
]
INSTANCE_TYPES = ["g5.xlarge", "g6.xlarge"]

AMI_SSM_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
)

STOP_KEY = f"checkpoints/{RUN_NAME}/STOP"
MIGRATION_KEY = f"checkpoints/{RUN_NAME}/.last-migration"
LIVE_STATES = ["pending", "running"]

ec2 = boto3.client("ec2", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)


def _missing(exc: ClientError) -> bool:
    return exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound")


def live_instances():
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [TAG_NAME]},
            {"Name": "instance-state-name", "Values": LIVE_STATES},
        ]
    )
    return [i for r in resp["Reservations"] for i in r["Instances"]]


def stop_requested() -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=STOP_KEY)
        return True
    except ClientError as exc:
        if _missing(exc):
            return False
        raise


def user_data() -> str:
    return s3.get_object(Bucket=BUCKET, Key=USER_DATA_KEY)["Body"].read().decode()


def best_spot_score() -> int:
    """Highest spot placement score (1-10) across our instance types."""
    resp = ec2.get_spot_placement_scores(
        InstanceTypes=INSTANCE_TYPES,
        TargetCapacity=1,
        TargetCapacityUnitType="units",
        SingleAvailabilityZone=True,
        RegionNames=[REGION],
    )
    scores = [s["Score"] for s in resp.get("SpotPlacementScores", [])]
    return max(scores) if scores else 0


def migration_on_cooldown() -> bool:
    try:
        head = s3.head_object(Bucket=BUCKET, Key=MIGRATION_KEY)
    except ClientError as exc:
        if _missing(exc):
            return False
        raise
    age = datetime.datetime.now(datetime.timezone.utc) - head["LastModified"]
    return age < datetime.timedelta(hours=MIGRATION_COOLDOWN_HOURS)


def mark_migration() -> None:
    s3.put_object(Bucket=BUCKET, Key=MIGRATION_KEY, Body=b"")


def launch(spot: bool):
    """Try every (instance type, AZ) pair. Returns (instance_id, detail)."""
    ami = ssm.get_parameter(Name=AMI_SSM_PARAM)["Parameter"]["Value"]
    script = user_data()
    market = (
        {
            "InstanceMarketOptions": {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }
        }
        if spot
        else {}
    )
    attempts = []
    for instance_type in INSTANCE_TYPES:
        for az, subnet in SUBNETS:
            try:
                resp = ec2.run_instances(
                    ImageId=ami,
                    InstanceType=instance_type,
                    MinCount=1,
                    MaxCount=1,
                    KeyName=KEY_NAME,
                    SubnetId=subnet,
                    SecurityGroupIds=[SECURITY_GROUP],
                    IamInstanceProfile={"Name": IAM_PROFILE},
                    UserData=script,
                    BlockDeviceMappings=[
                        {
                            "DeviceName": "/dev/sda1",
                            "Ebs": {
                                "VolumeSize": ROOT_VOLUME_GB,
                                "VolumeType": "gp3",
                                "DeleteOnTermination": True,
                            },
                        }
                    ],
                    TagSpecifications=[
                        {
                            "ResourceType": rt,
                            "Tags": [{"Key": "Name", "Value": TAG_NAME}],
                        }
                        for rt in ("instance", "volume")
                    ],
                    **market,
                )
                iid = resp["Instances"][0]["InstanceId"]
                return iid, f"{'spot' if spot else 'on-demand'} {instance_type}/{az}"
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                attempts.append(f"{instance_type}/{az}:{code}")
                if code in ("VcpuLimitExceeded", "MaxSpotInstanceCountExceeded"):
                    return None, "; ".join(attempts)
    return None, "; ".join(attempts)


def launch_preferring_spot():
    iid, detail = launch(spot=True)
    if iid:
        return iid, detail, "spot"
    iid, od_detail = launch(spot=False)
    if iid:
        return iid, od_detail, "on-demand"
    return None, f"spot[{detail}] on-demand[{od_detail}]", None


def handler(event, context):
    out = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    if stop_requested():
        out["action"] = "stopped-by-marker"
        print(json.dumps(out))
        return out

    live = live_instances()

    if not live:
        iid, detail, kind = launch_preferring_spot()
        out["action"] = "launched" if iid else "launch-failed"
        out["instance_id"] = iid
        out["detail"] = detail
        out["lifecycle"] = kind
        print(json.dumps(out))
        return out

    inst = live[0]
    out["instance_id"] = inst["InstanceId"]
    out["lifecycle"] = inst.get("InstanceLifecycle", "on-demand")
    out["state"] = inst["State"]["Name"]

    if out["lifecycle"] == "spot":
        out["action"] = "ok"
        print(json.dumps(out))
        return out

    # An on-demand instance is carrying the run at ~3x the spot price. Move back
    # to spot when capacity looks solid, but never churn: a failed migration
    # costs a warm-up plus the steps since the last checkpoint.
    if migration_on_cooldown():
        out["action"] = "on-demand-migration-cooldown"
        print(json.dumps(out))
        return out

    score = best_spot_score()
    out["spot_score"] = score
    if score < MIGRATION_SCORE_THRESHOLD:
        out["action"] = "on-demand-waiting-for-spot"
        print(json.dumps(out))
        return out

    mark_migration()
    ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
    iid, detail = launch(spot=True)
    if iid:
        out["action"] = "migrated-to-spot"
        out["new_instance_id"] = iid
        out["detail"] = detail
        print(json.dumps(out))
        return out

    # Spot vanished between the score check and the request: get the run back
    # on-demand immediately rather than leaving it dead until the next tick.
    iid, od_detail = launch(spot=False)
    out["action"] = "migration-failed-relaunched-on-demand" if iid else "migration-failed-dead"
    out["new_instance_id"] = iid
    out["detail"] = f"spot[{detail}] on-demand[{od_detail}]"
    print(json.dumps(out))
    return out
