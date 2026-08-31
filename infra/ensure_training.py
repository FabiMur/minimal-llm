#!/usr/bin/env python3
"""Watchdog for the minimal-llm spot training run.

Checks whether a `minimal-llm-train` instance is alive in us-east-2. If none is,
launches a new spot instance that resumes from the latest S3 checkpoint.

The run is interruption-driven: g5/g6 spot capacity in us-east-2 comes and goes,
and `user-data.sh` resumes from `checkpoints/<run>/latest.pt` on every boot, so
relaunching is the whole recovery story.

Usage:
    AWS_PROFILE=minimal-llm python3 infra/ensure_training.py [--launch]

Without --launch it only reports state (dry run), so it is safe to poll.
Set the STOP key in S3 to disable relaunching without touching this file:
    aws s3 cp - s3://minimal-llm-fabimur/checkpoints/aws-g5-run1/STOP </dev/null
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-2"
BUCKET = "minimal-llm-fabimur"
RUN_NAME = "aws-g5-run1"
TAG_NAME = "minimal-llm-train"
KEY_NAME = "minimal-llm-train"
IAM_PROFILE = "minimal-llm-training-profile"
SECURITY_GROUP = "sg-0f131240a8a6b9350"
ROOT_VOLUME_GB = 100

# Only us-east-2 has approved spot GPU quota (8 vCPUs). us-west-2/us-east-1 have
# zero quota — never launch there.
SUBNETS = [
    ("us-east-2c", "subnet-005ec8d61a5865041"),
    ("us-east-2b", "subnet-0133b5c88737bd22e"),
    ("us-east-2a", "subnet-0cab4cb88f1bf13a6"),
]

# Both are 4 vCPU / 24 GB VRAM and fit the quota. user-data namespaces the
# torch.compile cache by instance type, so switching between them is safe.
INSTANCE_TYPES = ["g5.xlarge", "g6.xlarge"]

AMI_SSM_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
)

STOP_KEY = f"checkpoints/{RUN_NAME}/STOP"
USER_DATA_PATH = pathlib.Path(__file__).with_name("user-data.sh")

LIVE_STATES = ["pending", "running"]


def live_instances(ec2):
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [TAG_NAME]},
            {"Name": "instance-state-name", "Values": LIVE_STATES},
        ]
    )
    return [i for r in resp["Reservations"] for i in r["Instances"]]


def stop_requested(s3) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=STOP_KEY)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def checkpoint_status(s3):
    """Return (LastModified, size) of latest.pt, or None if absent."""
    try:
        head = s3.head_object(Bucket=BUCKET, Key=f"checkpoints/{RUN_NAME}/latest.pt")
        return head["LastModified"], head["ContentLength"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def resolve_ami(ssm) -> str:
    return ssm.get_parameter(Name=AMI_SSM_PARAM)["Parameter"]["Value"]


def launch(ec2, ami: str, user_data: str):
    """Try every (instance type, AZ) pair until spot capacity is found."""
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
                    UserData=user_data,
                    InstanceMarketOptions={
                        "MarketType": "spot",
                        "SpotOptions": {
                            "SpotInstanceType": "one-time",
                            "InstanceInterruptionBehavior": "terminate",
                        },
                    },
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
                )
                inst = resp["Instances"][0]
                return inst["InstanceId"], instance_type, az, attempts
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                attempts.append(f"{instance_type}/{az}: {code}")
                # Quota errors will not be fixed by trying another AZ.
                if code in ("VcpuLimitExceeded", "MaxSpotInstanceCountExceeded"):
                    return None, None, None, attempts
    return None, None, None, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch",
        action="store_true",
        help="actually launch when nothing is running (default: report only)",
    )
    args = parser.parse_args()

    session = boto3.Session(region_name=REGION)
    ec2 = session.client("ec2")
    s3 = session.client("s3")
    ssm = session.client("ssm")

    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"[{now.isoformat()}] checking {TAG_NAME} in {REGION}")

    ckpt = checkpoint_status(s3)
    if ckpt:
        mtime, size = ckpt
        age = str(now - mtime).split(".")[0]
        print(f"  latest.pt: {size / 1e9:.3f} GB, {mtime.isoformat()} (age {age})")
    else:
        print("  latest.pt: absent — a launch would start from scratch")

    running = live_instances(ec2)
    if running:
        for i in running:
            uptime = str(now - i["LaunchTime"]).split(".")[0]
            print(
                f"  ALIVE {i['InstanceId']} {i['InstanceType']} "
                f"{i['State']['Name']} az={i['Placement']['AvailabilityZone']} "
                f"ip={i.get('PublicIpAddress', '-')} uptime={uptime}"
            )
        return 0

    print("  no live instance")

    if stop_requested(s3):
        print(f"  STOP marker present (s3://{BUCKET}/{STOP_KEY}) — not relaunching")
        return 0

    if not args.launch:
        print("  dry run — pass --launch to relaunch")
        return 1

    user_data = USER_DATA_PATH.read_text()
    ami = resolve_ami(ssm)
    print(f"  resolved AMI {ami}; launching…")

    instance_id, instance_type, az, attempts = launch(ec2, ami, user_data)
    for line in attempts:
        print(f"    tried {line}")
    if instance_id:
        print(f"  LAUNCHED {instance_id} ({instance_type} in {az})")
        return 0

    print("  launch failed on every instance type / AZ combination")
    return 2


if __name__ == "__main__":
    sys.exit(main())
