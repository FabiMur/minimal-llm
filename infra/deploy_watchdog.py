#!/usr/bin/env python3
"""Deploy (or update) the Lambda watchdog and its EventBridge schedule.

Idempotent: re-run it after editing lambda_watchdog.py or user-data.sh and it
updates in place.

    AWS_PROFILE=minimal-llm python3 infra/deploy_watchdog.py
    AWS_PROFILE=minimal-llm python3 infra/deploy_watchdog.py --teardown
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-2"
BUCKET = "minimal-llm-fabimur"
USER_DATA_KEY = "infra/user-data.sh"
FUNCTION = "minimal-llm-watchdog"
ROLE = "minimal-llm-watchdog-role"
RULE = "minimal-llm-watchdog-schedule"
SCHEDULE = "rate(5 minutes)"
IAM_PROFILE = "minimal-llm-training-profile"

HERE = pathlib.Path(__file__).parent
TRUST = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def policy_doc(pass_role_arn: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "Logs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": "arn:aws:logs:*:*:*",
            },
            {
                "Sid": "InspectAndLaunch",
                "Effect": "Allow",
                "Action": [
                    "ec2:DescribeInstances",
                    "ec2:DescribeImages",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeSecurityGroups",
                    "ec2:GetSpotPlacementScores",
                    "ec2:RunInstances",
                    "ec2:CreateTags",
                ],
                "Resource": "*",
            },
            {
                # Scoped to our own instances so a bug cannot reap anything else.
                "Sid": "TerminateOwnOnly",
                "Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"ec2:ResourceTag/Name": "minimal-llm-train"}
                },
            },
            {
                "Sid": "PassTrainingProfile",
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": pass_role_arn,
            },
            {
                "Sid": "ResolveAmi",
                "Effect": "Allow",
                "Action": ["ssm:GetParameter"],
                "Resource": "arn:aws:ssm:*::parameter/aws/service/deeplearning/*",
            },
            {
                "Sid": "RunState",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": f"arn:aws:s3:::{BUCKET}/*",
            },
        ],
    }


def zip_handler() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "lambda_watchdog.py", (HERE / "lambda_watchdog.py").read_text()
        )
    return buf.getvalue()


def ensure_role(iam, pass_role_arn: str) -> str:
    try:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
        print(f"  role exists: {arn}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        arn = iam.create_role(
            RoleName=ROLE,
            AssumeRolePolicyDocument=json.dumps(TRUST),
            Description="minimal-llm training watchdog",
        )["Role"]["Arn"]
        print(f"  role created: {arn}")
        time.sleep(10)  # IAM propagation before Lambda can assume it
    iam.put_role_policy(
        RoleName=ROLE,
        PolicyName="watchdog",
        PolicyDocument=json.dumps(policy_doc(pass_role_arn)),
    )
    print("  role policy written")
    return arn


def ensure_function(lam, role_arn: str, code: bytes) -> str:
    env = {"Variables": {"BUCKET": BUCKET, "USER_DATA_KEY": USER_DATA_KEY}}
    try:
        lam.get_function(FunctionName=FUNCTION)
        lam.update_function_code(FunctionName=FUNCTION, ZipFile=code)
        waiter = lam.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=FUNCTION)
        lam.update_function_configuration(
            FunctionName=FUNCTION, Timeout=120, MemorySize=256, Environment=env
        )
        waiter.wait(FunctionName=FUNCTION)
        print("  function updated")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        for attempt in range(6):
            try:
                lam.create_function(
                    FunctionName=FUNCTION,
                    Runtime="python3.12",
                    Role=role_arn,
                    Handler="lambda_watchdog.handler",
                    Code={"ZipFile": code},
                    Timeout=120,
                    MemorySize=256,
                    Environment=env,
                    Description="Keeps one minimal-llm training instance alive",
                )
                break
            except ClientError as inner:
                # The freshly created role may not be assumable yet.
                if inner.response["Error"]["Code"] != "InvalidParameterValueException":
                    raise
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("Lambda could not assume the role after retries")
        print("  function created")
    return lam.get_function(FunctionName=FUNCTION)["Configuration"]["FunctionArn"]


def ensure_schedule(events, lam, fn_arn: str) -> None:
    events.put_rule(Name=RULE, ScheduleExpression=SCHEDULE, State="ENABLED")
    try:
        lam.add_permission(
            FunctionName=FUNCTION,
            StatementId="eventbridge-invoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=events.describe_rule(Name=RULE)["Arn"],
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceConflictException":
            raise
    events.put_targets(Rule=RULE, Targets=[{"Id": "watchdog", "Arn": fn_arn}])
    print(f"  schedule set: {SCHEDULE}")


def teardown(iam, lam, events) -> None:
    for fn, kwargs in [
        (events.remove_targets, {"Rule": RULE, "Ids": ["watchdog"]}),
        (events.delete_rule, {"Name": RULE}),
        (lam.delete_function, {"FunctionName": FUNCTION}),
        (iam.delete_role_policy, {"RoleName": ROLE, "PolicyName": "watchdog"}),
        (iam.delete_role, {"RoleName": ROLE}),
    ]:
        try:
            fn(**kwargs)
            print(f"  removed via {fn.__name__}")
        except ClientError as exc:
            print(f"  skip {fn.__name__}: {exc.response['Error']['Code']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teardown", action="store_true")
    args = parser.parse_args()

    session = boto3.Session(region_name=REGION)
    iam, lam = session.client("iam"), session.client("lambda")
    events, s3 = session.client("events"), session.client("s3")

    if args.teardown:
        teardown(iam, lam, events)
        return 0

    # The boot script lives in S3 so it can change without a redeploy.
    s3.put_object(
        Bucket=BUCKET,
        Key=USER_DATA_KEY,
        Body=(HERE / "user-data.sh").read_bytes(),
    )
    print(f"  uploaded s3://{BUCKET}/{USER_DATA_KEY}")

    profile = iam.get_instance_profile(InstanceProfileName=IAM_PROFILE)
    pass_role_arn = profile["InstanceProfile"]["Roles"][0]["Arn"]
    print(f"  training role to pass: {pass_role_arn}")

    role_arn = ensure_role(iam, pass_role_arn)
    fn_arn = ensure_function(lam, role_arn, zip_handler())
    ensure_schedule(events, lam, fn_arn)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
