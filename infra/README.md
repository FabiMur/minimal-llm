# AWS training infrastructure

Operational notes for the `aws-g5-run1` training run. Everything here used to
live only in a local `/tmp/user-data.sh` and in session memory; it is committed
so an interrupted run can be recovered from any machine.

## Fixed resources

| Resource | Value |
| --- | --- |
| Region | `us-east-2` (only region with approved spot GPU quota — 8 vCPUs) |
| Account | `589609245910`, IAM user `minimal-llm-admin` |
| S3 bucket | `minimal-llm-fabimur` |
| Run name | `aws-g5-run1` |
| Docker image | `public.ecr.aws/d5g0x1k7/minimal-llm-train:latest` (public, no login) |
| IAM instance profile | `minimal-llm-training-profile` |
| EC2 key pair | `minimal-llm-train` |
| Security group | `sg-0f131240a8a6b9350` |
| Instance tag | `Name=minimal-llm-train` |
| Root volume | 100 GB gp3 |

Subnets, in the order the launcher tries them:

| AZ | Subnet |
| --- | --- |
| `us-east-2c` | `subnet-005ec8d61a5865041` |
| `us-east-2b` | `subnet-0133b5c88737bd22e` |
| `us-east-2a` | `subnet-0cab4cb88f1bf13a6` |

`us-west-2` and `us-east-1` have zero spot GPU quota — never launch there.

The AMI is resolved at launch time rather than pinned:

```bash
aws ssm get-parameter --region us-east-2 \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
  --query "Parameter.Value" --output text
```

## `user-data.sh`

Boot script passed to every launch. It pulls the image, downloads `train.bin` /
`val.bin` / `meta.json` and the checkpoint prefix from S3, resumes from
`latest.pt` when present, syncs checkpoints back to S3 every 60s, and caches
`torch.compile` artifacts in S3 namespaced by instance type (Inductor/Triton
output is tied to GPU arch, so g5's A10G and g6's L4 must not share a cache).

## Watchdog

The run is kept alive by a Lambda on a 5-minute EventBridge schedule. It lives
in AWS on purpose: an earlier version depended on a chat session re-arming a
timer each cycle, that chain broke silently, and the run sat dead for 22 hours.

| Piece | Name |
| --- | --- |
| Function | `minimal-llm-watchdog` |
| Role | `minimal-llm-watchdog-role` |
| Schedule | `minimal-llm-watchdog-schedule` (`rate(5 minutes)`) |
| Boot script | `s3://minimal-llm-fabimur/infra/user-data.sh` |

Each tick:

* nothing running → launch spot, falling back to on-demand (~3x the price)
* spot running → nothing to do
* on-demand running → migrate back to spot once capacity returns

Migration is gated on `GetSpotPlacementScores` (a non-destructive capacity
probe, 1-10) reaching 8, plus a 2-hour cooldown, because a failed migration
costs a warm-up plus every step since the last checkpoint. If the spot request
fails after the on-demand instance is terminated, the function relaunches
on-demand immediately rather than leaving the run dead until the next tick.
`TerminateInstances` is scoped by a tag condition so a bug cannot reap anything
but our own instances.

```bash
export AWS_PROFILE=minimal-llm
python3 infra/deploy_watchdog.py             # create or update, idempotent
python3 infra/deploy_watchdog.py --teardown  # remove it all

aws lambda invoke --region us-east-2 --function-name minimal-llm-watchdog /dev/stdout
aws logs tail /aws/lambda/minimal-llm-watchdog --region us-east-2 --follow
```

Redeploy after editing `user-data.sh` — the script is uploaded to S3 on every
deploy, and the function reads it from there at launch time.

### `ensure_training.py`

The same logic as a local CLI, for manual checks and one-off launches.

```bash
python3 infra/ensure_training.py                            # report only
python3 infra/ensure_training.py --launch                   # spot only
python3 infra/ensure_training.py --launch --allow-on-demand # spot, then on-demand
```

It never launches while an instance is `pending` or `running`, and it tries
`g5.xlarge` then `g6.xlarge` across all three AZs.

To stop it from relaunching (e.g. the run is finished), set the STOP marker:

```bash
aws s3 cp - s3://minimal-llm-fabimur/checkpoints/aws-g5-run1/STOP </dev/null
aws s3 rm s3://minimal-llm-fabimur/checkpoints/aws-g5-run1/STOP   # to re-enable
```

## Manual checks

```bash
export AWS_PROFILE=minimal-llm

# running instance
aws ec2 describe-instances --region us-east-2 \
  --filters "Name=tag:Name,Values=minimal-llm-train" \
            "Name=instance-state-name,Values=pending,running" \
  --query "Reservations[].Instances[].[InstanceId,State.Name,LaunchTime,PublicIpAddress]" \
  --output text

# checkpoint freshness
aws s3api head-object --bucket minimal-llm-fabimur \
  --key checkpoints/aws-g5-run1/latest.pt --query "LastModified" --output text
date -u

# why the previous instance died
aws ec2 describe-spot-instance-requests --region us-east-2 \
  --query "SpotInstanceRequests[].[SpotInstanceRequestId,State,Status.Code,CreateTime]" \
  --output text

# boot progress without SSH
aws ec2 get-console-output --region us-east-2 --instance-id <id> --latest \
  --query Output --output text | tail -40
```

SSH needs the `minimal-llm-train` `.pem` and the public IP from
`describe-instances`: `ssh -i <key>.pem ubuntu@<ip>`.
