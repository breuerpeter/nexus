# GPU runner watchdog

`runner-watchdog.yaml` is a CloudFormation stack that ends GPU CI runners that outlive their run. It closes the leak where a cancel lands while `gpu / start-runner` still executes. That job then publishes no instance id, so `stop-runner` has nothing to stop.

Every box `gpu-runner.yml` launches carries the tag `created-by=nexus-gpu-ci`, set on the launch call itself. Every 15 minutes a scheduled Lambda function finds the running boxes with that tag in each monitored region. It terminates each box older than `TerminateAfterMinutes`, 90 by default. The run job's `timeout-minutes` is 45, so a working box never reaches that age. The Lambda role can terminate only an instance with the tag.

Two alarms send their state changes to an SNS topic you already have:

- `nexus-gpu-runner-overdue` fires when a tagged box is 120 minutes old, or when the watchdog stops reporting.
- `nexus-gpu-runner-watchdog-errors` fires when a watchdog run fails to scan or terminate.

CI doesn't deploy the stack. A maintainer deploys it by hand, from this file.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `AlertTopicArn` | none | The SNS topic for the alarms. Required. |
| `TerminateAfterMinutes` | `90` | The age at which the watchdog terminates a tagged box, 1 to 120. |
| `TerminationEnabled` | `false` | `false` is shadow mode: the log lists each box as `would_terminate`, and the watchdog terminates nothing. |
| `RequiredRepository` | empty | When set, only boxes whose `GitHub-Repository` tag equals it are candidates. |
| `MonitoredRegions` | `us-west-2,eu-central-1,us-east-1,us-east-2` | The regions to scan. Keep it in step with `AWS_REGION_CANDIDATES`. |

## Grant the CI role the tag right

The OpenID Connect (OIDC) role in `AWS_GPU_ROLE_ARN` must be able to tag at launch, or every GPU run fails at `start-runner`. Add this statement to its policy before a run uses the tags:

```json
{
  "Effect": "Allow",
  "Action": "ec2:CreateTags",
  "Resource": "arn:aws:ec2:*:*:*/*",
  "Condition": { "StringEquals": { "ec2:CreateAction": "RunInstances" } }
}
```

## Deploy

Check the template, then deploy it in shadow mode:

```bash
uvx --from cfn-lint cfn-lint scripts/ci/aws/runner-watchdog.yaml
aws sts get-caller-identity
aws cloudformation deploy --region us-west-2 \
  --stack-name nexus-gpu-runner-watchdog \
  --template-file scripts/ci/aws/runner-watchdog.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides AlertTopicArn=<topic-arn>
```

To try it on live runners, deploy with `TerminateAfterMinutes=1 RequiredRepository=breuerpeter/nexus`, start a GPU run, and read the log after the next scan:

```bash
aws logs tail /aws/lambda/nexus-gpu-runner-watchdog --region us-west-2 --since 20m
```

The run's box shows as `would_terminate`, and no box without the tag shows at all. Then deploy again with `TerminateAfterMinutes=90 TerminationEnabled=true` and no `RequiredRepository`. Parameters you don't name keep their last value, so pass `RequiredRepository=""` to clear it.

## Check the ownership boundary

The policy simulator shows the role can terminate a tagged box and nothing else:

```bash
role=$(aws cloudformation describe-stacks --region us-west-2 --stack-name nexus-gpu-runner-watchdog \
  --query "Stacks[0].Outputs[?OutputKey=='RoleArn'].OutputValue" --output text)
for owner in nexus-gpu-ci other; do
  aws iam simulate-principal-policy --policy-source-arn "$role" \
    --action-names ec2:TerminateInstances \
    --resource-arns "arn:aws:ec2:us-west-2:$(aws sts get-caller-identity --query Account --output text):instance/i-0123456789abcdef0" \
    --context-entries "ContextKeyName=ec2:ResourceTag/created-by,ContextKeyValues=$owner,ContextKeyType=string" \
    --query "EvaluationResults[0].EvalDecision" --output text
done
```

It prints `allowed`, then `implicitDeny`.

## Check the alarms

Turn off the schedule for one alarm period, 30 minutes. `nexus-gpu-runner-overdue` goes to `ALARM` and the topic delivers. Turn the schedule on again, and the alarm returns to `OK` after the next period.

```bash
aws events disable-rule --region us-west-2 --name nexus-gpu-runner-watchdog
aws events enable-rule --region us-west-2 --name nexus-gpu-runner-watchdog
```

## Roll back

Before an update, save the current template and parameters:

```bash
aws cloudformation get-template --region us-west-2 --stack-name nexus-gpu-runner-watchdog > watchdog-template.json
aws cloudformation describe-stacks --region us-west-2 --stack-name nexus-gpu-runner-watchdog > watchdog-stack.json
```

To roll back, deploy the saved template with the saved parameters. To remove the watchdog, run `aws cloudformation delete-stack --region us-west-2 --stack-name nexus-gpu-runner-watchdog`.
