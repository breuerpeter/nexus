# Adapted from newton-physics/newton scripts/ci/discover_aws_runner_config.py
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Discover AWS EC2 runner placements for gpu-runner.yml (#26: capacity via breadth, not retries).

Runs in the start-runner job after AWS credentials are configured and emits GitHub Actions step
outputs consumed by ``machulav/ec2-github-runner``:

``availability-zones-config``
    JSON array of ``{imageId, subnetId, securityGroupId, region}`` candidates, tried in order.
``root-device``
    Root device name of the first candidate's AMI (for the block-device size override).

Candidate order (first success wins; the action falls through on capacity/quota failures):
1. STATIC entries from ``$STATIC_AZ_CONFIG``, if set (an escape hatch to pin known-good
   provisioned subnets ahead of discovery; unused since the us-west-2 consolidation: entries
   with an empty ``subnetId`` are dropped, so callers may template repo vars in freely).
2. Per region in ``$AWS_REGION_CANDIDATES``: subnets/SGs tagged ``$AWS_RUNNER_RESOURCE_TAG=true``,
   skipped where that variable names no key,
   (the upstream newton convention: how infra opts resources in), then the region's DEFAULT VPC
   subnets with its default egress-allowing security group (zero provisioning: this is what adds
   the AZs and regions we never provisioned). Regions without G-instance quota simply fail over
   at launch time (~10 s each).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

AwsCall = Callable[..., Any]
Warn = Callable[[str], None]
AWS_CLI_TIMEOUT_SECONDS = 120

AMI_NAME_FILTER = "Deep Learning Base AMI with Single CUDA (Ubuntu 22.04) ????????"


def warning(message: str) -> None:
    """Emit a GitHub Actions warning annotation."""
    print(f"::warning::{message}", flush=True)


def error(message: str) -> None:
    """Emit a GitHub Actions error annotation."""
    print(f"::error::{message}", flush=True)


def print_aws_cli_output(label: str, output: str | bytes | None) -> None:
    """Print captured Amazon Web Services (AWS) command-line tool output to stderr when available."""
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    print(f"AWS CLI {label}:", file=sys.stderr)
    print(output, file=sys.stderr)


def aws(region: str, *args: str) -> Any | None:
    """Run an AWS EC2 command-line tool command and parse the JSON response; None on any failure."""
    cmd = ["aws", "ec2", *args, "--region", region, "--output", "json"]
    command_label = f"aws ec2 {args[0]}" if args else "aws ec2"
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=AWS_CLI_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        warning(f"{region}: AWS CLI command timed out after {AWS_CLI_TIMEOUT_SECONDS}s: {command_label}")
        print_aws_cli_output("stdout before timeout", exc.stdout)
        print_aws_cli_output("stderr before timeout", exc.stderr)
        return None
    except subprocess.CalledProcessError as exc:
        warning(f"{region}: AWS CLI command failed: {command_label}")
        print_aws_cli_output("stdout", exc.stdout)
        print_aws_cli_output("stderr", exc.stderr)
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    try:
        return json.loads(output)
    except ValueError as exc:
        warning(f"{region}: AWS CLI command returned invalid JSON: {command_label}")
        print(f"Invalid JSON output from AWS CLI: {exc}", file=sys.stderr)
        print(f"Output prefix: {output[:1000]!r}", file=sys.stderr)
        return None


def allows_outbound_internet(security_group: Mapping[str, Any]) -> bool:
    """Return whether a security group has IPv4 internet egress, which is all the runner needs."""
    for permission in security_group.get("IpPermissionsEgress", []):
        for ip_range in permission.get("IpRanges", []):
            if ip_range.get("CidrIp") == "0.0.0.0/0":
                return True
    return False


def _region_ami(region: str, aws_call: AwsCall, warn: Warn) -> str | None:
    images = aws_call(
        region,
        "describe-images",
        "--owners", "amazon",
        "--filters", f"Name=name,Values={AMI_NAME_FILTER}", "Name=state,Values=available",
        "--query", "reverse(sort_by(Images, &CreationDate))[:1].ImageId",
    )  # fmt: skip
    if not images:
        warn(f"{region}: no Deep Learning Base GPU AMI found; skipping region")
        return None
    return images[0]


def _offered_zone_ids(region: str, instance_type: str, aws_call: AwsCall, warn: Warn) -> set[str]:
    offered = aws_call(
        region,
        "describe-instance-type-offerings",
        "--location-type", "availability-zone-id",
        "--filters", f"Name=instance-type,Values={instance_type}",
        "--query", "InstanceTypeOfferings[].Location",
    )  # fmt: skip
    if not offered:
        warn(f"{region}: {instance_type} is not offered in any availability zone; skipping region")
        return set()
    return set(offered)


def _entries_from_subnets(
    region: str,
    image_id: str,
    subnets: Sequence[Mapping[str, Any]],
    group_for_vpc: Mapping[str, str],
    offered_zone_ids: set[str],
    seen_zone_ids: set[str],
    warn: Warn,
    source: str,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    ordered = sorted(subnets, key=lambda s: (s.get("AvailabilityZoneId", ""), s["SubnetId"]))
    for subnet in ordered:
        subnet_id = subnet["SubnetId"]
        zone_id = subnet.get("AvailabilityZoneId", "")
        zone_name = subnet.get("AvailabilityZone", "")
        if not zone_id or zone_id in seen_zone_ids:
            continue
        if zone_id not in offered_zone_ids:
            continue
        group_id = group_for_vpc.get(subnet["VpcId"])
        if not group_id:
            warn(f"{region}: subnet {subnet_id} has no eligible security group; skipping")
            continue
        seen_zone_ids.add(zone_id)
        entries.append({"imageId": image_id, "subnetId": subnet_id, "securityGroupId": group_id, "region": region})
        print(
            f"Candidate ({source}): region={region} az={zone_name} ({zone_id}) subnet={subnet_id} sg={group_id}",
            flush=True,
        )
    return entries


def discover_candidates(
    regions: Sequence[str],
    instance_type: str,
    tag_key: str,
    aws_call: AwsCall = aws,
    warn: Warn = warning,
) -> list[dict[str, str]]:
    """Discover eligible runner placements: tagged resources first, then the default VPC."""
    candidates: list[dict[str, str]] = []
    for region in regions:
        print(f"Checking {region} for {instance_type}", flush=True)
        offered = _offered_zone_ids(region, instance_type, aws_call, warn)
        if not offered:
            continue
        image_id = _region_ami(region, aws_call, warn)
        if not image_id:
            continue
        seen_zone_ids: set[str] = set()

        # 1. Opt-in tagged subnets and SGs, the upstream newton convention. With no key set nothing
        #    opts in, and placement is the default VPC below.
        tagged_subnets: list = []
        tagged_groups: list = []
        if tag_key:
            tagged_subnets = aws_call(
                region, "describe-subnets",
                "--filters", f"Name=tag:{tag_key},Values=true", "--query", "Subnets[]",
            ) or []  # fmt: skip
            tagged_groups = aws_call(
                region, "describe-security-groups",
                "--filters", f"Name=tag:{tag_key},Values=true", "--query", "SecurityGroups[]",
            ) or []  # fmt: skip
        group_for_vpc: dict[str, str] = {}
        for sg in sorted(tagged_groups, key=lambda g: g["GroupId"]):
            if allows_outbound_internet(sg):
                group_for_vpc.setdefault(sg["VpcId"], sg["GroupId"])
        if tagged_subnets:
            candidates += _entries_from_subnets(
                region, image_id, tagged_subnets, group_for_vpc, offered, seen_zone_ids, warn, "tagged"
            )

        # 2. The region's default VPC: a subnet per AZ plus an egress-allowing default SG, with no
        #    provisioning. This is what contributes the AZs and regions nobody set up by hand.
        default_vpcs = aws_call(
            region, "describe-vpcs",
            "--filters", "Name=is-default,Values=true", "--query", "Vpcs[].VpcId",
        ) or []  # fmt: skip
        if default_vpcs:
            vpc_id = default_vpcs[0]
            default_groups = aws_call(
                region, "describe-security-groups",
                "--filters", f"Name=vpc-id,Values={vpc_id}", "Name=group-name,Values=default",
                "--query", "SecurityGroups[]",
            ) or []  # fmt: skip
            egress_groups = [g["GroupId"] for g in default_groups if allows_outbound_internet(g)]
            default_sg = {vpc_id: egress_groups[0]} if egress_groups else {}
            if default_sg:
                default_subnets = aws_call(
                    region, "describe-subnets",
                    "--filters", f"Name=vpc-id,Values={vpc_id}",
                    "Name=default-for-az,Values=true",
                    "--query", "Subnets[]",
                ) or []  # fmt: skip
                candidates += _entries_from_subnets(
                    region, image_id, default_subnets, default_sg, offered, seen_zone_ids, warn, "default-vpc"
                )
            else:
                warn(f"{region}: default VPC {vpc_id} has no egress-allowing default SG")
        else:
            warn(f"{region}: no default VPC")
        if not seen_zone_ids:
            warn(f"{region}: no eligible placements found")
    return candidates


def set_output(name: str, value: str) -> None:
    """Write a key-value pair to the GitHub Actions step-output file."""
    path = os.environ.get("GITHUB_OUTPUT", "")
    if path:
        with open(path, "a", encoding="utf-8") as output_file:
            output_file.write(f"{name}={value}\n")


def main() -> int:
    """Entry point for the workflow step."""
    regions = os.environ["AWS_REGION_CANDIDATES"].split()
    instance_type = os.environ["AWS_INSTANCE_TYPE"]
    tag_key = os.environ.get("AWS_RUNNER_RESOURCE_TAG", "")  # a name in the account: a setting, not a default here

    # Known-good provisioned entries lead the list, with networking and quota proven. Entries can omit
    # imageId: this script resolves the region's Amazon Machine Image (AMI) so the workflow doesn't have to.
    static_config = os.environ.get("STATIC_AZ_CONFIG", "").strip()
    candidates: list[dict[str, str]] = []
    if static_config:
        try:
            candidates = [e for e in json.loads(static_config) if e.get("subnetId")]
        except ValueError:
            warning("STATIC_AZ_CONFIG is not valid JSON; ignoring")
        for entry in candidates:
            entry.setdefault("region", regions[0])
            if not entry.get("imageId"):
                entry["imageId"] = _region_ami(entry["region"], aws, warning) or ""
        candidates = [e for e in candidates if e.get("imageId")]

    static_subnets = {e["subnetId"] for e in candidates}
    discovered = discover_candidates(regions, instance_type, tag_key)
    candidates += [e for e in discovered if e["subnetId"] not in static_subnets]

    if not candidates:
        error("No eligible EC2 runner placements were discovered.")
        return 1

    first = candidates[0]
    root_device = aws(
        first.get("region", regions[0]),
        "describe-images",
        "--image-ids", first["imageId"],
        "--query", "Images[0].RootDeviceName",
    ) or "/dev/sda1"  # fmt: skip

    print("Generated availability-zones-config:", flush=True)
    print(json.dumps(candidates, indent=2), flush=True)
    set_output("availability-zones-config", json.dumps(candidates, separators=(",", ":")))
    set_output("root-device", root_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
