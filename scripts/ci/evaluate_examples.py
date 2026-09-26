"""CI evaluation harness for the examples: flight performance, evo pose Absolute Pose Error (APE) plus
example stats, and speed, the Real-Time Factor (RTF), regression gates, plus the benchmark data feed.

Runs each example as its own subprocess, ``uv run [--extra X] -m nexus.examples <name>``,
reads the artifacts every example dumps via ``nexus.examples._lib.eval_dump``, where the harness sets
``$NEXUS_EVAL_OUT``, scores them, pose APE with **evo** for translation and, where the example defines
a reference attitude, rotation, the example's own stats, and the orchestrator's steady RTF, then
gates everything against the committed ``scripts/ci/examples_baselines.json`` and writes
``benchmark.json``, github-action-benchmark-style ``{name, unit, value}`` entries plus a
``biggerIsBetter`` direction, for the trend record. With ``--upload`` it publishes each flight's
``.rrd`` and the benchmark data to the CI artifacts bucket.

evo is under the General Public License (GPL) and heavy, so it lives only here, in the ``ci``
dependency-group, never a shipped dependency; the examples dump plain numpy/JSON.

    uv run --group ci --extra examples python scripts/ci/evaluate_examples.py [--only n1,n2]
        [--skip-missing] [--upload] [--update-baselines] [--out <dir>]

Baseline rules per metric: a bare value means exact match, for bools, ints and config; a dict combines an
optional recorded ``value`` with guards: absolute ``min``/``max``, always applied, and relative
``min_frac``/``max_frac``, applied once ``value`` has a recording; ``--update-baselines`` records fresh
values in place, run it on the CI runner hardware the baselines belong to.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINES = ROOT / "scripts" / "ci" / "examples_baselines.json"


def bucket() -> str:
    """The artifacts bucket, from ``$NEXUS_BUCKET``: CI artifacts under public/ci/, docs .rrds plus bench feed.

    A name outside the tree is a setting, not a line here, so a checkout carries no account's bucket.
    Only ``--upload`` needs it, and a run that asks to upload without it fails here rather than
    half-way through.
    """
    name = os.environ.get("NEXUS_BUCKET")
    if not name:
        raise SystemExit("--upload needs $NEXUS_BUCKET: the artifacts bucket to write to")
    return name


# name, the harness key, -> how to run it: "uv" the extras, "launcher" the example when the key is a
# second flight of one example, "requires" gates availability: --skip-missing skips, else fails.
EXAMPLES: dict[str, dict] = {
    "pid": {"uv": []},
    "gain_tuning": {"uv": ["--extra", "examples"]},
    "mass_recovery": {"uv": []},
    "sampling_mpc": {"uv": ["--extra", "examples"]},
    "acados_nmpc": {"uv": ["--extra", "acados"], "requires": "acados"},
    # goto_policy flies the hosted policy, the zero-arg example: given a policy the flight is
    # bit-reproducible, so it carries the tight deploy gates. goto_policy_fresh flies the policy the
    # rl leg just trained, one draw from a training that's not reproducible, so its baselines block
    # is empty: it gates on completing only, and the harness reports its numbers.
    "goto_policy": {"uv": ["--extra", "policy"]},
    "goto_policy_fresh": {"uv": ["--extra", "policy"], "launcher": "goto_policy", "requires": "policy"},
    "px4_sitl": {"uv": [], "requires": "px4"},
}
# The default set = everything the consolidated gpu-examples leg runs. The workflow provides the PX4
# checkout via scripts/ci/provision_px4.sh and acados via scripts/setup_acados.sh; main() below warms
# the PX4 *build*, from the one container definition in nexus._src.vehicle.controllers.px4.sitl.
# goto_policy_fresh rides the gpu-rl workflow: --only goto_policy_fresh --policy <the fresh export>.
# Local runs without the PX4/acados prerequisites: add --skip-missing.
DEFAULT_SET = ["pid", "gain_tuning", "mass_recovery", "sampling_mpc", "acados_nmpc", "goto_policy", "px4_sitl"]

# Metrics where bigger is better; for everything else numeric, smaller is better.
_BIGGER = {"rtf", "reached", "gradient_cosine", "deploy_steps_per_sec", "climb_m", "min_clearance_m"}
_UNITS = {"_m": "m", "_deg": "deg", "_s": "s", "rtf": "x realtime", "_per_sec": "1/s"}


def _available(requires: str | None, args: argparse.Namespace) -> tuple[bool, str]:
    # External tool locations come from the one definition of the env-overridable defaults the
    # examples themselves resolve: acados' from nexus.examples._external, PX4's from the
    # launcher that mounts them, nexus._src.vehicle.controllers.px4.sitl.
    from nexus._src.vehicle.controllers.px4.sitl import px4_dir
    from nexus.examples._external import acados_dir

    if requires is None:
        return True, ""
    if requires == "acados":
        if not (acados_dir() / "lib" / "libacados.so").exists():
            return False, "acados not provisioned (scripts/setup_acados.sh)"
        if not (ROOT / ".acados").exists():
            return False, ".acados/ path source missing (scripts/setup_acados.sh)"
        return True, ""
    if requires == "policy":
        p = args.policy or ""
        return (bool(p) and os.path.isfile(p)), "--policy not given / not a file (an exported policy.pt)"
    if requires == "px4":
        if not px4_dir().is_dir():
            return False, f"PX4_DIR not found ({px4_dir()})"
        if shutil.which("docker") is None:
            return False, "docker not available"
        return True, ""
    return False, f"unknown requirement {requires!r}"


def _dump_dir(name: str, spec: dict, out: pathlib.Path) -> pathlib.Path:
    """Where *name*'s example dumps its artifacts: ``out`` itself, or a subdirectory of it for a
    second flight of one example, since the example names its dump after its launcher.
    """
    return out / name if spec.get("launcher", name) != name else out


def _run_example(name: str, spec: dict, dump: pathlib.Path, timeout: float, extra_args: list[str]) -> bool:
    # Examples run with zero REQUIRED args: their configuration lives in the script, and recording is
    # always on. Per-example inputs such as a fresh policy checkpoint ride optional flags.
    cmd = ["uv", "run", *spec.get("uv", []), "-m", "nexus.examples", spec.get("launcher", name), *extra_args]
    print(f"\n===== {name}: {' '.join(cmd)} =====", flush=True)
    dump.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NEXUS_EVAL_OUT": str(dump)}
    try:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        print(f"[eval] {name}: TIMEOUT after {timeout:.0f}s", flush=True)
        return False
    if rc != 0:
        print(f"[eval] {name}: exit code {rc}", flush=True)
    return rc == 0


def _waypoint_reference(est_t: np.ndarray, est_pos: np.ndarray, waypoints: np.ndarray, arrival_t: np.ndarray):
    """Piecewise-linear position reference through the mission, time-anchored at the waypoint
    arrival times: start pose -> wp0 @ arrival_0 -> wp1 @ arrival_1 ... , held at the last reached
    waypoint afterwards; np.interp clamps at the edges.
    """
    knots_t = np.concatenate([[est_t[0]], arrival_t])
    knots_p = np.vstack([est_pos[0], waypoints[: len(arrival_t)]])
    return np.stack([np.interp(est_t, knots_t, knots_p[:, i]) for i in range(3)], axis=1)


def _ape(t, ref_pos, est_pos, ref_q_wxyz=None, est_q_wxyz=None) -> dict:
    """Pose APE via evo, in the world frame on time-aligned pairs, with no alignment step. Rotation APE only
    when the example defines a reference attitude.
    """
    from evo.core import metrics as em
    from evo.core.trajectory import PoseTrajectory3D

    ident = np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))
    ref = PoseTrajectory3D(
        positions_xyz=ref_pos, orientations_quat_wxyz=ident if ref_q_wxyz is None else ref_q_wxyz, timestamps=t
    )
    est = PoseTrajectory3D(
        positions_xyz=est_pos, orientations_quat_wxyz=ident if est_q_wxyz is None else est_q_wxyz, timestamps=t
    )
    out = {}
    ape_t = em.APE(em.PoseRelation.translation_part)
    ape_t.process_data((ref, est))
    out["ape_trans_rmse_m"] = float(ape_t.get_statistic(em.StatisticsType.rmse))
    out["ape_trans_max_m"] = float(ape_t.get_statistic(em.StatisticsType.max))
    if ref_q_wxyz is not None:
        ape_r = em.APE(em.PoseRelation.rotation_angle_deg)
        ape_r.process_data((ref, est))
        out["ape_rot_rmse_deg"] = float(ape_r.get_statistic(em.StatisticsType.rmse))
        out["ape_rot_max_deg"] = float(ape_r.get_statistic(em.StatisticsType.max))
    return out


def _score(name: str, out: pathlib.Path) -> tuple[dict, dict]:
    """Read one example's dumped artifacts and score it. Returns (metrics, meta-json)."""
    meta = json.loads((out / f"{name}.json").read_text())
    metrics: dict = {}
    for k, v in (meta.get("stats") or {}).items():
        if isinstance(v, (bool, int, float)):
            metrics[k] = v
    rtf = (meta.get("results") or {}).get("rtf")
    if rtf is not None:
        metrics["rtf"] = float(rtf)
    npz = out / f"{name}.npz"
    if npz.exists():
        d = np.load(npz)
        est_t, est_pos, est_q = d["est_t"], d["est_pos"], d["est_quat_xyzw"]
        if "ref_pos" in d.files:  # continuous tracking reference: pos plus attitude, time-aligned
            sel = np.isin(est_t, d["ref_t"])
            # The vehicle USDs are Front-Right-Down (FRD) authored, body +z down, while a flat
            # reference's attitude is upright, +z thrust up. This is the exact adapter the tracking
            # controller applies to its own measurement: q_upright = q_meas ⊗ q_flip, a 180° turn about
            # body x, so wxyz = (-x, w, z, -y) from the recorded xyzw; see acados_nmpc/controller.py,
            # the FRD → Nonlinear Model Predictive Control (NMPC) upright adapter.
            qx, qy, qz, qw = (est_q[sel][:, i] for i in range(4))
            est_q_wxyz = np.stack([-qx, qw, qz, -qy], axis=1)
            metrics.update(_ape(d["ref_t"], d["ref_pos"], est_pos[sel], d["ref_quat_wxyz"], est_q_wxyz))
        elif "waypoints" in d.files and d["arrival_t"].size:  # goal-sequencing mission
            ref_pos = _waypoint_reference(est_t, est_pos, d["waypoints"], d["arrival_t"])
            metrics.update(_ape(est_t, ref_pos, est_pos))
            metrics["final_err_m"] = float(np.linalg.norm(est_pos[-1] - d["waypoints"][-1]))
    return metrics, meta


def _check(rule, got) -> tuple[bool, str]:
    """One baseline rule against a fresh metric. A bare value = exact match; a dict combines
    absolute guards, min/max, always applied, with relative ones, min_frac/max_frac compared to the
    recorded value, skipped while value is null, meaning unbaselined.
    """
    if not isinstance(rule, dict):
        return got == rule, f"== {rule}"
    parts, ok = [], True
    v = rule.get("value")
    if "min" in rule:
        parts.append(f">= {rule['min']}")
        ok &= float(got) >= rule["min"]
    if "max" in rule:
        parts.append(f"<= {rule['max']}")
        ok &= float(got) <= rule["max"]
    if v is not None:
        if "min_frac" in rule:
            lo = v * rule["min_frac"]
            parts.append(f">= {lo:.4g}")
            ok &= float(got) >= lo
        if "max_frac" in rule:
            hi = v * rule["max_frac"]
            parts.append(f"<= {hi:.4g}")
            ok &= float(got) <= hi
    elif "min_frac" in rule or "max_frac" in rule:
        parts.append("(unbaselined)")
    return bool(ok), " & ".join(parts) or "(no guard)"


def _gate(name: str, metrics: dict, baselines: dict) -> list[str]:
    rules = baselines.get(name, {})
    failed = []
    print(f"\n--- {name} ---")
    print(f"{'metric':<26}{'fresh':>14}{'guard':>26}  status")
    for key, rule in rules.items():
        if key not in metrics:
            print(f"{key:<26}{'MISSING':>14}{'':>26}  REGRESSED")
            failed.append(f"{name}.{key} (missing)")
            continue
        ok, guard = _check(rule, metrics[key])
        print(f"{key:<26}{metrics[key]!s:>14}{guard:>26}  {'ok' if ok else 'REGRESSED'}")
        if not ok:
            failed.append(f"{name}.{key} ({metrics[key]} vs {guard})")
    for key in sorted(set(metrics) - set(rules)):  # monitored but not gated
        print(f"{key:<26}{metrics[key]!s:>14}{'(monitored)':>26}  -")
    return failed


def _unit(metric: str) -> str:
    for suffix, unit in _UNITS.items():
        if metric == suffix or metric.endswith(suffix):
            return unit
    return ""


def _bench_entries(scored: dict[str, dict]) -> list[dict]:
    entries = []
    for name, metrics in scored.items():
        for k, v in metrics.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            entries.append({"name": f"{k}[{name}]", "unit": _unit(k), "value": v, "biggerIsBetter": k in _BIGGER})
    return entries


def _sha() -> str:
    return (
        os.environ.get("GITHUB_SHA")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )


def _gpu_name() -> str:
    """The GPU's name from nvidia-smi, for the bench feed's ``_meta``, or ``unknown`` on a box without one."""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except FileNotFoundError:  # no nvidia-smi on the box, such as CPU CI
        return "unknown"
    return gpu.splitlines()[0] if gpu else "unknown"


def _merge_entries(prior: list, fresh: list[dict]) -> list[dict]:
    """Merge ``fresh`` entries over ``prior`` by entry name, fresh winning."""
    merged = {e["name"]: e for e in prior if isinstance(e, dict) and "name" in e}
    merged.update({e["name"]: e for e in fresh})
    return list(merged.values())


def _merged_bench(fresh: list[dict], key: str) -> list[dict]:
    """Merge fresh entries over the bucket's current ``key`` content, by entry name.

    The examples, RL and matrix CI legs publish disjoint metric sets, so a plain
    overwrite would clobber the other legs' entries at the fixed keys.
    """
    try:
        prior = json.loads(
            subprocess.run(["aws", "s3", "cp", f"s3://{bucket()}/{key}", "-"], capture_output=True, check=True).stdout
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        prior = []
    return _merge_entries(prior, fresh)


def _merged_local(path: pathlib.Path, fresh: list[dict]) -> list[dict]:
    """Merge fresh entries over ``path``'s current entries, by entry name.

    Split CI invocations sharing one --out dir, as the flight gate runs the eval once per PX4
    pin, would otherwise clobber the earlier invocation's entries; this is _merged_bench's local twin.
    """
    try:
        data = json.loads(path.read_text())
        prior = data["entries"] if isinstance(data, dict) else data
    except (OSError, ValueError, KeyError):
        prior = []
    return _merge_entries(prior, fresh)


def _upload(out: pathlib.Path, metas: dict[str, dict]) -> None:
    sha = _sha()

    def cp(src: pathlib.Path, key: str, ctype: str) -> None:
        print(f"uploading {src.name} -> s3://{bucket()}/{key}", flush=True)
        # Fixed keys, for stable docs URLs; max-age=300 so a re-upload propagates within ~5 min.
        subprocess.run(
            ["aws", "s3", "cp", str(src), f"s3://{bucket()}/{key}",
             "--content-type", ctype, "--cache-control", "max-age=300"],
            check=True,
        )  # fmt: skip

    for name, meta in metas.items():
        rrd = meta.get("rrd")
        if rrd and pathlib.Path(rrd).is_file():
            cp(pathlib.Path(rrd), f"public/ci/logs/{name}.rrd", "application/octet-stream")
            cp(pathlib.Path(rrd), f"public/ci/logs/{sha[:12]}/{name}.rrd", "application/octet-stream")
    bench = out / "benchmark.json"
    fresh = json.loads(bench.read_text()) if bench.exists() else []
    if not fresh:
        print("no scored metrics, skipping bench feed upload", flush=True)
        return
    for key in (f"public/ci/bench/{sha[:12]}.json", "public/ci/bench/latest.json"):
        merged = out / f"merged-{pathlib.Path(key).name}"
        merged.write_text(json.dumps(_merged_bench(fresh, key), indent=2))
        cp(merged, key, "application/json")
        merged.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated example names (default: the standalone set)")
    ap.add_argument("--skip-missing", action="store_true", help="skip examples whose requirement is unmet")
    ap.add_argument("--upload", action="store_true", help="publish .rrds + benchmark data to the CI bucket")
    ap.add_argument("--update-baselines", action="store_true", help="record fresh metric values into the baselines")
    ap.add_argument("--out", default=str(ROOT / ".eval-artifacts"), help="artifact dir (also $NEXUS_EVAL_OUT)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-example subprocess budget [s]")
    ap.add_argument("--policy", default=None, help="exported policy.pt that goto_policy_fresh flies")
    ap.add_argument("--baselines", default=str(BASELINES))
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",")] if args.only else list(DEFAULT_SET)
    unknown = [n for n in names if n not in EXAMPLES]
    if unknown:
        ap.error(f"unknown example(s): {unknown} (known: {list(EXAMPLES)})")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    baselines = json.loads(pathlib.Path(args.baselines).read_text())

    # Warm the PX4 build once, here rather than in _available(), which stays a pure predicate, and
    # outside --timeout, which budgets each example's subprocess: a px4 example's own launch re-runs
    # `make px4_sitl <airframe>` and must reach the sim inside its 30 s preroll window, GH #39; an
    # incremental no-op fits, a cold build never does. Same placement benchmark_matrix.py uses.
    if any(EXAMPLES[n].get("requires") == "px4" for n in names) and _available("px4", args)[0]:
        from nexus._src.vehicle.controllers.px4.sitl import build_px4_sitl

        print("pre-building PX4 SITL (one-time, outside any example budget)...", flush=True)
        build_px4_sitl()

    failed_runs: list[str] = []
    scored: dict[str, dict] = {}
    metas: dict[str, dict] = {}
    for name in names:
        ok, why = _available(EXAMPLES[name].get("requires"), args)
        if not ok:
            if args.skip_missing:
                print(f"\n===== {name} (skipped: {why}) =====", flush=True)
                continue
            print(f"\n===== {name}: requirement unmet: {why} =====", flush=True)
            failed_runs.append(name)
            continue
        extra = ["--policy", args.policy] if EXAMPLES[name].get("requires") == "policy" else []
        dump = _dump_dir(name, EXAMPLES[name], out)
        if not _run_example(name, EXAMPLES[name], dump, args.timeout, extra):
            failed_runs.append(name)
            continue
        scored[name], metas[name] = _score(EXAMPLES[name].get("launcher", name), dump)
        # Recordings otherwise live only in the cache under the home directory and die with the ephemeral runner
        # unless --upload runs; a copy here rides the GitHub artifact too.
        rrd = metas[name].get("rrd")
        if rrd and pathlib.Path(rrd).is_file():
            shutil.copy2(rrd, out / f"{name}.rrd")

    gate_failures: list[str] = []
    for name, metrics in scored.items():
        gate_failures += _gate(name, metrics, baselines)

    entries = _bench_entries(scored)
    (out / "benchmark.json").write_text(json.dumps(_merged_local(out / "benchmark.json", entries), indent=2))
    # The docs-data twin, docs/data/examples_bench.json via the data-refresh PR: the same entries,
    # self-describing with the hardware the numbers belong to. _meta refreshes on every write.
    meta = {"sha": _sha()[:12], "recorded": time.strftime("%Y-%m-%d"), "gpu": _gpu_name()}
    (out / "examples_bench.json").write_text(
        json.dumps({"_meta": meta, "entries": _merged_local(out / "examples_bench.json", entries)}, indent=2) + "\n"
    )

    if args.update_baselines:
        for name, metrics in scored.items():
            for key, rule in baselines.get(name, {}).items():
                if isinstance(rule, dict) and "value" in rule and key in metrics:
                    rule["value"] = round(float(metrics[key]), 4)
        pathlib.Path(args.baselines).write_text(json.dumps(baselines, indent=2) + "\n")
        shutil.copy2(args.baselines, out / "examples_baselines.json")  # rides the artifact off the box
        print(f"\nbaselines updated: {args.baselines}")

    if args.upload:
        _upload(out, metas)

    if failed_runs:
        print(f"\nFAILED examples: {', '.join(failed_runs)}", flush=True)
    if gate_failures:
        print(f"\nREGRESSED: {len(gate_failures)} gate(s):\n  " + "\n  ".join(gate_failures), flush=True)
    if not failed_runs and not gate_failures:
        print("\nAll examples passed their gates.", flush=True)
    return 1 if (failed_runs or gate_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
