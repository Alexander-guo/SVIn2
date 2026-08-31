#!/usr/bin/env python3

"""Summarize pose_graph loop_closure_funnel.csv without external dependencies."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import median


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def parse_float(value: str):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def format_rate(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def print_stage(name, entered, passed):
    rejected = entered - passed
    print(f"{name:28} entered={entered:6d}  passed={passed:6d}  "
          f"rejected={rejected:6d}  pass_rate={format_rate(passed, entered):>7}")


def print_distribution(name, values):
    if not values:
        print(f"{name:28} no finite values")
        return
    print(f"{name:28} min={min(values):.3f}  p25={percentile(values, 0.25):.3f}  "
          f"median={median(values):.3f}  p75={percentile(values, 0.75):.3f}  max={max(values):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="loop_closure_funnel.csv or the debug_output directory containing it",
    )
    args = parser.parse_args()

    csv_path = args.input / "loop_closure_funnel.csv" if args.input.is_dir() else args.input
    debug_output_path = args.input if args.input.is_dir() else csv_path.parent
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    dbow_path = debug_output_path / "loop_closure_dbow_funnel.csv"
    if dbow_path.exists():
        with dbow_path.open(newline="", encoding="utf-8") as stream:
            dbow_rows = list(csv.DictReader(stream))
        dbow_reasons = Counter(row["decision"] for row in dbow_rows)
        post_warmup = len(dbow_rows) - dbow_reasons["warmup"]
        selected = dbow_reasons["candidate_selected"]
        print(f"DBoW input: {dbow_path}")
        print("\nDBoW funnel")
        print_stage("Queried keyframes", len(dbow_rows), post_warmup)
        print_stage("Post-warmup selection", post_warmup, selected)
        print("\nDBoW terminal outcomes")
        for reason, count in dbow_reasons.most_common():
            print(f"{reason:28} {count:6d}  {format_rate(count, len(dbow_rows)):>7}")

    total = len(rows)
    vio_eligible = sum(row["rejection_reason"] != "old_keyframe_not_vio" for row in rows)
    pnp_attempted = sum(parse_bool(row["pnp_attempted"]) for row in rows)
    pnp_succeeded = sum(parse_bool(row["pnp_solver_succeeded"]) for row in rows)
    pnp_exceptions = sum(parse_bool(row["pnp_exception"]) for row in rows)
    inlier_gate_passed = sum(
        int(row["pnp_inliers"]) > int(row["min_correspondences"]) for row in rows
    )
    yaw_gate_passed = sum(parse_bool(row["yaw_gate_passed"]) for row in rows)
    position_gate_passed = sum(parse_bool(row["position_gate_passed"]) for row in rows)
    accepted = sum(parse_bool(row["accepted"]) for row in rows)

    print(f"\nCandidate input: {csv_path}")
    print("\nPost-DBoW candidate funnel")
    print_stage("Selected-candidate records", total, vio_eligible)
    print_stage("Descriptor-match gate", vio_eligible, pnp_attempted)
    print_stage("PnP solver status (info)", pnp_attempted, pnp_succeeded)
    print(f"{'PnP exceptions (info)':28} count={pnp_exceptions:6d}  "
          f"rate={format_rate(pnp_exceptions, pnp_attempted):>7}")
    print_stage("PnP inlier gate", pnp_attempted, inlier_gate_passed)
    print_stage("Yaw gate", inlier_gate_passed, yaw_gate_passed)
    print_stage("Position gate", inlier_gate_passed, position_gate_passed)
    print_stage("Both pose gates / accepted", inlier_gate_passed, accepted)

    print("\nTerminal outcomes")
    reasons = Counter(row["rejection_reason"] for row in rows)
    for reason, count in reasons.most_common():
        print(f"{reason:28} {count:6d}  {format_rate(count, total):>7}")

    print("\nValue distributions")
    print_distribution("Descriptor matches", [float(row["descriptor_matches"]) for row in rows])
    print_distribution("PnP inliers", [float(row["pnp_inliers"]) for row in rows if parse_bool(row["pnp_attempted"])])
    print_distribution(
        "Tested relative yaw [deg]",
        [value for row in rows if (value := parse_float(row["relative_yaw_deg"])) is not None],
    )
    print_distribution(
        "Tested translation norm [m]",
        [value for row in rows if (value := parse_float(row["relative_translation_m"])) is not None],
    )


if __name__ == "__main__":
    main()
