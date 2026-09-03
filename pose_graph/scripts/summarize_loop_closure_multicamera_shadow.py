#!/usr/bin/env python3

"""Summarize ordered-camera loop-closure shadow diagnostics.

The input is the ``loop_closure_multicamera_shadow.csv`` file written by
``pose_graph`` or its containing debug-output directory.  A row represents a
retrieved candidate, so query rows are counted as rows rather than deduplicated
keyframe IDs. Scores summarize all retrieved rows; descriptor matches summarize
temporally eligible shadow-verification rows; and PnP inliers summarize rows
that passed the descriptor gate. This avoids zero-filled, not-attempted rows
skewing the verification distributions while keeping the legacy score decision
as an independent measurement. Optional pose-gate and terminal-decision
booleans are counted when present; older 17-column files simply report zero
true values for those optional fields.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Optional


def parse_bool(value: object) -> bool:
    """Return the boolean representation used by the C++ CSV writer."""

    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_number(value: object) -> Optional[float]:
    """Parse a finite CSV number, returning ``None`` for missing values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _present(row: Mapping[str, str], name: str) -> bool:
    value = row.get(name)
    return value is not None and value.strip() != ""


def _camera_name(value: object) -> str:
    """Normalize numeric camera IDs while allowing descriptive IDs in logs."""

    if value is None:
        raise ValueError("camera ID must not be missing")
    text = str(value).strip()
    if not text:
        raise ValueError("camera ID must not be empty")
    try:
        return str(int(text))
    except ValueError:
        return text


def _camera_sort_key(camera: str):
    try:
        return (0, int(camera))
    except ValueError:
        return (1, camera)


def _score_passed(row: Mapping[str, str]) -> bool:
    """Use the recorded score decision, with a score-threshold fallback."""

    if _present(row, "score_passed"):
        return parse_bool(row["score_passed"])
    score = parse_number(row.get("dbow_score"))
    threshold = parse_number(row.get("score_threshold"))
    return score is not None and threshold is not None and score > threshold


def _median_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return float(median(values)) if values else None


@dataclass
class PairSummary:
    """Stage counts and distributions for one ordered camera pair."""

    query_rows: int = 0
    score_passing_candidates: int = 0
    verification_attempts: int = 0
    descriptor_gate_passes: int = 0
    pnp_solver_successes: int = 0
    pnp_inlier_gate_passes: int = 0
    yaw_gate_passes: int = 0
    position_gate_passes: int = 0
    accepted: int = 0
    selected: int = 0
    _scores: list[float] = field(default_factory=list, repr=False)
    _matches: list[float] = field(default_factory=list, repr=False)
    _inliers: list[float] = field(default_factory=list, repr=False)

    @property
    def median_score(self) -> Optional[float]:
        return _median_or_none(self._scores)

    @property
    def median_matches(self) -> Optional[float]:
        return _median_or_none(self._matches)

    @property
    def median_inliers(self) -> Optional[float]:
        return _median_or_none(self._inliers)


def summarize_rows(rows: Iterable[Mapping[str, str]], min_correspondences: int = 20):
    """Return ``(current_camera, historical_camera)`` summaries.

    Both correspondence gates use the strict comparison from the pose-graph
    implementation: a value passes only when it is greater than
    ``min_correspondences``.  PnP solver success is counted after the
    descriptor gate.  The inlier gate is counted directly from the recorded
    inlier count after that gate; normally a failed solver has zero inliers.
    """

    if min_correspondences < 0:
        raise ValueError("min_correspondences must be non-negative")

    summaries: dict[tuple[str, str], PairSummary] = defaultdict(PairSummary)
    for row in rows:
        try:
            pair = (_camera_name(row.get("current_camera")),
                    _camera_name(row.get("historical_camera")))
        except ValueError as error:
            raise ValueError(f"invalid camera pair in CSV row: {error}") from error

        summary = summaries[pair]
        summary.query_rows += 1

        score = parse_number(row.get("dbow_score"))
        if score is not None:
            summary._scores.append(score)

        score_passed = _score_passed(row)
        if score_passed:
            summary.score_passing_candidates += 1

        # These columns were added after the original 17-column CSV schema.
        # DictReader returns None for an absent column, and parse_bool treats
        # that as false, preserving compatibility with older files.
        if parse_bool(row.get("yaw_gate_passed", "")):
            summary.yaw_gate_passes += 1
        if parse_bool(row.get("position_gate_passed", "")):
            summary.position_gate_passes += 1
        if parse_bool(row.get("accepted", "")):
            summary.accepted += 1
        if parse_bool(row.get("selected", "")):
            summary.selected += 1

        verification_attempted = (parse_bool(row["verification_attempted"])
                                  if _present(row, "verification_attempted")
                                  else score_passed)
        if not verification_attempted:
            continue
        summary.verification_attempts += 1

        matches = parse_number(row.get("descriptor_matches"))
        if matches is not None:
            summary._matches.append(matches)
        descriptor_gate_passed = matches is not None and matches > min_correspondences
        if not descriptor_gate_passed:
            continue
        summary.descriptor_gate_passes += 1

        if parse_bool(row.get("pnp_solver_succeeded", "")):
            summary.pnp_solver_successes += 1

        inliers = parse_number(row.get("pnp_inliers"))
        if inliers is not None:
            summary._inliers.append(inliers)
        if inliers is not None and inliers > min_correspondences:
            summary.pnp_inlier_gate_passes += 1

    return dict(sorted(summaries.items(),
                       key=lambda item: (_camera_sort_key(item[0][0]),
                                         _camera_sort_key(item[0][1]))))


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read and minimally validate a shadow CSV."""

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        required = {"current_camera", "historical_camera"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))
        return list(reader)


def resolve_csv_path(input_path: Path) -> Path:
    """Resolve either a CSV path or a pose-graph debug-output directory."""

    csv_path = input_path / "loop_closure_multicamera_shadow.csv" if input_path.is_dir() else input_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"shadow CSV not found: {csv_path}")
    return csv_path


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def combine_summaries(summaries, same_camera: bool) -> PairSummary:
    """Combine pair summaries into same-camera or cross-camera totals."""

    combined = PairSummary()
    for (current_camera, historical_camera), summary in summaries.items():
        if (current_camera == historical_camera) != same_camera:
            continue
        combined.query_rows += summary.query_rows
        combined.score_passing_candidates += summary.score_passing_candidates
        combined.verification_attempts += summary.verification_attempts
        combined.descriptor_gate_passes += summary.descriptor_gate_passes
        combined.pnp_solver_successes += summary.pnp_solver_successes
        combined.pnp_inlier_gate_passes += summary.pnp_inlier_gate_passes
        combined.yaw_gate_passes += summary.yaw_gate_passes
        combined.position_gate_passes += summary.position_gate_passes
        combined.accepted += summary.accepted
        combined.selected += summary.selected
        combined._scores.extend(summary._scores)
        combined._matches.extend(summary._matches)
        combined._inliers.extend(summary._inliers)
    return combined


def _summary_metrics(label: str, summary: PairSummary) -> str:
    return (
        f"{label}: "
        f"query_rows={summary.query_rows} "
        f"score_passing_candidates={summary.score_passing_candidates} "
        f"verification_attempts={summary.verification_attempts} "
        f"descriptor_gate_passes={summary.descriptor_gate_passes} "
        f"pnp_solver_successes={summary.pnp_solver_successes} "
        f"pnp_inlier_gate_passes={summary.pnp_inlier_gate_passes} "
        f"yaw_gate_passes={summary.yaw_gate_passes} "
        f"position_gate_passes={summary.position_gate_passes} "
        f"accepted={summary.accepted} "
        f"selected={summary.selected} "
        f"median_score={_format_metric(summary.median_score)} "
        f"median_matches={_format_metric(summary.median_matches)} "
        f"median_inliers={_format_metric(summary.median_inliers)}"
    )


def format_summary(csv_path: Path, summaries, min_correspondences: int) -> str:
    """Format a deterministic, grep-friendly human-readable report."""

    lines = [
        f"Input: {csv_path}",
        f"Correspondence gates: value > {min_correspondences}",
    ]
    if not summaries:
        lines.append("No diagnostic rows.")
        return "\n".join(lines)

    lines.append(_summary_metrics("Class same-camera", combine_summaries(summaries, True)))
    lines.append(_summary_metrics("Class cross-camera", combine_summaries(summaries, False)))
    for (current_camera, historical_camera), summary in summaries.items():
        lines.append(_summary_metrics(f"Pair {current_camera}->{historical_camera}", summary))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="loop_closure_multicamera_shadow.csv or its debug-output directory",
    )
    parser.add_argument(
        "--min-correspondences",
        "--min_correspondences",
        dest="min_correspondences",
        type=int,
        default=20,
        metavar="N",
        help="strict descriptor/inlier gate threshold (default: 20)",
    )
    args = parser.parse_args(argv)
    try:
        csv_path = resolve_csv_path(args.input)
        summaries = summarize_rows(read_rows(csv_path), args.min_correspondences)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(format_summary(csv_path, summaries, args.min_correspondences))
    return 0


if __name__ == "__main__":
    main()
