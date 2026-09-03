#!/usr/bin/env python3

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import summarize_loop_closure_multicamera_shadow as summarizer  # noqa: E402


FIELDNAMES = [
    "current_kf_id",
    "current_timestamp",
    "current_camera",
    "historical_camera",
    "rank",
    "candidate_kf_id",
    "dbow_score",
    "score_threshold",
    "score_passed",
    "verification_attempted",
    "tracked_points",
    "descriptor_matches",
    "pnp_solver_succeeded",
    "pnp_exception",
    "pnp_inliers",
    "relative_yaw_deg",
    "relative_translation_m",
]

OPTIONAL_FIELDNAMES = [
    "yaw_gate_passed",
    "position_gate_passed",
    "accepted",
    "selected",
]

EXTENDED_FIELDNAMES = FIELDNAMES + OPTIONAL_FIELDNAMES


def row(current_camera, historical_camera, score, score_passed,
        matches, solver_succeeded, inliers, verification_attempted=True,
        gate_flags=None):
    values = {
        "current_kf_id": "100",
        "current_timestamp": "1000",
        "current_camera": str(current_camera),
        "historical_camera": str(historical_camera),
        "rank": "1",
        "candidate_kf_id": "10",
        "dbow_score": str(score),
        "score_threshold": "0.3",
        "score_passed": str(int(score_passed)),
        "verification_attempted": str(int(verification_attempted)),
        "tracked_points": str(matches),
        "descriptor_matches": str(matches),
        "pnp_solver_succeeded": str(int(solver_succeeded)),
        "pnp_exception": "0",
        "pnp_inliers": str(inliers),
        "relative_yaw_deg": "nan",
        "relative_translation_m": "nan",
    }
    if gate_flags is not None:
        values.update({name: str(int(gate_flags.get(name, False)))
                       for name in OPTIONAL_FIELDNAMES})
    return values


class MulticameraShadowSummaryTest(unittest.TestCase):
    def test_groups_ordered_pairs_and_applies_strict_gates(self):
        rows = [
            row(0, 0, 0.20, False, 0, False, 0, verification_attempted=False),
            row(1, 0, 0.80, True, 21, True, 25,
                gate_flags={"yaw_gate_passed": True, "position_gate_passed": True,
                            "accepted": True, "selected": True}),
            row(1, 0, 0.40, True, 20, True, 21,
                gate_flags={"yaw_gate_passed": False, "position_gate_passed": True}),
            row(1, 0, 0.10, False, 99, True, 99,
                gate_flags={"yaw_gate_passed": True}),
            row(0, 1, 0.70, True, 25, False, 19,
                gate_flags={"yaw_gate_passed": True, "accepted": False}),
            row(1, 1, 0.30, False, 0, False, 0,
                gate_flags={"position_gate_passed": True}),
        ]

        summaries = summarizer.summarize_rows(rows, min_correspondences=20)

        self.assertEqual(list(summaries), [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")])
        reverse = summaries[("1", "0")]
        self.assertEqual(reverse.query_rows, 3)
        self.assertEqual(reverse.score_passing_candidates, 2)
        self.assertEqual(reverse.verification_attempts, 3)
        self.assertEqual(reverse.descriptor_gate_passes, 2)
        self.assertEqual(reverse.pnp_solver_successes, 2)
        self.assertEqual(reverse.pnp_inlier_gate_passes, 2)
        self.assertEqual(reverse.yaw_gate_passes, 2)
        self.assertEqual(reverse.position_gate_passes, 2)
        self.assertEqual(reverse.accepted, 1)
        self.assertEqual(reverse.selected, 1)
        self.assertAlmostEqual(reverse.median_score, 0.40)
        self.assertAlmostEqual(reverse.median_matches, 21.0)
        self.assertAlmostEqual(reverse.median_inliers, 62.0)

        forward = summaries[("0", "1")]
        self.assertEqual(forward.descriptor_gate_passes, 1)
        self.assertEqual(forward.pnp_solver_successes, 0)
        self.assertEqual(forward.pnp_inlier_gate_passes, 0)
        self.assertEqual(forward.yaw_gate_passes, 1)
        self.assertEqual(forward.position_gate_passes, 0)
        self.assertEqual(forward.accepted, 0)
        self.assertEqual(forward.selected, 0)

        same_camera = summaries[("0", "0")]
        self.assertEqual(same_camera.query_rows, 1)
        self.assertEqual(same_camera.score_passing_candidates, 0)
        self.assertEqual(same_camera.yaw_gate_passes, 0)
        self.assertEqual(same_camera.position_gate_passes, 0)
        self.assertEqual(same_camera.accepted, 0)
        self.assertEqual(same_camera.selected, 0)
        self.assertIsNone(same_camera.median_matches)

        report = summarizer.format_summary(Path("synthetic.csv"), summaries, 20)
        self.assertIn("yaw_gate_passes=2", report)
        self.assertIn("position_gate_passes=2", report)
        self.assertIn("accepted=1", report)
        self.assertIn("selected=1", report)

    def test_directory_input_and_cli_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "loop_closure_multicamera_shadow.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerow(row(0, 0, 0.90, True, 21, True, 21))
                writer.writerow(row(0, 0, 0.80, True, 11, True, 21))

            output = io.StringIO()
            with redirect_stdout(output):
                result = summarizer.main([directory, "--min-correspondences", "10"])

        self.assertEqual(result, 0)
        report = output.getvalue()
        self.assertIn("Correspondence gates: value > 10", report)
        self.assertIn("Class same-camera: query_rows=2", report)
        self.assertIn("verification_attempts=2", report)
        self.assertIn("Class cross-camera: query_rows=0", report)
        self.assertIn("Pair 0->0:", report)
        self.assertIn("query_rows=2", report)
        self.assertIn("score_passing_candidates=2", report)
        self.assertIn("descriptor_gate_passes=2", report)
        self.assertIn("pnp_solver_successes=2", report)
        self.assertIn("pnp_inlier_gate_passes=2", report)
        self.assertIn("yaw_gate_passes=0", report)
        self.assertIn("position_gate_passes=0", report)
        self.assertIn("accepted=0", report)
        self.assertIn("selected=0", report)
        self.assertIn("median_score=0.850", report)
        self.assertIn("median_matches=16.000", report)
        self.assertIn("median_inliers=21.000", report)

    def test_extended_csv_reports_optional_pose_and_selection_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "loop_closure_multicamera_shadow.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=EXTENDED_FIELDNAMES)
                writer.writeheader()
                writer.writerow(row(
                    0, 1, 0.90, True, 21, True, 21,
                    gate_flags={"yaw_gate_passed": True,
                                "position_gate_passed": True,
                                "accepted": True,
                                "selected": True}))
                writer.writerow(row(
                    0, 1, 0.80, True, 21, True, 19,
                    gate_flags={"yaw_gate_passed": False,
                                "position_gate_passed": True,
                                "accepted": False,
                                "selected": False}))

            output = io.StringIO()
            with redirect_stdout(output):
                result = summarizer.main([str(csv_path)])

        self.assertEqual(result, 0)
        report = output.getvalue()
        self.assertIn("Pair 0->1:", report)
        self.assertIn("yaw_gate_passes=1", report)
        self.assertIn("position_gate_passes=2", report)
        self.assertIn("accepted=1", report)
        self.assertIn("selected=1", report)


if __name__ == "__main__":
    unittest.main()
