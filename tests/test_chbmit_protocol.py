from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from run_chbmit_locked import (
    RecordFeatures,
    build_context_lookup,
    holm_adjust,
    paired_sign_flip_pvalue,
    preictal_overlaps_prior_seizure,
    run_fold,
)


class LockedChbmitProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = json.loads((PROJECT / "configs/chbmit_locked_protocol.json").read_text())

    def test_context_lookup_maps_target_to_predecessor(self) -> None:
        manifest = [
            {
                "subject": "chb01",
                "record": "chb01_03",
                "context_for": ["chb01/chb01_04.edf"],
            }
        ]
        self.assertEqual(
            build_context_lookup(manifest)[("chb01", "chb01_04")], "chb01_03"
        )

    def test_prior_seizure_inside_preictal_interval_is_excluded(self) -> None:
        self.assertTrue(
            preictal_overlaps_prior_seizure(
                target_onset=5000.0,
                all_seizures=[(2990.0, 3040.0), (5000.0, 5040.0)],
                cfg=self.cfg,
            )
        )
        self.assertFalse(
            preictal_overlaps_prior_seizure(
                target_onset=5000.0,
                all_seizures=[(1000.0, 1040.0), (5000.0, 5040.0)],
                cfg=self.cfg,
            )
        )

    def test_paired_inference_is_deterministic_and_holm_monotone(self) -> None:
        difference = np.asarray([0.10, 0.20, 0.15, 0.30])
        first = paired_sign_flip_pvalue(difference, np.random.default_rng(7))
        second = paired_sign_flip_pvalue(difference, np.random.default_rng(8))
        self.assertEqual(first, second)  # exact enumeration for small cohorts
        raw = np.asarray([0.01, 0.04, 0.03])
        adjusted = holm_adjust(raw)
        self.assertTrue(np.all(adjusted >= raw))
        ordered = adjusted[np.argsort(raw)]
        self.assertTrue(np.all(np.diff(ordered) >= 0))

    def test_locked_fold_is_end_to_end_deterministic(self) -> None:
        cfg = dict(self.cfg)
        cfg.update(
            {
                "window_epochs": 20,
                "stride_epochs": 5,
                "trend_span_windows": 4,
                "trailing_smooth_windows": 2,
                "seizure_prediction_horizon_seconds": 20,
                "seizure_occurrence_period_seconds": 100,
                "graph_top_k": 3,
            }
        )
        channels = [f"C{i:02d}" for i in range(16)]

        def record(name: str, seed: int, n: int, kind: str,
                   seizures: list[tuple[float, float]]) -> RecordFeatures:
            rng = np.random.default_rng(seed)
            common = rng.normal(size=(n, 1))
            features = 0.25 * common + rng.normal(size=(n, len(channels)))
            return RecordFeatures("chb99", name, kind, channels, features, 2.0, seizures)

        context = record("chb99_01", 1, 180, "context", [])
        seizure = record("chb99_02", 2, 300, "seizure", [(300.0, 330.0)])
        held_non = record("chb99_03", 3, 300, "nonseizure", [])
        training = [
            record("chb99_04", 4, 300, "nonseizure", []),
            record("chb99_05", 5, 300, "nonseizure", []),
        ]
        first = run_fold("chb99", 0, seizure, context, held_non, training, channels, cfg)
        second = run_fold("chb99", 0, seizure, context, held_non, training, channels, cfg)
        self.assertGreater(len(first), 0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
