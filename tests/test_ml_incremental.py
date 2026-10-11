from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ml_incremental import (
    fit_grouped_elastic_net,
    grouped_out_of_fold_probabilities,
)


class IncrementalMlTests(unittest.TestCase):
    def test_grouped_elastic_net_is_deterministic_and_predictive(self) -> None:
        rng = np.random.default_rng(14)
        rows, labels, groups = [], [], []
        for group_index in range(6):
            label = group_index % 2
            values = rng.normal(size=(40, 4))
            values[:, 0] += 1.5 * label
            rows.append(values)
            labels.extend([label] * len(values))
            groups.extend([f"record_{group_index}"] * len(values))
        x = np.vstack(rows)
        y = np.asarray(labels)
        groups = np.asarray(groups)
        candidates = [
            {"C": 0.1, "l1_ratio": 0.0},
            {"C": 1.0, "l1_ratio": 0.5},
        ]
        first = fit_grouped_elastic_net(x, y, groups, candidates, seed=22)
        second = fit_grouped_elastic_net(x, y, groups, candidates, seed=22)
        np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))
        self.assertGreater(first.inner_auc, 0.70)

    def test_groups_cannot_be_replaced_by_single_window_groups(self) -> None:
        x = np.random.default_rng(2).normal(size=(20, 3))
        y = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
        groups = np.asarray(["negative"] * 10 + ["positive"] * 10)
        with self.assertRaises(ValueError):
            fit_grouped_elastic_net(
                x, y, groups, [{"C": 1.0, "l1_ratio": 0.5}], seed=1
            )

    def test_oof_probabilities_hold_out_complete_records(self) -> None:
        rng = np.random.default_rng(31)
        x = rng.normal(size=(80, 3))
        y = np.repeat([0, 1, 0, 1], 20)
        groups = np.repeat(["n1", "p1", "n2", "p2"], 20)
        x[:, 0] += y
        probability = grouped_out_of_fold_probabilities(
            x, y, groups, C=0.1, l1_ratio=0.0, seed=9
        )
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))


if __name__ == "__main__":
    unittest.main()
