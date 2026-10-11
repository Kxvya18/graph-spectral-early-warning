from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamic_gsews import compute_dynamic_scores
from graphs import eigensystem, laplacian, ring_graph


class DynamicGsewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adjacency = ring_graph(12)
        self.eigenvalues, self.eigenvectors = eigensystem(laplacian(self.adjacency))

    def simulate_modal_var(self, seed: int = 4) -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(seed)
        critical = 5
        rho = np.full(12, 0.25)
        rho[critical] = 0.94
        y = np.zeros((900, 12))
        for index in range(1, len(y)):
            y[index] = rho * y[index - 1] + rng.normal(scale=0.25, size=12)
        return y @ self.eigenvectors.T, critical

    def test_graph_constrained_covariance_lag_recovers_slow_mode(self) -> None:
        x, critical = self.simulate_modal_var()
        scores = compute_dynamic_scores(
            x, self.eigenvectors, 160, 20, 400,
            ridge_fraction=0.02, trailing_smooth=2, trend_span=6,
        )
        recovery = np.mean(scores.dominant_mode == critical)
        self.assertGreater(recovery, 0.80)
        truth = np.abs(self.eigenvectors[:, critical])
        estimated = np.abs(scores.generalized_vector[-1])
        self.assertGreater(np.corrcoef(truth, estimated)[0, 1], 0.80)

    def test_true_graph_concentrates_slow_direction_more_than_identity(self) -> None:
        x, _ = self.simulate_modal_var()
        true = compute_dynamic_scores(x, self.eigenvectors, 160, 20, 400, ridge_fraction=0.02)
        identity = compute_dynamic_scores(
            x, np.eye(x.shape[1]), 160, 20, 400,
            ridge_fraction=0.02, exclude_constant=False,
        )
        self.assertGreater(
            float(np.mean(true.graph_concentration)),
            float(np.mean(identity.graph_concentration)) + 0.20,
        )

    def test_dynamic_scores_do_not_use_future_samples(self) -> None:
        x, _ = self.simulate_modal_var()
        first = compute_dynamic_scores(x, self.eigenvectors, 160, 20, 400)
        changed = x.copy()
        changed[620:] += np.random.default_rng(12).normal(scale=100.0, size=changed[620:].shape)
        second = compute_dynamic_scores(changed, self.eigenvectors, 160, 20, 400)
        eligible = first.ends <= 620
        for method in first.scores:
            np.testing.assert_allclose(first.scores[method][eligible], second.scores[method][eligible])
        np.testing.assert_allclose(
            first.generalized_vector[eligible], second.generalized_vector[eligible]
        )


if __name__ == "__main__":
    unittest.main()
