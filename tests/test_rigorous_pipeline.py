from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from causal_gsews import compute_causal_scores, trailing_matrix
from graphs import eigensystem, laplacian, ring_graph
from rigorous_evaluate import evaluate_causal_run
from rigorous_simulate import simulate_graph_commuting_ou, spectral_drift_schedule


class RigorousPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adjacency = ring_graph(12)
        self.evals, self.evecs = eigensystem(laplacian(self.adjacency))

    def test_spectral_schedule_crosses_at_declared_transition(self) -> None:
        alpha, critical = spectral_drift_schedule(self.evals, 500, 400)
        self.assertGreater(critical, 0)
        self.assertLess(np.max(alpha[399, 1:]), 0.0)
        self.assertAlmostEqual(alpha[400, critical], 0.0, places=12)
        self.assertGreater(alpha[401, critical], 0.0)

    def test_trailing_smoothing_never_uses_future_values(self) -> None:
        values = np.arange(12, dtype=float)[:, None]
        original = trailing_matrix(values, 3)
        changed = values.copy()
        changed[7:] = 10000.0
        revised = trailing_matrix(changed, 3)
        np.testing.assert_allclose(original[:7], revised[:7])

    def test_scores_before_cutoff_are_invariant_to_future_data(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(500, 12))
        first = compute_causal_scores(x, self.evecs, self.adjacency, 80, 20, 220, 3, 6)
        changed = x.copy()
        changed[320:] += 1000.0 * rng.normal(size=changed[320:].shape)
        second = compute_causal_scores(changed, self.evecs, self.adjacency, 80, 20, 220, 3, 6)
        eligible = first.ends <= 320
        for method in first.scores:
            np.testing.assert_allclose(first.scores[method][eligible], second.scores[method][eligible])

    def test_evaluation_uses_only_complete_pretransition_windows(self) -> None:
        rng = np.random.default_rng(11)
        sim = simulate_graph_commuting_ou(
            self.evals, self.evecs, 600, 0.02, 480, 0.2, rng
        )
        scores = compute_causal_scores(
            sim.x, self.evecs, self.adjacency, 80, 20, 220, 3, 6
        )
        rows = evaluate_causal_run(
            "test", "ring", sim.model, 11, scores, 480, 220, 340, 440,
            sim.critical_mode, self.evecs, self.evals, 40,
        )
        expected = int(np.sum((scores.ends > 340) & (scores.ends <= 440)))
        self.assertGreater(expected, 0)
        self.assertTrue(all(row["n_positive_windows"] == expected for row in rows))
        self.assertTrue(np.all(scores.ends[(scores.ends > 340) & (scores.ends <= 440)] <= 440))

    def test_equal_node_noise_does_not_privilege_target_at_first_step(self) -> None:
        # Across many one-step simulations, modal innovation variances should be
        # approximately equal under orthonormal rotation of isotropic node noise.
        increments = []
        for seed in range(500):
            sim = simulate_graph_commuting_ou(
                self.evals, self.evecs, 2, 0.02, 1, 0.2,
                np.random.default_rng(seed),
            )
            increments.append(sim.x[1] @ self.evecs)
        variances = np.var(np.asarray(increments), axis=0)
        self.assertLess(float(variances.max() / variances.min()), 1.45)


if __name__ == "__main__":
    unittest.main()
