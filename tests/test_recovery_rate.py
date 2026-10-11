from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from graphs import eigensystem, laplacian, ring_graph
from recovery_rate import (
    alignment_excess,
    compute_recovery_trajectory,
    estimate_recovery,
    graph_alignment,
    phenotype_scores,
    realign_trajectory,
)


class RecoveryRateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adjacency = ring_graph(12)
        self.eigenvalues, self.eigenvectors = eigensystem(laplacian(self.adjacency))

    def simulate(self, seed: int = 7) -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(seed)
        critical = 5
        rho = np.full(12, 0.25)
        rho[critical] = 0.94
        modal = np.zeros((2400, 12))
        for index in range(1, len(modal)):
            modal[index] = rho * modal[index - 1] + rng.normal(scale=0.20, size=12)
        return modal @ self.eigenvectors.T, critical

    def test_population_like_recovery_direction_maps_to_state_space(self) -> None:
        x, critical = self.simulate()
        estimate = estimate_recovery(x, lag_steps=1, ridge_fraction=0.01)
        correlation = abs(float(estimate.direction @ self.eigenvectors[:, critical]))
        self.assertGreater(estimate.rho, 0.90)
        self.assertGreater(correlation, 0.90)

    def test_graph_alignment_recovers_exact_mode_and_rejects_constant(self) -> None:
        direction = self.eigenvectors[:, 4]
        aligned = graph_alignment(direction, self.eigenvectors, self.eigenvalues)
        self.assertEqual(aligned["dominant_graph_mode"], 4)
        self.assertAlmostEqual(aligned["concentration"], 1.0, places=8)
        constant = graph_alignment(self.eigenvectors[:, 0], self.eigenvectors, self.eigenvalues)
        self.assertLess(constant["concentration"], 1e-10)
        self.assertGreater(constant["constant_mass"], 0.999999)

    def test_trajectory_is_strictly_causal(self) -> None:
        x, _ = self.simulate()
        first = compute_recovery_trajectory(
            x, self.eigenvectors, self.eigenvalues, 160, 20,
            ridge_fraction=0.02, isolated_gap=0.05,
        )
        changed = x.copy()
        changed[1200:] += np.random.default_rng(91).normal(scale=100.0, size=changed[1200:].shape)
        second = compute_recovery_trajectory(
            changed, self.eigenvectors, self.eigenvalues, 160, 20,
            ridge_fraction=0.02, isolated_gap=0.05,
        )
        past = first.ends <= 1200
        np.testing.assert_allclose(first.rho[past], second.rho[past])
        np.testing.assert_allclose(first.direction[past], second.direction[past])

    def test_true_graph_phenotype_exceeds_identity_for_graph_mode(self) -> None:
        x, _ = self.simulate()
        true = compute_recovery_trajectory(
            x, self.eigenvectors, self.eigenvalues, 160, 20,
            ridge_fraction=0.02, isolated_gap=0.05,
        )
        identity = realign_trajectory(true, np.eye(12), np.arange(12, dtype=float))
        baseline = true.ends <= 900
        scores = phenotype_scores(true, [identity], baseline)
        self.assertEqual(scores["RecoveryPhenotype"].shape, true.rho.shape)
        self.assertGreater(float(np.mean(alignment_excess(true, [identity]))), 0.50)

    def test_negative_absolute_alignment_excess_cannot_create_phenotype(self) -> None:
        x, _ = self.simulate()
        true = compute_recovery_trajectory(
            x, np.eye(12), np.arange(12, dtype=float), 160, 20,
            ridge_fraction=0.02, isolated_gap=0.05, exclude_constant=False,
        )
        better = realign_trajectory(true, self.eigenvectors, self.eigenvalues)
        baseline = true.ends <= 900
        scores = phenotype_scores(true, [better], baseline)
        nonpositive = alignment_excess(true, [better]) <= 0.0
        self.assertTrue(np.all(scores["RecoveryPhenotype"][nonpositive] == 0.0))


if __name__ == "__main__":
    unittest.main()
