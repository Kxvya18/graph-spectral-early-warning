from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dynamic_gsews import compute_dynamic_scores
from graphs import eigensystem, laplacian, ring_graph
from v3_simulate import discrete_spectral_schedule, simulate_v3


class VersionThreePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adjacency = ring_graph(16)
        self.eigenvalues, self.eigenvectors = eigensystem(laplacian(self.adjacency))

    def test_isolated_schedule_has_declared_gap_and_transition(self) -> None:
        rho, critical = discrete_spectral_schedule(
            self.eigenvalues, 501, 200, 500, "isolated_mode",
            isolated_baseline_gap=0.15,
        )
        competitor = np.max(np.delete(rho[0], critical))
        self.assertGreaterEqual(rho[0, critical] - competitor, 0.149999)
        self.assertLess(np.max(rho[499, 1:]), 1.0)
        self.assertAlmostEqual(rho[500, critical], 1.0)

    def test_v3_simulation_is_reproducible(self) -> None:
        kwargs = dict(
            eigenvalues=self.eigenvalues, eigenvectors=self.eigenvectors,
            n_steps=501, ramp_start=200, transition_step=500,
            regime="near_degenerate_band", model="nonlinear_graph_map",
            noise_sigma=0.1,
        )
        first = simulate_v3(**kwargs, rng=np.random.default_rng(10))
        second = simulate_v3(**kwargs, rng=np.random.default_rng(10))
        np.testing.assert_allclose(first.x, second.x)

    def test_control_scoring_skips_graph_independent_eigendecomposition(self) -> None:
        x = np.random.default_rng(5).normal(size=(500, 16))
        scores = compute_dynamic_scores(
            x, self.eigenvectors, 80, 20, 220,
            include_generalized=False,
        )
        self.assertNotIn("SlowMode_Level", scores.scores)
        self.assertNotIn("GraphAligned_Level", scores.scores)
        self.assertTrue(np.all(np.isnan(scores.generalized_persistence)))


if __name__ == "__main__":
    unittest.main()
