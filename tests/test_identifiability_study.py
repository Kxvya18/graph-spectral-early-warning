from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from graphs import adjacency_matrix, eigensystem, laplacian
from run_identifiability_study import (
    fit_diagonal_dynamics,
    modal_dynamics,
    population_metrics,
    simulate_stationary_modal,
)
from run_rigorous_synthetic import weighted_adjacency


class IdentifiabilityStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(31)
        self.adjacency = weighted_adjacency(adjacency_matrix("small_world", 16, rng), rng)
        self.eigenvalues, self.eigenvectors = eigensystem(laplacian(self.adjacency))

    def test_population_true_basis_has_zero_operator_residual(self) -> None:
        rho, _ = modal_dynamics(self.eigenvalues, 0.10)
        rows = population_metrics(
            self.eigenvectors, rho,
            [("TrueGraph", self.eigenvectors), ("IdentityBasis", np.eye(16))],
        )
        true, identity = rows
        self.assertLess(true["operator_residual_ratio"], 1e-24)
        self.assertGreater(identity["operator_residual_ratio"], 1e-5)
        self.assertLess(true["expected_test_mse"], identity["expected_test_mse"])

    def test_separated_mode_is_recoverable_with_long_training_data(self) -> None:
        rho, critical = modal_dynamics(self.eigenvalues, 0.15)
        train, test = simulate_stationary_modal(
            rho, 500, 4000, 1000, 0.0, self.eigenvectors,
            np.random.default_rng(77),
        )
        fitted = fit_diagonal_dynamics(
            train, test, self.eigenvectors, ridge_fraction=0.001,
            exclude_constant=True,
        )
        self.assertEqual(fitted["dominant_mode"], critical)


if __name__ == "__main__":
    unittest.main()
