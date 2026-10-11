from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from run_chbmit_v3_ml import fit_multi_record_dynamics


class ChbmitVersionThreeTests(unittest.TestCase):
    def test_dynamics_fit_ignores_only_nonfinite_adjacent_epochs(self) -> None:
        rng = np.random.default_rng(42)
        angle = np.pi / 4
        basis = np.asarray(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )

        def simulate(length: int) -> np.ndarray:
            modal = np.zeros((length, 2))
            for index in range(1, length):
                modal[index] = (
                    modal[index - 1] * np.asarray([0.92, 0.15])
                    + rng.normal(scale=0.4, size=2)
                )
            values = modal @ basis.T
            values[:5] = np.nan
            return values

        train = [simulate(2500), simulate(2500)]
        validation = simulate(1200)
        fitted = fit_multi_record_dynamics(
            train, validation, basis, ridge_fraction=0.001,
            exclude_constant=False,
        )
        identity = fit_multi_record_dynamics(
            train, validation, np.eye(2), ridge_fraction=0.001,
            exclude_constant=False,
        )
        self.assertTrue(np.isfinite(fitted["validation_mse"]))
        self.assertGreater(fitted["estimated_max_rho"], 0.85)
        self.assertLess(fitted["validation_mse"], identity["validation_mse"])

    def test_dynamics_fit_rejects_all_nonfinite_training_data(self) -> None:
        values = np.full((20, 3), np.nan)
        with self.assertRaises(RuntimeError):
            fit_multi_record_dynamics(
                [values], np.zeros((20, 3)), np.eye(3),
                ridge_fraction=0.01, exclude_constant=False,
            )


if __name__ == "__main__":
    unittest.main()
