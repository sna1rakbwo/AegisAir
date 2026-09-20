"""Tests for the Phase 6 SharedStateEstimator and RA covariance projection."""

from __future__ import annotations

import unittest

import numpy as np

from swarm.estimation import (
    EstimatedState,
    SharedStateEstimator,
    SharedStateEstimatorConfig,
)
from swarm.ra.runtime_assurance import RuntimeAssurance, _projected_sigma
from swarm.safety import DroneSnapshot


def _snap(drone: int, position: tuple[float, float, float]) -> DroneSnapshot:
    return DroneSnapshot(
        drone_id=drone,
        position=position,
        velocity=(0.0, 0.0, 0.0),
        timestamp_ms=0,
    )


class SharedStateEstimatorTest(unittest.TestCase):
    def test_fresh_measurement_has_base_covariance(self) -> None:
        estimator = SharedStateEstimator()
        out = estimator.step({0: _snap(0, (1.0, 2.0, 0.0))}, now_ms=1000)
        state = out[0]
        self.assertEqual(state.position, (1.0, 2.0, 0.0))
        self.assertEqual(state.timestamp_ms, 1000)
        self.assertFalse(state.dropped)
        self.assertAlmostEqual(state.covariance[0], 0.01)

    def test_delay_outputs_older_state(self) -> None:
        estimator = SharedStateEstimator(
            SharedStateEstimatorConfig(delay_ms=100)
        )
        estimator.step({0: _snap(0, (0.0, 0.0, 0.0))}, now_ms=0)
        estimator.step({0: _snap(0, (1.0, 0.0, 0.0))}, now_ms=100)
        state = estimator.step(
            {0: _snap(0, (2.0, 0.0, 0.0))}, now_ms=200
        )[0]
        self.assertEqual(state.position, (1.0, 0.0, 0.0))
        self.assertEqual(state.timestamp_ms, 100)
        self.assertFalse(state.dropped)

    def test_dropout_holds_estimate_and_grows_covariance(self) -> None:
        estimator = SharedStateEstimator(
            SharedStateEstimatorConfig(
                dropout_rate=1.0,
                process_noise_m2_per_s=0.02,
            )
        )
        rng = np.random.default_rng(0)
        estimator.step({0: _snap(0, (0.0, 0.0, 0.0))}, now_ms=0, rng=rng)
        state = estimator.step(
            {0: _snap(0, (5.0, 0.0, 0.0))}, now_ms=1000, rng=rng
        )[0]
        self.assertEqual(state.position, (0.0, 0.0, 0.0))
        self.assertTrue(state.dropped)
        self.assertAlmostEqual(state.covariance[0], 0.03, places=6)

    def test_covariance_clamps_to_max(self) -> None:
        estimator = SharedStateEstimator(
            SharedStateEstimatorConfig(
                dropout_rate=1.0,
                process_noise_m2_per_s=10.0,
                max_cov_m2=0.5,
            )
        )
        rng = np.random.default_rng(0)
        estimator.step({0: _snap(0, (0.0, 0.0, 0.0))}, now_ms=0, rng=rng)
        state = estimator.step(
            {0: _snap(0, (1.0, 0.0, 0.0))}, now_ms=10000, rng=rng
        )[0]
        self.assertAlmostEqual(state.covariance[0], 0.5)


class CovarianceProjectionTest(unittest.TestCase):
    class _WithCov:
        def __init__(self, covariance) -> None:
            self.covariance = covariance

    def test_projects_covariance_onto_direction(self) -> None:
        snapshot = self._WithCov((0.04, 0.0, 0.0, 0.01))
        sigma_x = _projected_sigma(snapshot, np.array([1.0, 0.0]), 0.1)
        sigma_y = _projected_sigma(snapshot, np.array([0.0, 1.0]), 0.1)
        self.assertAlmostEqual(sigma_x, 0.2)
        self.assertAlmostEqual(sigma_y, 0.1)

    def test_missing_covariance_falls_back_to_default(self) -> None:
        snapshot = DroneSnapshot(
            drone_id=0, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)
        )
        sigma = _projected_sigma(snapshot, np.array([1.0, 0.0]), 0.1)
        self.assertAlmostEqual(sigma, 0.1)

    def test_invalid_covariance_cannot_remove_uncertainty_margin(self) -> None:
        zero = self._WithCov((0.0, 0.0, 0.0, 0.0))
        negative = self._WithCov((-1.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(_projected_sigma(zero, np.array([1.0, 0.0]), 0.1), 0.1)
        self.assertAlmostEqual(_projected_sigma(negative, np.array([1.0, 0.0]), 0.1), 0.1)

    def test_pair_sigmas_uses_line_of_sight(self) -> None:
        ra = RuntimeAssurance()
        a = EstimatedState(
            drone_id=0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            covariance=(0.04, 0.0, 0.0, 0.04),
            timestamp_ms=0,
            dropped=False,
        )
        b = EstimatedState(
            drone_id=1,
            position=(2.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            covariance=(0.01, 0.0, 0.0, 0.01),
            timestamp_ms=0,
            dropped=False,
        )
        sigma_a, sigma_b = ra._pair_sigmas(a, b)
        self.assertAlmostEqual(sigma_a, 0.2)
        self.assertAlmostEqual(sigma_b, 0.1)


if __name__ == "__main__":
    unittest.main()
