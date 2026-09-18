"""Frozen tests for the Phase 5 command path (no PX4 runtime required)."""

from __future__ import annotations

import unittest

import numpy as np

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import (
    CRUISE_ALTITUDE_M,
    OBSERVATION_LOCAL_FRESH_SELF,
    _audit_exact_zoh_interval,
    _command_authority,
    _priority_order,
    _reset_velocity_command,
    _step_exact_zoh_execution,
    _go_to_goal,
    _local_ra_view,
    _latency_summary_ms,
    _propagate_states,
    _scenario,
    _snapshots,
    _estimator_config_for_fault,
    _estimated_states_and_aoi,
    _freeze_holding_points,
    build_phase5_command,
    build_phase5_velocity_command,
    flu_snapshot_to_telemetry_state,
    run_sim_episode,
    run_mqtt_loop,
    snapshot_from_telemetry,
    validate_command_path,
)
from px4_adapter.mqtt_codec import decode_command, normalize_command_to_ned
from swarm.estimation import EstimatedState, SharedStateEstimator, SharedStateEstimatorConfig
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import RecoveryOverrides
from swarm.safety import DroneSnapshot


class Phase5CommandEncodingTest(unittest.TestCase):
    def test_live_command_authority_is_fail_closed(self) -> None:
        self.assertEqual(
            _command_authority(failed_or_aborted=False), ("ra_filtered", False)
        )
        self.assertEqual(
            _command_authority(failed_or_aborted=True), ("failed_zero", False)
        )

    def test_latency_summary_keeps_deadline_outlier(self) -> None:
        summary = _latency_summary_ms([1.0, 2.0, 51.0], deadline_ms=50.0)
        self.assertEqual(summary["max"], 51.0)
        self.assertEqual(summary["deadline_misses"], 1)

    def test_live_runner_defaults_to_landing_after_a_standalone_episode(self) -> None:
        import inspect

        self.assertTrue(
            inspect.signature(run_mqtt_loop).parameters["land_at_end"].default
        )
    def test_priority_order_puts_urgent_first(self) -> None:
        self.assertEqual(_priority_order([2, 3, 4, 5], urgent_drone=4), [4, 2, 3, 5])
        self.assertEqual(_priority_order([2, 3, 4, 5]), [2, 3, 4, 5])

    def test_sequential_holding_points_are_frozen_at_activation(self) -> None:
        positions = {2: np.array([-1.0, 0.0]), 3: np.array([1.0, 0.0])}
        holding = _freeze_holding_points([2, 3], positions)
        positions[2][:] = (-5.0, 0.0)
        positions[3][:] = (5.0, 0.0)
        self.assertEqual(holding[2], (-2.5, 0.0, 0.0))
        self.assertEqual(holding[3], (2.5, 0.0, 0.0))

    def test_high_closing_scenario_has_higher_speed_limit(self) -> None:
        self.assertEqual(_scenario("high_closing")["scenario"].speed_limit, 2.0)

    def test_exact_zoh_execution_uses_integrated_position(self) -> None:
        spec = _scenario("head_on")
        env = MultiUAVEnv(spec["scenario"])
        env.reset(seed=1)
        env.positions[:] = 0.0
        env.velocities[:] = 0.0
        _step_exact_zoh_execution(
            env,
            {0: np.array([0.5, 0.0]), 1: np.array([-0.5, 0.0])},
            tau_s=0.2,
        )
        alpha = 1.0 - np.exp(-env.scenario.dt / 0.2)
        beta = env.scenario.dt - 0.2 * alpha
        self.assertAlmostEqual(float(env.velocities[0, 0]), 0.5 * alpha)
        self.assertAlmostEqual(float(env.positions[0, 0]), 0.5 * beta)

    def test_exact_zoh_audit_matches_endpoint_propagation(self) -> None:
        spec = _scenario("head_on")
        env = MultiUAVEnv(spec["scenario"])
        env.reset(seed=1)
        env.positions[:] = 0.0
        env.velocities[:] = 0.0
        commands = {0: np.array([0.5, 0.0]), 1: np.array([-0.5, 0.0])}
        ra = RuntimeAssurance(params=RuntimeAssuranceParams(tau_ctrl=0.0))
        audit = _audit_exact_zoh_interval(env, commands, 0.2, ra, samples=100)
        _step_exact_zoh_execution(env, commands, tau_s=0.2)
        self.assertAlmostEqual(
            audit["endpoint_min_distance_m"],
            float(np.linalg.norm(env.positions[0] - env.positions[1])),
        )
        self.assertLessEqual(
            audit["intersample_min_distance_m"], audit["endpoint_min_distance_m"]
        )

    def test_build_phase5_command_integrates_safe_velocity(self) -> None:
        command = build_phase5_command(
            drone=2,
            safe_velocity=(1.0, -0.5),
            position_flu=(1.0, 2.0, 0.0),
            timestamp_ms=123,
        )
        self.assertEqual(command["action"], "move_to")
        self.assertEqual(command["source_frame"], "FLU")
        # target = position + velocity * 0.5, altitude clamped to cruise alt.
        self.assertEqual(command["target"], [1.5, 1.75, CRUISE_ALTITUDE_M])
        self.assertEqual(command["drone"], 2)

    def test_command_decodes_and_normalizes_to_ned(self) -> None:
        command = build_phase5_command(
            drone=3,
            safe_velocity=(0.0, 1.0),
            position_flu=(0.0, 0.0, CRUISE_ALTITUDE_M),
            timestamp_ms=1,
        )
        decoded = normalize_command_to_ned(
            decode_command(command, default_source_frame="PX4_NED")
        )
        # FLU target y = 0 + 1.0*0.5 = +0.5 maps to NED y=-0.5;
        # FLU altitude +2.5 maps to NED z=-2.5.
        self.assertEqual(decoded.source_frame, "PX4_NED")
        self.assertAlmostEqual(decoded.target[1], -0.5)
        self.assertAlmostEqual(decoded.target[2], -CRUISE_ALTITUDE_M)

    def test_build_phase5_velocity_command(self) -> None:
        command = build_phase5_velocity_command(
            drone=2,
            safe_velocity=(1.0, -0.5),
            vertical_velocity=0.2,
            timestamp_ms=123,
        )
        self.assertEqual(command["action"], "velocity")
        self.assertEqual(command["source_frame"], "FLU")
        self.assertEqual(command["velocity"], [1.0, -0.5, 0.2])

    def test_reset_velocity_homes_in_flu_without_exceeding_limits(self) -> None:
        velocity = _reset_velocity_command(
            position=(-3.0, 1.0, 2.0),
            target=(3.0, 1.0, 2.5),
        )
        self.assertAlmostEqual(velocity[0], 1.0)
        self.assertAlmostEqual(velocity[1], 0.0)
        self.assertAlmostEqual(velocity[2], 0.5)

    def test_reset_velocity_has_norm_bounded_diagonal_speed(self) -> None:
        velocity = _reset_velocity_command(
            position=(0.0, 0.0, 2.5),
            target=(3.0, 4.0, 2.5),
        )
        self.assertAlmostEqual(float(np.linalg.norm(velocity[:2])), 1.0)

    def test_velocity_command_passes_local_safety(self) -> None:
        timestamp_ms = 1000
        command = build_phase5_velocity_command(
            drone=2,
            safe_velocity=(1.0, 0.0),
            vertical_velocity=0.1,
            timestamp_ms=timestamp_ms,
        )
        state = flu_snapshot_to_telemetry_state(
            2,
            (0.0, 0.0, CRUISE_ALTITUDE_M),
            (0.0, 0.0, 0.0),
            timestamp_ms=timestamp_ms,
        )
        allowed, reason = validate_command_path(
            command, state, now_ms=timestamp_ms
        )
        self.assertTrue(allowed, reason)
        self.assertEqual(reason, "ok")

    def test_normal_command_passes_local_safety(self) -> None:
        timestamp_ms = 1000
        command = build_phase5_command(
            drone=2,
            safe_velocity=(1.0, 0.0),
            position_flu=(0.0, 0.0, CRUISE_ALTITUDE_M),
            timestamp_ms=timestamp_ms,
        )
        state = flu_snapshot_to_telemetry_state(
            2,
            (0.0, 0.0, CRUISE_ALTITUDE_M),
            (0.0, 0.0, 0.0),
            timestamp_ms=timestamp_ms,
        )
        allowed, reason = validate_command_path(
            command, state, now_ms=timestamp_ms
        )
        self.assertTrue(allowed, reason)
        self.assertEqual(reason, "ok")

    def test_out_of_limits_command_is_rejected(self) -> None:
        timestamp_ms = 1000
        command = build_phase5_command(
            drone=2,
            safe_velocity=(100.0, 0.0),
            position_flu=(0.0, 0.0, CRUISE_ALTITUDE_M),
            timestamp_ms=timestamp_ms,
        )
        state = flu_snapshot_to_telemetry_state(
            2,
            (0.0, 0.0, CRUISE_ALTITUDE_M),
            (0.0, 0.0, 0.0),
            timestamp_ms=timestamp_ms,
        )
        allowed, reason = validate_command_path(
            command, state, now_ms=timestamp_ms
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "horizontal_distance_out_of_limits")

    def test_telemetry_payload_parses_flu_position(self) -> None:
        payload = {
            "drone": 2,
            "position": [1.0, 2.0, 3.0],
            "velocity": [0.1, 0.2, 0.3],
            "timestamp_ms": 7,
            "status": "armed",
        }
        snapshot = snapshot_from_telemetry(payload)
        self.assertEqual(snapshot.drone_id, 2)
        self.assertEqual(snapshot.position, (1.0, 2.0, 3.0))
        self.assertEqual(snapshot.velocity, (0.1, 0.2, 0.3))


class Phase6FaultInjectionTest(unittest.TestCase):
    def _run(self, fault: dict) -> dict:
        spec = _scenario("head_on")
        env = MultiUAVEnv(spec["scenario"])
        ra = RuntimeAssurance(
            params=RuntimeAssuranceParams(tau_ctrl=0.0),
            v_max=1.5,
            sampled_data=True,
            gamma=0.1,
        )
        return run_sim_episode(
            spec=spec,
            env=env,
            seed=1,
            ra=ra,
            mode="CBF_ONLY",
            llm_client=None,
            llm_fallback=None,
            max_steps=60,
            real_time=False,
            fault=fault,
        )

    def test_command_latency_fails_closed(self) -> None:
        run = self._run({"command_latency_ms": 1100})
        self.assertFalse(run["collision"])
        self.assertGreater(run["rejected_reasons"].get("command_expired", 0), 0)

    def test_stale_telemetry_fails_closed(self) -> None:
        run = self._run({"telemetry_stale_ms": 3000})
        self.assertFalse(run["collision"])
        self.assertGreater(run["rejected_reasons"].get("telemetry_stale", 0), 0)

    def test_estimator_dropout_runs_through_ra(self) -> None:
        run = self._run({"estimator_dropout_rate": 0.3})
        self.assertIn("min_rho", run)
        self.assertIsNotNone(run["min_pairwise_distance_m"])
        self.assertGreater(run["emitted_commands"], 0)
        self.assertIn("path_length_m", run)
        self.assertIn("max_repeated_cbf_duration_s", run)


class EstimatorWiringTest(unittest.TestCase):
    def test_estimated_states_and_aoi_uses_estimator_age(self) -> None:
        estimator = SharedStateEstimator(
            SharedStateEstimatorConfig(delay_ms=100)
        )
        snapshots = {
            2: DroneSnapshot(
                drone_id=2, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)
            ),
            3: DroneSnapshot(
                drone_id=3, position=(1.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)
            ),
        }
        estimator.step(snapshots, now_ms=0, rng=None)
        estimator.step(snapshots, now_ms=100, rng=None)
        estimated, aoi = _estimated_states_and_aoi(
            snapshots, estimator, now_ms=200, rng=None
        )
        self.assertEqual(estimated[2].timestamp_ms, 100)
        self.assertAlmostEqual(aoi[(2, 3)], 0.1)

    def test_propagate_states_dead_reckons_position(self) -> None:
        state = DroneSnapshot(
            drone_id=2,
            position=(1.0, 2.0, 3.0),
            velocity=(0.5, 0.0, 0.0),
        )
        out = _propagate_states({2: state}, {2: 0.5})
        self.assertAlmostEqual(out[2].position[0], 1.25)
        self.assertAlmostEqual(out[2].position[1], 2.0)
        self.assertAlmostEqual(out[2].position[2], 3.0)

    def test_propagate_states_dead_reckons_estimated_state(self) -> None:
        state = EstimatedState(
            drone_id=2,
            position=(1.0, 0.0, 0.0),
            velocity=(0.5, 0.0, 0.0),
            covariance=(0.04, 0.0, 0.0, 0.04),
            timestamp_ms=100,
            dropped=True,
        )
        out = _propagate_states({2: state}, {2: 0.5})
        self.assertAlmostEqual(out[2].position[0], 1.25)
        self.assertAlmostEqual(out[2].position[1], 0.0)
        self.assertAlmostEqual(out[2].covariance[0], 0.04)

    def test_estimator_config_covariance_matches_injected_noise(self) -> None:
        cfg = _estimator_config_for_fault(
            {
                "estimator_dropout_rate": 0.3,
                "perception_noise_pos_m": 0.2,
            }
        )
        self.assertAlmostEqual(cfg.measurement_cov_m2, 0.04)
        self.assertEqual(cfg.dropout_rate, 0.3)

    def test_local_ra_view_keeps_self_fresh_and_peer_stale(self) -> None:
        fresh = {
            0: DroneSnapshot(0, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            1: DroneSnapshot(1, (4.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        }
        peers = {
            0: EstimatedState(0, (1.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.01, 0.0, 0.0, 0.01), 700, False),
            1: EstimatedState(1, (3.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.01, 0.0, 0.0, 0.01), 700, False),
        }
        view, aoi = _local_ra_view(
            observer=0, fresh_states=fresh, peer_states=peers, now_ms=1000
        )
        self.assertEqual(view[0].position, fresh[0].position)
        self.assertEqual(view[0].timestamp_ms, 1000)
        self.assertEqual(view[1].timestamp_ms, 700)
        self.assertAlmostEqual(aoi[(0, 1)], 0.3)

    def test_local_ra_view_treats_raw_peer_as_current(self) -> None:
        raw = {
            0: DroneSnapshot(0, (0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)),
            1: DroneSnapshot(1, (1.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)),
        }
        _, aoi = _local_ra_view(
            observer=0, fresh_states=raw, peer_states=raw, now_ms=5000
        )
        self.assertEqual(aoi[(0, 1)], 0.0)

    def test_nominal_can_use_observed_state(self) -> None:
        spec = _scenario("head_on")
        env = MultiUAVEnv(spec["scenario"])
        env.reset(seed=1)
        base_goals = {
            i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0)
            for i in env.agent_ids
        }
        observed = _snapshots(env)
        observed[0] = DroneSnapshot(0, base_goals[0], (0.0, 0.0, 0.0))
        actions = _go_to_goal(
            env,
            base_goals=base_goals,
            overrides=RecoveryOverrides(),
            failed=set(),
            aborted=set(),
            observed_states=observed,
        )
        self.assertTrue(np.allclose(actions[0], np.zeros(2)))

    def test_local_fresh_self_mode_runs(self) -> None:
        spec = _scenario("head_on")
        env = MultiUAVEnv(spec["scenario"])
        ra = RuntimeAssurance(params=RuntimeAssuranceParams(tau_ctrl=0.0), v_max=1.5)
        local = run_sim_episode(
            spec=spec,
            env=env,
            seed=1,
            ra=ra,
            mode="CBF_ONLY",
            llm_client=None,
            llm_fallback=None,
            max_steps=20,
            real_time=False,
            fault={"estimator_delay_ms": 300},
            observation_mode=OBSERVATION_LOCAL_FRESH_SELF,
        )
        self.assertEqual(local["observation_mode"], OBSERVATION_LOCAL_FRESH_SELF)


if __name__ == "__main__":
    unittest.main()
