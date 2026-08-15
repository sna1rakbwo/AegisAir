"""Tests for the asynchronous Semantic Mission Manager."""

from __future__ import annotations

import unittest

from swarm.ra.runtime_assurance import FilterResult
from swarm.recovery import AsyncMissionReplanner, DeterministicRecoveryClient, ReplanConfig
from swarm.safety import DroneSnapshot


def _snapshots(*positions) -> dict[int, DroneSnapshot]:
    return {
        i: DroneSnapshot(drone_id=i, position=pos, velocity=(0.0, 0.0, 0.0))
        for i, pos in enumerate(positions)
    }


def _result(drone: int, *, intervened: bool = False) -> FilterResult:
    return FilterResult(
        drone=drone,
        mode="normal",
        nominal_action=(1.0, 0.0),
        safe_action=(1.0, 0.0),
        safety_margin=1.0,
        predicted_margin=1.0,
        time_to_min_margin=0.0,
        time_to_safety_boundary=None,
        prediction_reliability_score=1.0,
        recovery_buffer=None,
        semantic_recovery_feasible=True,
        degradation=0.0,
        worst_pair=None,
        intervened=intervened,
        proactive=False,
    )


def _results(*drones) -> dict[int, FilterResult]:
    return {i: _result(i) for i in drones}


class AsyncMissionReplannerTest(unittest.TestCase):
    def _step(
        self,
        replanner: AsyncMissionReplanner,
        t: float,
        positions,
        *,
        mission_change=None,
    ) -> None:
        snapshots = _snapshots(*positions)
        replanner.step(
            t=t,
            snapshots=snapshots,
            results=_results(*range(len(positions))),
            current_goals={0: (4.0, 0.0, 0.0)},
            base_goals={0: (4.0, 0.0, 0.0)},
            mission_change=mission_change,
        )

    def test_route_deviation_triggers_and_commits_async(self) -> None:
        replanner = AsyncMissionReplanner(
            client=DeterministicRecoveryClient(),
            config=ReplanConfig(route_deviation_threshold_m=1.5),
        )
        self._step(replanner, 0.0, [(0.0, 0.0, 0.0)])
        self._step(replanner, 0.1, [(2.0, 3.0, 0.0)])
        self.assertEqual(replanner.counters.triggers, 1)
        self.assertEqual(replanner.counters.trigger_causes, ["ROUTE_DEVIATION"])
        # Async: generation is submitted but not committed in the same step.
        self.assertEqual(replanner.counters.plans_committed, 0)

        self._step(replanner, 0.2, [(2.0, 3.0, 0.0)])
        self.assertEqual(replanner.counters.plans_committed, 1)
        self.assertEqual(replanner.counters.mission_changes, 1)
        replanner.shutdown()

    def test_stall_triggers_coordination_degradation(self) -> None:
        replanner = AsyncMissionReplanner(
            client=DeterministicRecoveryClient(),
            config=ReplanConfig(stall_window_s=1.0),
        )
        self._step(replanner, 0.0, [(0.0, 0.0, 0.0)])
        # No progress toward the goal for >= stall_window_s.
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            self._step(replanner, t, [(0.0, 0.0, 0.0)])
        self.assertEqual(replanner.counters.triggers, 1)
        self.assertEqual(
            replanner.counters.trigger_causes, ["COORDINATION_DEGRADATION"]
        )
        replanner.shutdown()

    def test_mission_change_triggers_first(self) -> None:
        replanner = AsyncMissionReplanner(client=DeterministicRecoveryClient())
        self._step(replanner, 0.0, [(0.0, 0.0, 0.0)], mission_change={"drone": 0})
        self.assertEqual(replanner.counters.triggers, 1)
        self.assertEqual(replanner.counters.trigger_causes, ["MISSION_CHANGE"])
        replanner.shutdown()

    def test_blocking_commits_same_step(self) -> None:
        replanner = AsyncMissionReplanner(
            client=DeterministicRecoveryClient(),
            config=ReplanConfig(route_deviation_threshold_m=1.5),
            blocking=True,
        )
        self._step(replanner, 0.0, [(0.0, 0.0, 0.0)])
        self._step(replanner, 0.1, [(2.0, 3.0, 0.0)])
        self.assertEqual(replanner.counters.plans_committed, 1)
        replanner.shutdown()


if __name__ == "__main__":
    unittest.main()
