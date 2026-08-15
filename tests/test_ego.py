import unittest

from swarm.ego import EgoStateStore, parse_ego_position
from swarm.safety import DroneSnapshot


class EgoStateStoreTest(unittest.TestCase):
    def test_parse_requires_three_position_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_ego_position({"drone": 2, "position": [1.0, 2.0]})

    def test_pilot_keeps_own_telemetry_and_uses_fresh_peer_estimate(self) -> None:
        store = EgoStateStore()
        store.update({"drone": 2, "position": [2.1, 0.1, 1.0], "estimated_depth": 2.1}, received_at=10.0)
        own = DroneSnapshot(1, (0.0, 0.0, 1.0))
        peer = DroneSnapshot(2, (2.0, 0.0, 1.0))

        observed = store.snapshot_for_pilot(own, [own, peer], ttl_sec=0.5, now=10.2)

        self.assertEqual(observed[0].position, own.position)
        self.assertEqual(observed[1].position, (2.1, 0.1, 1.0))

    def test_stale_estimates_are_dropped_not_reused(self) -> None:
        store = EgoStateStore()
        store.update({"drone": 1, "position": [0.0, 0.0, 1.0]}, received_at=10.0)
        store.update({"drone": 2, "position": [2.0, 0.0, 1.0]}, received_at=10.0)
        snapshots = [DroneSnapshot(1, (0.0, 0.0, 1.0)), DroneSnapshot(2, (2.0, 0.0, 1.0))]

        self.assertEqual(store.snapshots_for_gate(snapshots, ttl_sec=0.5, now=10.6), [])


if __name__ == "__main__":
    unittest.main()
