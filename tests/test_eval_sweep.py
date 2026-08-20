from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from marllib.eval_sweep import evaluate_checkpoint


class EvalSweepTest(unittest.TestCase):
    def test_uses_requested_disjoint_seed_interval(self) -> None:
        seen = []

        class FakeEnv:
            num_agents = 1
            obs_dim = 1

            def __init__(self, *args, **kwargs):
                pass

            def reset(self, seed):
                seen.append(seed)
                return {0: __import__("numpy").zeros(1)}, {}

            def step(self, actions):
                return {0: __import__("numpy").zeros(1)}, {}, {0: True}, {}, {0: {"collided": False, "reached": True}}

        class FakeActor:
            def deterministic(self, value):
                return value.new_zeros((1, 2))

        class FakeModel:
            actor = FakeActor()

            def __init__(self, *args, **kwargs):
                pass

            def load_state_dict(self, state):
                pass

        scenario = type("Scenario", (), {"speed_limit": 1.5})()
        with patch("marllib.eval_sweep.MultiUAVEnv", FakeEnv), patch("marllib.eval_sweep.MAPPO", FakeModel), patch("marllib.eval_sweep.torch.load", return_value={}):
            result = evaluate_checkpoint(scenario, Path("unused.pt"), eval_seeds=3, eval_seed_start=101)
        self.assertEqual(seen, [101, 102, 103])
        self.assertEqual(result["eval_seed_start"], 101)
        self.assertEqual(result["reached"], 3)


if __name__ == "__main__":
    unittest.main()
