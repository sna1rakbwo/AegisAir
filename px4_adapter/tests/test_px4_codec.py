from __future__ import annotations

import unittest

from px4_adapter.px4_codec import (
    VEHICLE_CMD_COMPONENT_ARM_DISARM,
    VEHICLE_CMD_DO_SET_MODE,
    VEHICLE_CMD_NAV_LAND,
    VEHICLE_CMD_NAV_RETURN_TO_LAUNCH,
    build_control_plan,
    flu_to_px4_ned,
    px4_ned_to_flu,
)


class Px4CodecTest(unittest.TestCase):
    def test_ned_flu_roundtrip(self) -> None:
        flu = px4_ned_to_flu(1.0, 2.0, -3.0)
        self.assertEqual(flu, (1.0, -2.0, 3.0))
        self.assertEqual(flu_to_px4_ned(*flu), (1.0, 2.0, -3.0))

    def test_arm_plan_contains_mode_and_arm(self) -> None:
        plan = build_control_plan("arm", current_position=(0.0, 0.0, 0.0))
        commands = [cmd.command for cmd in plan.commands]
        self.assertIn(VEHICLE_CMD_DO_SET_MODE, commands)
        self.assertIn(VEHICLE_CMD_COMPONENT_ARM_DISARM, commands)
        self.assertTrue(plan.offboard.position)

    def test_move_to_requires_target(self) -> None:
        with self.assertRaises(ValueError):
            build_control_plan("move_to")

    def test_move_to_plan_uses_ned_target(self) -> None:
        plan = build_control_plan("move_to", target=(4.0, -1.0, -2.0))
        self.assertEqual(plan.trajectory.position, (4.0, -1.0, -2.0))

    def test_velocity_plan_uses_velocity_and_nan_position(self) -> None:
        plan = build_control_plan("velocity", velocity=(1.0, -2.0, -0.5))
        self.assertTrue(plan.offboard.velocity)
        self.assertFalse(plan.offboard.position)
        self.assertEqual(plan.trajectory.velocity, (1.0, -2.0, -0.5))
        self.assertIsNone(plan.trajectory.position)

    def test_velocity_plan_requires_velocity(self) -> None:
        with self.assertRaises(ValueError):
            build_control_plan("velocity")

    def test_takeoff_plan_uses_negative_altitude(self) -> None:
        plan = build_control_plan("takeoff", takeoff_altitude_m=2.5)
        self.assertAlmostEqual(plan.trajectory.position[2], -2.5)

    def test_land_and_rtl_plans(self) -> None:
        land = build_control_plan("land", current_position=(0.0, 0.0, -2.5))
        self.assertIn(VEHICLE_CMD_NAV_LAND, [c.command for c in land.commands])
        rtl = build_control_plan("rtl")
        self.assertIn(VEHICLE_CMD_NAV_RETURN_TO_LAUNCH, [c.command for c in rtl.commands])


if __name__ == "__main__":
    unittest.main()
