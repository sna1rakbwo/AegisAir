"""PX4 SITL/Gazebo/ROS 2 adapter for SafeDrones.

The adapter is intentionally isolated from ``mock_drone.py`` and the
rule/MARL pilot.  It owns the local safety boundary between SafeDrones'
task-level MQTT intent and PX4's native flight stack.
"""

__version__ = "0.1.0"
