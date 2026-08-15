# P5 adapter local fault-scan protocol

This is a deterministic, least-cost first gate. It tests the adapter's local
safety boundary in a pure event loop, not a PX4/Gazebo flight. Live PX4 fault
scanning is a later, more expensive gate and must reuse the frozen protocol and
stop rules established here.

## Frozen objects

- protocol id: `safedrones-px4-adapter-p5-local-v1`
- seeds: `1..10`
- duration: `6000 ms`
- command interval: `500 ms`
- command mix: `arm, takeoff, move_to, hold, land`
- safety limits: as in `p5_config.json`

## Faults

- `NONE`
- `COMMAND_LOSS_10/20/30`
- `COMMAND_LATENCY_100MS`
- `TELEMETRY_STALE`
- `GCS_CONNECTION_LOST`
- `OFFBOARD_LOSS`

## Per-episode metrics

- generated, dropped, evaluated, accepted, rejected command counts
- rejection-reason distribution
- local-limit bypass count (must be `0`)
- first fail-closed action latency for telemetry/GCS/offboard faults

## Stop rules

- If any accepted command bypasses a frozen local limit, the run is
  `ENGINEERING_INVALID`.
- If a telemetry/GCS/offboard fault produces no fail-closed action before the
  episode ends, that fault is failed.
- Do not change thresholds, add seeds, or widen limits to rescue a failed
  fault.

## Claim boundary

This gate validates deterministic adapter behavior only. It does not establish
PX4 flight safety, collision avoidance, or end-to-end fault statistics.
