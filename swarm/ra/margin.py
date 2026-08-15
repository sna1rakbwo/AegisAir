"""Normalized safety margin rho and its degradation rate g (plan.md 15-16)."""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm.ra.margins import RuntimeAssuranceParams


def normalized_margin(distance: float, d_safe: float) -> float:
    """rho = (d - d_safe) / d_safe."""
    if d_safe <= 0:
        raise ValueError("d_safe must be positive")
    return (distance - d_safe) / d_safe


@dataclass
class PairMarginTracker:
    """Tracks one pair's rho history and computes EMA-smoothed degradation."""

    params: RuntimeAssuranceParams
    prev_rho: float | None = None
    ema_g: float = 0.0
    _prev_t: float | None = field(default=None, init=False)

    def update(self, rho: float, t: float) -> float:
        """Return the EMA degradation g after observing rho at time t."""
        if self.prev_rho is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 0:
                g = (self.prev_rho - rho) / dt
                self.ema_g = (
                    self.params.ema_lambda * self.ema_g
                    + (1.0 - self.params.ema_lambda) * g
                )
        self.prev_rho = rho
        self._prev_t = t
        return self.ema_g

    def reset(self) -> None:
        self.prev_rho = None
        self.ema_g = 0.0
        self._prev_t = None
