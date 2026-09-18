"""Small, dependency-free vector helpers shared across the swarm runtime."""

from __future__ import annotations

import math


Vector3 = tuple[float, float, float]


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vector3, factor: float) -> Vector3:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vector3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vector3) -> Vector3:
    length = norm(a)
    if length == 0:
        raise ValueError("cannot normalize a zero vector")
    return scale(a, 1.0 / length)


def clamp_magnitude(a: Vector3, max_magnitude: float) -> Vector3:
    length = norm(a)
    if length <= max_magnitude:
        return a
    return scale(a, max_magnitude / length)
