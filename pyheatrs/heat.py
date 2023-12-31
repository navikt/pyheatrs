#!/usr/bin/env python3

"""
Implementation of the heat equations in Python
"""

import numpy as np


def default_field(size: tuple[int, int]) -> np.ndarray:
    """Create a new 2D field with a default initialization"""
    field = np.full(size, 65.0, dtype=np.float64)
    # Setup boundary conditions for our field
    field[:, 0] = 85.0  # Top row
    field[:, -1] = 5.0  # Bottom row
    field[0, :] = 20.0  # Left column
    field[-1, :] = 70.0  # Right column

    radius = size[0] // 6
    orig_x = size[0] // 2
    orig_y = size[1] // 2
    for i in range(orig_x - radius, orig_x + radius):
        dx = i - orig_x + 1
        for j in range(orig_y - radius, orig_y + radius):
            dy = j - orig_y + 1
            if (dx**2 + dy**2) < radius**2:
                field[i, j] = 5.0
    return field


def estimate_dt(dxdy: tuple[float, float], diffusion: float) -> float:
    """
    Calculate the largest stable time step (delta time) for the given field size and
    diffusion constant

    :param dxdy: Tuple representing the delta between x and y coordinates
    :param diffusion: Diffusion constant that will be used in the heat equations
    :returns: time delta in seconds
    """
    dx = float(dxdy[0] ** 2)
    dy = float(dxdy[1] ** 2)
    return (dx * dy) / (2.0 * diffusion * (dx + dy))


def evolve(
    field: np.ndarray, dxdy: tuple[float, float], a: float, dt: float, iter: int
) -> np.ndarray:
    """
    Evolve the heat equation over the given field
    """
    assert iter > 0, "Number of iterations must be positive above zero!"
    curr = field.copy()
    next = field.copy()
    x_size, y_size = curr.shape
    dx = dxdy[0] ** 2
    dy = dxdy[1] ** 2
    for _ in range(iter):
        for i in range(1, x_size - 1):
            for j in range(1, y_size - 1):
                left = curr[i - 1, j]
                right = curr[i + 1, j]
                up = curr[i, j - 1]
                down = curr[i, j + 1]
                mid = curr[i, j]
                next[i, j] = mid + a * dt * (
                    (right - 2.0 * mid + left) / dx + (down - 2.0 * mid + up) / dy
                )
        curr, next = next, curr
    if iter % 2 == 0:
        return next
    return curr
