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
    field[0, :] = 20.0  # Right column
    field[-1, :] = 70.0  # Left column

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


def estimate_dt(size: tuple[int, int], diffusion: float) -> float:
    """
    Calculate the largest stable time step (delta time) for the given field size and diffusion constant

    :param size: Tuple representing the field size
    :param diffusion: Diffusion constant that will be used in the heat equations
    :returns: time delta in seconds
    """
    dx = size[0] ** 2
    dy = size[1] ** 2
    return (dx * dy) / (2.0 * diffusion * (dx + dy))


def evolve(field: np.ndarray, a: float, dt: float, iter: int) -> np.ndarray:
    """
    Evolve the heat equation over the given field
    """
    assert iter > 0, "Number of iterations must be positive above zero!"
    curr = field.copy()
    next = field.copy()
    x_size, y_size = curr.shape
    # Subtract 2 from size to account for boundary
    dx = float((x_size - 2) ** 2)
    dy = float((y_size - 2) ** 2)
    for _ in range(iter):
        for i in range(1, x_size - 1):
            for j in range(1, y_size - 1):
                up = curr[i - 1, j]
                down = curr[i + 1, j]
                left = curr[i, j - 1]
                right = curr[i, j + 1]
                mid = curr[i, j]
                next[i, j] = mid + a * dt * (
                    (up - 2.0 * mid + down) / dx + (left - 2.0 * mid + right) / dy
                )
        curr, next = next, curr
    if iter % 2 == 0:
        return next
    return curr
