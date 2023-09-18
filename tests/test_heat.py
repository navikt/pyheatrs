#!/usr/bin/env python3

"""
Unit tests and benchmarks for PyHeatrs

Run with `pytest`, `nox -s test` or `nox -s bench`
"""

import numpy as np
import pyheatrs
import pytest


@pytest.fixture
def diffusion() -> float:
    return 0.5


@pytest.fixture
def delta() -> tuple[float, float]:
    return (0.01, 0.01)


@pytest.fixture
def small_field(diffusion, delta):
    size = (200, 200)
    dt = pyheatrs.py.estimate_dt(delta, diffusion)
    return pyheatrs.py.default_field(size), delta, diffusion, dt


@pytest.fixture
def medium_field(diffusion, delta):
    size = (400, 400)
    dt = pyheatrs.py.estimate_dt(delta, diffusion)
    return pyheatrs.py.default_field(size), delta, diffusion, dt


@pytest.fixture
def large_field(diffusion, delta):
    size = (1000, 1000)
    dt = pyheatrs.py.estimate_dt(delta, diffusion)
    return pyheatrs.py.default_field(size), delta, diffusion, dt


@pytest.mark.benchmark(group="small")
def test_python_small(small_field, benchmark):
    field, dxdy, a, dt = small_field
    benchmark(pyheatrs.py.evolve, field, dxdy, a, dt, 10)


@pytest.mark.benchmark(group="medium")
def test_python_medium(medium_field, benchmark):
    field, dxdy, a, dt = medium_field
    benchmark(pyheatrs.py.evolve, field, dxdy, a, dt, 10)


@pytest.mark.benchmark(group="small")
def test_rust_small(small_field, benchmark):
    field, dxdy, a, dt = small_field
    res = benchmark(pyheatrs.rs.evolve, field, dxdy, a, dt, 10)
    true = pyheatrs.py.evolve(field, dxdy, a, dt, 10)
    assert np.allclose(res, true)


@pytest.mark.benchmark(group="medium")
def test_rust_medium(medium_field, benchmark):
    field, dxdy, a, dt = medium_field
    res = benchmark(pyheatrs.rs.evolve, field, dxdy, a, dt, 10)
    true = pyheatrs.py.evolve(field, dxdy, a, dt, 10)
    assert np.allclose(res, true)


@pytest.mark.benchmark(group="large")
def test_rust_large(large_field, benchmark):
    field, dxdy, a, dt = large_field
    benchmark(pyheatrs.rs.evolve, field, dxdy, a, dt, 100)
