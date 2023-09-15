#!/usr/bin/env python3

"""
Unit tests and benchmarks for PyHeatrs

Run with `pytest`, `nox -s test` or `nox -s bench`
"""

import numpy as np
import pyheatrs
import pytest


@pytest.fixture
def small_field():
    a = 0.5
    size = (200, 200)
    dt = pyheatrs.py.estimate_dt(size, a)
    return pyheatrs.py.default_field(size), a, dt


@pytest.fixture
def medium_field():
    a = 0.5
    size = (400, 400)
    dt = pyheatrs.py.estimate_dt(size, a)
    return pyheatrs.py.default_field(size), a, dt


@pytest.mark.benchmark(group="small")
def test_python_small(small_field, benchmark):
    field, a, dt = small_field
    benchmark(pyheatrs.py.evolve, field, a, dt, 10)


@pytest.mark.benchmark(group="medium")
def test_python_medium(medium_field, benchmark):
    field, a, dt = medium_field
    benchmark(pyheatrs.py.evolve, field, a, dt, 10)


@pytest.mark.benchmark(group="small")
def test_rust_small(small_field, benchmark):
    field, a, dt = small_field
    res = benchmark(pyheatrs.rs.evolve, field, a, dt, 10)
    true = pyheatrs.py.evolve(field, a, dt, 10)
    assert np.allclose(res, true)


@pytest.mark.benchmark(group="medium")
def test_rust_medium(medium_field, benchmark):
    field, a, dt = medium_field
    res = benchmark(pyheatrs.rs.evolve, field, a, dt, 10)
    true = pyheatrs.py.evolve(field, a, dt, 10)
    assert np.allclose(res, true)
