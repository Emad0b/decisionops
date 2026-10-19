import itertools
import numpy as np
import pytest
from decisionops.core import *


def test_allocation_matches_exhaustive_integer_optimum():
    samples = np.array([[1, 4], [2, 3], [5, 1]], float)
    budget = 6
    selected = allocate(samples, budget, under=3, over=1)
    achieved = costs(samples, selected, 3, 1).mean(axis=0).sum()
    brute = min(
        costs(samples, np.array([a, b]), 3, 1).mean(axis=0).sum()
        for a in range(7)
        for b in range(7 - a)
    )
    assert abs(achieved - brute) < 1e-10 and selected.sum() <= budget


def test_budget_zero_and_nonnegative():
    assert not allocate(np.ones((4, 3)), 0).any()
    assert (allocate(np.ones((4, 3)), 2) >= 0).all()


def test_cost_asymmetry_and_quantile():
    assert costs(np.array([10]), np.array([9]))[0] == 9
    assert costs(np.array([10]), np.array([11]))[0] == 1
    assert optimal_quantile(9, 1) == 0.9


def test_features_use_only_prior_observations():
    y = simulate(days=100)
    days, x = features(y)
    altered = y.copy()
    altered[70:] += 10000
    _, changed = features(altered)
    assert np.array_equal(x[days <= 70], changed[days <= 70])


def test_invalid_budget():
    with pytest.raises(ValueError):
        allocate(np.ones((2, 2)), -1)
