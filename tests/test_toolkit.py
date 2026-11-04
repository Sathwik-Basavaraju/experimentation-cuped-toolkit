import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from experimentation.cuped import cuped_adjust, cuped_theta, sample_size_per_arm, variance_reduction
from experimentation.sequential import launch_decision, obf_boundaries


def test_cuped_theta_and_adjustment_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(50, 10, size=5000)
    y = 50 + 0.55 * (x - 50) + np.sqrt(1 - 0.55**2) * rng.normal(0, 10, size=5000)
    theta = cuped_theta(y, x)
    assert 0.3 < theta < 0.8
    adj = cuped_adjust(y, x)
    assert np.var(adj, ddof=1) < np.var(y, ddof=1)
    rho, factor, _, _ = variance_reduction(y, x)
    assert 0.4 < rho < 0.7
    assert 0.5 < factor < 0.9


def test_sample_size_decreases_with_cuped_sigma():
    n_full = sample_size_per_arm(10.0, 1.5)
    n_cuped = sample_size_per_arm(10.0 * np.sqrt(1 - 0.55**2), 1.5)
    assert n_cuped < n_full
    assert n_full > 0 and n_cuped > 0


def test_obf_boundaries_decrease_and_end_near_fixed():
    b = obf_boundaries(5)
    assert len(b) == 5
    assert all(b[i] > b[i + 1] for i in range(4))
    assert 2.0 < b[-1] < 2.2
    assert b[0] > 4.0


def test_launch_rule_blocks_guardrail_harm():
    # Primary significant at final look, guardrails fine -> launch.
    assert launch_decision(2.5, 4, [-0.5, 0.3]) is True
    # Primary significant but a guardrail shows harm -> no launch.
    assert launch_decision(2.5, 4, [-2.5, 0.3]) is False
    # Primary not significant -> no launch even with fine guardrails.
    assert launch_decision(1.0, 4, [0.0, 0.0]) is False
    # Early OBF boundary is strict: z=3.0 at look 0 does not cross 4.56.
    assert launch_decision(3.0, 0, [0.0, 0.0]) is False
