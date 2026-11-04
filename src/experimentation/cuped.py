"""CUPED (Controlled-experiment Using Pre-Experiment Data) adjustment.

Method (Deng et al.): for post-period outcome Y and pre-period
covariate X, the CUPED-adjusted outcome is

    Y_cuped = Y - theta * (X - E[X]),  theta = Cov(Y, X) / Var(X)

This removes the part of Y's variance explained by X, reducing the
variance of the treatment-effect estimator by a factor (1 - rho^2),
where rho is the pre/post correlation, without biasing the effect
when X is measured before randomisation.
"""
import math

import numpy as np
from scipy import stats


def cuped_theta(y, x):
    """Return theta = Cov(Y, X) / Var(X)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return 0.0
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def cuped_adjust(y, x, theta=None):
    """Return CUPED-adjusted outcomes, using the sample theta if not given.

    Note: x here should be centred by the pooled/overall pre-period
    mean when comparing arms (see scripts/simulate_cuped.py). This
    function centres by mean(x) of the array passed in, so pass the
    pooled data (both arms together) or an explicitly centred x when
    the goal is a treatment-effect comparison; centring each arm by
    its own mean leaves arm means unchanged and does not correct
    pre-period imbalance.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if theta is None:
        theta = cuped_theta(y, x)
    return y - theta * (x - np.mean(x))


def variance_reduction(y, x):
    """Return (rho, variance_reduction_factor, var_original, var_adjusted)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    rho = float(np.corrcoef(y, x)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    adj = cuped_adjust(y, x)
    return rho, float(1 - rho**2), float(np.var(y, ddof=1)), float(np.var(adj, ddof=1))


def sample_size_per_arm(sigma, delta, alpha=0.05, power=0.80):
    """Two-sided, two-sample normal-approximation sample size per arm.

    n = 2 * sigma^2 * (z_{1-alpha/2} + z_power)^2 / delta^2
    """
    if sigma <= 0 or delta == 0:
        raise ValueError("sigma must be > 0 and delta must be non-zero")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * sigma**2 * (z_alpha + z_beta) ** 2 / delta**2
    return int(math.ceil(n))
