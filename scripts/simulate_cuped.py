"""CUPED power simulation (vectorised NumPy).

Design (all numbers below are inputs, results are computed):
- Pre-period metric X ~ N(baseline_mean, baseline_sd)
- Post-period control Y = baseline_mean + rho*(X-baseline_mean)
  + sqrt(1-rho^2)*noise, noise ~ N(0, baseline_sd), so Corr(X,Y)=rho
  and Var(Y)=baseline_sd^2.
- Treatment adds delta_absolute to post-period Y.
- Two-sided test, alpha=0.05, target power=0.80, 3% relative lift
  on a baseline mean of 50 (delta=1.5).

Outputs outputs/cuped_metrics.json with theoretical sample sizes
(with/without CUPED) and empirical power from the simulations.
Run from the project root: python3 scripts/simulate_cuped.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from experimentation.cuped import cuped_adjust, cuped_theta, sample_size_per_arm  # noqa: E402

BASELINE_MEAN = 50.0
BASELINE_SD = 10.0
RHO_TARGET = 0.55
DELTA_REL = 0.03
DELTA = BASELINE_MEAN * DELTA_REL  # 1.5
ALPHA = 0.05
TARGET_POWER = 0.80
N_EXPERIMENTS = 2000
SEED = 42


def _two_sample_p(diff, se):
    z = diff / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def simulate(n_per_arm, rng):
    """Run N_EXPERIMENTS experiments at n_per_arm per arm.

    Returns (power_naive, power_cuped, rho_observed).
    Vectorised: shape (N_EXPERIMENTS, n_per_arm) per arm.
    """
    n = n_per_arm
    # Pre-period for control and treatment (same distribution).
    x_c = rng.normal(BASELINE_MEAN, BASELINE_SD, size=(N_EXPERIMENTS, n))
    x_t = rng.normal(BASELINE_MEAN, BASELINE_SD, size=(N_EXPERIMENTS, n))
    noise_c = rng.normal(0, BASELINE_SD, size=(N_EXPERIMENTS, n))
    noise_t = rng.normal(0, BASELINE_SD, size=(N_EXPERIMENTS, n))
    y_c = BASELINE_MEAN + RHO_TARGET * (x_c - BASELINE_MEAN) + np.sqrt(1 - RHO_TARGET**2) * noise_c
    y_t = (BASELINE_MEAN + DELTA + RHO_TARGET * (x_t - BASELINE_MEAN)
           + np.sqrt(1 - RHO_TARGET**2) * noise_t)
    # Naive: two-sample test on post-period means.
    diff = y_t.mean(axis=1) - y_c.mean(axis=1)
    se = np.sqrt(y_t.var(axis=1, ddof=1) / n + y_c.var(axis=1, ddof=1) / n)
    power_naive = float(np.mean(_two_sample_p(diff, se) < ALPHA))
    # CUPED: theta estimated per experiment from pooled pre/post data.
    # Pooled theta (common practice): use all users in the experiment.
    x_all = np.concatenate([x_c, x_t], axis=1)
    y_all = np.concatenate([y_c, y_t], axis=1)
    # Theta per experiment: cov/var.
    x_mean = x_all.mean(axis=1, keepdims=True)
    y_mean = y_all.mean(axis=1, keepdims=True)
    cov = ((x_all - x_mean) * (y_all - y_mean)).sum(axis=1) / (x_all.shape[1] - 1)
    var_x = ((x_all - x_mean) ** 2).sum(axis=1) / (x_all.shape[1] - 1)
    theta = cov / var_x
    # CUPED must centre by the POOLED pre-period mean, not the arm
    # mean: centring by arm mean leaves each arm's mean unchanged
    # (mean(Y - theta*(X - mean_arm(X))) = mean(Y)), so it cannot
    # correct random pre-period imbalance between arms.
    x_pooled_mean = x_all.mean(axis=1, keepdims=True)
    y_c_adj = y_c - theta[:, None] * (x_c - x_pooled_mean)
    y_t_adj = y_t - theta[:, None] * (x_t - x_pooled_mean)
    diff_adj = y_t_adj.mean(axis=1) - y_c_adj.mean(axis=1)
    se_adj = np.sqrt(y_t_adj.var(axis=1, ddof=1) / n + y_c_adj.var(axis=1, ddof=1) / n)
    power_cuped = float(np.mean(_two_sample_p(diff_adj, se_adj) < ALPHA))
    rho_obs = float(np.corrcoef(x_all.ravel(), y_all.ravel())[0, 1])
    return power_naive, power_cuped, rho_obs


def main():
    rng = np.random.default_rng(SEED)
    sigma_adj = BASELINE_SD * np.sqrt(1 - RHO_TARGET**2)
    n_without = sample_size_per_arm(BASELINE_SD, DELTA, ALPHA, TARGET_POWER)
    n_with = sample_size_per_arm(sigma_adj, DELTA, ALPHA, TARGET_POWER)
    reduction = (n_without - n_with) / n_without
    print(f"Theoretical n/arm without CUPED: {n_without}")
    print(f"Theoretical n/arm with CUPED: {n_with} (reduction {reduction:.1%})")
    # Empirical power at the CUPED-sized experiment.
    p_naive_at_cuped_n, p_cuped_at_cuped_n, rho_obs = simulate(n_with, rng)
    print(f"At n={n_with}/arm: naive power={p_naive_at_cuped_n:.3f}, CUPED power={p_cuped_at_cuped_n:.3f}, rho_obs={rho_obs:.3f}")
    # Empirical power at the unadjusted-sized experiment (fresh RNG stream).
    p_naive_at_full_n, p_cuped_at_full_n, _ = simulate(n_without, rng)
    print(f"At n={n_without}/arm: naive power={p_naive_at_full_n:.3f}, CUPED power={p_cuped_at_full_n:.3f}")
    metrics = {
        "baseline_mean": BASELINE_MEAN,
        "baseline_sd": BASELINE_SD,
        "delta_absolute": DELTA,
        "delta_relative": DELTA_REL,
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "rho_target": RHO_TARGET,
        "rho_observed_pooled": round(rho_obs, 4),
        "n_experiments": N_EXPERIMENTS,
        "seed": SEED,
        "sample_size_per_arm_without_cuped": n_without,
        "sample_size_per_arm_with_cuped": n_with,
        "sample_size_reduction_fraction": round(reduction, 4),
        "sample_size_reduction_percent": round(reduction * 100, 2),
        "empirical_power_at_cuped_n": {
            "naive": round(p_naive_at_cuped_n, 4),
            "cuped": round(p_cuped_at_cuped_n, 4),
        },
        "empirical_power_at_full_n": {
            "naive": round(p_naive_at_full_n, 4),
            "cuped": round(p_cuped_at_full_n, 4),
        },
    }
    out = ROOT / "outputs" / "cuped_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return metrics


if __name__ == "__main__":
    main()
