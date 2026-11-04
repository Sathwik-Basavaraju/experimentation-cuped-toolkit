"""A/A peeking simulation (vectorised NumPy).

Design: n_tests A/A experiments (true effect zero), 5 equally spaced
interim looks, final sample 1,000 users per arm. At each look, the
cumulative z-score is built from independent segment increments, so
z-scores across looks are correctly correlated (Brownian motion).
Guardrails: two independent null metrics, generated the same way.
A launch is blocked if either guardrail shows significant harm
(z < -1.96) at the look where the primary first crosses.

Naive rule: launch if |z| > 1.96 at any look.
OBF rule: launch if |z| > OBF boundary at any look.
False-launch rate applies the guardrail block to the same rule.

Outputs outputs/aa_metrics.json.
Run from the project root: python3 scripts/simulate_aa.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from experimentation.sequential import obf_boundaries, launch_decision  # noqa: E402

N_TESTS = 10000
N_LOOKS = 5
FINAL_N_PER_ARM = 1000
SEED = 123
GUARDRAIL_THRESHOLD = -1.96


def _cumulative_z(rng, n_tests, n_looks, final_n):
    """Return cumulative z-scores, shape (n_tests, n_looks)."""
    segment_n = final_n / n_looks
    # Segment mean-diffs: Var = 2*sigma^2/segment_n, sigma=1.
    seg_diff = rng.normal(0, np.sqrt(2 / segment_n), size=(n_tests, n_looks))
    # Cumulative mean-diff is the AVERAGE of segment mean-diffs, not
    # their sum: mean_k = (seg_1 + ... + seg_k) / k. Using the sum
    # would inflate z by a factor of k (a bug found in review).
    cum_diff = np.cumsum(seg_diff, axis=1) / np.arange(1, n_looks + 1)[None, :]
    look_n = segment_n * np.arange(1, n_looks + 1)
    se = np.sqrt(2 / look_n)  # shape (n_looks,)
    return cum_diff / se


def _first_cross(z, boundaries):
    """Return (crossed_bool, first_cross_index_or_-1) per test."""
    crossed = np.abs(z) > np.array(boundaries)[None, :]
    any_cross = crossed.any(axis=1)
    first_idx = np.where(any_cross, crossed.argmax(axis=1), -1)
    return any_cross, first_idx


def main():
    rng = np.random.default_rng(SEED)
    bounds = obf_boundaries(N_LOOKS)
    naive_bounds = [1.96] * N_LOOKS
    z_primary = _cumulative_z(rng, N_TESTS, N_LOOKS, FINAL_N_PER_ARM)
    z_g1 = _cumulative_z(rng, N_TESTS, N_LOOKS, FINAL_N_PER_ARM)
    z_g2 = _cumulative_z(rng, N_TESTS, N_LOOKS, FINAL_N_PER_ARM)

    def false_positive_and_launch(z, boundaries):
        crossed, first_idx = _first_cross(z, boundaries)
        fpr = float(crossed.mean())
        # Guardrail check at the first-cross look; no cross => no launch.
        launched = np.zeros(N_TESTS, dtype=bool)
        idx = first_idx[first_idx >= 0]
        rows = np.where(first_idx >= 0)[0]
        if len(rows):
            blocked = (z_g1[rows, idx] < GUARDRAIL_THRESHOLD) | (z_g2[rows, idx] < GUARDRAIL_THRESHOLD)
            launched[rows] = ~blocked
        return fpr, float(launched.mean())

    fpr_naive, fl_naive = false_positive_and_launch(z_primary, naive_bounds)
    fpr_obf, fl_obf = false_positive_and_launch(z_primary, bounds)
    print(f"Naive: FPR={fpr_naive:.4f}, false-launch={fl_naive:.4f}")
    print(f"OBF:   FPR={fpr_obf:.4f}, false-launch={fl_obf:.4f}")
    print(f"OBF boundaries: {bounds}")
    metrics = {
        "n_tests": N_TESTS,
        "n_looks": N_LOOKS,
        "final_n_per_arm": FINAL_N_PER_ARM,
        "alpha": 0.05,
        "seed": SEED,
        "guardrail_threshold": GUARDRAIL_THRESHOLD,
        "guardrails": 2,
        "obf_boundaries": bounds,
        "false_positive_rate": {"naive_peeking": round(fpr_naive, 4), "obf": round(fpr_obf, 4)},
        "false_launch_rate_with_guardrails": {"naive_peeking": round(fl_naive, 4), "obf": round(fl_obf, 4)},
        "wrong_launches_per_10000_tests": {
            "naive_peeking": round(fl_naive * 10000),
            "obf": round(fl_obf * 10000),
        },
        "fewer_wrong_launches_per_10000_with_obf": round((fl_naive - fl_obf) * 10000),
    }
    out = ROOT / "outputs" / "aa_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return metrics


if __name__ == "__main__":
    main()
