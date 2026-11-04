"""Sequential testing: O'Brien-Fleming boundaries and a launch rule.

O'Brien-Fleming (OBF) boundaries via the Lan-DeMets alpha-spending
approximation. The critical values below are the standard OBF
boundaries for a two-sided alpha = 0.05 test with 5 equally spaced
interim looks, as used in the simulation in scripts/simulate_aa.py.
They are conservative early (large critical value) and approach the
fixed-sample 1.96 at the final look, which is what keeps the overall
false-positive rate near 5% despite repeated peeking. The simulation
in this repo empirically validates that claim — see outputs/metrics.json
for the actual false-positive rate measured here, not a textbook value.

Launch rule: launch only if the primary metric is significant at the
OBF boundary at the look where it first crosses AND neither guardrail
metric shows significant harm (guardrail z < -1.96) at that look.
Guardrails in the simulation are null metrics (true effect zero).
"""
import numpy as np

# Standard OBF critical values, two-sided alpha=0.05, 5 looks.
_OBF_5 = [4.56, 3.23, 2.64, 2.33, 2.04]


def obf_boundaries(n_looks=5, alpha=0.05):
    """Return OBF critical z-values for n_looks equally spaced looks."""
    if n_looks == 5 and alpha == 0.05:
        return list(_OBF_5)
    if n_looks <= 0:
        raise ValueError("n_looks must be positive")
    # Fallback approximation for other look counts: scale the OBF
    # spending shape. Only the 5-look case is used and validated here.
    t = np.linspace(1 / n_looks, 1.0, n_looks)
    z_alpha = 1.96 if alpha == 0.05 else float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - alpha / 2))
    # OBF spending: critical value approx z_alpha / sqrt(t), capped sensibly.
    return [float(min(5.0, z_alpha / np.sqrt(ti))) for ti in t]


def is_significant(z, look_index, boundaries=None):
    """True if |z| exceeds the OBF boundary at look_index (0-based)."""
    if boundaries is None:
        boundaries = obf_boundaries(5)
    return abs(float(z)) > boundaries[look_index]


def launch_decision(primary_z, look_index, guardrail_zs, boundaries=None, guardrail_threshold=-1.96):
    """Launch decision.

    Returns True only if the primary metric is OBF-significant at
    look_index AND every guardrail z is above guardrail_threshold
    (i.e. no significant harm). Otherwise False.
    """
    if not is_significant(primary_z, look_index, boundaries):
        return False
    return all(float(g) > guardrail_threshold for g in guardrail_zs)
