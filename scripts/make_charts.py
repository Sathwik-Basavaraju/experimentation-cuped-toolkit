"""Charts for the experimentation toolkit (matplotlib, optional).

Reads outputs/cuped_metrics.json and outputs/aa_metrics.json and
writes outputs/charts/power_comparison.png and
outputs/charts/fpr_comparison.png. If matplotlib is unavailable,
prints a note and exits cleanly — charts are illustrative, the
metrics JSON files are the source of truth.
Run from the project root: python3 scripts/make_charts.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping charts")
        return
    outdir = ROOT / "outputs" / "charts"
    outdir.mkdir(parents=True, exist_ok=True)
    cuped = json.loads((ROOT / "outputs" / "cuped_metrics.json").read_text())
    aa = json.loads((ROOT / "outputs" / "aa_metrics.json").read_text())
    # Power comparison at the CUPED-sized sample.
    labels = ["Naive", "CUPED"]
    vals = [cuped["empirical_power_at_cuped_n"]["naive"], cuped["empirical_power_at_cuped_n"]["cuped"]]
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.set_ylim(0, 1)
    ax.axhline(0.80, linestyle="--")
    ax.set_ylabel("Empirical power")
    ax.set_title(f"Power at n={cuped['sample_size_per_arm_with_cuped']}/arm (target 0.80)")
    fig.tight_layout()
    fig.savefig(outdir / "power_comparison.png", dpi=150)
    plt.close(fig)
    # FPR / false-launch comparison.
    cats = ["False-positive\nrate", "False-launch\nrate (guardrails)"]
    naive = [aa["false_positive_rate"]["naive_peeking"], aa["false_launch_rate_with_guardrails"]["naive_peeking"]]
    obf = [aa["false_positive_rate"]["obf"], aa["false_launch_rate_with_guardrails"]["obf"]]
    x = range(len(cats))
    fig, ax = plt.subplots()
    ax.bar([i - 0.2 for i in x], naive, width=0.4, label="Naive peeking")
    ax.bar([i + 0.2 for i in x], obf, width=0.4, label="O'Brien-Fleming")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("Rate")
    ax.legend()
    ax.set_title(f"A/A simulation, {aa['n_tests']:,} tests, 5 looks")
    fig.tight_layout()
    fig.savefig(outdir / "fpr_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Wrote charts to {outdir}")


if __name__ == "__main__":
    main()
