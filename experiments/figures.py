"""README figures, regenerated from the results manifests — pipeline-generated
like every quoted number (run after the eval battery; `run_all.sh` does).

  assets/beta_ladder.svg — β̂ per ranker on the held-out test split, 95%
  match-clustered CIs. Transparent background + mid-gray text so it reads on
  both GitHub themes; `svg.hashsalt` pinned so regeneration diffs cleanly.

  assets/dbeta_power.svg — the open Q−BC gap vs corpus size: Δβ̂ is a
  match-clustered mean, so its CI half-width shrinks as 1/√(corpus multiple);
  the marked crossing (from the manifest's nstar keys) is conditional on the
  point estimate being the true effect and the split shares staying fixed.

Run:
  DOTA2AD_ROOT=work pixi run python experiments/figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dota2ad.core import default_paths  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INK = "#7d8590"          # legible on light and dark GitHub themes
MODEL = "#388bfd"
TABLE = "#8b949e"
CONTROL = "#d29922"


def main() -> int:
    res = default_paths().root / "results"

    def load(name):
        return json.loads((res / f"{name}.json").read_text())

    cr, sr, tr = load("stats-causal-rank"), load("static-rank"), load("trial")
    rows = [  # bottom-to-top display order
        ("scrambled control", cr["beta_perm"], cr["beta_perm_lo"], cr["beta_perm_hi"], CONTROL),
        ("win-rate table", sr["wr_raw_beta"], sr["wr_raw_beta_lo"], sr["wr_raw_beta_hi"], TABLE),
        ("pair-synergy table", sr["pair_raw_beta"], sr["pair_raw_beta_lo"], sr["pair_raw_beta_hi"], TABLE),
        ("popularity", sr["static_beta"], sr["static_beta_lo"], sr["static_beta_hi"], TABLE),
        ("Trial — direct causal fit", tr["beta_trial"], tr["beta_trial_lo"], tr["beta_trial_hi"], MODEL),
        ("BC — human consensus", cr["beta_bc"], cr["beta_bc_lo"], cr["beta_bc_hi"], MODEL),
        ("Q — recommender", cr["beta_q"], cr["beta_q_lo"], cr["beta_q_hi"], MODEL),
    ]

    plt.rcParams.update({
        "svg.hashsalt": "dota2ad",
        "font.size": 11,
        "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ys = range(len(rows))
    for y, (_label, b, lo, hi, color) in zip(ys, rows, strict=True):
        ax.errorbar(b, y, xerr=[[b - lo], [hi - b]], fmt="o", color=color,
                    ecolor=color, elinewidth=2, capsize=3, markersize=6)
    ax.axvline(0.0, color=INK, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.set_xlabel("β̂ — causal ranking skill (composite, held-out test; 95% match-clustered CI)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(y=0.12)
    fig.tight_layout()
    out = REPO / "assets"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "beta_ladder.svg", transparent=True, metadata={"Date": None},
                bbox_inches="tight")
    print(f"wrote {out / 'beta_ladder.svg'}")

    b, lo, hi = cr["dbeta"], cr["dbeta_lo"], cr["dbeta_hi"]
    h0 = (hi - lo) / 2
    xstar, n_raw = cr["nstar_dbeta_x"], cr["n_raw_matches"]
    xs = np.linspace(1.0, math.ceil(xstar * 1.35), 300)
    hw = h0 / np.sqrt(xs)
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.fill_between(xs, b - hw, b + hw, color=MODEL, alpha=0.15, linewidth=0)
    ax.plot(xs, b - hw, color=MODEL, linewidth=1.0, linestyle="--")
    ax.plot(xs, b + hw, color=MODEL, linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.errorbar([1.0], [b], yerr=[[b - lo], [hi - b]], fmt="o", color=MODEL,
                ecolor=MODEL, elinewidth=2, capsize=3, markersize=6)
    ax.annotate(f"today\n{b:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                (1.0, b), xytext=(1.25, b + 0.001), va="top", color=INK, fontsize=10)
    ax.axvline(xstar, color=INK, linewidth=0.8, linestyle=":", alpha=0.8)
    ax.annotate(f"CI excludes 0 at ≈{xstar:.1f}×\n(≈{cr['nstar_dbeta_matches'] / 1000:,.0f}k "
                "collected matches) —\nif the point estimate is the true gap",
                (xstar, b + h0 / math.sqrt(xstar)),
                xytext=(xstar - 0.2, hi + 0.004), ha="right", va="bottom",
                color=INK, fontsize=10)
    ax.set_xlabel(f"corpus size, × the current {n_raw / 1000:,.0f}k collected matches "
                  "(fixed split shares)")
    ax.set_ylabel("Δβ̂  (Q − BC)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out / "dbeta_power.svg", transparent=True, metadata={"Date": None},
                bbox_inches="tight")
    print(f"wrote {out / 'dbeta_power.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
