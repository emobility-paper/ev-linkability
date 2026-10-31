#!/usr/bin/env python3
"""Generate the data-driven paper figures (section 26).

Figure 3  retrieval Top-1/Top-5 vs candidate-set size K (per observer), with
          identity-level CI bands and the 1/K random baseline.
Figure 4  privacy-utility Pareto: attack ROC-AUC vs operational utility loss
          across mitigation profiles P0-P4, as small multiples over KPIs.

Auto-discovers outputs/tables/retrieval_*.csv and mitigation_*.csv. Schematic
figures (ecosystem, pipeline) are drawn separately and not produced here.

Run (after evaluate_retrieval / evaluate_mitigation):
    python src/make_figures.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_FLOAT_HINT = ("top", "mrr", "roc", "pr_", "auc", "energy", "js", "occupancy",
               "util", "random", "median", "shift", "prevalence")


def read_table(path: Path) -> List[Dict]:
    rows = []
    for r in csv.DictReader(path.open(newline="", encoding="utf-8")):
        parsed = {}
        for k, v in r.items():
            if any(h in k for h in _FLOAT_HINT):
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = float("nan")
            elif k in ("m", "K", "n_targets", "n_trials", "peak_hour_match"):
                parsed[k] = int(float(v)) if v not in ("", None) else 0
            else:
                parsed[k] = v
        rows.append(parsed)
    return rows


# --- Figure 3: retrieval curves ---------------------------------------------
def plot_retrieval(rows: List[Dict], out_base: Path, title: str) -> None:
    Ks = sorted({r["K"] for r in rows})
    ms = sorted({r["m"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)

    for m in ms:
        sub = sorted([r for r in rows if r["m"] == m], key=lambda r: r["K"])
        x = [r["K"] for r in sub]
        t1 = [r["top1"] for r in sub]
        lo = [r["top1_lo"] for r in sub]
        hi = [r["top1_hi"] for r in sub]
        t5 = [r["top5"] for r in sub]
        line, = ax.plot(x, t1, "-o", label=f"Top-1 (m={m})")
        ax.fill_between(x, lo, hi, alpha=0.15, color=line.get_color())
        ax.plot(x, t5, "--s", color=line.get_color(), alpha=0.7, label=f"Top-5 (m={m})")

    rnd = [1.0 / k for k in Ks]
    ax.plot(Ks, rnd, ":", color="0.5", label="random Top-1")

    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("Candidate-set size $K$")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Figure 4: privacy-utility Pareto ---------------------------------------
_UTIL = [
    ("daily_energy_mape", "Daily-energy MAPE"),
    ("arrival_js", "Arrival-hour JS div."),
    ("occupancy_mae", "Occupancy MAE"),
]


def plot_pareto(rows: List[Dict], out_base: Path, title: str) -> None:
    rows = sorted(rows, key=lambda r: r["profile"])
    profiles = [r["profile"] for r in rows]
    y = np.array([r["roc_auc"] for r in rows])
    ylo = np.array([r["roc_auc_lo"] for r in rows])
    yhi = np.array([r["roc_auc_hi"] for r in rows])

    fig, axes = plt.subplots(1, len(_UTIL), figsize=(12.5, 4.2),
                             constrained_layout=True, sharey=True)
    for ax, (key, label) in zip(axes, _UTIL):
        x = np.array([r.get(key, float("nan")) for r in rows])
        ax.plot(x, y, "-", color="0.6", zorder=1)
        ax.errorbar(x, y, yerr=[y - ylo, yhi - y], fmt="o", capsize=3, zorder=2)
        for xi, yi, p in zip(x, y, profiles):
            ax.annotate(p, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel(f"{label}  (utility loss \u2192)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Attack ROC-AUC (privacy risk $\\uparrow$)")
    fig.suptitle(title)

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args(argv)

    tdir = Path(args.tables_dir)
    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    made = []

    for path in sorted(tdir.glob("retrieval_*.csv")):
        obs = "v2" if path.stem.endswith("v2") else ("v1" if path.stem.endswith("v1") else "")
        out = odir / f"figure3_retrieval_{obs or path.stem}"
        plot_retrieval(read_table(path), out, f"Candidate-set retrieval ({obs})")
        made.append(f"{out}.pdf")

    for path in sorted(tdir.glob("mitigation_*.csv")):
        obs = "v2" if path.stem.endswith("v2") else ("v1" if path.stem.endswith("v1") else "")
        out = odir / f"figure4_privacy_utility_{obs or path.stem}"
        plot_pareto(read_table(path), out, f"Privacy\u2013utility trade-off ({obs})")
        made.append(f"{out}.pdf")

    if not made:
        print(f"No retrieval_*/mitigation_* tables found in {tdir}/")
        return 1
    print("Wrote figures:")
    for f in made:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
