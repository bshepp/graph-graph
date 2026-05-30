"""
cap -> dimension finite-size scaling for the `grown` generator.

Question
--------
The `grown` topology (degree-capped frontier growth) produces a tunable
emergent dimension -- cap 6 -> ~2D, 8 -> ~3D at modest N. But is that law
*stable* and does it *converge to clean integers* as N grows, or does d_eff
drift? This is the cheapest scaling check in the program and the prerequisite
for trusting any larger run: it establishes that the dimension instrument and
the generator behave at scale, and it lets us extrapolate an asymptotic d_inf
per cap rather than quoting a single-N number. See FINDINGS.md ->
"Scaling directions" (this is step 1 of the budget-honest sequence).

Method
------
For each degree cap, each N on a geometric ladder, and several seeds:
  * build a `grown` graph (create_initial_graph, the single topology source),
  * measure the local effective dimension field on a sample of nodes using the
    *fixed* max_radius sparse path (dimension.fast_dimension_field) -- a fixed
    radius keeps the comparison across N honest: the estimator's regime gate,
    not a varying radius, decides where dimension is resolvable. Small N simply
    fails to resolve high d (defined_frac drops), which is itself informative.
We record, per (cap, N), the seed-averaged mean d_eff over *defined* nodes, its
spread, and the defined fraction. Per cap we then extrapolate d_inf by a linear
fit of d_eff vs 1/ln(N) (leading finite-size correction), using only points
where dimension is well enough resolved (defined_frac >= --min-defined).

Usage
-----
    python cap_dimension_scaling.py                              # default ladder
    python cap_dimension_scaling.py --caps 6 7 8 --nodes 1000 2000 5000 10000 20000 --seeds 3
    python cap_dimension_scaling.py --nodes 5000 20000 50000 100000 --samples 300
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import networkx as nx

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience, not a hard dependency
    def tqdm(x, **kwargs):
        return x

from simulation import create_initial_graph
from dimension import fast_dimension_field, dimension_stats


def measure_grown(cap: int, n: int, seed: int, max_radius: int,
                  samples: int) -> dict:
    """One (cap, N, seed): build a grown graph and measure its dimension field."""
    random.seed(seed)
    np.random.seed(seed)
    G = create_initial_graph(n, topology="grown", k=cap, seed=seed)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    field = fast_dimension_field(A, max_radius=max_radius, n_samples=samples)
    stats = dimension_stats(field, G.number_of_nodes())
    return {
        "cap": cap, "N": G.number_of_nodes(), "seed": seed,
        "d_eff_mean": stats["d_eff_mean"], "d_eff_median": stats["d_eff_median"],
        "d_eff_std": stats["d_eff_std"], "defined_frac": stats["defined_frac"],
        "n_defined": stats["n_defined"], "n_sampled": stats["n_sampled"],
    }


def aggregate(rows: list) -> dict:
    """Collapse per-seed rows into per-(cap, N) seed means + spreads."""
    agg = {}
    caps = sorted({r["cap"] for r in rows})
    for cap in caps:
        for n in sorted({r["N"] for r in rows if r["cap"] == cap}):
            grp = [r for r in rows if r["cap"] == cap and r["N"] == n]
            # Only average d_eff over seeds where dimension was actually defined.
            defined = [r for r in grp if r["n_defined"] > 0]
            d_vals = np.array([r["d_eff_mean"] for r in defined]) \
                if defined else np.array([])
            agg[(cap, n)] = {
                "cap": cap, "N": n, "n_seeds": len(grp),
                "d_eff": float(d_vals.mean()) if d_vals.size else float("nan"),
                "d_eff_seed_spread": float(d_vals.std()) if d_vals.size else 0.0,
                "defined_frac": float(np.mean([r["defined_frac"] for r in grp])),
            }
    return agg


def convergence(agg: dict, cap: int, min_defined: float, tol: float = 0.05):
    """Plateau diagnostic for d_eff(N).

    At a *fixed* radius, d_eff rises with N only until N is large enough that
    every radius clears the saturation gate, then it plateaus -- so the honest
    question is "has it plateaued, and at what integer?", not a 1/ln(N)
    extrapolation (which overshoots a saturating curve). We report the value at
    the largest resolved N, the step from the previous resolved N (delta), and
    whether it has plateaued (|delta| < tol).

    Returns dict with d_last, N_last, delta, nearest_int, plateaued, n_points.
    """
    pts = [v for (c, _), v in agg.items()
           if c == cap and v["defined_frac"] >= min_defined
           and np.isfinite(v["d_eff"])]
    pts.sort(key=lambda v: v["N"])
    if not pts:
        return {"d_last": float("nan"), "N_last": None, "delta": float("nan"),
                "nearest_int": None, "plateaued": False, "n_points": 0}
    d_last = pts[-1]["d_eff"]
    delta = (pts[-1]["d_eff"] - pts[-2]["d_eff"]) if len(pts) >= 2 else float("nan")
    nearest = int(round(d_last))
    plateaued = (len(pts) >= 2 and abs(delta) < tol)
    return {"d_last": d_last, "N_last": pts[-1]["N"], "delta": delta,
            "nearest_int": nearest, "plateaued": plateaued, "n_points": len(pts)}


def plot(agg: dict, dinf: dict, save_path: str):
    import matplotlib.pyplot as plt

    caps = sorted({k[0] for k in agg})
    cmap = plt.get_cmap("viridis")
    colors = {c: cmap(i / max(len(caps) - 1, 1)) for i, c in enumerate(caps)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for cap in caps:
        pts = sorted([v for (c, _), v in agg.items() if c == cap],
                     key=lambda v: v["N"])
        Ns = [v["N"] for v in pts]
        d = [v["d_eff"] for v in pts]
        err = [v["d_eff_seed_spread"] for v in pts]
        df = [v["defined_frac"] for v in pts]
        axes[0].errorbar(Ns, d, yerr=err, fmt="o-", color=colors[cap],
                         capsize=3, label=f"cap {cap}")
        conv = dinf.get(cap, {})
        if conv.get("N_last") is not None:
            tag = "plateau" if conv["plateaued"] else "rising"
            axes[0].annotate(f"{conv['d_last']:.2f} ({tag})",
                             xy=(Ns[-1], conv["d_last"]), fontsize=8,
                             color=colors[cap], va="bottom", ha="right")
        axes[1].plot(Ns, df, "o-", color=colors[cap], label=f"cap {cap}")

    for d_int in (2, 3, 4):
        axes[0].axhline(d_int, color="0.7", ls="--", lw=0.8, zorder=0)
    axes[0].set(xscale="log", xlabel="N (nodes)", ylabel="mean d_eff (defined nodes)",
                title="cap → dimension vs scale")
    axes[1].set(xscale="log", xlabel="N (nodes)", ylabel="defined_frac",
                title="dimension resolvability vs scale", ylim=(-0.02, 1.02))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("grown generator: does cap→d converge to clean integers as N grows?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    p = argparse.ArgumentParser(
        description="cap -> dimension finite-size scaling for the grown generator",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caps", type=int, nargs="+", default=[6, 7, 8])
    p.add_argument("--nodes", type=int, nargs="+",
                   default=[1000, 2000, 5000, 10000, 20000])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--samples", type=int, default=200,
                   help="nodes sampled per graph for the dimension field")
    p.add_argument("--max-radius", type=int, default=10,
                   help="fixed BFS radius (kept constant across N for comparability)")
    p.add_argument("--min-defined", type=float, default=0.3,
                   help="only extrapolate from (cap,N) points this well resolved")
    p.add_argument("--seed", type=int, default=0, help="base seed")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"cap->d scaling: caps={args.caps}, N={args.nodes}, seeds={args.seeds}, "
          f"max_radius={args.max_radius}, samples={args.samples}")

    rows = []
    jobs = [(cap, n, args.seed + s)
            for cap in args.caps for n in args.nodes for s in range(args.seeds)]
    for cap, n, seed in tqdm(jobs, desc="measuring"):
        rows.append(measure_grown(cap, n, seed, args.max_radius, args.samples))

    agg = aggregate(rows)
    dinf = {cap: convergence(agg, cap, args.min_defined) for cap in args.caps}

    print("\nPer-cap convergence (does d_eff plateau on a clean integer as N grows?):")
    print(f"  {'cap':>4} {'N_last':>7} {'d_eff':>7} {'delta':>8} "
          f"{'->int':>6} {'status':>10}")
    for cap in args.caps:
        c = dinf[cap]
        if c["N_last"] is None:
            print(f"  {cap:>4} {'--':>7} {'nan':>7}   (never resolved at "
                  f"defined_frac >= {args.min_defined})")
            continue
        status = "plateau" if c["plateaued"] else "still rising"
        print(f"  {cap:>4} {c['N_last']:>7} {c['d_last']:>7.3f} "
              f"{c['delta']:>8.3f} {c['nearest_int']:>6} {status:>12}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    csv_path = f"results/cap_dimension_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")
    plot(agg, dinf, f"plots/cap_dimension_{ts}.png")


if __name__ == "__main__":
    main()
