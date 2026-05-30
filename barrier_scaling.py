"""
Quantify the bootstrapping barrier: extent-growth vs rewiring steps and vs N.

The bootstrapping-barrier claim (FINDINGS.md) is that *fixed-N local rewiring
cannot grow extent (diameter) from an expander*. So far that rests on three
rules failing at one N plus a structural diameter argument. This driver turns
the assertion into a measurement:

  * **vs steps** -- does the achieved diameter grow as a rewiring rule runs, or
    plateau? (the "crumple, don't unfold" claim);
  * **vs N (the flagship)** -- fit extent ~ N^alpha for each rewiring rule. The
    barrier predicts alpha ~ 0 (extent stays ~log N, expander-like, independent
    of N), versus a true d-dimensional graph's alpha = 1/d. The `grown`
    generator is the positive control: it should give alpha ~ 1/2 (cap 6 ~ 2D).
A *measured* obstruction exponent (alpha ~ 0 for rewiring, ~0.5 for grown) is
far more citable than an asserted one, and the gap between the two is the whole
result -- it widens with N.

Extent is estimated by the double-sweep heuristic (BFS to the farthest node,
then BFS from there) on the largest connected component -- a tight diameter
lower bound, far cheaper than exact all-pairs. Fragmentation is itself part of
the barrier ("either fragments it or stalls"), so we also track the
largest-component fraction.

Usage
-----
    python barrier_scaling.py                                   # default sweep
    python barrier_scaling.py --rules triadic geometrize ricci --nodes 500 1000 2000 4000 8000
    python barrier_scaling.py --steps 400 --seeds 5
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
except ImportError:
    def tqdm(x, **kwargs):
        return x

from simulation import create_initial_graph
from rules import get_rule


def estimate_extent(G: nx.Graph, n_sweeps: int = 4) -> tuple:
    """(diameter_estimate, mean_eccentricity, lcc_frac) on the largest component.

    Double-sweep: from a random source, BFS to the farthest node u, then BFS
    from u; the farthest distance is a tight lower bound on the diameter. Take
    the max over a few sweeps.
    """
    n_total = G.number_of_nodes()
    if n_total == 0:
        return 0.0, 0.0, 0.0
    cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(cc)
    nodes = list(cc)
    if len(nodes) < 2:
        return 0.0, 0.0, len(cc) / n_total

    sweep_diams = []
    for _ in range(n_sweeps):
        s = nodes[np.random.randint(len(nodes))]
        d1 = nx.single_source_shortest_path_length(H, s)
        u = max(d1, key=d1.get)
        d2 = nx.single_source_shortest_path_length(H, u)
        sweep_diams.append(max(d2.values()))
    return float(max(sweep_diams)), float(np.mean(sweep_diams)), len(cc) / n_total


def run_trajectory(rule_name: str, n: int, seed: int, steps: int,
                   interval: int, start: str, mean_degree: int) -> list:
    """Apply `rule_name` to a `start` graph; record extent every `interval`."""
    random.seed(seed)
    np.random.seed(seed)
    topo = {"grown": "grown", "lattice": "lattice"}.get(start, "random")
    G = create_initial_graph(n, topology=topo, k=mean_degree, seed=seed)

    traj = []
    d, mecc, lcc = estimate_extent(G)
    traj.append({"step": 0, "diam": d, "mean_ecc": mecc, "lcc_frac": lcc})

    # Static reference series (no rewiring): measure the start graph only.
    #   lattice -> positive control, diameter ~ 2*sqrt(N) (alpha = 0.5)
    #   grown   -> locally-2D growth, but globally compressed (alpha < 0.5)
    #   none    -> the random/expander baseline (alpha ~ 0, diameter ~ log N)
    if rule_name in ("none", "grown", "lattice"):
        return traj

    rule = get_rule(rule_name)
    for step in range(1, steps + 1):
        rule(G)
        if step % interval == 0:
            d, mecc, lcc = estimate_extent(G)
            traj.append({"step": step, "diam": d, "mean_ecc": mecc,
                         "lcc_frac": lcc})
    return traj


def fit_exponent(Ns: np.ndarray, extents: np.ndarray):
    """Fit extent ~ N^alpha (log-log slope) and extent ~ a + b*ln(N).

    Returns (alpha, loglog_r2, b_logN, logN_r2). alpha ~ 0 with a good a+b*lnN
    fit is the expander/barrier signature; alpha ~ 1/d is a d-dimensional graph.
    """
    mask = (extents > 0) & np.isfinite(extents)
    if mask.sum() < 2:
        return float("nan"), 0.0, float("nan"), 0.0
    lN, lE = np.log(Ns[mask]), np.log(extents[mask])
    alpha, c = np.polyfit(lN, lE, 1)
    r2_ll = _r2(lN, lE, alpha, c)
    b, a = np.polyfit(np.log(Ns[mask]), extents[mask], 1)
    r2_log = _r2(np.log(Ns[mask]), extents[mask], b, a)
    return float(alpha), float(r2_ll), float(b), float(r2_log)


def _r2(x, y, slope, intercept):
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def plot(rows, finals, exps, save_path):
    import matplotlib.pyplot as plt

    rules = sorted({r["rule"] for r in rows})
    cmap = plt.get_cmap("tab10")
    colors = {rl: cmap(i) for i, rl in enumerate(sorted(
        set(rules) | {"grown", "none"}))}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel 1: extent vs rewiring step (largest N, rewiring rules only) ---
    rewiring = [r for r in rules if r not in ("none", "grown")]
    Nmax = max(r["N"] for r in rows)
    for rl in rewiring:
        steps = sorted({r["step"] for r in rows if r["rule"] == rl and r["N"] == Nmax})
        means = [np.mean([r["diam"] for r in rows
                          if r["rule"] == rl and r["N"] == Nmax and r["step"] == s])
                 for s in steps]
        axes[0].plot(steps, means, "o-", color=colors[rl], label=rl)
    axes[0].set(xlabel="rewiring step", ylabel="diameter (double-sweep)",
                title=f"extent vs steps (N={Nmax}) — does it unfold or crumple?")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # --- Panel 2: final extent vs N (the flagship), log-log ---
    series = sorted(finals.keys())
    for rl in series:
        pts = sorted(finals[rl], key=lambda p: p[0])
        Ns = [p[0] for p in pts]
        ext = [p[1] for p in pts]
        alpha = exps.get(rl, (float("nan"),))[0]
        axes[1].plot(Ns, ext, "o-", color=colors.get(rl, "k"),
                     label=f"{rl}  (α={alpha:.2f})")
    # Reference slopes anchored at the smallest N.
    Ns_ref = np.array(sorted({p[0] for v in finals.values() for p in v}))
    if Ns_ref.size:
        base = Ns_ref[0]
        anchor = np.median([p[1] for v in finals.values() for p in v
                            if p[0] == base] or [3.0])
        for d_int, style in ((2, "--"), (3, ":")):
            axes[1].plot(Ns_ref, anchor * (Ns_ref / base) ** (1.0 / d_int),
                         style, color="0.5", lw=1,
                         label=f"N^(1/{d_int}) ref")
        axes[1].plot(Ns_ref, anchor + 2.0 * (np.log(Ns_ref) - np.log(base)),
                     "-.", color="0.7", lw=1, label="~log N ref")
    axes[1].set(xscale="log", yscale="log", xlabel="N (nodes)",
                ylabel="final diameter", title="extent vs N — barrier (α≈0) vs growth (α≈1/d)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle("Quantifying the bootstrapping barrier: "
                 "local rewiring cannot grow extent from an expander", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    p = argparse.ArgumentParser(
        description="Quantify the bootstrapping barrier (extent-growth scaling)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rules", nargs="+", default=["triadic", "geometrize", "ricci"],
                   help="rewiring rules to test (started from a random graph)")
    p.add_argument("--nodes", type=int, nargs="+",
                   default=[1000, 2000, 4000, 8000, 16000])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--interval", type=int, default=50)
    p.add_argument("--mean-degree", type=int, default=6,
                   help="mean degree of the random start / cap of the grown control")
    p.add_argument("--no-control", action="store_true",
                   help="skip the grown (positive) and none (baseline) controls")
    p.add_argument("--seed", type=int, default=0, help="base seed")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Rewiring rules (random start) + baseline + positive control + grown.
    series_specs = [(rl, "random") for rl in args.rules]
    if not args.no_control:
        series_specs += [("none", "random"), ("lattice", "lattice"),
                         ("grown", "grown")]

    print(f"barrier scaling: rules={args.rules}, N={args.nodes}, "
          f"seeds={args.seeds}, steps={args.steps}, mean_degree={args.mean_degree}")

    rows = []
    jobs = [(rl, start, n, args.seed + s)
            for (rl, start) in series_specs
            for n in args.nodes for s in range(args.seeds)]
    for rl, start, n, seed in tqdm(jobs, desc="trajectories"):
        traj = run_trajectory(rl, n, seed, args.steps, args.interval, start,
                              args.mean_degree)
        for rec in traj:
            rows.append({"rule": rl, "N": n, "seed": seed, **rec})

    # Final extent per (series, N): mean over seeds of the last-step diameter.
    finals = {}
    series = [s[0] for s in series_specs]
    for rl in series:
        pts = []
        for n in args.nodes:
            last_step = max(r["step"] for r in rows if r["rule"] == rl and r["N"] == n)
            vals = [r["diam"] for r in rows
                    if r["rule"] == rl and r["N"] == n and r["step"] == last_step]
            pts.append((n, float(np.mean(vals))))
        finals[rl] = pts

    exps = {}
    print("\nMeasured extent-vs-N scaling  (extent ~ N^alpha):")
    print(f"  {'series':>11} {'alpha':>7} {'loglog_R2':>10} {'b(lnN)':>8} "
          f"{'lnN_R2':>7}   interpretation")
    for rl in series:
        Ns = np.array([p[0] for p in finals[rl]], dtype=float)
        ext = np.array([p[1] for p in finals[rl]], dtype=float)
        alpha, r2ll, b, r2log = fit_exponent(Ns, ext)
        exps[rl] = (alpha, r2ll, b, r2log)
        if rl == "lattice":
            interp = "POSITIVE CONTROL: polynomial extent (expect alpha~0.5)"
        elif rl == "grown":
            interp = "locally-2D growth, globally compressed"
        elif rl == "none":
            interp = "expander baseline (diameter ~ log N)"
        elif alpha < 0.15:
            interp = "BARRIER: stays ~log N, no polynomial growth"
        else:
            interp = "extent grows with N (check vs baseline)"
        print(f"  {rl:>11} {alpha:>7.3f} {r2ll:>10.3f} {b:>8.3f} {r2log:>7.3f}"
              f"   {interp}")

    # Growth factor: how much did rewiring move the diameter off its start?
    print("\nExtent growth factor (final diameter / initial), rewiring rules:")
    for rl in args.rules:
        ratios = []
        for n in args.nodes:
            for s in range(args.seeds):
                seed = args.seed + s
                t0 = [r["diam"] for r in rows if r["rule"] == rl and r["N"] == n
                      and r["seed"] == seed and r["step"] == 0]
                tl_step = max(r["step"] for r in rows if r["rule"] == rl
                              and r["N"] == n and r["seed"] == seed)
                tl = [r["diam"] for r in rows if r["rule"] == rl and r["N"] == n
                      and r["seed"] == seed and r["step"] == tl_step]
                if t0 and tl and t0[0] > 0:
                    ratios.append(tl[0] / t0[0])
        if ratios:
            print(f"  {rl:>11}: {np.mean(ratios):.2f}x  "
                  f"(>1 grows extent, ~1 stalls, <1 crumples)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    csv_path = f"results/barrier_scaling_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")
    plot(rows, finals, exps, f"plots/barrier_scaling_{ts}.png")


if __name__ == "__main__":
    main()
