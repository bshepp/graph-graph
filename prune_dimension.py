"""
prune + shortcut-density -> a tunable emergent dimension (a continuum knob).

Question
--------
FINDINGS flagged `prune` as the candidate for a *phase transition* hunt
("shortcut-density -> dimensional onset"). A scout overturned that premise:
pruning a Watts-Strogatz ring (k=6) to convergence does NOT show a sharp
dimensional onset -- `defined_frac` stays ~1 across the whole range -- and
instead the pruned graph's emergent dimension slides *continuously* from ~1
(an intact ring) to ~2 as the rewire probability `p` rises, with a curve that
is essentially the SAME at N = 2e3, 8e3, 3.2e4. That is the signature of a
converged *crossover*, not a critical point: there is no sharpening with N for
finite-size scaling to latch onto. So `prune` is not a phase transition -- it
is a third continuum dimension knob, alongside the `grown` degree cap.

This driver banks that result honestly:
  1. maps d(p) across N and confirms it is N-independent (a crossover);
  2. quantifies "not a transition" by showing max |dd/dp| does NOT grow with N
     (the dual of a diverging susceptibility -- it stays bounded);
  3. keeps the "is it real geometry?" controls inline (clustering stays high;
     `--validate-real` compares to an Erdos-Renyi graph at matched mean degree,
     which is an undefined expander -- so the dimension is structure, not a
     low-degree artifact).

Mechanism: `prune` removes zero-triangle ("shortcut") edges to convergence, so
it peels a WS graph back toward the high-overlap backbone that survived
rewiring. At low p that backbone is a near-pure ring (d~1); as p rises the
rewired-in edges that happen to sit in triangles survive and cross-link the
backbone into a more 2D mesh (d->~2). The local effective dimension is genuine
(clean power-law ball growth, high clustering), not a sparse-random artifact.

Usage
-----
    python prune_dimension.py                       # default p x N sweep
    python prune_dimension.py --validate-real        # ER-control sanity check
    python prune_dimension.py --nodes 2000 8000 32000 --ps 0.05 0.1 0.2 0.4 0.8 --seeds 3
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
from simulation_fast import run_fast_simulation
from dimension import fast_dimension_field, dimension_stats, estimate_max_radius


def prune_to_convergence(G: nx.Graph, chunk: int = 25,
                         max_rounds: int = 400) -> nx.Graph:
    """Apply the fast `prune` rule until the edge count stops changing.

    `prune` self-terminates once every surviving edge sits in a triangle, so a
    fixed-point is well defined; we detect it by a stable edge count.
    """
    prev = G.number_of_edges()
    for _ in range(max_rounds):
        G = run_fast_simulation(G, ['prune'], chunk)['final_graph']
        now = G.number_of_edges()
        if now == prev:
            break
        prev = now
    return G


def measure_pruned(p: float, n: int, seed: int, k: int,
                   max_radius: int, samples: int) -> dict:
    """One (p, N, seed): build a WS ring, prune to convergence, measure it."""
    random.seed(seed)
    np.random.seed(seed)
    G = create_initial_graph(n, topology="small_world", k=k, p=p, seed=seed)
    G = prune_to_convergence(G)

    n_now = G.number_of_nodes()
    lcc = len(max(nx.connected_components(G), key=len)) / n_now
    mean_deg = 2 * G.number_of_edges() / n_now
    clustering = nx.average_clustering(G)

    A = nx.to_scipy_sparse_array(G, weight=None, format="csr", dtype=np.float32)
    field = fast_dimension_field(A, max_radius=max_radius, n_samples=samples)
    stats = dimension_stats(field, n_now)
    return {
        "p": p, "N": n, "seed": seed,
        "d_eff_median": stats["d_eff_median"], "d_eff_mean": stats["d_eff_mean"],
        "defined_frac": stats["defined_frac"], "clustering": clustering,
        "mean_deg": mean_deg, "lcc_frac": lcc,
    }


def aggregate(rows: list) -> dict:
    """Collapse per-seed rows into per-(p, N) seed means + spreads."""
    agg = {}
    ps = sorted({r["p"] for r in rows})
    Ns = sorted({r["N"] for r in rows})
    for p in ps:
        for n in Ns:
            grp = [r for r in rows if r["p"] == p and r["N"] == n]
            if not grp:
                continue
            d = np.array([r["d_eff_median"] for r in grp], dtype=float)
            agg[(p, n)] = {
                "p": p, "N": n, "n_seeds": len(grp),
                "d_eff": float(np.nanmean(d)),
                "d_eff_seed_spread": float(np.nanstd(d)),
                "defined_frac": float(np.mean([r["defined_frac"] for r in grp])),
                "clustering": float(np.mean([r["clustering"] for r in grp])),
                "mean_deg": float(np.mean([r["mean_deg"] for r in grp])),
                "lcc_frac": float(np.mean([r["lcc_frac"] for r in grp])),
            }
    return agg


def crossover_diagnostics(agg: dict, ps: list, Ns: list) -> dict:
    """Two numbers that distinguish a crossover from a critical point.

    * N-independence: at each p, the spread of d_eff across N. Small spread ->
      the d(p) curve has converged (a crossover); a true transition would keep
      shifting/sharpening with N.
    * Non-sharpening: per N, the maximum adjacent slope |Δd/Δp|. If this peak
      slope does NOT grow with N, there is no diverging response -- the dual of
      a non-diverging susceptibility, i.e. no critical point.
    """
    # N-independence: max over p of std_N(d_eff)
    per_p_spread = []
    for p in ps:
        ds = [agg[(p, n)]["d_eff"] for n in Ns if (p, n) in agg
              and np.isfinite(agg[(p, n)]["d_eff"])]
        if len(ds) >= 2:
            per_p_spread.append(np.std(ds))
    max_N_spread = float(np.max(per_p_spread)) if per_p_spread else float("nan")

    # Non-sharpening: peak adjacent slope per N
    peak_slope = {}
    for n in Ns:
        curve = [(p, agg[(p, n)]["d_eff"]) for p in ps if (p, n) in agg
                 and np.isfinite(agg[(p, n)]["d_eff"])]
        curve.sort()
        slopes = [abs((curve[i + 1][1] - curve[i][1]) /
                      (curve[i + 1][0] - curve[i][0]))
                  for i in range(len(curve) - 1)]
        peak_slope[n] = float(np.max(slopes)) if slopes else float("nan")
    return {"max_N_spread": max_N_spread, "peak_slope": peak_slope}


def validate_real(n: int, k: int, max_radius: int, samples: int):
    """Control: is the dimension real geometry or a low-degree artifact?

    Compare the pruned WS graph to an Erdos-Renyi graph at the SAME mean degree.
    Real geometry -> pruned-WS defined with high clustering; ER -> undefined
    expander with ~0 clustering. Decisive separation = the crossover is real.
    """
    print(f"Reality control (N={n}): pruned-WS vs ER at matched mean degree\n")
    print(f"  {'p':>5} {'graph':16s} {'deg':>5} {'clus':>6} {'def':>6} {'d_med':>7}")
    for p in (0.1, 0.3, 0.5):
        random.seed(0); np.random.seed(0)
        G = create_initial_graph(n, topology="small_world", k=k, p=p, seed=0)
        G = prune_to_convergence(G)
        R = estimate_max_radius(G)
        md = 2 * G.number_of_edges() / G.number_of_nodes()
        A = nx.to_scipy_sparse_array(G, weight=None, format="csr", dtype=np.float32)
        st = dimension_stats(fast_dimension_field(A, max_radius=R,
                                                  n_samples=samples), n)
        print(f"  {p:>5.2f} {'pruned-WS':16s} {md:>5.2f} "
              f"{nx.average_clustering(G):>6.3f} {st['defined_frac']:>6.3f} "
              f"{st['d_eff_median']:>7.2f}")
        random.seed(0); np.random.seed(0)
        ER = nx.erdos_renyi_graph(n, md / (n - 1), seed=0)
        Ae = nx.to_scipy_sparse_array(ER, weight=None, format="csr", dtype=np.float32)
        ste = dimension_stats(fast_dimension_field(Ae, max_radius=R,
                                                   n_samples=samples), n)
        print(f"  {'':>5} {'ER matched':16s} {2*ER.number_of_edges()/n:>5.2f} "
              f"{nx.average_clustering(ER):>6.3f} {ste['defined_frac']:>6.3f} "
              f"{ste['d_eff_median']:>7.2f}")


def plot(agg: dict, ps: list, Ns: list, save_path: str):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(i / max(len(Ns) - 1, 1)) for i, n in enumerate(Ns)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for n in Ns:
        pts = sorted([agg[(p, n)] for p in ps if (p, n) in agg],
                     key=lambda v: v["p"])
        xs = [v["p"] for v in pts]
        axes[0].errorbar(xs, [v["d_eff"] for v in pts],
                         yerr=[v["d_eff_seed_spread"] for v in pts],
                         fmt="o-", color=colors[n], capsize=3, label=f"N={n}")
        axes[1].plot(xs, [v["clustering"] for v in pts], "o-", color=colors[n],
                     label=f"N={n}")
        axes[2].plot(xs, [v["defined_frac"] for v in pts], "o-",
                     color=colors[n], label=f"N={n}")

    for d_int in (1, 2):
        axes[0].axhline(d_int, color="0.7", ls="--", lw=0.8, zorder=0)
    axes[0].set(xlabel="rewire prob p (shortcut density)",
                ylabel="median d_eff (defined nodes)",
                title="d(p): a continuum knob, N-independent")
    axes[1].set(xlabel="rewire prob p", ylabel="avg clustering",
                title="clustering stays high → real local structure")
    axes[2].set(xlabel="rewire prob p", ylabel="defined_frac",
                title="defined everywhere → no dimensional onset", ylim=(-0.02, 1.05))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("prune + shortcut-density: a tunable emergent dimension "
                 "(crossover, not a transition)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    ap = argparse.ArgumentParser(
        description="prune + shortcut-density -> tunable emergent dimension",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, nargs="+", default=[2000, 8000, 32000])
    ap.add_argument("--ps", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--k", type=int, default=6, help="WS ring range (mean degree)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=300,
                    help="nodes sampled per graph for the dimension field")
    ap.add_argument("--max-radius", type=int, default=10,
                    help="fixed BFS radius (kept constant across N)")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--validate-real", action="store_true",
                    help="run the ER-control reality check and exit")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate_real:
        validate_real(max(args.nodes), args.k, args.max_radius, args.samples)
        return

    print(f"prune d(p) scaling: N={args.nodes}, ps={args.ps}, k={args.k}, "
          f"seeds={args.seeds}, max_radius={args.max_radius}")

    jobs = [(p, n, args.seed + s)
            for p in args.ps for n in args.nodes for s in range(args.seeds)]
    rows = [measure_pruned(p, n, seed, args.k, args.max_radius, args.samples)
            for p, n, seed in tqdm(jobs, desc="measuring")]

    agg = aggregate(rows)
    Ns = sorted({r["N"] for r in rows})
    diag = crossover_diagnostics(agg, args.ps, Ns)

    print("\nd(p) master curve (seed-averaged median d_eff; ~N-independent):")
    head = f"  {'p':>5} " + " ".join(f"N={n:<7d}" for n in Ns) + "  clus  def"
    print(head)
    for p in args.ps:
        cells = []
        for n in Ns:
            v = agg.get((p, n))
            cells.append(f"{v['d_eff']:8.2f}" if v else f"{'--':>8}")
        ref = agg.get((p, Ns[-1]))
        tail = (f"  {ref['clustering']:.2f}  {ref['defined_frac']:.2f}"
                if ref else "")
        print(f"  {p:>5.2f} " + " ".join(cells) + tail)

    print("\nCrossover, not a transition:")
    print(f"  N-independence : max over p of std_N(d_eff) = "
          f"{diag['max_N_spread']:.3f}  (small -> curve converged)")
    print("  Non-sharpening : peak |dd/dp| per N "
          "(flat -> no diverging response):")
    for n in Ns:
        print(f"      N={n:<7d} peak slope = {diag['peak_slope'][n]:.2f}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    csv_path = f"results/prune_dimension_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")
    plot(agg, args.ps, Ns, f"plots/prune_dimension_{ts}.png")


if __name__ == "__main__":
    main()
