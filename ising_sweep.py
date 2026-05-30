"""
Finite-size-scaling validation: the `majority` rule as a kinetic Ising /
majority-vote model on a 2D lattice.

Purpose
-------
Before trusting finite-size scaling (FSS) to hunt for *novel* phase transitions
in this project (e.g. shortcut-density -> dimensional onset under `prune`), we
validate the machinery on a transition whose physics is already known. The
`majority` rule with noise is the noisy majority-vote model: on a 2D lattice it
has a continuous order/disorder transition in the 2D Ising universality class,
driven by the noise q (which plays the role of temperature).

If our FSS pipeline -- order parameter, susceptibility, and Binder cumulant --
locates a single critical noise q_c (Binder curves for different L cross at one
point; susceptibility peaks grow with L) and the data collapse with 2D Ising
exponents works, then the machinery is trustworthy. See FINDINGS.md ->
"Scaling directions".

Method
------
For each lattice side L (N = L^2), each noise q, and several seeds:
  * start from a random binary configuration (create_initial_graph already
    randomises `state`), relax for `--equil` steps under `majority_vote(noise=q)`,
  * then time-sample the magnetization m = (1/N) sum_i (2 s_i - 1) every
    `--sample-interval` steps for `--sample` steps.
Per (L, q) we accumulate, over all samples and seeds:
  M   = <|m|>                          (order parameter)
  chi = N (<m^2> - <|m|>^2)            (susceptibility)
  U   = 1 - <m^4> / (3 <m^2>^2)        (Binder cumulant)
The Binder cumulant is the exponent-free locator: curves U(q) for different L
intersect at q_c.

Usage
-----
    python ising_sweep.py --quick                         # fast smoke (small L)
    python ising_sweep.py --sides 16 24 32 48 --seeds 6   # a real validation run
    python ising_sweep.py --noise-min 0.05 --noise-max 0.30 --noise-steps 14

Caveats
-------
* `grid_2d_graph` has *open* boundaries, so FSS corrections are larger than the
  textbook periodic case -- q_c and exponents come out approximate.
* The rule breaks ties with argmax (deterministically toward state 0), a weak
  symmetry-breaking field that rounds the Binder crossing slightly. We measure
  |m| and start symmetric to keep this small; it is a property of the rule, not
  the FSS machinery.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import networkx as nx

from simulation import create_initial_graph
from simulation_fast import FastGraph


# 2D Ising critical exponents (for the optional data collapse).
ISING_2D = {"beta_over_nu": 0.125, "gamma_over_nu": 1.75, "inv_nu": 1.0}


def _moments(abs_m, m2, m4, n):
    """Pack magnetization moments + derived FSS quantities into a result dict."""
    abs_m = np.array(abs_m)
    m2 = np.array(m2)
    m4 = np.array(m4)
    mean_absm = float(abs_m.mean())
    mean_m2 = float(m2.mean())
    mean_m4 = float(m4.mean())
    chi = n * (mean_m2 - mean_absm ** 2)
    binder = 1.0 - mean_m4 / (3.0 * mean_m2 ** 2) if mean_m2 > 0 else 0.0
    return {"M": mean_absm, "M2": mean_m2, "M4": mean_m4,
            "chi": float(chi), "binder": float(binder),
            "n_samples": int(abs_m.size)}


def measure_point_mvm(L, q, n_seeds, equil, sample, sample_interval, base_seed):
    """Textbook Z2-symmetric majority-vote model (de Oliveira) on the lattice.

    Checkerboard update: the 2D grid is bipartite, so we update one colour
    using the other's spins, then swap -- sequential-correct and vectorized,
    and free of the period-2 "blinking" that a fully synchronous update creates
    on a bipartite lattice. Each node takes its neighbour-majority spin with
    prob (1-q) and the minority spin with prob q; exact ties are broken at
    random. This is symmetric under sigma -> -sigma, so it has the clean
    2D-Ising-class order/disorder transition (q_c ~ 0.075 on a square lattice)
    that the FSS machinery should recover. Topology comes from
    create_initial_graph; only the update dynamics are local to this harness.
    """
    n = L * L
    # Two-colour the lattice from its row-major integer index (grid_2d_graph is
    # relabelled in row order, so node k sits at (k // L, k % L)).
    idx = np.arange(n)
    parity = ((idx // L) + (idx % L)) % 2
    colors = (np.where(parity == 0)[0], np.where(parity == 1)[0])
    abs_m, m2, m4 = [], [], []

    for s in range(n_seeds):
        seed = base_seed + s
        random.seed(seed)
        np.random.seed(seed)
        G = create_initial_graph(n, topology="lattice", seed=seed)
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
        sigma = np.where(np.random.random(n) < 0.5, 1.0, -1.0)

        def step(sig):
            for cells in colors:
                h = (A @ sig)[cells]
                maj = np.sign(h)
                ties = maj == 0.0
                if ties.any():
                    maj[ties] = np.where(
                        np.random.random(int(ties.sum())) < 0.5, 1.0, -1.0)
                flip = np.random.random(cells.size) < q
                sig[cells] = np.where(flip, -maj, maj)
            return sig

        for _ in range(equil):
            sigma = step(sigma)
        steps_since = 0
        for _ in range(sample):
            sigma = step(sigma)
            steps_since += 1
            if steps_since >= sample_interval:
                steps_since = 0
                m = float(sigma.mean())
                abs_m.append(abs(m)); m2.append(m * m); m4.append(m ** 4)

    res = _moments(abs_m, m2, m4, n)
    res.update({"L": L, "N": n, "q": q})
    return res


def measure_point_project(L, q, n_seeds, equil, sample, sample_interval,
                          base_seed):
    """The project's actual `majority` rule (rules.py / FastGraph), reset-noise.

    Kept for honest comparison: this rule breaks ties deterministically toward
    state 0, a strong symmetry-breaking field on an even-degree lattice, so it
    does NOT show a clean Z2 transition -- it is not a faithful Ising validator.
    """
    n = L * L
    abs_m, m2, m4 = [], [], []

    for s in range(n_seeds):
        seed = base_seed + s
        random.seed(seed)
        np.random.seed(seed)
        # create_initial_graph randomises `state` using np.random, so the seed
        # above controls the initial configuration too.
        G = create_initial_graph(n, topology="lattice", seed=seed)
        fg = FastGraph(G)

        for _ in range(equil):
            fg.majority_vote(num_states=2, noise=q)

        steps_since = 0
        for _ in range(sample):
            fg.majority_vote(num_states=2, noise=q)
            steps_since += 1
            if steps_since >= sample_interval:
                steps_since = 0
                m = 2.0 * fg.state.mean() - 1.0
                abs_m.append(abs(m))
                m2.append(m * m)
                m4.append(m ** 4)

    res = _moments(abs_m, m2, m4, n)
    res.update({"L": L, "N": n, "q": q})
    return res


def estimate_qc_from_binder(rows: list) -> float | None:
    """Locate q_c as the mean pairwise crossing of consecutive-L Binder curves."""
    sides = sorted({r["L"] for r in rows})
    if len(sides) < 2:
        return None

    def curve(L):
        pts = sorted([(r["q"], r["binder"]) for r in rows if r["L"] == L])
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    crossings = []
    for L1, L2 in zip(sides[:-1], sides[1:]):
        q1, u1 = curve(L1)
        q2, u2 = curve(L2)
        # Common q grid (they share the swept noises in practice).
        qs = np.array(sorted(set(q1).intersection(set(q2))))
        if len(qs) < 2:
            continue
        d = np.array([u1[list(q1).index(q)] - u2[list(q2).index(q)] for q in qs])
        sign = np.sign(d)
        for i in range(len(qs) - 1):
            if sign[i] == 0:
                crossings.append(qs[i])
            elif sign[i] * sign[i + 1] < 0:
                # Linear interpolation of the zero crossing of the difference.
                t = d[i] / (d[i] - d[i + 1])
                crossings.append(qs[i] + t * (qs[i + 1] - qs[i]))
    return float(np.mean(crossings)) if crossings else None


def plot_sweep(rows: list, qc: float | None, save_path: str,
               collapse: bool = False):
    import matplotlib.pyplot as plt

    sides = sorted({r["L"] for r in rows})
    cmap = plt.get_cmap("viridis")
    colors = {L: cmap(i / max(len(sides) - 1, 1)) for i, L in enumerate(sides)}

    npanels = 4 if collapse else 3
    fig, axes = plt.subplots(1, npanels, figsize=(5 * npanels, 4.2))

    def curve(L, key):
        pts = sorted([(r["q"], r[key]) for r in rows if r["L"] == L])
        return [p[0] for p in pts], [p[1] for p in pts]

    for L in sides:
        qx, my = curve(L, "M")
        axes[0].plot(qx, my, "o-", color=colors[L], ms=4, label=f"L={L}")
        qx, cy = curve(L, "chi")
        axes[1].plot(qx, cy, "o-", color=colors[L], ms=4, label=f"L={L}")
        qx, uy = curve(L, "binder")
        axes[2].plot(qx, uy, "o-", color=colors[L], ms=4, label=f"L={L}")

    axes[0].set(xlabel="noise q", ylabel="M = <|m|>", title="Order parameter")
    axes[1].set(xlabel="noise q", ylabel="chi", title="Susceptibility")
    axes[2].set(xlabel="noise q", ylabel="Binder U", title="Binder cumulant")
    for ax in axes[:3]:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    if qc is not None:
        for ax in axes[:3]:
            ax.axvline(qc, color="red", ls="--", lw=1, alpha=0.6)
        axes[2].annotate(f"q_c = {qc:.4f}", xy=(qc, axes[2].get_ylim()[0]),
                         color="red", fontsize=8, ha="left", va="bottom")

    if collapse and qc is not None:
        bn = ISING_2D["beta_over_nu"]
        inv_nu = ISING_2D["inv_nu"]
        for L in sides:
            qx, my = curve(L, "M")
            qx = np.array(qx); my = np.array(my)
            x = (qx - qc) * L ** inv_nu
            y = my * L ** bn
            axes[3].plot(x, y, "o-", color=colors[L], ms=4, label=f"L={L}")
        axes[3].set(xlabel=r"$(q-q_c)\,L^{1/\nu}$", ylabel=r"$M\,L^{\beta/\nu}$",
                    title="Data collapse (2D Ising)")
        axes[3].legend(fontsize=7)
        axes[3].grid(alpha=0.3)

    fig.suptitle("majority-vote FSS on a 2D lattice "
                 "(validation of the scaling machinery)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    p = argparse.ArgumentParser(
        description="Majority-vote / Ising finite-size-scaling validation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mvm", "project"], default="mvm",
                   help="mvm = symmetric majority-vote (clean Ising validator); "
                        "project = the actual rules.py majority rule (biased)")
    p.add_argument("--sides", type=int, nargs="+", default=[16, 24, 32],
                   help="lattice sides L (N = L^2)")
    p.add_argument("--noise-min", type=float, default=0.04,
                   help="below ~0.03 the dynamics freeze into metastable "
                        "domains; default brackets the transition cleanly")
    p.add_argument("--noise-max", type=float, default=0.12)
    p.add_argument("--noise-steps", type=int, default=13)
    p.add_argument("--noises", type=float, nargs="+", default=None,
                   help="explicit noise grid (overrides --noise-min/max/steps)")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--equil", type=int, default=1500, help="relaxation steps")
    p.add_argument("--sample", type=int, default=1500, help="sampling steps")
    p.add_argument("--sample-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="base seed")
    p.add_argument("--collapse", action="store_true",
                   help="add a 2D-Ising data-collapse panel")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke preset (small L, short runs)")
    args = p.parse_args()

    if args.quick:
        args.sides = [12, 18, 24]
        args.equil, args.sample, args.sample_interval = 400, 400, 4
        args.seeds = 2
        if args.noises is None:
            args.noise_steps = 9

    random.seed(args.seed)
    np.random.seed(args.seed)

    noises = (args.noises if args.noises is not None
              else list(np.linspace(args.noise_min, args.noise_max,
                                     args.noise_steps)))

    measure = measure_point_mvm if args.model == "mvm" else measure_point_project
    print(f"FSS sweep [{args.model}]: sides={args.sides}, {len(noises)} noises "
          f"in [{noises[0]:.3f}, {noises[-1]:.3f}], seeds={args.seeds}, "
          f"equil={args.equil}, sample={args.sample}")

    rows = []
    total = len(args.sides) * len(noises)
    done = 0
    for L in args.sides:
        for q in noises:
            row = measure(L, float(q), args.seeds, args.equil,
                          args.sample, args.sample_interval, args.seed)
            rows.append(row)
            done += 1
            print(f"  [{done}/{total}] L={L:3d} q={q:.4f}  "
                  f"M={row['M']:.3f} chi={row['chi']:.2f} U={row['binder']:.3f}")

    qc = estimate_qc_from_binder(rows)
    if qc is not None:
        print(f"\nBinder-crossing estimate: q_c ~= {qc:.4f}")
    elif len({r['L'] for r in rows}) < 2:
        print("\nNeed >= 2 lattice sizes to locate a Binder crossing.")
    else:
        print("\nNo Binder crossing found in this noise window -- the curves "
              "don't intersect here. Widen --noise-min/--noise-max to bracket "
              "the transition (the ordered phase has U -> 2/3).")

    ts = time.strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    csv_path = f"results/ising_sweep_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")

    plot_sweep(rows, qc, f"plots/ising_sweep_{ts}.png", collapse=args.collapse)


if __name__ == "__main__":
    main()
