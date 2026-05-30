"""
Generate a curated set of showcase animations for graph-graph.

Each animation is designed to highlight a specific rule, graph-theory
concept, or emergent phenomenon with parameters tuned for visual impact.

Usage:
    python showcase.py                # generate all showcases
    python showcase.py --pick 1 3 5   # generate only specific ones
    python showcase.py --list         # list available showcases
"""

import argparse
import sys
from pathlib import Path
from typing import List

from animate import run_animation
from traverse import run_traversal


# ═══════════════════════════════════════════════════════════════════════
# Showcase definitions
# ═══════════════════════════════════════════════════════════════════════
# Each entry: (filename, title, description, kwargs for run_animation)

SHOWCASES = [
    # ── 1. Epidemic spreading (activation only) ──────────────────────
    (
        "01_epidemic_spread.gif",
        "Epidemic Spreading (SIS Model)",
        "Pure activation rule on a small-world network.  Watch a handful\n"
        "of 'infected' nodes (orange) seed an epidemic that sweeps through\n"
        "the graph, reaches a dynamic equilibrium, then fluctuates as nodes\n"
        "recover and get re-infected.  Classic SIS dynamics emerge from\n"
        "nothing but a local spread-or-decay coin flip.",
        dict(
            n_nodes=350,
            n_steps=300,
            rules=["activation"],
            topology="small_world",
            seed=12,
            frame_interval=2,
            fps=14,
            dpi=110,
        ),
    ),
    # ── 2. Hebbian edge reinforcement ────────────────────────────────
    (
        "02_hebbian_reinforcement.gif",
        "Hebbian Learning -- Edges That Fire Together Wire Together",
        "Activation + edge reinforcement on a scale-free hub network.\n"
        "Edges between co-active nodes thicken and glow; all others slowly\n"
        "fade.  Over time the network develops a backbone of strong\n"
        "connections tracing the most-used pathways -- a Hebbian memory\n"
        "forming in the edge weights with no central controller.",
        dict(
            n_nodes=300,
            n_steps=400,
            rules=["activation", "reinforcement"],
            topology="scale_free",
            seed=7,
            frame_interval=2,
            fps=14,
            dpi=110,
        ),
    ),
    # ── 3. Majority vote — domain formation ─────────────────────────
    (
        "03_majority_domains.gif",
        "Majority Vote -- Spontaneous Domain Formation",
        "Pure majority-vote rule on a 2D lattice.  Nodes adopt whichever\n"
        "binary state most of their neighbors hold; a pinch of noise\n"
        "prevents instant freezing.  Watch coherent domains of same-state\n"
        "nodes nucleate, grow, and compete -- a phase-ordering process\n"
        "analogous to crystal grain growth or Ising-model coarsening.",
        dict(
            n_nodes=400,
            n_steps=250,
            rules=["majority"],
            topology="lattice",
            seed=21,
            frame_interval=1,
            fps=12,
            dpi=110,
        ),
    ),
    # ── 4. Random rewiring — small-world emergence ──────────────────
    (
        "04_rewiring_small_world.gif",
        "Random Rewiring -- Small-World Emergence",
        "Activation + rewiring on a lattice.  Each step a few edges are\n"
        "randomly relocated, gradually adding long-range shortcuts.\n"
        "Watch the clustering coefficient drop and the largest component\n"
        "stay intact -- the signature of a small-world transition.\n"
        "The graph starts local and becomes globally connected.",
        dict(
            n_nodes=250,
            n_steps=500,
            rules=["activation", "rewire"],
            topology="lattice",
            seed=33,
            frame_interval=3,
            fps=14,
            dpi=110,
        ),
    ),
    # ── 5. Scale-free hubs — preferential attachment topology ───────
    (
        "05_scale_free_hubs.gif",
        "Scale-Free Network -- Hub Dynamics",
        "Activation spreading on a Barabasi-Albert scale-free graph.\n"
        "A few hub nodes have many connections and act as super-spreaders;\n"
        "the epidemic ignites fast through hubs then trickles into the\n"
        "periphery.  Contrast with the slower, more uniform spread on\n"
        "small-world or lattice topologies.",
        dict(
            n_nodes=350,
            n_steps=250,
            rules=["activation"],
            topology="scale_free",
            seed=5,
            frame_interval=2,
            fps=14,
            dpi=110,
        ),
    ),
    # ── 6. Full emergence — all rules combined ──────────────────────
    (
        "06_full_emergence.gif",
        "Full Emergence -- All Four Rules",
        "All rules active together on a small-world network: activation\n"
        "spreads, edges reinforce, majority vote forms domains, and\n"
        "random rewiring reshapes the topology.  This is the main\n"
        "experiment: does combining simple local rules produce structure\n"
        "that none of them would create alone?  Watch for coordinated\n"
        "clustering shifts, domain locking, and backbone formation.",
        dict(
            n_nodes=350,
            n_steps=500,
            rules=["activation", "reinforcement", "majority", "rewire"],
            topology="small_world",
            seed=42,
            frame_interval=3,
            fps=15,
            dpi=110,
        ),
    ),
    # ── 7. Erdős-Rényi random graph — baseline ─────────────────────
    (
        "07_random_baseline.gif",
        "Random Graph Baseline (Erdos-Renyi)",
        "Activation + majority vote on a purely random Erdos-Renyi graph.\n"
        "With no geometric structure or preferential attachment, does\n"
        "anything interesting still emerge?  This is the null-hypothesis\n"
        "control -- any structure seen here must come purely from the\n"
        "rules, not from the initial topology.",
        dict(
            n_nodes=350,
            n_steps=300,
            rules=["activation", "majority"],
            topology="random",
            seed=17,
            frame_interval=2,
            fps=14,
            dpi=110,
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Traversal showcases (walk diffusion / ball growth -- see traverse.py)
# ═══════════════════════════════════════════════════════════════════════
# Each entry: (filename, title, description, mode, kwargs for run_traversal)

TRAVERSE_SHOWCASES = [
    # ── 8. Quantum vs classical walk on a grown geometric graph ──────
    (
        "08_walk_quantum_vs_classical.gif",
        "Quantum vs Classical Walk -- Ballistic Meets Diffusive",
        "A walk spreading from one seed node on a grown (emergent-geometry)\n"
        "graph, classical on the left, quantum on the right.  The classical\n"
        "walk stays balled up near the seed (diffusive, ~sqrt(t) spread);\n"
        "the quantum walk rushes outward along the branches (ballistic, ~t)\n"
        "with interference fringes.  Same graph, same start -- the physics\n"
        "of the walk is the only difference.",
        "walk",
        dict(topology="grown", nodes=600, seed=0, frames=60, walk_time=14.0,
             fps=12, dpi=110),
    ),
    # ── 9. Ball growth on a lattice -- dimension locks in at d=2 ─────
    (
        "09_ball_growth_lattice.gif",
        "Ball Growth -- Watching Dimension Become Defined (d -> 2)",
        "The geodesic ball |B(v, r)| growing ring-by-ring on a 2D lattice --\n"
        "the exact BFS the dimension estimator runs.  The side panel plots\n"
        "log|B| vs log r and calls the real local_dimension(); watch the\n"
        "points fall onto a clean line and d_eff lock in at ~2.0 once there\n"
        "are enough unsaturated radii to earn the verdict.",
        "ball",
        dict(topology="lattice", nodes=2500, seed=0, max_radius=12, fps=5,
             dpi=110),
    ),
    # ── 10. Ball growth on a random graph -- dimension never defines ─
    (
        "10_ball_growth_expander.gif",
        "Ball Growth on an Expander -- Dimension Stays Undefined",
        "The same ball-growth traversal on an Erdos-Renyi random graph.\n"
        "With no geometry, the ball engulfs the whole graph in two or three\n"
        "hops -- there is no power-law regime, so the estimator honestly\n"
        "reports d_eff = undefined.  The direct visual contrast with the\n"
        "lattice is why dimension is a real, falsifiable signal here.",
        "ball",
        dict(topology="random", nodes=2000, seed=0, max_radius=8, fps=4,
             dpi=110),
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def _registry():
    """Unified list of (kind, filename, title, desc, kwargs) for all showcases.

    ``kind`` is ``"animate"`` for rule-dynamics showcases (run via
    ``animate.run_animation``) or a traversal mode (``"walk"`` / ``"ball"``,
    run via ``traverse.run_traversal``).
    """
    reg = [("animate", f, t, d, kw) for (f, t, d, kw) in SHOWCASES]
    reg += [(mode, f, t, d, kw) for (f, t, d, mode, kw) in TRAVERSE_SHOWCASES]
    return reg


def list_showcases():
    print("\nAvailable showcases:\n")
    for i, (_kind, filename, title, desc, _) in enumerate(_registry(), 1):
        print(f"  {i}. {title}")
        for line in desc.strip().splitlines():
            print(f"     {line.strip()}")
        print(f"     -> {filename}")
        print()


def generate(indices: List[int], output_dir: str):
    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True)
    registry = _registry()

    total = len(indices)
    for count, idx in enumerate(indices, 1):
        kind, filename, title, desc, kwargs = registry[idx]
        save_path = str(outdir / filename)

        print(f"\n{'=' * 70}")
        print(f"[{count}/{total}]  {title}")
        print(f"{'=' * 70}")
        for line in desc.strip().splitlines():
            print(f"  {line.strip()}")
        print(f"  -> {save_path}\n")

        if kind == "animate":
            run_animation(save_path=save_path, **kwargs)
        else:
            run_traversal(kind, save=save_path, **kwargs)

    print(f"\n{'=' * 70}")
    print(f"Done -- {total} animations saved to {outdir}/")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate showcase animations for graph-graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pick", type=int, nargs="+", default=None,
        help="Generate only these showcase numbers (1-indexed)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available showcases and exit",
    )
    parser.add_argument(
        "--output-dir", type=str, default="showcase",
        help="Output directory (default: showcase/)",
    )
    args = parser.parse_args()

    if args.list:
        list_showcases()
        return

    n_showcases = len(_registry())
    if args.pick:
        indices = []
        for p in args.pick:
            if p < 1 or p > n_showcases:
                print(f"Error: showcase {p} doesn't exist (range 1-{n_showcases})")
                sys.exit(1)
            indices.append(p - 1)
    else:
        indices = list(range(n_showcases))

    generate(indices, args.output_dir)


if __name__ == "__main__":
    main()
