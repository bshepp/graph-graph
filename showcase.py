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
        "Hebbian Learning — Edges That Fire Together Wire Together",
        "Activation + edge reinforcement on a scale-free hub network.\n"
        "Edges between co-active nodes thicken and glow; all others slowly\n"
        "fade.  Over time the network develops a backbone of strong\n"
        "connections tracing the most-used pathways — a Hebbian memory\n"
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
        "Majority Vote — Spontaneous Domain Formation",
        "Pure majority-vote rule on a 2D lattice.  Nodes adopt whichever\n"
        "binary state most of their neighbors hold; a pinch of noise\n"
        "prevents instant freezing.  Watch coherent domains of same-state\n"
        "nodes nucleate, grow, and compete — a phase-ordering process\n"
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
        "Random Rewiring — Small-World Emergence",
        "Activation + rewiring on a lattice.  Each step a few edges are\n"
        "randomly relocated, gradually adding long-range shortcuts.\n"
        "Watch the clustering coefficient drop and the largest component\n"
        "stay intact — the signature of a small-world transition.\n"
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
        "Scale-Free Network — Hub Dynamics",
        "Activation spreading on a Barabási-Albert scale-free graph.\n"
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
        "Full Emergence — All Four Rules",
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
        "Random Graph Baseline (Erdős–Rényi)",
        "Activation + majority vote on a purely random Erdős-Rényi graph.\n"
        "With no geometric structure or preferential attachment, does\n"
        "anything interesting still emerge?  This is the null-hypothesis\n"
        "control — any structure seen here must come purely from the\n"
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
# Runner
# ═══════════════════════════════════════════════════════════════════════

def list_showcases():
    print("\nAvailable showcases:\n")
    for i, (filename, title, desc, _) in enumerate(SHOWCASES, 1):
        print(f"  {i}. {title}")
        for line in desc.strip().splitlines():
            print(f"     {line.strip()}")
        print(f"     → {filename}")
        print()


def generate(indices: List[int], output_dir: str):
    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True)

    total = len(indices)
    for count, idx in enumerate(indices, 1):
        filename, title, desc, kwargs = SHOWCASES[idx]
        save_path = str(outdir / filename)

        print(f"\n{'═' * 70}")
        print(f"[{count}/{total}]  {title}")
        print(f"{'═' * 70}")
        for line in desc.strip().splitlines():
            print(f"  {line.strip()}")
        print(f"  → {save_path}\n")

        run_animation(save_path=save_path, **kwargs)

    print(f"\n{'═' * 70}")
    print(f"Done — {total} animations saved to {outdir}/")
    print(f"{'═' * 70}\n")


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

    if args.pick:
        indices = []
        for p in args.pick:
            if p < 1 or p > len(SHOWCASES):
                print(f"Error: showcase {p} doesn't exist (range 1-{len(SHOWCASES)})")
                sys.exit(1)
            indices.append(p - 1)
    else:
        indices = list(range(len(SHOWCASES)))

    generate(indices, args.output_dir)


if __name__ == "__main__":
    main()
