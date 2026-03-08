"""
Animated visualization for graph-graph simulations.

Renders a live dashboard showing the graph evolving under local rules:
  - Nodes glow when active, dim when inactive
  - Edge thickness tracks reinforcement weights
  - Domain coloring shows majority-vote regions
  - Sparkline panels track key metrics over time

Usage:
    python animate.py --nodes 300 --steps 400 --rules activation majority --seed 42
    python animate.py --nodes 500 --steps 600 --rules activation reinforcement --save anim.gif
    python animate.py --nodes 200 --steps 300 --rules activation majority rewire --fps 20
"""

import argparse
import random
from typing import List, Dict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from simulation import create_initial_graph
from simulation_fast import FastGraph, FAST_RULES


# ── Colour palette (dark theme) ─────────────────────────────────────
BG        = "#0f0f1a"
PANEL_BG  = "#161625"
GRID_CLR  = "#2a2a40"
TEXT_CLR   = "#c8c8d8"
ACTIVE_CLR = "#ff6f3c"   # warm orange
INACTIVE_CLR = "#1e1e3a" # dark indigo
EDGE_CLR  = "#3a3a5c"
EDGE_HI   = "#ff9f6c"    # warm edge between co-active nodes
METRIC_CLRS = ["#00d4aa", "#ff6f3c", "#6fa8ff", "#c678dd"]


def _layout(G: nx.Graph, n: int) -> np.ndarray:
    """Compute a stable 2-D layout. Returns (n, 2) array."""
    if n <= 500:
        pos = nx.spring_layout(G, k=1.2 / np.sqrt(n), iterations=80, seed=0)
    else:
        pos = nx.spring_layout(G, k=1.2 / np.sqrt(n), iterations=40, seed=0)
    xy = np.array([pos[node] for node in G.nodes()])
    return xy


def _node_colors(active: np.ndarray) -> np.ndarray:
    """Map active (float 0/1) → RGBA array."""
    act_rgba = np.array(mcolors.to_rgba(ACTIVE_CLR))
    inact_rgba = np.array(mcolors.to_rgba(INACTIVE_CLR))
    a = active[:, None]
    return a * act_rgba + (1 - a) * inact_rgba


def _node_sizes(active: np.ndarray, base: float = 18, boost: float = 40) -> np.ndarray:
    return base + active * boost


def run_animation(
    n_nodes: int,
    n_steps: int,
    rules: List[str],
    topology: str = "small_world",
    seed: int | None = None,
    frame_interval: int = 1,
    fps: int = 15,
    save_path: str | None = None,
    dpi: int = 120,
):
    """Build and play/save the animation."""

    # ── Seed ─────────────────────────────────────────────────────────
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ── Build graph & fast representation ────────────────────────────
    G = create_initial_graph(n_nodes, topology, seed=seed)
    fg = FastGraph(G)

    rule_methods = []
    for r in rules:
        if r not in FAST_RULES:
            raise ValueError(f"Unknown rule: {r}")
        rule_methods.append(getattr(fg, FAST_RULES[r]))

    # ── Pre-compute layout on initial graph ──────────────────────────
    xy = _layout(G, n_nodes)

    # ── Pre-compute edge index arrays (row, col) for drawing ─────────
    coo = fg.A.tocoo()
    upper = coo.row < coo.col
    edge_r = coo.row[upper]
    edge_c = coo.col[upper]

    # ── Record frames ────────────────────────────────────────────────
    frame_active: List[np.ndarray] = []
    frame_weights: List[np.ndarray] = []  # weight per upper-triangle edge
    frame_metrics: Dict[str, list] = {
        "active_frac": [],
        "mean_weight": [],
        "clustering": [],
        "largest_cc": [],
    }

    def _snapshot():
        frame_active.append(fg.active.copy())
        # Extract weights for the upper-triangle edges
        w_coo = fg.weights.tocoo()
        w_upper = w_coo.row < w_coo.col
        frame_weights.append(w_coo.data[w_upper].copy())
        m = fg.compute_metrics()
        frame_metrics["active_frac"].append(m["n_active"] / n_nodes)
        frame_metrics["mean_weight"].append(m["mean_weight"])
        frame_metrics["clustering"].append(m["clustering"])
        frame_metrics["largest_cc"].append(m["largest_component"])

    _snapshot()  # step 0

    print(f"Simulating {n_steps} steps ({n_nodes} nodes, rules={rules})...")
    for step in range(1, n_steps + 1):
        for rm in rule_methods:
            rm()
        # Re-derive edge indices if rewire is active (topology changes)
        if "rewire" in rules:
            coo = fg.A.tocoo()
            upper = coo.row < coo.col
            edge_r = coo.row[upper]
            edge_c = coo.col[upper]
        if step % frame_interval == 0:
            _snapshot()

    n_frames = len(frame_active)
    print(f"Captured {n_frames} frames. Building animation...")

    # ── Figure layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = fig.add_gridspec(4, 5, hspace=0.45, wspace=0.5,
                          left=0.03, right=0.97, top=0.92, bottom=0.06)

    ax_graph = fig.add_subplot(gs[:, :3])
    ax_graph.set_facecolor(BG)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    metric_names = ["active_frac", "mean_weight", "clustering", "largest_cc"]
    metric_labels = ["Active Fraction", "Mean Edge Weight", "Clustering Coeff", "Largest Component"]
    ax_metrics = []
    for i in range(4):
        ax = fig.add_subplot(gs[i, 3:])
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=7)
        ax.set_ylabel(metric_labels[i], fontsize=7, color=TEXT_CLR, labelpad=2)
        ax.grid(True, color=GRID_CLR, linewidth=0.4, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
        ax_metrics.append(ax)

    title = fig.suptitle("", fontsize=13, color=TEXT_CLR, fontweight="bold")

    # ── Initial draw objects ─────────────────────────────────────────
    # Edges as a LineCollection-style scatter (we use plot segments)
    edge_lines = []
    for idx in range(len(edge_r)):
        u, v = edge_r[idx], edge_c[idx]
        (ln,) = ax_graph.plot(
            [xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
            color=EDGE_CLR, linewidth=0.3, alpha=0.25, solid_capstyle="round",
        )
        edge_lines.append(ln)

    colors0 = _node_colors(frame_active[0])
    sizes0 = _node_sizes(frame_active[0])
    scatter = ax_graph.scatter(
        xy[:, 0], xy[:, 1], c=colors0, s=sizes0,
        edgecolors="none", zorder=3,
    )

    # Metric sparklines
    metric_lines = []
    for i, key in enumerate(metric_names):
        (ln,) = ax_metrics[i].plot([], [], color=METRIC_CLRS[i], linewidth=1.2)
        metric_lines.append(ln)

    # ── Update function ──────────────────────────────────────────────
    def update(frame_idx):
        active = frame_active[frame_idx]

        # --- Nodes ---
        scatter.set_color(_node_colors(active))
        scatter.set_sizes(_node_sizes(active))

        # --- Edges ---
        if frame_idx < len(frame_weights):
            weights = frame_weights[frame_idx]

            # Rebuild edge lines if topology changed (rewire rule)
            if "rewire" in rules:
                # Edge count may have changed; redraw all edge lines
                # Remove old lines
                for ln in edge_lines:
                    ln.remove()
                edge_lines.clear()

                # Current edges from the stored frame's perspective:
                # We need matching edge indices — re-derive from fg at this point
                # Since we stored per-frame weights aligned with upper-triangle
                # at capture time, and edge_r/edge_c also update, we use the
                # last captured set. For simplicity with rewire, just update
                # coloring on the final set.
                coo_now = fg.A.tocoo()
                up = coo_now.row < coo_now.col
                er = coo_now.row[up]
                ec = coo_now.col[up]
                w = weights if len(weights) == len(er) else np.full(len(er), 0.5)
                active_bool = active.astype(bool)
                for idx in range(len(er)):
                    u, v = er[idx], ec[idx]
                    co_active = active_bool[u] and active_bool[v]
                    lw = 0.3 + w[idx] * 1.5 if co_active else 0.3
                    clr = EDGE_HI if co_active else EDGE_CLR
                    alph = 0.55 if co_active else 0.15
                    (ln,) = ax_graph.plot(
                        [xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                        color=clr, linewidth=lw, alpha=alph, solid_capstyle="round",
                    )
                    edge_lines.append(ln)
            else:
                active_bool = active.astype(bool)
                for idx in range(min(len(edge_lines), len(weights))):
                    u, v = edge_r[idx], edge_c[idx]
                    w = weights[idx]
                    co_active = active_bool[u] and active_bool[v]
                    lw = 0.3 + w * 1.5 if co_active else 0.3
                    clr = EDGE_HI if co_active else EDGE_CLR
                    alph = 0.55 if co_active else 0.15
                    edge_lines[idx].set_linewidth(lw)
                    edge_lines[idx].set_color(clr)
                    edge_lines[idx].set_alpha(alph)

        # --- Sparklines ---
        x_data = list(range(frame_idx + 1))
        for i, key in enumerate(metric_names):
            metric_lines[i].set_data(x_data, frame_metrics[key][: frame_idx + 1])
            ax_metrics[i].set_xlim(0, max(n_frames - 1, 1))
            vals = frame_metrics[key][: frame_idx + 1]
            if vals:
                lo, hi = min(vals), max(vals)
                margin = max((hi - lo) * 0.15, 0.01)
                ax_metrics[i].set_ylim(lo - margin, hi + margin)

        step_num = frame_idx * frame_interval
        title.set_text(
            f"graph-graph  ·  {topology}  ·  {' + '.join(rules)}  ·  "
            f"step {step_num}/{n_steps}  ·  {n_nodes} nodes"
        )

        return [scatter] + edge_lines + metric_lines + [title]

    # ── Build animation ──────────────────────────────────────────────
    anim = FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=False, repeat=True,
    )

    if save_path:
        ext = save_path.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            writer = "pillow"
        elif ext in ("mp4", "webm"):
            writer = "ffmpeg"
        else:
            writer = "pillow"
            save_path = save_path.rsplit(".", 1)[0] + ".gif"
        print(f"Saving to {save_path} ({writer}, {fps} fps)...")
        anim.save(save_path, writer=writer, fps=fps, dpi=dpi,
                  savefig_kwargs={"facecolor": BG})
        print(f"Saved {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Animated graph-graph visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--topology", type=str, default="small_world",
                        choices=["small_world", "scale_free", "lattice", "random"])
    parser.add_argument("--rules", type=str, nargs="+", default=["activation"],
                        choices=list(FAST_RULES.keys()))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frame-interval", type=int, default=2,
                        help="Capture a frame every N steps (default: 2)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--save", type=str, default=None,
                        help="Save to file (.gif or .mp4) instead of showing")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    run_animation(
        n_nodes=args.nodes,
        n_steps=args.steps,
        rules=args.rules,
        topology=args.topology,
        seed=args.seed,
        frame_interval=args.frame_interval,
        fps=args.fps,
        save_path=args.save,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
