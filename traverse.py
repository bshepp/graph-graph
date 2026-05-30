"""
Animated graph *traversals* for graph-graph.

Two complementary traversal animations on a single graph (a results pickle or a
freshly-built topology), sharing one layout + frame-render core:

  walk -- a walk spreading from a seed node. Classical random walk (diffusive)
          and continuous-time quantum walk (ballistic, interference) shown
          side-by-side, nodes coloured by probability at time t. Reuses the
          matrix walks in braket_walks.py.

  ball -- the geodesic ball |B(v, r)| growing ring-by-ring: the exact BFS
          traversal the dimension estimator runs. A side panel shows log|B|
          vs log r accumulating and calls the *real* local_dimension() from
          dimension.py, so you watch d_eff become defined (a disk on a
          lattice) or never define (an expander engulfed in a few hops).

Usage:
    # Build a graph on the fly
    python traverse.py --mode ball --topology lattice --nodes 900 --seed 0 --save ball.gif
    python traverse.py --mode walk --topology grown   --nodes 600 --seed 0 --save walk.gif

    # Traverse a completed simulation's final graph
    python traverse.py results/run_TIMESTAMP.pkl --mode walk
    python traverse.py results/run_TIMESTAMP.pkl --mode ball --start 0
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

from simulation import create_initial_graph
from dimension import ball_sizes, local_dimension, estimate_max_radius
from braket_walks import quantum_walk_ctqw


# ── Colour palette (shared dark theme, matches animate.py) ──────────────
BG        = "#0f0f1a"
PANEL_BG  = "#161625"
GRID_CLR  = "#2a2a40"
TEXT_CLR  = "#c8c8d8"
EDGE_CLR  = "#2a2a44"
SEED_CLR  = "#36e0ff"   # cyan ring on the start node
FRONT_CLR = "#ff6f3c"   # warm orange frontier
DIM_CLR   = "#1e1e3a"   # unvisited / background


# ======================================================================
# Shared: load-or-build the graph, lay it out, pick a start node
# ======================================================================

def load_or_build(args) -> tuple[nx.Graph, str]:
    """Return (graph relabelled to 0..n-1, label string for the title)."""
    if args.input:
        with open(args.input, "rb") as f:
            results = pickle.load(f)
        G = results["final_graph"]
        params = results.get("params", {})
        rules = params.get("rules", "?")
        label = f"{Path(args.input).name}  ·  rules={rules}"
    else:
        G = create_initial_graph(args.nodes, args.topology, seed=args.seed)
        label = f"{args.topology}  ·  {G.number_of_nodes()} nodes"

    # Consecutive integer labels so node id == array index everywhere.
    G = nx.convert_node_labels_to_integers(G)
    return G, label


def layout(G: nx.Graph, seed: int | None) -> np.ndarray:
    """Stable 2-D spring layout (display only -- no coordinates feed any rule)."""
    n = G.number_of_nodes()
    iters = 80 if n <= 500 else 40
    pos = nx.spring_layout(G, k=1.2 / np.sqrt(max(n, 1)), iterations=iters,
                           seed=0 if seed is None else seed)
    return np.array([pos[v] for v in G.nodes()])


def pick_start(G: nx.Graph, xy: np.ndarray, start_arg: int | None) -> int:
    """Chosen start node, or the node nearest the layout centroid for framing."""
    if start_arg is not None:
        if not 0 <= start_arg < G.number_of_nodes():
            raise ValueError(f"--start {start_arg} out of range [0, {G.number_of_nodes()})")
        return start_arg
    centroid = xy.mean(axis=0)
    return int(np.argmin(((xy - centroid) ** 2).sum(axis=1)))


def edge_collection(G: nx.Graph, xy: np.ndarray) -> LineCollection:
    """One LineCollection for all edges (static across frames)."""
    segs = np.array([[xy[u], xy[v]] for u, v in G.edges()])
    return LineCollection(segs, colors=EDGE_CLR, linewidths=0.35, alpha=0.35,
                          zorder=1)


# ======================================================================
# Mode: ball growth (the dimension estimator's BFS, animated)
# ======================================================================

def animate_ball(G, xy, start, label, args):
    n = G.number_of_nodes()
    dist = nx.single_source_shortest_path_length(G, start)
    dist_arr = np.full(n, -1)
    for node, d in dist.items():
        dist_arr[node] = d
    reachable_max = int(dist_arr.max())
    max_r = min(reachable_max, args.max_radius or reachable_max)

    # Full ball-count profile once; each frame uses the prefix up to radius r.
    full_counts = ball_sizes(G, start, max_r)         # [(r, |B(r)|), ...]
    full_log_r = np.log([r for r, _ in full_counts])
    full_log_c = np.log([c for _, c in full_counts])

    fig = plt.figure(figsize=(13, 7), facecolor=BG)
    gs = fig.add_gridspec(1, 5, wspace=0.28, left=0.02, right=0.97,
                          top=0.9, bottom=0.1)
    ax_g = fig.add_subplot(gs[0, :3]); ax_g.set_facecolor(BG)
    ax_g.set_aspect("equal"); ax_g.axis("off")
    ax_f = fig.add_subplot(gs[0, 3:]); ax_f.set_facecolor(PANEL_BG)
    ax_f.tick_params(colors=TEXT_CLR, labelsize=8)
    for s in ax_f.spines.values():
        s.set_color(GRID_CLR)
    ax_f.grid(True, color=GRID_CLR, lw=0.4, alpha=0.5)
    ax_f.set_xlabel("log r", color=TEXT_CLR, fontsize=9)
    ax_f.set_ylabel("log |B(v, r)|", color=TEXT_CLR, fontsize=9)
    ax_f.set_xlim(full_log_r.min() - 0.15, full_log_r.max() + 0.15)
    ax_f.set_ylim(full_log_c.min() - 0.3, full_log_c.max() + 0.3)

    ax_g.add_collection(edge_collection(G, xy))
    base_size = max(6, 1600 / np.sqrt(n))
    scatter = ax_g.scatter(xy[:, 0], xy[:, 1], s=base_size, zorder=3,
                           edgecolors="none")
    # Cyan ring marks the seed.
    ax_g.scatter([xy[start, 0]], [xy[start, 1]], s=base_size * 3,
                 facecolors="none", edgecolors=SEED_CLR, linewidths=1.6, zorder=4)

    fit_pts, = ax_f.plot([], [], "o", color=FRONT_CLR, ms=7, zorder=3)
    fit_line, = ax_f.plot([], [], "-", color=SEED_CLR, lw=1.6, alpha=0.9, zorder=2)
    verdict = ax_f.text(0.04, 0.96, "", transform=ax_f.transAxes, va="top",
                        ha="left", color=TEXT_CLR, fontsize=11, fontfamily="monospace")
    title = fig.suptitle("", color=TEXT_CLR, fontsize=13, fontweight="bold")

    interior = plt.get_cmap("viridis")

    def update(frame):
        r = frame + 1
        colors = np.tile(mcolors.to_rgba(DIM_CLR), (n, 1))
        sizes = np.full(n, base_size)
        inside = (dist_arr >= 0) & (dist_arr < r)
        frontier = dist_arr == r
        # Interior shaded by depth; frontier flares orange.
        if inside.any():
            shade = dist_arr[inside] / max(r, 1)
            colors[inside] = interior(0.25 + 0.6 * shade)
        colors[frontier] = mcolors.to_rgba(FRONT_CLR)
        sizes[frontier] = base_size * 2.2
        colors[start] = mcolors.to_rgba(SEED_CLR)
        scatter.set_color(colors)
        scatter.set_sizes(sizes)

        prefix = full_counts[:r]
        fit_pts.set_data(np.log([rr for rr, _ in prefix]),
                         np.log([cc for _, cc in prefix]))
        ball_now = prefix[-1][1]
        d_eff, r2 = local_dimension(prefix, n)
        if np.isnan(d_eff):
            fit_line.set_data([], [])
            verdict.set_text(f"r = {r}\n|B| = {ball_now}\nd_eff = undefined")
            verdict.set_color("#ff9f6c")
        else:
            lr = np.log([rr for rr, _ in prefix])
            lc = np.log([cc for _, cc in prefix])
            # Reference line of slope d_eff anchored at the data's centroid.
            anchor_x, anchor_y = lr.mean(), lc.mean()
            xs = np.array([full_log_r.min(), full_log_r.max()])
            fit_line.set_data(xs, anchor_y + d_eff * (xs - anchor_x))
            verdict.set_text(f"r = {r}\n|B| = {ball_now}\nd_eff = {d_eff:.2f}  (R²={r2:.3f})")
            verdict.set_color("#7CFC9A")

        title.set_text(f"ball growth from node {start}  ·  {label}  ·  "
                       f"radius {r}/{max_r}")
        return [scatter, fit_pts, fit_line, verdict, title]

    anim = FuncAnimation(fig, update, frames=max_r, interval=1000 // args.fps,
                         blit=False, repeat=True)
    _finish(anim, fig, args)


# ======================================================================
# Mode: walk diffusion (classical vs quantum, side-by-side)
# ======================================================================

def animate_walk(G, xy, start, label, args):
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)

    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply

    times = np.linspace(args.t0, args.walk_time, args.frames)

    # Quantum: one batched expm_multiply over the whole time grid.
    psi0 = np.zeros(n, dtype=np.complex128); psi0[start] = 1.0
    psi_all = expm_multiply(-1j * A, psi0, start=float(times[0]),
                            stop=float(times[-1]), num=len(times), endpoint=True)
    q_frames = np.abs(psi_all) ** 2

    # Classical: incremental mat-vec at integer steps mapped from the times.
    deg = np.asarray(A.sum(axis=1)).flatten(); deg[deg == 0] = 1
    P = (A @ sp.diags(1.0 / deg)).tocsr()
    p = np.zeros(n); p[start] = 1.0
    c_frames, step = [], 0
    for t in times:
        target = max(int(round(t)), 1)
        while step < target:
            p = P @ p; step += 1
        c_frames.append(p.copy())
    c_frames = np.array(c_frames)

    tvd = 0.5 * np.abs(c_frames - q_frames).sum(axis=1)

    fig = plt.figure(figsize=(14, 7.5), facecolor=BG)
    gs = fig.add_gridspec(5, 2, hspace=0.0, wspace=0.04, left=0.02, right=0.98,
                          top=0.9, bottom=0.08)
    ax_c = fig.add_subplot(gs[:4, 0]); ax_q = fig.add_subplot(gs[:4, 1])
    ax_t = fig.add_subplot(gs[4, :])
    for ax, ttl in ((ax_c, "classical (diffusive)"), (ax_q, "quantum (ballistic)")):
        ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl, color=TEXT_CLR, fontsize=11, pad=2)
        ax.add_collection(edge_collection(G, xy))

    base_size = max(6, 1600 / np.sqrt(n))
    cmap = plt.get_cmap("inferno")

    def make_scatter(ax):
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=base_size, c=np.zeros(n),
                        cmap=cmap, vmin=0, vmax=1, zorder=3, edgecolors="none")
        ax.scatter([xy[start, 0]], [xy[start, 1]], s=base_size * 3,
                   facecolors="none", edgecolors=SEED_CLR, linewidths=1.6, zorder=4)
        return sc

    sc_c, sc_q = make_scatter(ax_c), make_scatter(ax_q)

    ax_t.set_facecolor(PANEL_BG); ax_t.tick_params(colors=TEXT_CLR, labelsize=8)
    for s in ax_t.spines.values():
        s.set_color(GRID_CLR)
    ax_t.grid(True, color=GRID_CLR, lw=0.4, alpha=0.5)
    ax_t.set_xlim(times[0], times[-1]); ax_t.set_ylim(0, max(tvd.max() * 1.1, 0.05))
    ax_t.set_ylabel("TVD", color=TEXT_CLR, fontsize=9)
    ax_t.set_xlabel("walk time", color=TEXT_CLR, fontsize=9)
    tvd_line, = ax_t.plot([], [], color="#00d4aa", lw=1.5)
    title = fig.suptitle("", color=TEXT_CLR, fontsize=13, fontweight="bold")

    def shade(probs):
        # Per-frame normalise + gamma so the spreading front stays visible.
        m = probs.max()
        return (probs / m) ** 0.5 if m > 0 else probs

    def grow(probs):
        return base_size + shade(probs) * base_size * 3

    def update(frame):
        c, q = c_frames[frame], q_frames[frame]
        sc_c.set_array(shade(c)); sc_c.set_sizes(grow(c))
        sc_q.set_array(shade(q)); sc_q.set_sizes(grow(q))
        tvd_line.set_data(times[: frame + 1], tvd[: frame + 1])
        title.set_text(f"walk from node {start}  ·  {label}  ·  "
                       f"t = {times[frame]:.2f}  ·  TVD = {tvd[frame]:.3f}")
        return [sc_c, sc_q, tvd_line, title]

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 // args.fps,
                         blit=False, repeat=True)
    _finish(anim, fig, args)


# ======================================================================
# Save / show
# ======================================================================

def _finish(anim, fig, args):
    if args.save:
        ext = args.save.rsplit(".", 1)[-1].lower()
        writer = "ffmpeg" if ext in ("mp4", "webm") else "pillow"
        out = args.save if ext in ("gif", "mp4", "webm") else args.save + ".gif"
        print(f"Saving to {out} ({writer}, {args.fps} fps)...")
        anim.save(out, writer=writer, fps=args.fps, dpi=args.dpi,
                  savefig_kwargs={"facecolor": BG})
        print(f"Saved {out}")
    else:
        plt.show()
    plt.close(fig)


# ======================================================================
# Programmatic entry point (used by the CLI and by showcase.py)
# ======================================================================

def run_traversal(mode: str, *, input: str | None = None,
                  topology: str = "grown", nodes: int = 600,
                  seed: int | None = None, start: int | None = None,
                  fps: int = 12, dpi: int = 120, save: str | None = None,
                  walk_time: float = 12.0, t0: float = 0.4, frames: int = 60,
                  max_radius: int = 0):
    """Build/load a graph and animate the chosen traversal (``walk`` or ``ball``)."""
    from types import SimpleNamespace
    args = SimpleNamespace(
        input=input, topology=topology, nodes=nodes, seed=seed, start=start,
        fps=fps, dpi=dpi, save=save, walk_time=walk_time, t0=t0,
        frames=frames, max_radius=max_radius, mode=mode,
    )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G, label = load_or_build(args)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. "
          f"Laying out...")
    xy = layout(G, seed)
    s = pick_start(G, xy, start)
    (animate_ball if mode == "ball" else animate_walk)(G, xy, s, label, args)


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description="Animated graph traversals (walk diffusion / ball growth)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", default=None,
                   help="results/*.pkl to traverse (omit to build from --topology)")
    p.add_argument("--mode", choices=["walk", "ball"], required=True)
    p.add_argument("--topology", default="grown",
                   choices=["small_world", "scale_free", "lattice", "random", "grown"])
    p.add_argument("--nodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--start", type=int, default=None,
                   help="seed node index (default: nearest layout centroid)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--save", default=None, help="write .gif/.mp4 instead of showing")
    # walk-only
    p.add_argument("--walk-time", type=float, default=12.0, help="[walk] max time")
    p.add_argument("--t0", type=float, default=0.4, help="[walk] start time")
    p.add_argument("--frames", type=int, default=60, help="[walk] number of frames")
    # ball-only
    p.add_argument("--max-radius", type=int, default=0,
                   help="[ball] cap radius (0 = full eccentricity from start)")
    args = p.parse_args()

    run_traversal(
        args.mode, input=args.input, topology=args.topology, nodes=args.nodes,
        seed=args.seed, start=args.start, fps=args.fps, dpi=args.dpi,
        save=args.save, walk_time=args.walk_time, t0=args.t0, frames=args.frames,
        max_radius=args.max_radius,
    )


if __name__ == "__main__":
    main()
