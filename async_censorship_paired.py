"""
Higher-power PAIRED estimate of the step-4 P2 (self-stabilization) comparison.

The two-gate `async_censorship.py --validate` runs only 3-5 seeds, enough for
Gate 1 (P1) but too noisy to quantify Gate 2's async-vs-sync difference: the
first-pass gate showed a spurious "~2x" woven attenuation that did NOT survive a
proper paired analysis. This driver runs the `triadic`+`prune` race through BOTH
schedules on the SAME injected base+portals per seed, over many seeds, and
reports each observable's paired difference (sync - async) with a rough paired-t,
plus how many seeds show async < sync. It reuses `async_censorship`'s own
`sync_condition` / `async_condition` / `summarize_portals`, so it measures
exactly what the checkpoint measures -- only with the seed count turned up.

Result at the defaults (N=1200, 12 seeds): long-survival is schedule-invariant
(sync 0.133 vs async 0.131, t=0.14) and woven shows at most a weak,
non-significant dip (3.50 vs 2.83, ratio 0.81, t=1.77). See FINDINGS.md
"censorship under async (step 4)".

Usage:
    python async_censorship_paired.py                 # N=1200, 12 seeds
    python async_censorship_paired.py --nodes 2000 --seeds 20
"""

import argparse
import random
from typing import List, Tuple

import numpy as np

from async_censorship import (build_base_and_portals, fabric_edges,
                              summarize_portals, async_condition,
                              sync_condition)

_REWIRE_PROB = 0.05  # must match run_condition's hardcoded triadic rate


def paired_race(n_nodes: int, cap: int, n_long: int, n_detour2: int,
                sweeps: int, seeds: int, prune_prob: float,
                seed_base: int) -> List[Tuple[float, float, float, float]]:
    """Per-seed (sync_woven, async_woven, sync_long, async_long) for the race."""
    rows: List[Tuple[float, float, float, float]] = []
    for s in range(seeds):
        base, longp, d2 = build_base_and_portals(n_nodes, cap, n_long,
                                                 n_detour2, s)
        fabric = fabric_edges(base, longp, d2)
        sr, sg = sync_condition(base, longp, d2, 'triadic+prune', sweeps,
                                prune_prob)
        ar, ag, _ = async_condition(
            base, longp, d2, ['triadic', 'prune'], [1.0, 1.0],
            [{'rewire_prob': _REWIRE_PROB}, {'prune_prob': prune_prob}],
            float(sweeps), seed=seed_base + s)
        ss = summarize_portals(sr, longp, d2, sg, fabric)
        asum = summarize_portals(ar, longp, d2, ag, fabric)
        rows.append((ss['woven'], asum['woven'],
                     ss['long_survival'], asum['long_survival']))
        print(f"seed {s:2d}: woven sync {ss['woven']:.0f} async "
              f"{asum['woven']:.0f} | long sync {ss['long_survival']:.3f} "
              f"async {asum['long_survival']:.3f}", flush=True)
    return rows


def _report(rows: List[Tuple[float, float, float, float]], n_nodes: int,
            sweeps: int) -> None:
    a = np.array(rows)
    sw, aw, sl, al = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    def ms(x: np.ndarray) -> str:
        return f"{x.mean():.3f}+-{x.std(ddof=1) / np.sqrt(len(x)):.3f}"

    def paired(x: np.ndarray, y: np.ndarray, label: str) -> None:
        d = x - y
        sem = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / sem if sem > 0 else float('nan')
        print(f"{label}: sync {ms(x)}  async {ms(y)}  ratio "
              f"{y.mean() / x.mean():.2f}  | paired diff {ms(d)}  "
              f"async<sync in {int((y < x).sum())}/{len(d)} seeds  t={t:.2f}")

    print(f"\n=== K={len(rows)} seeds, N={n_nodes}, {sweeps} sweeps "
          f"(triadic+prune, paired sync vs async) ===")
    paired(sw, aw, 'woven        ')
    paired(sl, al, 'long_survival')


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Higher-power paired estimate of the step-4 P2 comparison.")
    ap.add_argument('--nodes', type=int, default=1200)
    ap.add_argument('--cap', type=int, default=6)
    ap.add_argument('--long', type=int, default=40)
    ap.add_argument('--detour2', type=int, default=20)
    ap.add_argument('--sweeps', type=int, default=120)
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--prune-prob', type=float, default=0.05)
    ap.add_argument('--seed-base', type=int, default=3000,
                    help='async event-RNG seed offset (independent of the '
                         'gate seeds so this is a fresh sample).')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = paired_race(args.nodes, args.cap, args.long, args.detour2,
                       args.sweeps, args.seeds, args.prune_prob, args.seed_base)
    _report(rows, args.nodes, args.sweeps)


if __name__ == '__main__':
    main()
