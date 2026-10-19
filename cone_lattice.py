"""Ring-built triangulated discs with prescribed vertex degrees (calibration side quest).

Builds a disc ring by ring around a centre vertex: every vertex on the current
outer ring is completed to its target degree by attaching new outer vertices,
consecutive ring vertices sharing their last/first new neighbour. Target 6
everywhere gives the triangular lattice (ring sizes 6r); target 7 everywhere
gives the regular {3,7} hyperbolic triangulation (sphere growth (3 + sqrt 5)/2);
target 7 with probability p and 6 otherwise gives a disc with random cone points
of density ~p per vertex. Ring index equals graph distance from the centre, so
V(r) is a ball volume, measured exactly as in `sheet_variants.grow_one`.

Used to test the wave 6 hyperbolic-disc law (`sheet_variants.hyperbolic_slope`)
on graphs whose curvature is set by construction, independent of `sheet`.

Usage:
    python cone_lattice.py --validate
    python cone_lattice.py --calibrate [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np

from sheet_variants import hyperbolic_slope, HEX_AREA


def ring_volumes(r_max: int, p7: float, rng: random.Random, n_cap: int = 30_000_000):
    """Ball volumes V(0..r) and interior degree-7 count of a ring-built disc.

    Returns (V, n7) where V[r] = vertices within distance r of the centre and
    n7 counts completed (interior) vertices with target degree 7.
    """
    def target() -> int:
        return 7 if rng.random() < p7 else 6

    t0 = target()
    n7 = int(t0 == 7)
    # Ring 1: t0 vertices in a cycle, each adjacent to the centre and two ring
    # neighbours -> degree 3.
    deg = [3] * t0
    V = [1, 1 + t0]
    for _ in range(2, r_max + 1):
        need = []
        for d in deg:
            t = target()
            n7 += int(t == 7)
            k = t - d
            if k < 1:
                raise ValueError('ring vertex already at target degree')
            need.append(k)
        m = len(deg)
        # Ring vertex i owns k_i new neighbours; its last is shared with the first
        # of i+1. A run of k == 1 vertices shares one new vertex with the owners on
        # either side: parents = 2 + run length, degree = parents + 2 (two ring
        # neighbours). Private new neighbours have one parent, degree 3.
        j = next((i for i in range(m) if need[i] >= 2), None)
        if j is None:
            raise ValueError('every ring vertex has k == 1')
        new_deg = []
        run = 0
        for t in range(1, m + 1):
            k = need[(j + t) % m]
            if k == 1:
                run += 1
                continue
            new_deg.append(2 + run + 2)
            run = 0
            new_deg.extend([3] * (k - 2))
        deg = new_deg
        V.append(V[-1] + len(deg))
        if V[-1] > n_cap:
            break
    return np.array(V), n7


def slope(V: np.ndarray, r: int) -> float:
    return float(np.log(V[int(r * 1.25)] / V[int(r / 1.25)]) / np.log(1.5625))


def validate() -> None:
    rng = random.Random(0)
    V, n7 = ring_volumes(50, 0.0, rng)
    assert all(V[r] == 3 * r * r + 3 * r + 1 for r in range(51)), V[:6]
    print('target 6: V(r) = 3r^2 + 3r + 1 for r <= 50  OK')
    V, n7 = ring_volumes(14, 1.0, rng)
    s = np.diff(V)
    print('target 7 sphere sizes:', s[:8].tolist(), ' ratio', s[-1] / s[-2],
          ' (3+sqrt5)/2 =', (3 + 5 ** 0.5) / 2)


HEX = HEX_AREA


def calibrate(seed: int, n_seeds: int = 32) -> None:
    """Ensemble test of the hyperbolic-disc law on random cone-point discs.

    For each cone density p, mean measured local slope over n_seeds discs
    against the law at lam = p, with x = r sqrt(K) (as registered in wave 6) and
    with x = HEX r sqrt(K) (graph ball read as a hexagon of equal area).
    """
    flat = ring_volumes(3000, 0.0, random.Random(0))[0]
    print('p7      x=r*sqrtK  r     measured (sd)     law   ratio | law_hex  ratio')
    rows = []
    for p in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1):
        sk = math.sqrt(2 * math.pi * p / (3 * math.sqrt(3)))
        radii = sorted({max(8, int(round(x / sk))) for x in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5)})
        rmax = int(radii[-1] * 1.25) + 1
        ms = {r: [] for r in radii}
        for k in range(n_seeds):
            V, _ = ring_volumes(rmax, p, random.Random(seed * 100003 + k), n_cap=10 ** 9)
            for r in radii:
                ms[r].append(slope(V, r))
        for r in radii:
            m = float(np.mean(ms[r]))
            sd = float(np.std(ms[r]))
            fs = slope(flat, r) / 2
            law = hyperbolic_slope(r, p) * fs
            lawh = hyperbolic_slope(HEX * r, p) * fs
            rows.append((p, r * sk, m, law, lawh))
            print(f'{p:<7g} {r * sk:6.2f}  {r:5d}  {m:7.3f} ({sd:5.3f})  {law:6.3f} {m / law:6.3f} |'
                  f' {lawh:6.3f} {m / lawh:6.3f}', flush=True)
    V, _ = ring_volumes(30, 1.0, random.Random(0))
    print(f'{{3,7}}: asymptotic log growth per layer {math.log(V[-1] - V[-2]) - math.log(V[-2] - V[-3]):.4f}'
          f' vs law sqrt(K) {math.sqrt(2 * math.pi / (3 * math.sqrt(3))):.4f},'
          f' hex {HEX * math.sqrt(2 * math.pi / (3 * math.sqrt(3))):.4f}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    if a.validate:
        validate()
    if a.calibrate:
        calibrate(a.seed)


if __name__ == '__main__':
    main()
