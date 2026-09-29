"""
Does `sheet` at beta = 2 turn tree-like near 1.6 million nodes?

Pre-registration:
docs/superpowers/specs/2026-09-28-sheet-crossover-4M-preregistration.md
(all definitions and verdict rules below are frozen there).

Three crossover estimators on the same growth, read at every factor 2:
  M1 boundary exponent   e_B = dln B / dln N   crosses 0.75 upward
  M2 diameter exponent   e_D = dln D / dln N   crosses 0.30 downward
  M3 arm fraction        A = share of nodes within 10 hops of the boundary;
                         dln A / dln N rises above -0.25
plus the window audit and spectral flow on the final graph (supplementary).

Usage:
    python sheet_crossover.py --validate
    python sheet_crossover.py --seeds 300 --n-max 4194304            # memory trial
    python sheet_crossover.py --seeds 301 302 303 --n-max 4194304 --jobs 3
    python sheet_crossover.py --rescan --jobs 12                     # Part B
    python sheet_crossover.py --analyze results/sheet_crossover_*.csv
    python sheet_crossover.py --judge results/sheet_cross_*.npz
"""

import argparse
import csv
import glob
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

from sheet_growth import grow_sheet, census, to_csr, save_edges, load_csr
from sheet_transition import (local_exponents, checkpoints, fit_laws,
                              crossover_size as cross_up, E_CROSS)

BETA = 2.0
CAP = 6
SEEDS_A = [300, 301, 302, 303]
N_MAX_A = 4194304
N_FIRST = 1024
ARM_DEPTH = 10
E_D_CROSS = 0.30
E_A_CROSS = -0.25
WINDOW = (262144, 4194304)
RESCAN_BETAS = [round(1.0 + 0.1 * i, 1) for i in range(8)]
RESCAN_SEEDS = list(range(304, 312))


def diameter_csr(A: sp.csr_array, rng: np.random.Generator,
                 sweeps: int = 2) -> float:
    n = A.shape[0]
    best = 0.0
    for _ in range(sweeps):
        d = dijkstra(A, indices=int(rng.integers(n)), unweighted=True)
        d2 = dijkstra(A, indices=int(np.argmax(d)), unweighted=True)
        best = max(best, float(d2.max()))
    return best


def arm_fraction(A: sp.csr_array, adj, depth: int = ARM_DEPTH) -> float:
    """Share of nodes within `depth` hops of a boundary edge."""
    n = A.shape[0]
    bnd = [u for u in range(n)
           if any(len(adj[u] & adj[v]) == 1 for v in adj[u])]
    if not bnd:
        return 0.0
    d = dijkstra(A, indices=bnd, unweighted=True, min_only=True,
                 limit=depth + 0.5)
    return float(np.mean(d <= depth))


def run_one(spec: Tuple[float, int, int, int, bool]) -> List[Dict]:
    beta, seed, n_max, n_first, save = spec
    rows: List[Dict] = []
    rng = np.random.default_rng(seed)

    def snap(n: int, adj) -> None:
        cs = census(adj, CAP)
        A = to_csr(adj)
        rows.append({'beta': beta, 'seed': seed, 'N': n,
                     'boundary_edges': cs['boundary_edges'],
                     'diameter': diameter_csr(A, rng),
                     'arm_fraction': arm_fraction(A, adj),
                     'interior_deg_c': cs['interior_deg_c'],
                     'nonmanifold_edges': cs['nonmanifold_edges'],
                     'seconds': time.time() - t0})

    t0 = time.time()
    adj = grow_sheet(n_max, CAP, beta, seed,
                     checkpoints=checkpoints(n_first, n_max),
                     on_checkpoint=snap)
    for r in rows:
        r['reached'] = len(adj)
    if save and len(adj) == n_max:
        Path("results").mkdir(exist_ok=True)
        save_edges(adj, f"results/sheet_cross_b{beta:g}_s{seed}_N{n_max}.npz")
    return rows


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------

def cross_down(mid: np.ndarray, e: np.ndarray, level: float
               ) -> Tuple[str, float]:
    """Where e falls below `level` and stays below (mirror of cross_up)."""
    # reflect about `level` and shift so that `level` maps onto E_CROSS
    return cross_up(mid, (level - e) + E_CROSS)


def seed_mean_curves(rows: List[Dict]) -> Tuple[List[int], Dict[str, list]]:
    seeds = sorted({int(r['seed']) for r in rows})
    Ns = sorted({int(r['N']) for r in rows})
    Ns = [n for n in Ns
          if len([r for r in rows if int(r['N']) == n]) == len(seeds)]
    curves = {}
    for key in ('boundary_edges', 'diameter', 'arm_fraction'):
        if any(r.get(key) in (None, '') for r in rows):
            continue                 # step-1 rows carry boundary_edges only
        curves[key] = [np.mean([float(r[key]) for r in rows
                                if int(r['N']) == n]) for n in Ns]
    return Ns, curves


def analyze_part_a(rows: List[Dict], n_max: int = N_MAX_A) -> Dict[str, object]:
    Ns, c = seed_mean_curves(rows)
    seeds = sorted({int(r['seed']) for r in rows})
    print(f"\nPart A: beta = {BETA}, {len(seeds)} seeds, N = {Ns[0]} .. {Ns[-1]}")
    mid, eB = local_exponents(Ns, c['boundary_edges'])
    _, eD = local_exponents(Ns, c['diameter'])
    _, eA = local_exponents(Ns, c['arm_fraction'])
    print(f"  {'N (mid)':>9} {'e_B':>6} {'e_D':>6} {'e_A':>6}   "
          f"(compact: 0.50 / 0.50 / -0.50;  tree-like: 1.0 / ~0.1 / ~0)")
    for m, b, d, a in zip(mid, eB, eD, eA):
        print(f"  {m:9.0f} {b:6.2f} {d:6.2f} {a:6.2f}")
    stB, nB = cross_up(mid, eB)
    stD, nD = cross_down(mid, eD, E_D_CROSS)
    stA, nA = _cross_up_level(mid, eA, E_A_CROSS)
    res = {'M1': (stB, nB), 'M2': (stD, nD), 'M3': (stA, nA)}
    print()
    for k, (st, ns) in res.items():
        txt = f"{ns:,.0f}" if st == 'finite' else st
        print(f"  {k} crossover size: {txt}")
    finite = {k: v[1] for k, v in res.items() if v[0] == 'finite'
              and WINDOW[0] <= v[1] <= WINDOW[1]}
    agree = (len(finite) >= 2 and
             max(finite.values()) / min(finite.values()) <= 4.0)
    last = {'e_B': float(eB[-1]), 'e_D': float(eD[-1]), 'e_A': float(eA[-1])}
    censored_all = all(v[0] == 'censored' for v in res.values())
    compact = (last['e_B'] <= 0.6 and last['e_D'] >= 0.4
               and last['e_A'] <= -0.35)
    if agree:
        verdict = 'CROSSOVER CONFIRMED'
    elif censored_all and compact and Ns[-1] >= n_max:
        verdict = 'TRANSITION BACK ON THE TABLE'
    else:
        verdict = 'UNDECIDED'
    print(f"  last step: e_B {last['e_B']:.2f}  e_D {last['e_D']:.2f}  "
          f"e_A {last['e_A']:.2f}")
    print(f"  VERDICT (frozen rules, Part A): {verdict}")
    return {'verdict': verdict, 'sizes': res, 'last': last, 'Ns': Ns}


def _cross_up_level(mid: np.ndarray, e: np.ndarray, level: float
                    ) -> Tuple[str, float]:
    """Where e rises above `level` and stays above (cross_up at any level)."""
    return cross_up(mid, e - level + E_CROSS)


def analyze_part_b(rows: List[Dict], old_csv: str | None
                   ) -> Dict[str, object]:
    """Refit N*(beta) on the re-scan (plus the step-1 seeds if given)."""
    allrows = list(rows)
    if old_csv and os.path.exists(old_csv):
        with open(old_csv) as f:
            allrows += list(csv.DictReader(f))
    betas = sorted({float(r['beta']) for r in allrows})
    print(f"\nPart B: N*(beta) on {len({int(r['seed']) for r in allrows})} "
          f"seeds")
    fin_b, fin_ln = [], []
    for b in betas:
        sel = [r for r in allrows if float(r['beta']) == b]
        Ns, c = seed_mean_curves(sel)
        mid, e = local_exponents(Ns, c['boundary_edges'])
        st, ns = cross_up(mid, e)
        nseeds = len({int(r['seed']) for r in sel})
        print(f"  beta {b:.1f} ({nseeds:2d} seeds): "
              + (f"N* = {ns:9,.0f}" if st == 'finite' else f"{st:>14}")
              + "   e: " + " ".join(f"{x:.2f}" for x in e))
        if st == 'finite':
            fin_b.append(b)
            fin_ln.append(np.log(ns))
    if len(fin_b) < 5:
        print("  fewer than 5 finite N*: no refit")
        return {}
    fit = fit_laws(np.array(fin_b), np.array(fin_ln))
    n = len(fin_b)
    # 95% prediction interval of the exponential law at beta = 2
    X = np.column_stack([np.ones(n), fin_b])
    resid_var = fit['rss_exp'] / (n - 2)
    x0 = np.array([1.0, BETA])
    lev = float(x0 @ np.linalg.inv(X.T @ X) @ x0)
    from scipy.stats import t as tdist
    half = tdist.ppf(0.975, n - 2) * np.sqrt(resid_var * (1 + lev))
    pred = fit['exp_a'] + fit['exp_b'] * BETA
    lo, hi = np.exp(pred - half), np.exp(pred + half)
    print(f"  exponential  ln N* = {fit['exp_a']:.2f} + {fit['exp_b']:.2f} "
          f"beta   AICc {fit['aicc_exp']:.2f}")
    print(f"  power law    ln N* = {fit['pow_a']:.2f} - {fit['nu']:.2f} "
          f"ln({fit['beta_c']:.3f} - beta)   AICc {fit['aicc_pow']:.2f}")
    print(f"  exponential prediction at beta = 2: N* = {np.exp(pred):,.0f}, "
          f"95% prediction interval [{lo:,.0f}, {hi:,.0f}]")
    return {**fit, 'pred_N': float(np.exp(pred)), 'pi': (float(lo),
                                                          float(hi))}


def judge(paths: Sequence[str]) -> None:
    from window_stability import audit_adjacency
    from spectral_flow import measure_adjacency, show
    for path in paths:
        A = load_csr(path)
        name = Path(path).stem
        seed = int(name.split('_s')[1].split('_')[0])
        import window_stability as ws
        ws.WINDOWS = (6, 10, 16, 25, 40, 63, 100, 160, 250, 400)
        r = audit_adjacency(name, A, 40, seed)
        print(f"  M4 window audit: {r['verdict']} (drift {r['drift']:.2f}, "
              f"d at R{r['table'][-1][0]} = {r['table'][-1][1]:.2f})")
        s = measure_adjacency(name, A, seed, 4)
        show(s)


def _validate() -> bool:
    ok = True
    print("[1] crossover readers on exact curves")
    Ns = checkpoints(1024, 4194304)
    mid, e = local_exponents(Ns, [3.0 * n ** 0.5 for n in Ns])
    good = cross_down(mid, e, E_D_CROSS)[0] == 'censored'
    ok &= good
    print(f"  diameter ~ sqrt(N): M2 censored  {'OK' if good else 'FAIL'}")
    D = [n ** 0.5 / np.sqrt(1 + n / 1.0e6) for n in Ns]   # -> log-like
    mid, e = local_exponents(Ns, D)
    st, ns = cross_down(mid, e, E_D_CROSS)
    good = st == 'finite' and 3e5 < ns < 3e6
    ok &= good
    print(f"  diameter bending at 1e6: M2 reads {ns:,.0f}  "
          f"{'OK' if good else 'FAIL'}")
    Aarm = [min(1.0, 40.0 / n ** 0.5 * np.sqrt(1 + n / 1.0e6)) for n in Ns]
    mid, e = local_exponents(Ns, Aarm)
    st, ns = _cross_up_level(mid, e, E_A_CROSS)
    good = st == 'finite' and 3e5 < ns < 3e6
    ok &= good
    print(f"  arm fraction levelling at 1e6: M3 reads {ns:,.0f}  "
          f"{'OK' if good else 'FAIL'}")

    print("[2] measurements on a small sheet")
    adj = grow_sheet(20000, CAP, 2.0, seed=7)
    A = to_csr(adj)
    af = arm_fraction(A, adj)
    D = diameter_csr(A, np.random.default_rng(0))
    good = 0.05 < af < 0.6 and 100 < D < 250
    ok &= good
    print(f"  N=20000 beta=2: arm fraction {af:.3f}, diameter {D:.0f}  "
          f"{'OK' if good else 'FAIL'}")
    print("\nPASS: sheet crossover instrument" if ok
          else "\nFAIL: sheet crossover instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="sheet at beta=2: crossover near 1.6M nodes?",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--n-max', type=int, default=N_MAX_A)
    ap.add_argument('--rescan', action='store_true', help='Part B')
    ap.add_argument('--analyze', type=str, nargs='+', default=None,
                    help='Part A CSVs to reduce together')
    ap.add_argument('--old-scan', type=str, default=None,
                    help='step-1 CSV to pool into the Part B refit')
    ap.add_argument('--judge', type=str, nargs='+', default=None)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.judge:
        judge(args.judge)
        return
    if args.analyze:
        rows: List[Dict] = []
        for path in args.analyze:
            with open(path) as f:
                rows += list(csv.DictReader(f))
        if any(float(r['beta']) != BETA for r in rows):
            analyze_part_b(rows, args.old_scan)
        else:
            analyze_part_a(rows, args.n_max)
        return

    Path("results").mkdir(exist_ok=True)
    if args.rescan:
        specs = [(b, s, 512000, 1000, False) for b in RESCAN_BETAS
                 for s in RESCAN_SEEDS]
        tag = 'rescan'
    else:
        seeds = args.seeds or SEEDS_A
        specs = [(BETA, s, args.n_max, N_FIRST, True) for s in seeds]
        tag = f"partA_N{args.n_max}"
    path = f"results/sheet_crossover_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for out in tqdm(ex.map(run_one, specs), total=len(specs),
                        desc=tag):
            rows.extend(out)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"Saved {path}")
    if args.rescan:
        analyze_part_b(rows, args.old_scan)
    else:
        analyze_part_a(rows, args.n_max)


if __name__ == '__main__':
    main()
