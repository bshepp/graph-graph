"""
Distribution of the sheet's crossover size at beta = 2, seed by seed.

Pre-registration: docs/superpowers/specs/2026-10-13-round2-preregistrations.md (item 4).

Reads Part A CSVs from `sheet_crossover.py` (seeds 300-303 of step 2 plus the
fresh seeds 304-311), computes the boundary-exponent crossover N*_B per seed
with the step-2 estimator (`sheet_transition.crossover_size` on the local
exponent of boundary_edges vs N), and the frozen statistics on the per-seed
values: coefficient of variation CV = sd / mean of the finite N*_B and the
censored fraction (seeds with no crossover by N_max).

Frozen readings:  MEMORYLESS  CV >= 0.7 | SHARP THRESHOLD  CV <= 0.3 |
                  IN BETWEEN  otherwise (reported with the CV).

Usage:
    python crossover_cv.py results/sheet_crossover_partA_N4194304_*.csv
    python crossover_cv.py --validate
"""

import argparse
import csv
import random
from typing import Dict, List, Sequence

import numpy as np

from sheet_transition import local_exponents, crossover_size


def per_seed_nstar(rows: List[Dict]) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for seed in sorted({int(r['seed']) for r in rows}):
        sel = sorted((int(r['N']), float(r['boundary_edges']))
                     for r in rows if int(r['seed']) == seed)
        Ns = [n for n, _ in sel]
        B = [b for _, b in sel]
        mid, e = local_exponents(Ns, B)
        st, ns = crossover_size(mid, e)
        out[seed] = {'status': st, 'N_star': ns, 'N_max': Ns[-1],
                     'B_over_sqrtN_last': B[-1] / np.sqrt(Ns[-1])}
    return out


def frozen_stats(ps: Dict[int, Dict[str, object]]) -> Dict[str, float]:
    vals = np.array([v['N_star'] for v in ps.values()
                     if v['status'] == 'finite'], dtype=float)
    n = len(ps)
    cens = sum(v['status'] != 'finite' for v in ps.values())
    if len(vals) >= 2:
        cv = float(vals.std(ddof=1) / vals.mean())
    else:
        cv = float('nan')
    if not np.isfinite(cv):
        verdict = 'UNDECIDED (fewer than 2 finite values)'
    elif cv >= 0.7:
        verdict = 'MEMORYLESS'
    elif cv <= 0.3:
        verdict = 'SHARP THRESHOLD'
    else:
        verdict = 'IN BETWEEN'
    # bootstrap SE of the CV, disclosure only
    rng = np.random.default_rng(12345)
    boots = []
    if len(vals) >= 3:
        for _ in range(2000):
            s = rng.choice(vals, size=len(vals), replace=True)
            if s.mean() > 0:
                boots.append(s.std(ddof=1) / s.mean())
    return {'n': n, 'n_finite': int(len(vals)), 'censored_frac': cens / n,
            'mean': float(vals.mean()) if len(vals) else float('nan'),
            'sd': float(vals.std(ddof=1)) if len(vals) >= 2 else float('nan'),
            'cv': cv, 'cv_se': float(np.std(boots)) if boots else float('nan'),
            'min': float(vals.min()) if len(vals) else float('nan'),
            'max': float(vals.max()) if len(vals) else float('nan'),
            'verdict': verdict}


def report(paths: Sequence[str]) -> Dict[str, float]:
    rows: List[Dict] = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    rows = [r for r in rows if float(r['beta']) == 2.0]
    ps = per_seed_nstar(rows)
    print(f"{'seed':>5} {'status':>9} {'N*_B':>11} {'B/sqrtN @Nmax':>14}")
    for s, v in ps.items():
        ns = f"{v['N_star']:,.0f}" if v['status'] == 'finite' else '--'
        print(f"{s:5d} {v['status']:>9} {ns:>11} {v['B_over_sqrtN_last']:14.2f}")
    st = frozen_stats(ps)
    print(f"\n{st['n_finite']}/{st['n']} seeds finite "
          f"(censored fraction {st['censored_frac']:.2f}); "
          f"N*_B mean {st['mean']:,.0f}, sd {st['sd']:,.0f}, "
          f"range {st['min']:,.0f} .. {st['max']:,.0f}")
    print(f"CV = {st['cv']:.3f} (bootstrap SE {st['cv_se']:.3f})")
    print(f"VERDICT (frozen rules, item 4): {st['verdict']}")
    return st


def _validate() -> bool:
    ok = True
    print("[1] per-seed estimator on an exact curve with a planted jump")
    Ns = [2 ** k for k in range(10, 23)]
    rows = []
    for seed, nj in ((1, 2 ** 19), (2, 2 ** 20.5)):
        for n in Ns:
            b = 3.8 * n ** 0.5 if n < nj else 3.8 * nj ** 0.5 * (n / nj) ** 1.2
            rows.append({'seed': seed, 'N': n, 'boundary_edges': b})
    for n in Ns:                      # a censored seed: compact throughout
        rows.append({'seed': 3, 'N': n, 'boundary_edges': 3.8 * n ** 0.5})
    ps = per_seed_nstar(rows)
    good = (ps[1]['status'] == 'finite' and 2 ** 18.5 < ps[1]['N_star'] < 2 ** 20
            and ps[2]['status'] == 'finite'
            and 2 ** 20 < ps[2]['N_star'] < 2 ** 21.5
            and ps[3]['status'] == 'censored')
    ok &= good
    print(f"  jumps at 2^19, 2^20.5, none: read "
          f"{ps[1]['N_star']:,.0f}, {ps[2]['N_star']:,.0f}, {ps[3]['status']}"
          f"  {'OK' if good else 'FAIL'}")
    print("[2] readings")
    mk = lambda vals: {i: {'status': 'finite', 'N_star': v} for i, v in enumerate(vals)}
    for name, vals, want in (
            ('exponential-like', [1, 2, 3, 5, 8, 13, 21, 34], 'MEMORYLESS'),
            ('threshold', [1.0, 1.05, 0.95, 1.1, 0.9], 'SHARP THRESHOLD'),
            ('between', [0.47, 0.89, 0.90, 1.09, 1.5, 0.6], 'IN BETWEEN')):
        st = frozen_stats(mk(vals))
        good = st['verdict'] == want
        ok &= good
        print(f"  {name:>17}: CV {st['cv']:.2f} -> {st['verdict']}  "
              f"{'OK' if good else 'FAIL'}")
    print("\nPASS: crossover CV" if ok else "\nFAIL: crossover CV")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="CV of the per-seed crossover size")
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if not args.paths:
        ap.error('give Part A CSVs')
    report(args.paths)


if __name__ == '__main__':
    main()
