"""
The portal program re-run on substrates that have a dimension (plan step 3).

Pre-registration:
docs/superpowers/specs/2026-09-28-portals-on-geometry-preregistration.md

E1 censorship and E2 tolerance are run here through the banked drivers'
own functions (`shortcut_censorship.run_condition`,
`shortcut_tolerance.measure_tolerance`) on `triangular`, `sheet` and, as the
paired reference, `grown`. E3 walkers is `shortcut_walkers.py --topology ...`.

Usage:
    python portals_on_geometry.py --e1 --jobs 12
    python portals_on_geometry.py --e2 --jobs 9
    python portals_on_geometry.py --report results/portals_e1_*.csv results/portals_e2_*.csv
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from shortcuts import inject_shortcuts
from shortcut_censorship import run_condition
from shortcut_tolerance import measure_tolerance

SUBSTRATES = ['triangular', 'sheet', 'grown']
E1 = dict(n=2000, long=40, detour2=20, steps=120, prob=0.05, seeds=[0, 1, 2, 3, 4],
          conditions=['prune', 'ricci', 'triadic+prune'])
E2 = dict(n=5000, counts=[0, 5, 10, 20, 50, 100, 200, 400], seeds=[0, 1, 2],
          perms=199)


def e1_job(spec: Tuple[str, int]) -> List[Dict]:
    topo, seed = spec
    random.seed(seed)
    np.random.seed(seed)
    base = create_initial_graph(E1['n'], topology=topo, k=6, seed=seed)
    long_p = inject_shortcuts(base, E1['long'], min_distance=6)
    short_p = inject_shortcuts(base, E1['detour2'], min_distance=2,
                               max_distance=2)
    portals = long_p + short_p
    rows = []
    for cond in E1['conditions']:
        random.seed(seed * 7 + 1)
        np.random.seed(seed * 7 + 1)
        res = run_condition(base, portals, cond, E1['steps'], E1['prob'])
        rs = res['removal_step']
        Gf = res['final_graph']
        removed = [(d, rs[(u, v)]) for (u, v, d) in long_p
                   if rs[(u, v)] is not None]
        corr = float('nan')
        if len(removed) >= 8:
            a = np.argsort(np.argsort([d for d, _ in removed])).astype(float)
            t = np.argsort(np.argsort([t for _, t in removed])).astype(float)
            if a.std() > 0 and t.std() > 0:
                corr = float(np.corrcoef(a, t)[0, 1])
        rows.append({
            'substrate': topo, 'seed': seed, 'condition': cond,
            'n_long': len(long_p), 'n_detour2': len(short_p),
            'mean_advantage': float(np.mean([d for _, _, d in long_p])),
            'long_survival': sum(rs[(u, v)] is None for u, v, _ in long_p)
            / max(len(long_p), 1),
            'detour2_survival': sum(rs[(u, v)] is None for u, v, _ in short_p)
            / max(len(short_p), 1),
            'mean_removal': float(np.mean([t for _, t in removed]))
            if removed else float('nan'),
            'adv_corr': corr,
            'woven': sum(1 for (u, v, _) in long_p if rs[(u, v)] is None
                         and len(set(Gf[u]) & set(Gf[v])) >= 1),
            'collateral': res['collateral']})
    return rows


def e2_job(spec: Tuple[str, int]) -> List[Dict]:
    topo, seed = spec
    rows = measure_tolerance(E2['n'], topo, 6, E2['counts'], seed,
                             n_perm=E2['perms'])
    for r in rows:
        r['substrate'] = topo
    return rows


def _run(jobs: int, specs, fn, tag: str) -> List[Dict]:
    rows: List[Dict] = []
    Path("results").mkdir(exist_ok=True)
    path = f"results/portals_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        for out in tqdm(ex.map(fn, specs), total=len(specs), desc=tag):
            rows.extend(out)
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    print(f"Saved {path}")
    return rows


def _m(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None, '', 'nan')]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def report_e1(rows: List[Dict]) -> None:
    print("\nE1 censorship -- N=2000, 40 long (adv>=6) + 20 detour-2, 120 "
          "steps, prob 0.05, 5 seeds")
    print(f"{'substrate':>10} {'condition':>14} {'<adv>':>6} {'long':>6} "
          f"{'d2':>5} {'<t_rm>':>7} {'rank':>6} {'woven':>6} {'collat':>7}")
    for topo in SUBSTRATES:
        for cond in E1['conditions']:
            sel = [r for r in rows if r['substrate'] == topo
                   and r['condition'] == cond]
            if not sel:
                continue
            print(f"{topo:>10} {cond:>14} {_m(sel, 'mean_advantage'):6.1f} "
                  f"{_m(sel, 'long_survival'):6.3f} "
                  f"{_m(sel, 'detour2_survival'):5.2f} "
                  f"{_m(sel, 'mean_removal'):7.1f} {_m(sel, 'adv_corr'):+6.2f} "
                  f"{_m(sel, 'woven'):6.1f} {_m(sel, 'collateral'):7.3f}")
    print("\n  P1 per substrate (prune): detour-2 ~1, |rank| < 0.2, "
          "collateral 0")
    for topo in SUBSTRATES:
        sel = [r for r in rows if r['substrate'] == topo
               and r['condition'] == 'prune']
        if not sel:
            continue
        ok = (_m(sel, 'detour2_survival') >= 0.95
              and abs(_m(sel, 'adv_corr')) < 0.2
              and _m(sel, 'collateral') < 0.01)
        print(f"    {topo:>10}: {'PASS' if ok else 'FAIL'}")
    print("  P2 per substrate (triadic+prune): woven > 0 and long survival "
          "> prune-only")
    for topo in SUBSTRATES:
        tp = [r for r in rows if r['substrate'] == topo
              and r['condition'] == 'triadic+prune']
        pr = [r for r in rows if r['substrate'] == topo
              and r['condition'] == 'prune']
        if not tp or not pr:
            continue
        ok = (_m(tp, 'woven') > 0
              and _m(tp, 'long_survival') > _m(pr, 'long_survival'))
        print(f"    {topo:>10}: {'PASS' if ok else 'FAIL'}")


def report_e2(rows: List[Dict]) -> None:
    print("\nE2 tolerance -- N=5000, fixed r0, 3 seeds")
    print(f"{'substrate':>10} {'k':>4} {'placed':>6} {'defined':>8} "
          f"{'drift':>6} {'stable':>7} {'d_eff':>6} {'MoranI':>7} "
          f"{'<dist>':>7}")
    kstar = {}
    for topo in SUBSTRATES:
        for k in E2['counts']:
            sel = [r for r in rows if r['substrate'] == topo
                   and int(r['k']) == k]
            if not sel:
                continue
            stab = [r['window_stable'] for r in sel]
            n_true = sum(1 for s in stab if str(s) == 'True')
            n_none = sum(1 for s in stab if str(s) == 'None')
            stab_txt = (f"{n_true}/{len(sel)}" if n_none == 0
                        else f"{n_true}/{len(sel)}?{n_none}")
            drift = _m(sel, 'median_drift')
            if topo not in kstar and np.isfinite(drift) and drift > 0.10:
                kstar[topo] = k
            print(f"{topo:>10} {k:4d} {_m(sel, 'k_placed'):6.1f} "
                  f"{_m(sel, 'defined_frac'):8.2f} {drift:+6.2f} "
                  f"{stab_txt:>7} {_m(sel, 'd_eff_mean'):6.2f} "
                  f"{_m(sel, 'moran_I'):7.3f} {_m(sel, 'mean_pair_dist'):7.1f}")
    print("\n  k* = first k with median drift > 0.10 (dimension gone):")
    for topo in SUBSTRATES:
        print(f"    {topo:>10}: {kstar.get(topo, 'not reached')}")


def main():
    ap = argparse.ArgumentParser(
        description="Portal program on substrates with a dimension",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--e1', action='store_true')
    ap.add_argument('--e2', action='store_true')
    ap.add_argument('--report', type=str, nargs='+', default=None)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.report:
        for path in args.report:
            with open(path) as f:
                rows = list(csv.DictReader(f))
            if rows and 'condition' in rows[0]:
                report_e1(rows)
            else:
                report_e2(rows)
        return
    if args.e1:
        rows = _run(args.jobs, [(t, s) for t in SUBSTRATES
                                for s in E1['seeds']], e1_job, 'e1')
        report_e1(rows)
    if args.e2:
        rows = _run(args.jobs, [(t, s) for t in SUBSTRATES
                                for s in E2['seeds']], e2_job, 'e2')
        report_e2(rows)


if __name__ == '__main__':
    main()
