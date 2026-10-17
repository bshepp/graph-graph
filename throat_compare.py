"""
Throat criticality across substrates: one table from the saved FSS CSVs.

Pre-registration: docs/superpowers/specs/2026-10-13-round2-preregistrations.md (item 2).

For each substrate (grown / triangular / sheet) and geometry r the newest
`results/throat_fss_<topo>_r<r>_*.csv` is read and the stage-1 statistics are
recomputed from its rows (same formulas as `throat_criticality.py`):
capacity, a50 (median critical density), A*50 (median strand count),
relative width (10-90 span of a* over a50), and the onset core
`core_frac @ A* x A*50` in strands. Then the frozen per-substrate reading:

  MOTIF ONSET HOLDS      onset core in [2, 3.5] at every r AND relative width
                         does not fall with capacity (rises, or flat within 10%)
  SUBSTRATE-DEPENDENT    onset core differs from grown's 2.6 by > 1.0 strand,
                         or relative width falls by > 10% across capacities
  UNDECIDED              otherwise

`--integer` (pre-registration 2026-10-17-wave1-preregistration.md, W1-a):
the per-draw core size at onset m = round(A_star * core_frac) is an integer
>= 2; its histogram per substrate and r, with the frozen readings
  INTEGER MIXTURE     P(m=2) + P(m=3) >= 0.80 in every cell
  BROAD               below 0.80 in at least three cells
  FRACTION UNIVERSAL  max - min of P(m=2) over cells <= 0.15

Usage:
    python throat_compare.py            # all substrates with saved FSS files
    python throat_compare.py --integer
    python throat_compare.py --validate
"""

import argparse
import csv
import glob
import os
import random
import re
from typing import Dict, List, Optional

import numpy as np

GROWN_ONSET = 2.6
TOPOS = ('grown', 'triangular', 'sheet')


def load_rows(path: str) -> List[Dict[str, float]]:
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    a = np.array([r['a_star'] for r in rows])
    A = np.array([r['A_star'] for r in rows])
    fin = np.isfinite(a) & (A > 0)
    a, A = a[fin], A[fin]
    q = np.quantile(a, [0.1, 0.5, 0.9])
    A50 = float(np.median(A))
    cf = float(np.mean([r['core_frac'] for r in rows if r['A_star'] > 0]))
    rng = np.random.default_rng(12345)
    boots = []
    for _ in range(1000):
        s = rng.choice(a, size=len(a), replace=True)
        qs = np.quantile(s, [0.1, 0.5, 0.9])
        boots.append((qs[2] - qs[0]) / qs[1])
    return {'n': int(len(rows)), 'n_finite': int(fin.sum()),
            'capacity': float(np.mean([r['capacity'] for r in rows])),
            'a50': float(q[1]), 'width': float(q[2] - q[0]),
            'rel_width': float((q[2] - q[0]) / q[1]),
            'rel_width_se': float(np.std(boots)),
            'A50': A50, 'core_frac': cf, 'onset_core': cf * A50,
            'regens': int(sum(r['substrate_regens'] for r in rows))}


def newest_files(results_dir: str = 'results') -> Dict[str, Dict[int, str]]:
    out: Dict[str, Dict[int, str]] = {t: {} for t in TOPOS}
    pat = re.compile(r'throat_fss_(?:(triangular|sheet)_)?r(\d)_(\d{8}_\d{6})\.csv$')
    for path in sorted(glob.glob(os.path.join(results_dir, 'throat_fss_*.csv'))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        topo = m.group(1) or 'grown'
        out[topo][int(m.group(2))] = path      # sorted -> newest wins
    return out


def reading(per_r: Dict[int, Dict[str, float]]) -> str:
    rs = sorted(per_r)
    cores = [per_r[r]['onset_core'] for r in rs]
    rel = [per_r[r]['rel_width'] for r in rs]
    if len(rs) < 3:
        return f'INCOMPLETE ({len(rs)}/3 geometries)'
    core_in_band = all(2.0 <= c <= 3.5 for c in cores)
    core_far = any(abs(c - GROWN_ONSET) > 1.0 for c in cores)
    # "does not fall with capacity": every step up in r keeps rel_width
    # within -10% of the previous (rises, or flat within 10%)
    steps = [(rel[i + 1] - rel[i]) / rel[i] for i in range(len(rel) - 1)]
    falls_10 = any(s < -0.10 for s in steps)
    if core_in_band and not falls_10:
        return 'MOTIF ONSET HOLDS'
    if core_far or falls_10:
        return 'SUBSTRATE-DEPENDENT'
    return 'UNDECIDED'


def compare(results_dir: str = 'results') -> Dict[str, Dict[int, Dict[str, float]]]:
    files = newest_files(results_dir)
    table: Dict[str, Dict[int, Dict[str, float]]] = {}
    print(f"{'substrate':>10} {'r':>2} {'capacity':>8} {'A*50':>5} {'a50':>7} "
          f"{'rel_width':>9} {'+/-':>5} {'core@A*':>7} {'onset':>6} "
          f"{'finite':>9} {'regens':>6}")
    for topo in TOPOS:
        table[topo] = {}
        for r in sorted(files[topo]):
            s = stats(load_rows(files[topo][r]))
            table[topo][r] = s
            print(f"{topo:>10} {r:2d} {s['capacity']:8.0f} {s['A50']:5.0f} "
                  f"{s['a50']:7.4f} {s['rel_width']:9.3f} "
                  f"{s['rel_width_se']:5.3f} {s['core_frac']:7.3f} "
                  f"{s['onset_core']:6.2f} "
                  f"{s['n_finite']:4d}/{s['n']:<4d} {s['regens']:6d}")
    print()
    for topo in TOPOS:
        if table[topo]:
            print(f"  {topo:>10}: {reading(table[topo])}")
    return table


def core_sizes(rows: List[Dict[str, float]]) -> np.ndarray:
    """Per finite draw, the number of strands in the peeling core at onset."""
    return np.array([int(round(r['A_star'] * r['core_frac'])) for r in rows
                     if r['A_star'] > 0], dtype=int)


def integer_reading(cells: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    p23 = [c['p2'] + c['p3'] for c in cells.values()]
    low = sum(x < 0.80 for x in p23)
    mix = ('INTEGER MIXTURE' if low == 0 else
           'BROAD' if low >= 3 else 'UNDECIDED')
    p2 = [c['p2'] for c in cells.values()]
    frac = ('FRACTION UNIVERSAL' if max(p2) - min(p2) <= 0.15
            else 'FRACTION VARIES')
    return {'mixture': mix, 'fraction': frac,
            'p2_range': f"{min(p2):.3f}-{max(p2):.3f}"}


def integer_mixture(results_dir: str = 'results') -> Dict[str, Dict[str, float]]:
    files = newest_files(results_dir)
    cells: Dict[str, Dict[str, float]] = {}
    print(f"{'substrate':>10} {'r':>2} {'draws':>5} {'min m':>5} {'P(2)':>6} "
          f"{'P(3)':>6} {'P(4)':>6} {'P(>=5)':>6} {'P(2)+P(3)':>9} {'mean m':>6} "
          f"{'mode':>4}")
    bad = 0
    for topo in TOPOS:
        for r in sorted(files[topo]):
            m = core_sizes(load_rows(files[topo][r]))
            bad += int((m < 2).sum())
            n = len(m)
            c = {'n': n, 'p2': float(np.mean(m == 2)), 'p3': float(np.mean(m == 3)),
                 'p4': float(np.mean(m == 4)), 'p5': float(np.mean(m >= 5)),
                 'mean': float(m.mean()), 'mode': int(np.bincount(m).argmax()),
                 'min': int(m.min())}
            cells[f"{topo} r{r}"] = c
            print(f"{topo:>10} {r:2d} {n:5d} {c['min']:5d} {c['p2']:6.3f} "
                  f"{c['p3']:6.3f} {c['p4']:6.3f} {c['p5']:6.3f} "
                  f"{c['p2'] + c['p3']:9.3f} {c['mean']:6.2f} {c['mode']:4d}")
    rd = integer_reading(cells)
    check = 'PASS' if bad == 0 else f'FAIL ({bad} draws)'
    print(f"\ninstrument check (m >= 2 in every draw): {check}")
    n2 = sum(c['mode'] == 2 for c in cells.values())
    print(f"W1-a: {rd['mixture']}; {rd['fraction']} (P(2) range {rd['p2_range']}); "
          f"mode 2 in {n2}/{len(cells)} cells")
    return cells


def _validate() -> bool:
    ok = True
    print("[1] synthetic ensembles with known statistics")
    rng = np.random.default_rng(0)
    # core_frac 0.5 at A*=5 -> onset 2.5; a* uniform on [0.01, 0.03]
    rows = [{'draw': i, 'capacity': 200.0, 'A_star': 5.0,
             'a_star': float(rng.uniform(0.01, 0.03)), 'core_frac': 0.5,
             'substrate_regens': 0.0} for i in range(4000)]
    s = stats(rows)
    good = abs(s['onset_core'] - 2.5) < 1e-9 and abs(s['a50'] - 0.02) < 0.001 \
        and abs(s['rel_width'] - 0.8) < 0.05
    ok &= good
    print(f"  onset {s['onset_core']:.2f} (2.5)  a50 {s['a50']:.4f} (0.02)  "
          f"rel_width {s['rel_width']:.3f} (0.80)  {'OK' if good else 'FAIL'}")
    print("[2] readings")
    flat = {2: {'onset_core': 2.5, 'rel_width': 1.30},
            3: {'onset_core': 2.7, 'rel_width': 1.31},
            4: {'onset_core': 2.6, 'rel_width': 1.25}}
    fall = {2: {'onset_core': 2.5, 'rel_width': 1.30},
            3: {'onset_core': 2.7, 'rel_width': 1.10},
            4: {'onset_core': 2.6, 'rel_width': 1.00}}
    far = {2: {'onset_core': 4.0, 'rel_width': 1.30},
           3: {'onset_core': 4.1, 'rel_width': 1.35},
           4: {'onset_core': 4.2, 'rel_width': 1.40}}
    for name, d, want in (('flat', flat, 'MOTIF ONSET HOLDS'),
                          ('falling', fall, 'SUBSTRATE-DEPENDENT'),
                          ('far core', far, 'SUBSTRATE-DEPENDENT'),
                          ('two r', {2: flat[2], 3: flat[3]},
                           'INCOMPLETE (2/3 geometries)')):
        got = reading(d)
        good = got == want
        ok &= good
        print(f"  {name:>9}: {got}  {'OK' if good else 'FAIL (want ' + want + ')'}")
    print("[3] integer core sizes and readings")
    rows3 = [{'A_star': 10.0, 'core_frac': 0.2}, {'A_star': 10.0, 'core_frac': 0.3},
             {'A_star': -1.0, 'core_frac': 0.0}, {'A_star': 7.0, 'core_frac': 4 / 7}]
    m = core_sizes(rows3)
    good = list(m) == [2, 3, 4]
    ok &= good
    print(f"  sizes {list(m)} (want [2, 3, 4])  {'OK' if good else 'FAIL'}")
    mix = {f"c{i}": {'p2': 0.5 + 0.01 * i, 'p3': 0.4} for i in range(9)}
    broad = {f"c{i}": {'p2': 0.4, 'p3': 0.3 if i < 3 else 0.5} for i in range(9)}
    for name, d, want in (('mixture', mix, ('INTEGER MIXTURE', 'FRACTION UNIVERSAL')),
                          ('broad', broad, ('BROAD', 'FRACTION UNIVERSAL'))):
        rd = integer_reading(d)
        good = (rd['mixture'], rd['fraction']) == want
        ok &= good
        print(f"  {name:>9}: {rd['mixture']}, {rd['fraction']}  "
              f"{'OK' if good else 'FAIL'}")
    print("\nPASS: throat comparison" if ok else "\nFAIL: throat comparison")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Throat criticality across substrates")
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--integer', action='store_true',
                    help='per-draw integer core size at onset (wave 1, W1-a)')
    ap.add_argument('--results', type=str, default='results')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.integer:
        integer_mixture(args.results)
    else:
        compare(args.results)


if __name__ == '__main__':
    main()
