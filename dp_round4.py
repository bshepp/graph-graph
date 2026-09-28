"""
d(p) fine structure, round 4: is the waveform just the ring -> small-world
crossover seen through a polynomial detrend?

Pre-registration:
docs/superpowers/specs/2026-09-28-dp-round4-crossover-preregistration.md

Fits a crossover form to the log ball counts L_r(p) on FRESH seeds and asks
whether what is left is seed noise:

    primary     F(p) = a + h*ln(1 + (p/p0)^g)                 4 parameters
    comparator  two such crossovers added                     7 parameters

against polynomial detrends with the same parameter counts (cubic, sextic).

Usage:
    python dp_round4.py --validate
    python dp_round4.py --jobs 5
    python dp_round4.py --analyze results/dp_round4_TIMESTAMP.csv
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

from dp_decomposition import (measure, banked_ps, fit_weights, N_NODES,
                              MAX_RADIUS)

SEEDS = [6, 7, 8, 9, 10, 11]
RADII = list(range(4, MAX_RADIUS + 1))
N_RESTARTS = 40


def primary(theta: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, h, lp0, g = theta
    return a + h * np.logaddexp(0.0, g * (np.log(p) - lp0))


def comparator(theta: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, h1, lp1, g1, h2, lp2, g2 = theta
    x = np.log(p)
    return (a + h1 * np.logaddexp(0.0, g1 * (x - lp1))
            + h2 * np.logaddexp(0.0, g2 * (x - lp2)))


def fit_form(form: Callable, n_par: int, p: np.ndarray, y: np.ndarray,
             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Best of N_RESTARTS least-squares fits; returns (theta, residual)."""
    x = np.log(p)
    best, best_cost = None, np.inf
    for _ in range(N_RESTARTS):
        if n_par == 4:
            t0 = [y.min(), rng.uniform(0.1, 2.0),
                  rng.uniform(x.min() - 1, x.max()), rng.uniform(0.5, 4.0)]
        else:
            t0 = [y.min(), rng.uniform(0.1, 2.0),
                  rng.uniform(x.min() - 1, x.max()), rng.uniform(0.5, 4.0),
                  rng.uniform(-1.0, 1.0),
                  rng.uniform(x.min(), x.max() + 1), rng.uniform(0.5, 4.0)]
        try:
            r = least_squares(lambda t: form(t, p) - y, t0, max_nfev=4000)
        except (ValueError, FloatingPointError):
            continue
        if r.cost < best_cost and np.all(np.isfinite(r.x)):
            best, best_cost = r.x, r.cost
    return best, y - form(best, p)


def poly_resid(p: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    x = np.log(p)
    x = (x - x.mean()) / x.std()
    return y - np.polyval(np.polyfit(x, y, order), x)


def table(rows: List[Dict]) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """(ps, seeds, L[p, seed, r]) with r = 1..MAX_RADIUS."""
    ps = np.array(sorted({float(r['p']) for r in rows}))
    seeds = sorted({int(r['seed']) for r in rows})
    key = {(float(r['p']), int(r['seed'])): r for r in rows}
    L = np.array([[[float(key[(p, s)][f"L{r}"])
                    for r in range(1, MAX_RADIUS + 1)]
                   for s in seeds] for p in ps])
    return ps, seeds, L


def analyze(rows: List[Dict], seed: int = 0) -> Dict[str, object]:
    ps, seeds, L = table(rows)
    rng = np.random.default_rng(seed)
    Lm = L.mean(axis=1)
    SE = (L.std(axis=1, ddof=1) / np.sqrt(len(seeds))).mean(axis=0)
    w = fit_weights(MAX_RADIUS)

    res = {k: np.zeros_like(Lm) for k in ('cubic', 'sextic', 'primary',
                                          'comparator')}
    for r in range(1, MAX_RADIUS + 1):
        y = Lm[:, r - 1]
        res['cubic'][:, r - 1] = poly_resid(ps, y, 3)
        res['sextic'][:, r - 1] = poly_resid(ps, y, 6)
        if r in RADII:
            res['primary'][:, r - 1] = fit_form(primary, 4, ps, y, rng)[1]
            res['comparator'][:, r - 1] = fit_form(comparator, 7, ps, y,
                                                   rng)[1]
        else:
            # r <= 3 carries no waveform (round 3); use the cubic there
            res['primary'][:, r - 1] = res['cubic'][:, r - 1]
            res['comparator'][:, r - 1] = res['sextic'][:, r - 1]

    # noise level of the waveform itself, straight from the seeds
    sigma_w = float(((L @ w).std(axis=1, ddof=1)
                     / np.sqrt(len(seeds))).mean())
    rms = {k: np.sqrt(np.mean(v ** 2, axis=0)) for k, v in res.items()}
    wave = {k: v @ w for k, v in res.items()}
    ptp = {k: float(np.ptp(v)) for k, v in wave.items()}

    print(f"\n   r   seed SE   RMS residual / seed SE")
    print(f"                 {'cubic':>8} {'sextic':>8} {'primary':>8} "
          f"{'compar.':>8}")
    for r in RADII:
        i = r - 1
        print(f"  {r:2d}   {SE[i]:.4f}   "
              + " ".join(f"{rms[k][i] / SE[i]:8.1f}"
                         for k in ('cubic', 'sextic', 'primary',
                                   'comparator')))
    print("\n  peak-to-peak of the d waveform: "
          + "  ".join(f"{k} {ptp[k]:.3f}" for k in
                      ('cubic', 'sextic', 'primary', 'comparator')))

    # split-half reproducibility of the primary residual at r = 10
    half = []
    for idx in ([0, 1, 2], [3, 4, 5]):
        y = L[:, idx, MAX_RADIUS - 1].mean(axis=1)
        half.append(fit_form(primary, 4, ps, y, rng)[1])
    split = float(np.corrcoef(half[0], half[1])[0, 1])
    print(f"  split-half correlation of the primary residual at r = 10: "
          f"{split:+.3f}")
    print(f"  waveform noise level sigma_w = {sigma_w:.4f}  "
          f"(5 sigma_w = {5 * sigma_w:.3f})")

    i10 = MAX_RADIUS - 1
    support = (all(rms['primary'][r - 1] <= 2 * SE[r - 1] for r in (8, 9, 10))
               and ptp['primary'] <= max(0.25 * ptp['cubic'],
                                         5.0 * sigma_w))
    refute = rms['primary'][i10] > 4 * SE[i10] and split > 0.7
    removed = 1.0 - float(np.var(wave['primary']) / np.var(wave['cubic']))
    removed_c = 1.0 - float(np.var(wave['comparator'])
                            / np.var(wave['sextic']))
    if support:
        verdict = 'H SUPPORTED'
    elif refute:
        verdict = 'H REFUTED'
    else:
        verdict = 'PARTIAL'
    print(f"  the primary form removes {100 * removed:.0f}% of the cubic "
          f"waveform's variance")
    print(f"  the comparator removes {100 * removed_c:.0f}% of the sextic "
          f"waveform's variance")
    print(f"  VERDICT (frozen rules): {verdict}")
    return {'verdict': verdict, 'rms': rms, 'SE': SE, 'ptp': ptp,
            'sigma_w': sigma_w,
            'split': split, 'removed': removed, 'ps': ps, 'res': res}


def _synthetic(bump: float, noise: float, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    ps = np.array(banked_ps())
    rows = []
    for s in SEEDS:
        for p in ps:
            row = {'p': float(p), 'seed': s, 'N': 0, 'd_eff_median': 0.0}
            for r in range(1, MAX_RADIUS + 1):
                base = np.log(6.0 * r + 1) + 0.12 * r * np.logaddexp(
                    0.0, 3.0 * (np.log(p) - np.log(1.0 / r)))
                wiggle = bump * np.sin(4.0 * np.log(p)) if r >= 4 else 0.0
                row[f"L{r}"] = base + wiggle + noise * rng.standard_normal()
            rows.append(row)
    return rows


def _validate() -> bool:
    ok = True
    print("[1] a pure crossover plus noise must read H SUPPORTED, although "
          "a cubic leaves a waveform")
    v = analyze(_synthetic(0.0, 0.01, 1))
    good = (v['verdict'] == 'H SUPPORTED'
            and v['ptp']['cubic'] > 3 * v['ptp']['primary'])
    ok &= good
    print(f"  -> {v['verdict']}  {'OK' if good else 'FAIL'}")
    print("\n[2] a crossover plus a planted reproducible ripple must read "
          "H REFUTED")
    v = analyze(_synthetic(0.05, 0.01, 2))
    good = v['verdict'] == 'H REFUTED'
    ok &= good
    print(f"  -> {v['verdict']}  {'OK' if good else 'FAIL'}")
    print("\nPASS: d(p) round-4 instrument" if ok
          else "\nFAIL: d(p) round-4 instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="d(p) round 4: crossover form vs polynomial detrend",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--analyze', type=str, default=None)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds the fit restarts; the grid uses the frozen '
                         'seeds 6-11')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.analyze:
        with open(args.analyze) as f:
            analyze(list(csv.DictReader(f)), args.seed)
        return

    specs = [(float(p), s, N_NODES) for p in banked_ps() for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        rows = list(tqdm(ex.map(measure, specs), total=len(specs),
                         desc='grid'))
    Path("results").mkdir(exist_ok=True)
    path = f"results/dp_round4_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Saved {path}")
    analyze(rows, args.seed)


if __name__ == '__main__':
    main()
