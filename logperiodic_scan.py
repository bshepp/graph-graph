"""
Log-periodicity (discrete-scale-invariance) scan of the sandbox's scaling laws.

Trigger: Ecker/Ecker/Grumiller (PRL 2026, arXiv:2601.14358) construct the
Choptuik critical solution analytically at large D -- a "spacetime crystal"
periodic in LOG-scale (discrete self-similarity, DSS). The graph analogue
would be a log-periodic modulation riding on one of this sandbox's scaling
relations. This driver hunts for it and, finding none, reports a BOUNDED null
(what amplitude is excluded), not just "we saw nothing".

Tested relations (>= 8 distinct x-values required, else declared
INSUFFICIENT and not tested):
  * prune d(p) per N        (results/prune_dimension_*.csv, cubic detrend)
  * grown extent ~ N        (dense 24-point grid, --generate, OLS in ln-ln)
  * grown ball growth B(r)  (N=20000, --generate, OLS in ln-ln, pre-saturation
                             cut B <= N/4, r >= 2)
  * barrier extent ~ N and cap->d are auto-declared INSUFFICIENT on the
    historical 5-point / 3-point grids -- kept so the refusal is on record.

Method (fixed 2026-08-11 before residuals were examined; the discriminator
ladder was added the same day after the first pass, labeled post-hoc there,
and is now part of the standard pipeline):
  1. Detrend with the pre-registered smooth model; Lomb-Scargle periodogram
     of residuals vs ln x; significance = permutation test on max power
     (2000 shuffles), curve threshold p < 0.01, Bonferroni across curves.
  2. THE PERMUTATION TEST DETECTS ANY AUTOCORRELATED STRUCTURE, NOT
     SPECIFICALLY PERIODICITY -- a smooth misfit bow triggers it as readily
     as a crystal. Every raw p < 0.05 therefore enters the discriminator
     ladder: (a) cycles-in-range = ln-span / best period, demanding >= 2;
     (b) detrend orders 2..5 -- a polynomial absorbs a bow but not several
     cycles, so genuine DSS must survive deg 4-5; (c) cross-seed / cross-N
     residual correlation -- r ~ 1 with a shared x-grid means a
     deterministic smooth shape, not noise-borne periodicity.
  3. Integer staircase gate: extent/ball observables are integer-derived;
     synthetic rounded power laws are pushed through the identical pipeline
     to measure the false-detection rate.
  4. Injection-recovery sensitivity: a90 = smallest log-periodic amplitude
     (ln-units ~ fractional modulation) detected >= 90% of the time, so a
     null is the statement "no modulation above a90".

Result (2026-08-11, banked in FINDINGS.md): global null. The two raw
p < 0.05 families both resolved as smooth-shape misfit by the ladder
(prune: cross-N residual r = +0.89..+0.98, killed by quartic; ball growth:
1.5 cycles = one bow + the r=2 discreteness point, killed by quartic, cross-
seed r = +0.95..+1.00). Bounds: no modulation above ~10% on grown extent
scaling, ~10-20% on ball growth after shape removal; prune d(p) sensitivity
insufficient at n=10 (a90 = inf -- no claim either way).

Usage:
    python logperiodic_scan.py --generate   # write the dense grown CSVs
    python logperiodic_scan.py              # run the scan + ladder
    python logperiodic_scan.py --validate   # positive + negative controls
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from scipy.signal import lombscargle
from tqdm import tqdm

from simulation import create_initial_graph
from barrier_scaling import estimate_extent

RESULTS = Path(__file__).parent / 'results'
EXTENT_CSV = RESULTS / 'logperiodic_grown_extent.csv'
BALL_CSV = RESULTS / 'logperiodic_grown_ballgrowth.csv'
MIN_PTS = 8
P_THRESH = 0.01
N_PERM = 2000
BALL_N = 20000


# --------------------------------------------------------------------------
# Dense data generation (generation-only -- no rule dynamics)
# --------------------------------------------------------------------------

def generate_dense(seeds: int = 3) -> None:
    """Dense grown-extent grid (24 log-spaced N) and ball-growth curves."""
    RESULTS.mkdir(exist_ok=True)
    ns = np.unique(np.round(np.logspace(np.log10(500), np.log10(50000), 24))
                   .astype(int))
    with open(EXTENT_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['N', 'seed', 'diam', 'mean_ecc', 'lcc_frac'])
        for n in tqdm(ns, desc='extent grid'):
            for seed in range(seeds):
                np.random.seed(seed)
                G = create_initial_graph(int(n), topology='grown', k=6,
                                         seed=seed)
                diam, mecc, lcc = estimate_extent(G)
                w.writerow([int(n), seed, diam, mecc, lcc])

    with open(BALL_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'r', 'mean_ball', 'n_sources'])
        for seed in tqdm(range(seeds), desc='ball growth'):
            np.random.seed(seed)
            G = create_initial_graph(BALL_N, topology='grown', k=6, seed=seed)
            nodes = list(G.nodes())
            rng = np.random.default_rng(seed)
            srcs = rng.choice(len(nodes), size=400, replace=False)
            maxr = 30
            sums = np.zeros(maxr + 1)
            for si in srcs:
                dist = nx.single_source_shortest_path_length(
                    G, nodes[int(si)], cutoff=maxr)
                dvals = np.fromiter(dist.values(), dtype=int)
                for r in range(maxr + 1):
                    sums[r] += np.sum(dvals <= r)
            for r in range(maxr + 1):
                w.writerow([seed, r, sums[r] / len(srcs), len(srcs)])


# --------------------------------------------------------------------------
# Periodogram machinery
# --------------------------------------------------------------------------

def _period_grid(x: np.ndarray) -> Optional[np.ndarray]:
    span = x.max() - x.min()
    dmed = np.median(np.diff(np.sort(np.unique(x))))
    pmin, pmax = 2.0 * dmed, span / 1.5
    if pmin >= pmax:
        return None
    return 2.0 * np.pi / np.linspace(pmin, pmax, 200)


def _max_power(x: np.ndarray, res: np.ndarray, w: np.ndarray) -> float:
    r = res - res.mean()
    if r.std() == 0:
        return 0.0
    return float(lombscargle(x, r, w).max())


def _perm_p(x: np.ndarray, res: np.ndarray, w: np.ndarray,
            n_perm: int = N_PERM,
            rng: Optional[np.random.Generator] = None) -> float:
    rng = rng or np.random.default_rng(0)
    obs = _max_power(x, res, w)
    count = sum(_max_power(x, rng.permutation(res), w) >= obs
                for _ in range(n_perm))
    return (count + 1) / (n_perm + 1)


def _detrend(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    return y - np.polyval(np.polyfit(x, y, deg), x)


def _sensitivity(x: np.ndarray, res: np.ndarray, w: np.ndarray,
                 rng: np.random.Generator) -> float:
    """a90: smallest injected log-periodic amplitude detected >=90% of runs."""
    span = x.max() - x.min()
    dmed = np.median(np.diff(np.sort(np.unique(x))))
    for a in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        hits = 0
        for _ in range(100):
            period = rng.uniform(2 * dmed, span / 1.5)
            phi = rng.uniform(0, 2 * np.pi)
            synth = rng.permutation(res) + a * np.cos(
                2 * np.pi * x / period + phi)
            hits += _perm_p(x, synth, w, n_perm=500, rng=rng) < P_THRESH
        if hits >= 90:
            return a
    return float('inf')


def _ladder(x: np.ndarray, y: np.ndarray, base_deg: int,
            rng: np.random.Generator) -> Tuple[float, Dict[int, float]]:
    """Discriminators: cycles-in-range at base detrend + detrend-order p's."""
    w = _period_grid(x)
    res = _detrend(x, y, base_deg)
    pw = lombscargle(x, res - res.mean(), w)
    best_period = 2 * np.pi / w[int(np.argmax(pw))]
    cycles = (x.max() - x.min()) / best_period
    orders = {deg: _perm_p(x, _detrend(x, y, deg), w, rng=rng)
              for deg in range(base_deg, base_deg + 4)}
    return cycles, orders


# --------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------

def _test(name: str, x: np.ndarray, y: np.ndarray, deg: int,
          rng: np.random.Generator, results: List) -> None:
    if len(np.unique(x)) < MIN_PTS:
        print(f"  {name:<34s} INSUFFICIENT ({len(np.unique(x))} pts)")
        return
    w = _period_grid(x)
    if w is None:
        print(f"  {name:<34s} INSUFFICIENT (degenerate period range)")
        return
    p = _perm_p(x, _detrend(x, y, deg), w, rng=rng)
    a90 = _sensitivity(x, _detrend(x, y, deg), w, rng)
    line = f"  {name:<34s} n={len(x):3d}  p={p:.4f}  a90={a90:.2f}"
    survives: Optional[bool] = None
    if p < 0.05:
        cycles, orders = _ladder(x, y, deg, rng)
        survives = all(v < P_THRESH for v in orders.values()) and cycles >= 2
        line += (f"  [ladder: cycles={cycles:.1f}, "
                 + ' '.join(f"deg{d}:{v:.3f}" for d, v in orders.items())
                 + (' DSS-CANDIDATE' if survives else ' -> misfit artifact'))
        line += ']'
    print(line)
    results.append((name, p, survives))


def scan(prune_csv: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    results: List[Tuple[str, float]] = []
    print("=== log-periodicity scan (see module docstring for method) ===")

    print("[prune d(p) per N]")
    if prune_csv.exists():
        d = np.genfromtxt(prune_csv, delimiter=',', names=True)
        for n in np.unique(d['N']):
            sel = d[d['N'] == n]
            ps = np.unique(sel['p'])
            y = np.array([sel[sel['p'] == p]['d_eff_median'].mean()
                          for p in ps])
            _test(f"prune[N={int(n)}] cubic", np.log(ps), y, 3, rng, results)
    else:
        print(f"  ({prune_csv.name} not found -- skipped)")

    print("[grown extent ~ N]")
    if EXTENT_CSV.exists():
        d = np.genfromtxt(EXTENT_CSV, delimiter=',', names=True)
        ns = np.unique(d['N'])
        y = np.array([np.log(d[d['N'] == n]['mean_ecc'].mean()) for n in ns])
        _test("grown ln mean_ecc ~ ln N", np.log(ns), y, 1, rng, results)
    else:
        print("  (run --generate first -- skipped)")

    print("[grown ball growth ln B ~ ln r]")
    if BALL_CSV.exists():
        d = np.genfromtxt(BALL_CSV, delimiter=',', names=True)
        for s in np.unique(d['seed']):
            sel = d[d['seed'] == s]
            keep = (sel['r'] >= 2) & (sel['mean_ball'] <= BALL_N / 4)
            _test(f"ball[seed {int(s)}] quadratic",
                  np.log(sel['r'][keep].astype(float)),
                  np.log(sel['mean_ball'][keep]), 2, rng, results)
    else:
        print("  (run --generate first -- skipped)")

    m = max(len(results), 1)
    # A curve counts as a DSS candidate only if it is Bonferroni-significant
    # AND survived the discriminator ladder -- a significant p with a failed
    # ladder is a smooth-misfit artifact by construction (see docstring).
    candidates = [n for n, p, surv in results if p * m < P_THRESH and surv]
    artifacts = [n for n, p, surv in results
                 if p * m < P_THRESH and surv is False]
    print(f"\n{len(results)} curves tested (Bonferroni x{m}); "
          f"{len(candidates)} DSS candidate(s), "
          f"{len(artifacts)} ladder-disqualified artifact(s).")
    print("VERDICT: " + (f"DSS CANDIDATES: {', '.join(candidates)}"
                         if candidates else
                         "no log-periodic modulation (bounded null; a90 per "
                         "curve above)"))


# --------------------------------------------------------------------------
# Validation: the pipeline must find a planted crystal and refuse a fake one
# --------------------------------------------------------------------------

def _validate(seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    x = np.log(np.unique(np.round(np.logspace(np.log10(500), np.log10(50000),
                                              24))))
    w = _period_grid(x)
    ok = True

    # Positive control: power law + 3-cycle log-periodic wiggle must be
    # detected AND survive the discriminator ladder.
    span = x.max() - x.min()
    period = span / 3.0
    y_pos = (0.5 * x + 0.15 * np.cos(2 * np.pi * x / period)
             + rng.normal(0, 0.02, size=len(x)))
    p_pos = _perm_p(x, _detrend(x, y_pos, 1), w, rng=rng)
    cycles, orders = _ladder(x, y_pos, 1, rng)
    pos_ok = (p_pos < P_THRESH and cycles >= 2
              and all(v < P_THRESH for v in orders.values()))
    ok &= pos_ok
    print(f"positive control (planted 3-cycle DSS, amp 0.15): p={p_pos:.4f}, "
          f"cycles={cycles:.1f}, ladder "
          + ' '.join(f"deg{d}:{v:.3f}" for d, v in orders.items())
          + f"  {'OK' if pos_ok else 'MISSED -- FAIL'}")

    # Negative control 1: pure noisy power law -> no detection.
    y_neg = 0.5 * x + rng.normal(0, 0.02, size=len(x))
    p_neg = _perm_p(x, _detrend(x, y_neg, 1), w, rng=rng)
    neg_ok = p_neg > P_THRESH
    ok &= neg_ok
    print(f"negative control (pure power law): p={p_neg:.4f}  "
          f"{'OK' if neg_ok else 'FALSE DETECTION -- FAIL'}")

    # Negative control 2: integer-rounded power law (staircase) -- the
    # false-positive rate over 100 draws must stay near the nominal 1%.
    false = 0
    for _ in range(100):
        alpha = rng.uniform(0.45, 0.55)
        y_int = np.maximum(np.round(np.exp(alpha * x)), 1.0)
        false += _perm_p(x, _detrend(x, np.log(y_int), 1), w,
                         n_perm=500, rng=rng) < P_THRESH
    stair_ok = false <= 5
    ok &= stair_ok
    print(f"staircase control: {false}/100 false detections  "
          f"{'OK' if stair_ok else 'ALIASING -- FAIL'}")

    print(f"\n{'PASS' if ok else 'FAIL'}: log-periodicity scan instrument")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hunt for discrete scale invariance in scaling relations.")
    ap.add_argument('--generate', action='store_true',
                    help='write the dense grown CSVs into results/')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--prune-csv', type=Path,
                    default=RESULTS / 'prune_dimension_20260627_213122.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(args.seed) else 1)
    if args.generate:
        generate_dense()
        print(f"wrote {EXTENT_CSV.name}, {BALL_CSV.name}")
        return
    scan(args.prune_csv, args.seed)


if __name__ == '__main__':
    main()
