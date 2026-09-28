"""
Is there a transition in the `sheet` rate strength beta, or a crossover?

Pre-registration:
docs/superpowers/specs/2026-09-28-sheet-beta-transition-preregistration.md
(every definition and verdict rule below is frozen there).

`sheet` (sheet_growth.py) is tree-like at beta = 1 and two-dimensional at
beta = 2. This driver scans beta between them, measures the boundary length
B(N) at every factor 2 in N, and reads off the crossover size N*(beta) at
which the local boundary exponent passes from the compact value 1/2 toward
the tree-like value 1. Then it asks which law N*(beta) follows:

    exponential   ln N* = a + b*beta             a crossover, no beta_c
    power law     ln N* = a - nu*ln(beta_c-beta) divergence at a finite beta_c

Usage:
    python sheet_transition.py --validate
    python sheet_transition.py --jobs 12
    python sheet_transition.py --analyze results/sheet_transition_TIMESTAMP.csv
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from sheet_growth import grow_sheet, census

# Frozen by the pre-registration.
BETAS = [round(1.0 + 0.1 * i, 1) for i in range(11)]
SEEDS = [200, 201, 202, 203]
N_MAX = 512000
N_FIRST = 1000
CAP = 6
E_CROSS = 0.75
E_COMPACT_MAX = 0.6
BETA_C_MAX = 2.2


def checkpoints(n_first: int, n_max: int) -> Tuple[int, ...]:
    out, n = [], n_first
    while n <= n_max:
        out.append(n)
        n *= 2
    return tuple(out)


def run_one(spec: Tuple[float, int, int, int]) -> List[Dict]:
    beta, seed, n_max, n_first = spec
    rows: List[Dict] = []

    def snap(n: int, adj) -> None:
        cs = census(adj, CAP)
        rows.append({'beta': beta, 'seed': seed, 'N': n,
                     'boundary_edges': cs['boundary_edges'],
                     'interior_deg_c': cs['interior_deg_c'],
                     'max_deg': cs['max_deg'],
                     'nonmanifold_edges': cs['nonmanifold_edges']})

    t0 = time.time()
    adj = grow_sheet(n_max, CAP, beta, seed,
                     checkpoints=checkpoints(n_first, n_max),
                     on_checkpoint=snap)
    for r in rows:
        r['reached'] = len(adj)
        r['seconds'] = time.time() - t0
    return rows


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------

def local_exponents(Ns: Sequence[int], B: Sequence[float]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """(geometric midpoints, e) with e = dln B / dln N per step."""
    Ns, B = np.asarray(Ns, float), np.asarray(B, float)
    e = np.log(B[1:] / B[:-1]) / np.log(Ns[1:] / Ns[:-1])
    return np.sqrt(Ns[1:] * Ns[:-1]), e


def crossover_size(mid: np.ndarray, e: np.ndarray
                   ) -> Tuple[str, float]:
    """
    ('finite', N*) | ('censored', nan) | ('below', nan).

    N* is where e crosses E_CROSS going up and stays above it at every
    later midpoint, interpolated linearly in ln N.
    """
    above = e >= E_CROSS
    if not above.any() or not above[-1]:
        return 'censored', float('nan')
    # first index from which e stays above to the end
    k = len(e) - 1
    while k > 0 and above[k - 1]:
        k -= 1
    if k == 0:
        return 'below', float('nan')
    x0, x1 = np.log(mid[k - 1]), np.log(mid[k])
    t = (E_CROSS - e[k - 1]) / (e[k] - e[k - 1])
    return 'finite', float(np.exp(x0 + t * (x1 - x0)))


def _aicc(rss: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return float('inf')
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_laws(betas: np.ndarray, lnN: np.ndarray) -> Dict[str, object]:
    """Exponential (2 params) vs power law with finite beta_c (3 params)."""
    n = len(betas)
    Xe = np.column_stack([np.ones(n), betas])
    ce = np.linalg.lstsq(Xe, lnN, rcond=None)[0]
    rss_e = float(np.sum((lnN - Xe @ ce) ** 2))

    best = (float('inf'), None, None)
    for bc in np.arange(betas.max() + 0.01, 4.0001, 0.005):
        Xp = np.column_stack([np.ones(n), -np.log(bc - betas)])
        cp = np.linalg.lstsq(Xp, lnN, rcond=None)[0]
        rss = float(np.sum((lnN - Xp @ cp) ** 2))
        if rss < best[0]:
            best = (rss, float(bc), cp)
    rss_p, beta_c, cp = best
    return {'exp_a': float(ce[0]), 'exp_b': float(ce[1]), 'rss_exp': rss_e,
            'aicc_exp': _aicc(rss_e, n, 2),
            'beta_c': beta_c, 'nu': float(cp[1]), 'pow_a': float(cp[0]),
            'rss_pow': rss_p, 'aicc_pow': _aicc(rss_p, n, 3)}


def verdict(table: Dict[float, Dict], n_max: int) -> Dict[str, object]:
    """The frozen verdict rules, applied mechanically."""
    finite = sorted(b for b, r in table.items() if r['status'] == 'finite')
    censored = sorted(b for b, r in table.items()
                      if r['status'] == 'censored')
    if len(finite) < 5:
        return {'verdict': 'INSUFFICIENT', 'n_finite': len(finite)}
    bf = np.array(finite)
    ln = np.log([table[b]['N_star'] for b in finite])
    fit = fit_laws(bf, ln)
    out = {'n_finite': len(finite), **fit}

    above_bc = [b for b in table if b > fit['beta_c']]
    compact_above = bool(above_bc) and all(
        table[b]['status'] == 'censored'
        and table[b]['e_last'] <= E_COMPACT_MAX for b in above_bc)
    if (fit['aicc_pow'] < fit['aicc_exp'] - 4
            and fit['beta_c'] <= BETA_C_MAX and compact_above):
        out['verdict'] = 'TRANSITION SUPPORTED'
        return out

    if fit['aicc_exp'] <= fit['aicc_pow'] + 2:
        if censored:
            b0 = censored[0]
            pred = float(np.exp(fit['exp_a'] + fit['exp_b'] * b0))
            out['exp_pred_beta'] = b0
            out['exp_pred_N'] = pred
            consistent = pred > n_max
        else:
            consistent = True
        if consistent:
            out['verdict'] = 'CROSSOVER SUPPORTED'
            return out
        out['note'] = ('exponential fits the finite points but predicts a '
                       'crossover inside the range for a beta that shows '
                       'none')
    out['verdict'] = 'UNDECIDED'
    return out


def analyze(rows: List[Dict], n_max: int) -> Dict[str, object]:
    betas = sorted({float(r['beta']) for r in rows})
    table: Dict[float, Dict] = {}
    print(f"\n{'beta':>5} {'status':>9} {'N*':>10}   local boundary exponent "
          f"e by size (compact 0.50, tree-like 1.00)")
    for b in betas:
        sel = [r for r in rows if float(r['beta']) == b]
        reached = min(int(r['reached']) for r in sel)
        Ns = sorted({int(r['N']) for r in sel})
        seeds = sorted({int(r['seed']) for r in sel})
        Ns = [n for n in Ns
              if len([r for r in sel if int(r['N']) == n]) == len(seeds)]
        B = [np.mean([float(r['boundary_edges']) for r in sel
                      if int(r['N']) == n]) for n in Ns]
        mid, e = local_exponents(Ns, B)
        status, ns = crossover_size(mid, e)
        # seed scatter of the last exponent
        e_seed = []
        for s in seeds:
            Bs = [float(r['boundary_edges']) for n in Ns[-2:] for r in sel
                  if int(r['N']) == n and int(r['seed']) == s]
            e_seed.append(np.log(Bs[1] / Bs[0]) / np.log(Ns[-1] / Ns[-2]))
        table[b] = {'status': status, 'N_star': ns, 'e_last': float(e[-1]),
                    'e_last_sd': float(np.std(e_seed, ddof=1)),
                    'mid': mid, 'e': e, 'reached': reached}
        ns_txt = f"{ns:10.0f}" if status == 'finite' else f"{'--':>10}"
        stall = '' if reached >= n_max else f"  STALLED at {reached}"
        print(f"{b:5.1f} {status:>9} {ns_txt}   "
              + " ".join(f"{x:.2f}" for x in e)
              + f"   (last: sd {table[b]['e_last_sd']:.2f} over seeds)"
              + stall)

    v = verdict(table, n_max)
    print()
    if 'rss_exp' in v:
        print(f"  exponential  ln N* = {v['exp_a']:.2f} + {v['exp_b']:.2f} "
              f"beta          RSS {v['rss_exp']:.4f}  AICc "
              f"{v['aicc_exp']:.2f}")
        print(f"  power law    ln N* = {v['pow_a']:.2f} - {v['nu']:.2f} "
              f"ln({v['beta_c']:.3f} - beta)  RSS {v['rss_pow']:.4f}  AICc "
              f"{v['aicc_pow']:.2f}")
        if 'exp_pred_N' in v:
            print(f"  exponential fit predicts N*({v['exp_pred_beta']}) = "
                  f"{v['exp_pred_N']:.3g}  (range ends at {n_max})")
    if 'note' in v:
        print(f"  note: {v['note']}")
    print(f"  finite N* at {v['n_finite']} values of beta")
    print(f"  VERDICT (frozen rules): {v['verdict']}")
    return {'table': table, **v}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _synthetic_rows(law: str, n_max: int) -> List[Dict]:
    """Boundary curves with a planted N*(beta), compact below and tree above."""
    rows = []
    Ns = checkpoints(N_FIRST, n_max)
    for b in BETAS:
        if law == 'exp':
            ns = np.exp(8.0 + 4.0 * (b - 1.0))
        else:
            ns = np.exp(8.0) * ((1.75 - 1.0) / (1.75 - b)) ** 2.0 \
                if b < 1.75 else np.inf
        for s in SEEDS:
            for n in Ns:
                # smooth interpolation: B ~ sqrt(N) below N*, ~ N above
                B = 4.0 * np.sqrt(n) * np.sqrt(1.0 + n / ns)
                rows.append({'beta': b, 'seed': s, 'N': n,
                             'boundary_edges': B, 'reached': n_max})
    return rows


def _validate() -> bool:
    ok = True
    print("[1] local exponents and crossover size on exact curves")
    Ns = checkpoints(1000, 512000)
    mid, e = local_exponents(Ns, [3.0 * n ** 0.5 for n in Ns])
    good = np.allclose(e, 0.5) and crossover_size(mid, e)[0] == 'censored'
    ok &= good
    print(f"  pure sqrt(N): e = {e[0]:.3f}, censored  "
          f"{'OK' if good else 'FAIL'}")
    mid, e = local_exponents(Ns, [0.6 * n for n in Ns])
    good = np.allclose(e, 1.0) and crossover_size(mid, e)[0] == 'below'
    ok &= good
    print(f"  pure N: e = {e[0]:.3f}, below range  {'OK' if good else 'FAIL'}")
    ns_true = 20000.0
    B = [4.0 * np.sqrt(n) * np.sqrt(1.0 + n / ns_true) for n in Ns]
    mid, e = local_exponents(Ns, B)
    st, ns = crossover_size(mid, e)
    # for this form e = 0.75 exactly at N = N*
    good = st == 'finite' and abs(np.log(ns / ns_true)) < 0.1
    ok &= good
    print(f"  planted N* = {ns_true:.0f}: recovered {ns:.0f}  "
          f"{'OK' if good else 'FAIL'}")

    print("[2] the verdict rules recover a planted law")
    for law, want in (('exp', 'CROSSOVER SUPPORTED'),
                      ('pow', 'TRANSITION SUPPORTED')):
        v = analyze(_synthetic_rows(law, N_MAX), N_MAX)
        good = v['verdict'] == want
        if law == 'pow':
            good = good and abs(v['beta_c'] - 1.75) < 0.05
        ok &= good
        print(f"  planted {law}: {v['verdict']}  {'OK' if good else 'FAIL'}")

    print("[3] the generator's endpoints, small scale")
    for beta, lo, hi in ((0.0, 0.9, 1.1), (3.0, 0.4, 0.6)):
        B = {}
        for n in (4000, 16000):
            B[n] = np.mean([census(grow_sheet(n, CAP, beta, seed=s), CAP)
                            ['boundary_edges'] for s in (1, 2)])
        ex = np.log(B[16000] / B[4000]) / np.log(4.0)
        good = lo <= ex <= hi
        ok &= good
        print(f"  beta={beta}: boundary exponent {ex:.2f} (expected "
              f"{lo}-{hi})  {'OK' if good else 'FAIL'}")

    print("\nPASS: sheet transition instrument" if ok
          else "\nFAIL: sheet transition instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Transition or crossover in the sheet rate strength?",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--analyze', type=str, default=None)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds global RNGs; the scan uses the frozen seeds '
                         '200-203, so this is a formality')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.analyze:
        with open(args.analyze) as f:
            analyze(list(csv.DictReader(f)), N_MAX)
        return

    specs = [(b, s, N_MAX, N_FIRST) for b in BETAS for s in SEEDS]
    rows: List[Dict] = []
    Path("results").mkdir(exist_ok=True)
    path = f"results/sheet_transition_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for out in tqdm(ex.map(run_one, specs), total=len(specs),
                        desc='sheets'):
            rows.extend(out)
            # rewrite after every run so a crash loses nothing
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"Saved {path}")
    analyze(rows, N_MAX)


if __name__ == '__main__':
    main()
