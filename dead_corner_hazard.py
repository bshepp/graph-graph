"""
Dead-corner hazard: when does a `sheet` acquire its first permanent dead corner?

Pre-registration: docs/superpowers/specs/2026-10-17-wave1-preregistration.md (W1-d).

A dead corner is a boundary node of degree >= c whose wheel can never be closed
(`sheet_growth._closable` is None); it is permanent. This driver grows fresh
sheets with an exact event-level detector (the `on_event` hook of
`grow_sheet`), records the node count N_first at which the first dead corner
appears together with the event that created it, and estimates the hazard per
added node: exposure MLE, censored Weibull shape, Nelson-Aalen, and the
Arrhenius fit of ln(rate) against beta.

Usage:
    python dead_corner_hazard.py --validate
    python dead_corner_hazard.py --scan --jobs 10            # the frozen scan
    python dead_corner_hazard.py --analyze results/dead_corner_hazard_*.csv
"""

import argparse
import csv
import glob
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from sheet_growth import grow_sheet, _closable, to_csr

CAP = 6
# Frozen scan (pre-registration W1-d).
SCAN_BETAS = (1.0, 1.25, 1.5, 1.75)
SCAN_NMAX = {1.0: 131072, 1.25: 524288, 1.5: 2097152, 1.75: 4194304}
SCAN_SEEDS = tuple(range(500, 540))
NSTAR_B_SLOPE = 5.38          # the N*_B law, FINDINGS 2026-09-29


# --------------------------------------------------------------------------
# Failure modes and the detector
# --------------------------------------------------------------------------

def is_boundary_node(adj, y: int) -> bool:
    return any(len(adj[y] & adj[z]) == 1 for z in adj[y])


def failure_modes(adj, x: int, c: int = CAP) -> str:
    """Which `_closable` conditions fail at x: subset of 'i', 'ii', 'iii', 'iv'.

    (i) not exactly two boundary edges; (ii) the link ends adjacent;
    (iii) the link ends share a neighbour other than x (pinch);
    (iv) a link end at the cap (degree guard). (ii)-(iv) are evaluated only
    when (i) holds."""
    ends = [y for y in adj[x] if len(adj[x] & adj[y]) == 1]
    if len(ends) != 2:
        return 'i'
    a, b = ends
    out = []
    if b in adj[a]:
        out.append('ii')
    if (adj[a] & adj[b]) != {x}:
        out.append('iii')
    if len(adj[a]) >= c or len(adj[b]) >= c:
        out.append('iv')
    return '+'.join(out)


class DeadDetector:
    """`on_event` hook: records every dead corner the moment it appears.

    After an event only nodes within one hop of a changed node can change
    closability (x's closability reads x's degree, x's boundary edges, its
    link ends' degrees and their common neighbours). Interior is permanent
    (edges and triangles are never removed), so interior nodes are cached.
    """

    def __init__(self, c: int = CAP, stop_at_first: bool = False):
        self.c = c
        self.stop_at_first = stop_at_first
        self.dead: Set[int] = set()
        self.interior: Set[int] = set()
        self.records: List[Dict] = []

    def __call__(self, kind: str, nodes: Tuple[int, ...], adj) -> bool:
        c = self.c
        cand = set(nodes)
        for y in nodes:
            cand |= adj[y]
        new = []
        for y in cand:
            if y in self.dead or y in self.interior or len(adj[y]) < c:
                continue
            if not is_boundary_node(adj, y):
                self.interior.add(y)
                continue
            if _closable(adj, y, c) is None:
                new.append(y)
        if not new:
            return False
        n = len(adj)
        newset = set(new)
        for y in sorted(new):
            self.dead.add(y)
            ends = [z for z in adj[y] if len(adj[y] & adj[z]) == 1]
            self.records.append({
                'N': n, 'node': y, 'mode': failure_modes(adj, y, c),
                'kind': kind, 'event_nodes': ' '.join(map(str, nodes)),
                'event_degs': ' '.join(str(len(adj[z])) for z in nodes),
                'ends': ' '.join(map(str, ends)),
                'end_degs': ' '.join(str(len(adj[z])) for z in ends),
                'partner_new': ' '.join(str(z) for z in ends if z in newset),
                'n_new': len(new)})
        return self.stop_at_first


# --------------------------------------------------------------------------
# One growth
# --------------------------------------------------------------------------

def grow_one(spec: Tuple[float, int, int]) -> List[Dict]:
    beta, seed, n_max = spec
    random.seed(seed)
    np.random.seed(seed)
    det = DeadDetector(CAP, stop_at_first=True)
    t0 = time.time()
    adj = grow_sheet(n_max, CAP, beta, seed, on_event=det)
    secs = time.time() - t0
    n_end = len(adj)
    base = {'beta': beta, 'seed': seed, 'n_max': n_max, 'n_end': n_end,
            'stalled': int(not det.records and n_end < n_max),
            'seconds': round(secs, 1)}
    if not det.records:
        return [{**base, 'event': 0, 'N_first': -1, 'exposure': n_end,
                 'node': -1, 'mode': '', 'kind': '', 'event_nodes': '',
                 'event_degs': '', 'ends': '', 'end_degs': '',
                 'partner_new': '', 'n_new': 0}]
    first_n = det.records[0]['N']
    return [{**base, 'event': 1, 'N_first': first_n, 'exposure': first_n, **r}
            for r in det.records if r['N'] == first_n]


def scan(jobs: int, out_csv: str, betas: Sequence[float] = SCAN_BETAS,
         seeds: Sequence[int] = SCAN_SEEDS) -> List[Dict]:
    # Most expensive first, so the long tails start early.
    specs = sorted(((b, s, SCAN_NMAX[b]) for b in betas for s in seeds),
                   key=lambda t: -t[0])
    rows: List[Dict] = []
    Path(out_csv).parent.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=jobs) as ex, \
            open(out_csv, 'w', newline='') as f:
        w = None
        for res in ex.map(grow_one, specs):
            for r in res:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(r.keys()))
                    w.writeheader()
                w.writerow(r)
                f.flush()
            rows += res
            r = res[0]
            print(f"beta {r['beta']:<5} seed {r['seed']}  N_first "
                  f"{r['N_first'] if r['event'] else 'censored':>9}  "
                  f"{r['mode']:<6} {r['kind']:<6} [{r['seconds']:.0f} s]",
                  flush=True)
    print(f"saved: {out_csv}")
    return rows


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def per_seed(rows: List[Dict]) -> List[Dict]:
    """One row per (beta, seed): t = N_first or censoring size, event flag."""
    seen, out = set(), []
    for r in rows:
        key = (float(r['beta']), int(r['seed']))
        if key in seen:
            continue
        seen.add(key)
        out.append({'beta': key[0], 'seed': key[1],
                    't': float(r['exposure']), 'event': int(r['event'])})
    return out


def exposure_rate(t: np.ndarray, e: np.ndarray) -> Dict[str, float]:
    d, T = int(e.sum()), float(t.sum())
    lo = stats.chi2.ppf(0.025, 2 * d) / 2 / T if d else 0.0
    hi = stats.chi2.ppf(0.975, 2 * d + 2) / 2 / T
    return {'d': d, 'n': len(t), 'T': T, 'rate': d / T, 'lo': lo, 'hi': hi}


def _weibull_profile_ll(k: float, groups: List[Tuple[np.ndarray, np.ndarray]]
                        ) -> float:
    """Censored Weibull log-likelihood at shape k, scale profiled per group."""
    ll = 0.0
    for t, e in groups:
        d = e.sum()
        if d == 0:
            continue
        lt = np.log(t)
        s = np.exp(k * lt).sum()
        ll += d * np.log(k) - d * np.log(s / d) + (k - 1) * lt[e == 1].sum() - d
    return float(ll)


def weibull_shape(groups: List[Tuple[np.ndarray, np.ndarray]]
                  ) -> Dict[str, float]:
    # Rescale times per group for numerical stability (shape is scale-free).
    g2 = [(t / np.exp(np.log(t).mean()), e) for t, e in groups]
    f = lambda lk: -_weibull_profile_ll(np.exp(lk), g2)
    res = minimize_scalar(f, bounds=(np.log(0.05), np.log(20)), method='bounded')
    lk_hat, ll_hat = res.x, -res.fun
    def bound(sign):
        a, b = (lk_hat, np.log(20)) if sign > 0 else (np.log(0.05), lk_hat)
        target = ll_hat - stats.chi2.ppf(0.95, 1) / 2
        far = b if sign > 0 else a
        if -f(far) > target:
            return float(np.exp(far))     # interval reaches the search bound
        for _ in range(80):
            m = (a + b) / 2
            if (-f(m) > target) == (sign > 0):
                a = m
            else:
                b = m
        return float(np.exp((a + b) / 2))
    return {'k': float(np.exp(lk_hat)), 'k_lo': bound(-1), 'k_hi': bound(+1)}


def nelson_aalen(t: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t, e = t[order], e[order]
    at_risk = len(t) - np.arange(len(t))
    H = np.cumsum(e / at_risk)
    return t[e == 1], H[e == 1]


def arrhenius(betas: np.ndarray, rates: np.ndarray, d: np.ndarray
              ) -> Dict[str, float]:
    """WLS of ln(rate) on beta (weights d = 1/var of ln rate), plus a
    quadratic term's 95% interval."""
    y, w = np.log(rates), d.astype(float)
    def wls(X):
        W = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
        resid = y - X @ coef
        dof = max(len(y) - X.shape[1], 1)
        s2 = max(float((w * resid ** 2).sum() / dof), 1.0)   # >= Poisson
        cov = s2 * np.linalg.inv(X.T @ (X * w[:, None]))
        return coef, cov, dof
    X1 = np.column_stack([np.ones_like(betas), betas])
    c1, cov1, dof1 = wls(X1)
    out = {'intercept': float(c1[0]), 'a': float(-c1[1]),
           'a_se': float(np.sqrt(cov1[1, 1]))}
    if len(betas) >= 4:
        X2 = np.column_stack([X1, betas ** 2])
        c2, cov2, dof2 = wls(X2)
        se = float(np.sqrt(cov2[2, 2]))
        tq = stats.t.ppf(0.975, dof2)
        out.update({'quad': float(c2[2]), 'quad_lo': float(c2[2] - tq * se),
                    'quad_hi': float(c2[2] + tq * se)})
    return out


def analyze(paths: Sequence[str], extra_b2: Optional[str] = None) -> Dict:
    rows: List[Dict] = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    ps = per_seed(rows)
    betas = sorted({r['beta'] for r in ps})
    print(f"{'beta':>5} {'n':>3} {'d':>3} {'rate/node':>11} {'95% CI':>23} "
          f"{'mean N':>9} {'Weibull k':>9} {'k 95% CI':>14}")
    groups, table = [], []
    for b in betas:
        sel = [r for r in ps if r['beta'] == b]
        t = np.array([r['t'] for r in sel])
        e = np.array([r['event'] for r in sel])
        er = exposure_rate(t, e)
        wk = weibull_shape([(t, e)])
        groups.append((t, e))
        table.append({'beta': b, **er, **wk})
        print(f"{b:5.2f} {er['n']:3d} {er['d']:3d} {er['rate']:11.3e} "
              f"[{er['lo']:.2e}, {er['hi']:.2e}] {1 / er['rate']:9.0f} "
              f"{wk['k']:9.2f} [{wk['k_lo']:.2f}, {wk['k_hi']:.2f}]")
    pooled = weibull_shape(groups)
    print(f"pooled common shape k = {pooled['k']:.2f} "
          f"[{pooled['k_lo']:.2f}, {pooled['k_hi']:.2f}]")

    # Nelson-Aalen linearity: cumulative hazard per node in the first and
    # second half of each beta's event range (equal slopes = constant hazard).
    print("\nNelson-Aalen slope ratio (late / early half; 1 = constant hazard):")
    for b, (t, e) in zip(betas, groups):
        tt, H = nelson_aalen(t, e)
        if len(tt) >= 6:
            mid = len(tt) // 2
            s1 = H[mid] / tt[mid]
            s2 = (H[-1] - H[mid]) / (tt[-1] - tt[mid])
            print(f"  beta {b:.2f}: {s2 / s1:.2f}")

    tab = np.array([[r['beta'], r['rate'], r['d']] for r in table if r['d'] > 0])
    arr = arrhenius(tab[:, 0], tab[:, 1], tab[:, 2])
    print(f"\nArrhenius: ln rate = {arr['intercept']:.2f} - {arr['a']:.2f} beta "
          f"(a se {arr['a_se']:.2f})")
    if 'quad' in arr:
        print(f"  quadratic term {arr['quad']:.2f} "
              f"[{arr['quad_lo']:.2f}, {arr['quad_hi']:.2f}]")

    # Frozen readings.
    covers1 = [r['k_lo'] <= 1 <= r['k_hi'] for r in table]
    if sum(covers1) >= 3 and pooled['k_lo'] <= 1 <= pooled['k_hi']:
        p1 = 'PER-NODE HAZARD'
    elif not (pooled['k_lo'] <= 1 <= pooled['k_hi']) and \
            pooled['k_lo'] <= 1.5 <= pooled['k_hi']:
        p1 = 'BOUNDARY-SITE HAZARD'
    else:
        p1 = 'UNDECIDED'
    p2 = 'n/a'
    if 'quad' in arr:
        p2 = 'ARRHENIUS' if arr['quad_lo'] <= 0 <= arr['quad_hi'] else 'NOT ARRHENIUS'
        p2 += (', SAME LAW' if abs(arr['a'] - NSTAR_B_SLOPE) <= 1.0
               else ', RATE LAW DIFFERS')
    firsts = [r for r in rows if int(r['event'])]
    by_seed: Dict[Tuple[float, int], List[Dict]] = {}
    for r in firsts:
        by_seed.setdefault((float(r['beta']), int(r['seed'])), []).append(r)
    lost = 0
    for recs in by_seed.values():
        ok = (len(recs) == 2 and all(r['mode'] == 'iv' for r in recs)
              and all(r['kind'] == 'attach' for r in recs)
              and recs[0]['partner_new'] == str(recs[1]['node'])
              and recs[1]['partner_new'] == str(recs[0]['node']))
        lost += ok
    frac = lost / max(len(by_seed), 1)
    p3 = 'LOST RACE' if frac >= 0.9 else 'see table'
    print(f"\nP1 clock: {p1}   (shape CI covers 1 at {sum(covers1)}/{len(table)} betas)")
    print(f"P2 law:   {p2}   (a = {arr['a']:.2f} vs N*_B law {NSTAR_B_SLOPE})")
    print(f"P3 mechanism: {p3}   (adjacent (iv)-only pair created by an attachment "
          f"in {lost}/{len(by_seed)} seeds = {frac:.2f})")

    # Creating-event table.
    combos: Dict[str, int] = {}
    for recs in by_seed.values():
        key = f"{len(recs)} node(s), modes {'/'.join(r['mode'] for r in recs)}, " \
              f"{recs[0]['kind']}"
        combos[key] = combos.get(key, 0) + 1
    print("\nfirst dead corner, by shape:")
    for k, v in sorted(combos.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    gaps: Dict[str, int] = {}
    for recs in by_seed.values():
        r = recs[0]
        if r['kind'] != 'attach':
            continue
        du, dv, _ = (int(x) for x in r['event_degs'].split())
        key = f"attach endpoints after (deg {min(du, dv)}, {max(du, dv)})"
        gaps[key] = gaps.get(key, 0) + 1
    print("creating attachment, endpoint degrees after the event:")
    for k, v in sorted(gaps.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    return {'table': table, 'pooled': pooled, 'arrhenius': arr,
            'P1': p1, 'P2': p2, 'P3': p3}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _validate() -> bool:
    from blocked_closure import blocked_nodes, first_blocked
    ok = True
    # 1. The hook does not change the growth.
    for beta, seed, n in ((1.0, 7, 20000), (2.0, 3, 20000)):
        a = grow_sheet(n, CAP, beta, seed)
        det = DeadDetector(CAP)
        b = grow_sheet(n, CAP, beta, seed, on_event=det)
        same = len(a) == len(b) and all(x == y for x, y in zip(a, b))
        print(f"[{'PASS' if same else 'FAIL'}] hook leaves growth identical "
              f"(beta {beta}, seed {seed}, N {n})")
        ok &= same
    # 2. Incremental dead set equals the static census of the final graph,
    #    and every recorded dead corner is still dead at the end.
    for seed in (11, 12, 13):
        det = DeadDetector(CAP)
        adj = grow_sheet(30000, CAP, 1.0, seed, on_event=det)
        static = set(blocked_nodes(adj, CAP))
        same = static == det.dead
        print(f"[{'PASS' if same else 'FAIL'}] incremental dead set == static "
              f"census (beta 1.0, seed {seed}: {len(det.dead)} dead)")
        ok &= same
        # 3. Prefix bisection can only fire at or before the exact history.
        if det.records:
            fb = first_blocked(to_csr(adj), n_lo=256)
            exact = det.records[0]['N']
            le = fb is not None and fb['n_first'] <= exact
            print(f"[{'PASS' if le else 'FAIL'}] prefix first_blocked "
                  f"{fb['n_first'] if fb else None} <= exact {exact} "
                  f"(gap {exact - fb['n_first'] if fb else 'n/a'} nodes)")
            ok &= le
    # 4. Failure modes on a hand case: x capped with a capped link end -> 'iv'.
    det = DeadDetector(CAP)
    adj = grow_sheet(30000, CAP, 1.0, 11, on_event=det)
    modes = {failure_modes(adj, x) for x in det.dead}
    sane = all(m and set(m.split('+')) <= {'i', 'ii', 'iii', 'iv'} for m in modes)
    print(f"[{'PASS' if sane else 'FAIL'}] every dead corner has a failure mode "
          f"({sorted(modes)})")
    ok &= sane
    # 5. Estimators on synthetic exponential data: k ~ 1, rate recovered.
    rng = np.random.default_rng(0)
    t = rng.exponential(1e4, 400)
    cens = 3e4
    e = (t < cens).astype(int)
    t = np.minimum(t, cens)
    er, wk = exposure_rate(t, e), weibull_shape([(t, e)])
    # Interval coverage over repeated small censored samples (n = 40, the
    # scan's size per beta) is the right check, not one draw.
    cov_r = cov_k = 0
    for _ in range(300):
        ts = rng.exponential(1.0, 40)
        es = (ts < 4).astype(int)
        ts = np.minimum(ts, 4)
        r_ = exposure_rate(ts, es)
        k_ = weibull_shape([(ts, es)])
        cov_r += r_['lo'] <= 1 <= r_['hi']
        cov_k += k_['k_lo'] <= 1 <= k_['k_hi']
    good = 0.91 <= cov_r / 300 <= 0.99 and 0.91 <= cov_k / 300 <= 0.99
    print(f"[{'PASS' if good else 'FAIL'}] synthetic exponential, n = 40: rate CI "
          f"coverage {cov_r / 300:.2f}, shape CI coverage {cov_k / 300:.2f} "
          f"(one n = 400 draw: rate x 1e4 = {er['rate'] * 1e4:.3f}, k = "
          f"{wk['k']:.2f} [{wk['k_lo']:.2f}, {wk['k_hi']:.2f}])")
    ok &= good
    t2 = rng.weibull(1.5, 400) * 1e4
    wk2 = weibull_shape([(t2, np.ones(400, int))])
    good2 = wk2['k_lo'] <= 1.5 <= wk2['k_hi'] and wk2['k_lo'] > 1
    print(f"[{'PASS' if good2 else 'FAIL'}] synthetic Weibull 1.5: k = "
          f"{wk2['k']:.2f} [{wk2['k_lo']:.2f}, {wk2['k_hi']:.2f}]")
    ok &= good2
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--scan', action='store_true')
    mode.add_argument('--analyze', nargs='+')
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0,
                    help='unused by the frozen scan (seeds are fixed); kept for '
                         'the repo convention')
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.scan:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        out = f"results/dead_corner_hazard_{stamp}.csv"
        scan(args.jobs, out)
        analyze([out])
    else:
        paths = [p for a in args.analyze for p in glob.glob(a)]
        analyze(paths)


if __name__ == '__main__':
    main()
