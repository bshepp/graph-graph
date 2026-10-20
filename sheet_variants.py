"""
Wave 5: labelled rule variants of `sheet` (relief, veto) against the default.

Pre-registration: docs/superpowers/specs/2026-10-17-wave5-preregistration.md.

  relief  a closure is admitted when a link end is at the cap (it rises to c + 1):
          each would-be dead pair becomes one interior degree-7 cone point
  veto    an attachment that would cap a node whose other boundary neighbour is
          already capped is refused: no defect forms

Both are default-off options of `sheet_growth.grow_sheet(variant=...)`; the default
generator is unchanged (checked by --validate). Neither is ever reported as `sheet`.

Usage:
    python sheet_variants.py --validate
    python sheet_variants.py --scan [--jobs 6]
    python sheet_variants.py --cap7
    python sheet_variants.py --scan6 [--jobs 24]     (wave 6)
    python sheet_variants.py --read6 CSV
    python sheet_variants.py --scan7 [--jobs 24]     (wave 7)
    python sheet_variants.py --read7 CSV
    python sheet_variants.py --scan8 [--jobs 24]     (wave 8)
    python sheet_variants.py --read8 CSV
    python sheet_variants.py --scan9 [--jobs 24]     (wave 9)
    python sheet_variants.py --read9 CSV
    python sheet_variants.py --scan10 [--jobs 16]    (wave 10)
    python sheet_variants.py --read10 CSV

Wave 6 (docs/superpowers/specs/2026-10-17-wave6-preregistration.md):
  strict  any event after which a node within one hop of it would be a dead
          corner is refused (applied, checked, reverted)
  relief  the hyperbolic-disc law: local slope of V(r) around node 0 against
          s(r) = x sinh x / (cosh x - 1), x = r sqrt(K), K = 2 pi lambda / (3 sqrt 3),
          lambda = interior degree-7 nodes per node, times the flat estimator's
          own slope / 2

Wave 8 (docs/superpowers/specs/2026-10-18-wave8-preregistration.md):
  relief  the same law with the graph ball read as a hexagon of equal area
          (x = HEX_AREA r sqrt(K)), flat slope from the exact triangular lattice

Wave 9 (docs/superpowers/specs/2026-10-19-wave9-preregistration.md), new variants of
`sheet_growth.grow_sheet` (see its docstring):
  disloc  relief, with every cone point paired with a degree-5 point (5-7 dislocations)
  pent    relief plus uncompensated degree-5 closures at a rate: the signed disc law
  eager   relief with the closure rate raised: the race law
  detach  sheet with reversible attachment (C4)
"""

import argparse
import csv
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from sheet_growth import grow_sheet, to_csr, _closable, _is_boundary, load_csr, _makes_dead
from dead_corner_hazard import DeadDetector

CAP = 6
SCAN_VARIANTS = ('sheet', 'relief', 'veto')
SCAN_BETAS = (1.0, 1.25)
SCAN_SEEDS = tuple(range(600, 608))
N_MAX = 2 ** 20
MARKS = [2 ** k for k in range(12, 21)]
COMPACT_B = 3.85
# 25 / 75 / 125 / 175 added in wave 8; read6 reads only the FLAT_SLOPE radii
RADII = (25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500)
# a graph ball on the triangular lattice is a hexagon of circumradius r; the disc of
# equal area has radius HEX_AREA * r (wave 8)
HEX_AREA = math.sqrt(3 * math.sqrt(3) / (2 * math.pi))
# wave 6: the flat estimator's slope (wave 5 veto growths with no dead corner,
# V(1.25 r) / V(r / 1.25) on a disc), fixed before any wave 6 growth
FLAT_SLOPE = {50: 1.94, 100: 1.99, 150: 1.99, 200: 1.99, 250: 2.00, 300: 2.00,
              400: 2.00, 500: 2.00}
# wave 9: small radii for the spherical side of the signed law
RADII9 = (10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500)
W9_SPECS = ([('detach', b, s, 2 ** 20, math.exp(b * mu)) for b in (1.0, 1.25) for mu in (3.0, 3.8)
             for s in range(1000, 1008)] +
            [('eager', b, s, 2 ** 20, rho) for b in (0.5, 0.75) for rho in (1.0, 4.0, 16.0, 64.0)
             for s in range(1000, 1004)] +
            [('disloc', b, s, 2 ** 20, 1.0) for b in (0.5, 0.75, 1.0) for s in range(1000, 1008)] +
            [('relief', 1.0, s, 2 ** 20, 1.0) for s in range(1000, 1004)] +
            [('pent', 1.0, s, 2 ** 20, p) for p in (3e-6, 6e-6, 1e-5)
             for s in range(1000, 1008)] +
            [('pent', 1.0, s, 2 ** 18, p) for p in (3e-5, 1e-4) for s in range(1000, 1004)])
W10_SPECS = [('disloc', b, s, 2 ** 22, 1.0) for b in (0.5, 0.75) for s in range(1100, 1108)]
W8_SPECS = [('relief', b, s, 2 ** 21) for b in (0.75, 0.875, 1.0) for s in range(900, 908)]
W7_SPECS = [('strict', b, s, 2 ** 20) for b in (1.0, 1.125) for s in range(800, 816)]
W6_SPECS = ([('relief', b, s, 2 ** 21) for b in (1.0, 1.25, 1.5) for s in range(700, 708)] +
            [('strict', b, s, 2 ** 20) for b in (0.5, 0.75, 1.0, 1.25) for s in range(700, 708)] +
            [('sheet', b, s, 2 ** 20) for b in (0.5, 0.75) for s in range(700, 708)])


def checkpoint_stats(adj, guard: int) -> Dict[str, float]:
    n = len(adj)
    b_edges = 0
    on_b = np.zeros(n, dtype=bool)
    for u in range(n):
        for v in adj[u]:
            if u < v and len(adj[u] & adj[v]) == 1:
                b_edges += 1
                on_b[u] = on_b[v] = True
    deg = np.fromiter((len(s) for s in adj), dtype=np.int64, count=n)
    interior = ~on_b
    capped_b = int((on_b & (deg >= CAP)).sum())
    dead = sum(1 for x in np.nonzero(on_b & (deg >= CAP))[0].tolist()
               if _closable(adj, x, CAP, guard) is None)
    A = to_csr(adj)
    d0 = dijkstra(A, indices=0, unweighted=True)
    return {'N': n, 'B': b_edges, 'B_sqrtN': b_edges / np.sqrt(n),
            'deg7_interior': int((interior & (deg == CAP + 1)).sum()),
            'deg5_interior': int((interior & (deg == CAP - 1)).sum()),
            'deg7_all': int((deg == CAP + 1).sum()), 'max_deg': int(deg.max()),
            'dead': dead, 'capped_b': capped_b, 'ecc0': float(d0[np.isfinite(d0)].max())}


def hyperbolic_slope(r: float, lam: float) -> float:
    """Local log-log slope of a hyperbolic disc's area with curvature from a
    density lam of degree-7 cone points per node (angle excess pi/3 each,
    area sqrt(3)/2 per node); 2 in the flat limit."""
    x = r * math.sqrt(2 * math.pi * lam / (3 * math.sqrt(3)))
    if x < 1e-6:
        return 2.0
    return x * math.sinh(x) / (math.cosh(x) - 1)


def lattice_slope(r: float) -> float:
    """The ball-slope estimator on the exact triangular lattice, V = 3r^2 + 3r + 1."""
    def v(q: int) -> int:
        return 3 * q * q + 3 * q + 1
    return math.log(v(int(r * 1.25)) / v(int(r / 1.25))) / math.log(1.5625)


def hyperbolic_slope_hex(r: float, lam: float) -> float:
    """Wave 8: `hyperbolic_slope` with the graph radius read as HEX_AREA * r."""
    return hyperbolic_slope(HEX_AREA * r, lam)


def frozen_set(adj) -> set:
    """`strict`: capped boundary nodes whose closure is allowed by `_closable`
    but refused by the strict veto (closing would leave a dead corner)."""
    out = set()
    for x in range(len(adj)):
        if len(adj[x]) < CAP or not any(_is_boundary(adj, x, z) for z in adj[x]):
            continue
        ab = _closable(adj, x, CAP)
        if ab is None:
            continue
        a, b = ab
        adj[a].add(b)
        adj[b].add(a)
        bad = _makes_dead(adj, (x, a, b), CAP)
        adj[a].discard(b)
        adj[b].discard(a)
        if bad:
            out.add(x)
    return out


def grow_one(spec) -> Dict:
    variant, beta, seed = spec[:3]
    n_max = spec[3] if len(spec) > 3 else N_MAX
    marks = [2 ** k for k in range(12, 30) if 2 ** k <= n_max]
    random.seed(seed)
    np.random.seed(seed)
    guard = CAP + 1 if variant == 'relief' else CAP
    det = DeadDetector(CAP, guard=guard)
    first7 = {'N': -1}

    def on_event(kind, nodes, adj):
        det(kind, nodes, adj)
        if first7['N'] < 0 and any(len(adj[x]) > CAP for x in nodes):
            first7['N'] = len(adj)
        return False

    rows: List[Dict] = []

    prev_frozen: Dict[str, set] = {'s': set()}
    persist = [0, 0]

    def on_ck(n, adj):
        st = checkpoint_stats(adj, guard)
        if variant == 'strict':
            fz = frozen_set(adj)
            persist[0] += len(prev_frozen['s'])
            persist[1] += len(prev_frozen['s'] & fz)
            prev_frozen['s'] = fz
            st['frozen'] = len(fz)
        rows.append(st)

    t0 = time.time()
    adj = grow_sheet(n_max, CAP, beta, seed, checkpoints=marks, on_checkpoint=on_ck,
                     on_event=on_event, variant=variant)
    A = to_csr(adj)
    d0 = dijkstra(A, indices=0, unweighted=True)
    h = np.bincount(d0[np.isfinite(d0)].astype(int))
    V = np.cumsum(h)
    slopes, fill = {}, {}
    for r in RADII:
        if int(r * 1.25) < len(V):
            slopes[r] = float(np.log(V[int(r * 1.25)] / V[int(r / 1.25)]) / np.log(1.5625))
            fill[r] = float(V[int(r * 1.25)] / len(adj))
    from curvature_census import disc_audit
    au = disc_audit(A, seed)
    last, prev = rows[-1], rows[-2]
    ecc_exp = float(np.log(last['ecc0'] / prev['ecc0']) / np.log(last['N'] / prev['N']))
    return {'variant': variant, 'beta': beta, 'seed': seed, 'n_end': len(adj),
            'stalled': int(len(adj) < n_max),
            'first_dead': det.records[0]['N'] if det.records else -1,
            'first_dead_mode': det.records[0]['mode'] if det.records else '',
            'first_deg7': first7['N'], 'ecc_exp_last': ecc_exp,
            'B_sqrtN_end': last['B_sqrtN'], 'dead_end': last['dead'],
            'deg7_end': last['deg7_interior'], 'deg7_prev': prev['deg7_interior'],
            'max_deg': last['max_deg'],
            'slopes0': ' '.join(f"{r}:{x:.2f}" for r, x in slopes.items()),
            'slopes_fill': ' '.join(f"{r}:{x:.4f}:{fill[r]:.4f}" for r, x in slopes.items()),
            'capped_b_end': last['capped_b'],
            'frozen_profile': ' '.join(f"{r['N']}:{r.get('frozen', -1)}" for r in rows),
            'frozen_persist': f"{persist[1]}/{persist[0]}",
            'slope_max': max(slopes.values()) if slopes else float('nan'),
            'audit': au['verdict'], 'audit_d': au['d_last'],
            'profile': ' '.join(f"{r['N']}:{r['B_sqrtN']:.2f}/{r['dead']}/{r['deg7_interior']}"
                                f"/{r['ecc0']:.0f}" for r in rows),
            'seconds': round(time.time() - t0)}


def read(rows: List[Dict]) -> None:
    for v in SCAN_VARIANTS:
        rs = [r for r in rows if r['variant'] == v]
        if not rs:
            continue
        near = sum(1 for r in rs if abs(r['B_sqrtN_end'] / COMPACT_B - 1) <= 0.10)
        ecc_ok = sum(1 for r in rs if 0.40 <= r['ecc_exp_last'] <= 0.60)
        cross = sum(1 for r in rs if r['B_sqrtN_end'] > 1.2 * COMPACT_B or r['ecc_exp_last'] < 0.35)
        nodead = sum(1 for r in rs if r['dead_end'] == 0)
        if near >= 14 and ecc_ok >= 14:
            verdict = 'NO CROSSOVER'
        elif cross >= 12:
            verdict = 'CROSSOVER'
        else:
            verdict = 'UNDECIDED'
        line = (f"{v:>6}: {verdict}  (B/sqrtN within 10% of 3.85 in {near}/{len(rs)}, ecc exponent "
                f"in [0.40, 0.60] in {ecc_ok}/{len(rs)}, crossed by the CROSSOVER clause in "
                f"{cross}/{len(rs)}; zero dead corners at 2^20 in {nodead}/{len(rs)})")
        if v == 'relief':
            crossed = [r for r in rs if r['B_sqrtN_end'] > 1.2 * COMPACT_B or r['ecc_exp_last'] < 0.35]
            hyp = [r for r in crossed if r['slope_max'] > 2.5 and r['dead_end'] == 0 and
                   r['deg7_prev'] > 0 and 1.6 <= r['deg7_end'] / r['deg7_prev'] <= 2.4]
            line += (f"; HYPERBOLIC INTERIOR in {len(hyp)}/{len(crossed)} crossed growths")
        print(line)
        for r in rs:
            print(f"    beta {r['beta']} seed {r['seed']}: B/sqrtN {r['B_sqrtN_end']:.2f}, ecc exp "
                  f"{r['ecc_exp_last']:.2f}, dead {r['dead_end']} (first {r['first_dead']} "
                  f"{r['first_dead_mode']}), deg-7 interior {r['deg7_end']} (first at "
                  f"{r['first_deg7']}), V(r) slopes {r['slopes0']}, audit {r['audit']} "
                  f"{r['audit_d']:.2f}")


def scan(jobs: int) -> None:
    specs = [(v, b, s) for b in SCAN_BETAS for v in SCAN_VARIANTS for s in SCAN_SEEDS]
    out = f"results/sheet_variants_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[Dict] = []
    with ProcessPoolExecutor(jobs) as ex, open(out, 'w', newline='') as f:
        w = None
        for r in ex.map(grow_one, specs):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                w.writeheader()
            w.writerow(r)
            f.flush()
            rows.append(r)
            print(f"{r['variant']:>6} beta {r['beta']} seed {r['seed']}: B/sqrtN "
                  f"{r['B_sqrtN_end']:.2f} ecc exp {r['ecc_exp_last']:.2f} dead {r['dead_end']} "
                  f"deg7 {r['deg7_end']} [{r['seconds']} s]", flush=True)
    print(f"saved: {out}")
    read(rows)


def read6(rows: List[Dict]) -> None:
    """Wave 6 frozen readings."""
    rel = [r for r in rows if r['variant'] == 'relief']
    if rel:
        pairs, last_ok, n_g = [], 0, 0
        print("W6-a relief, hyperbolic-disc law (measured / predicted local slope of V(r)):")
        for r in rel:
            lam = r['deg7_end'] / r['n_end']
            el = []
            for t in r['slopes_fill'].split():
                rr, sl, fr = t.split(':')
                rr, sl, fr = int(rr), float(sl), float(fr)
                if fr <= 0.5 and rr in FLAT_SLOPE:
                    el.append((rr, sl, hyperbolic_slope(rr, lam) * FLAT_SLOPE[rr] / 2))
            ok = [abs(sl / pr - 1) <= 0.10 for _, sl, pr in el]
            pairs += ok
            if el:
                n_g += 1
                last_ok += int(ok[-1])
            print(f"    beta {r['beta']} seed {r['seed']}: lambda {lam:.2e}; " +
                  ' '.join(f"{rr}:{sl:.2f}/{pr:.2f}" for rr, sl, pr in el))
        frac = sum(pairs) / max(len(pairs), 1)
        v = ('HYPERBOLIC DISC LAW' if frac >= 0.90 and last_ok >= math.ceil(20 / 24 * n_g)
             else 'LAW MISSED')
        print(f"  W6-a: {v}  ({sum(pairs)}/{len(pairs)} eligible pairs within 10% = {frac:.2f}; "
              f"largest eligible r within 10% in {last_ok}/{n_g} growths)")
        for b in sorted({r['beta'] for r in rel}):
            rb = [r for r in rel if r['beta'] == b]
            print(f"    beta {b}: B/sqrtN {min(x['B_sqrtN_end'] for x in rb):.2f}-"
                  f"{max(x['B_sqrtN_end'] for x in rb):.2f}, dead {sum(x['dead_end'] for x in rb)}, "
                  f"ecc exp {min(x['ecc_exp_last'] for x in rb):.2f}-"
                  f"{max(x['ecc_exp_last'] for x in rb):.2f}")
    for v in ('strict', 'sheet'):
        rs = [r for r in rows if r['variant'] == v]
        if not rs:
            continue
        print(f"W6-b {v}:")
        if v == 'strict':
            dead = sum(1 for r in rs if r['first_dead'] >= 0 or r['dead_end'] > 0)
            print(f"  gate (no dead corner ever, in every growth): {'PASS' if dead == 0 else 'FAIL'} "
                  f"({dead} growths with a dead corner)")
        for b in sorted({r['beta'] for r in rs}):
            rb = [r for r in rs if r['beta'] == b]
            flat = cross = stall = 0
            for r in rb:
                prof = {int(t.split(':')[0]): float(t.split(':')[1].split('/')[0])
                        for t in r['profile'].split()}
                ratio = r['B_sqrtN_end'] / prof.get(2 ** 16, float('nan'))
                r['_ratio'] = ratio
                stall += r['stalled']
                if 0.85 <= ratio <= 1.15 and 0.40 <= r['ecc_exp_last'] <= 0.60:
                    flat += 1
                if ratio > 1.3 or r['ecc_exp_last'] < 0.35:
                    cross += 1
            if stall >= 4:
                verdict = 'STALLED'
            elif flat >= 7:
                verdict = 'FLAT'
            elif cross >= 6:
                verdict = 'CROSSOVER'
            else:
                verdict = 'UNDECIDED'
            print(f"  beta {b}: {verdict}  (flat {flat}/8, crossed {cross}/8, stalled {stall}/8)")
            for r in rb:
                print(f"      seed {r['seed']}: N {r['n_end']}, B/sqrtN {r['B_sqrtN_end']:.2f} "
                      f"(x{r['_ratio']:.2f} since 2^16), ecc exp {r['ecc_exp_last']:.2f}, dead "
                      f"{r['dead_end']} (first {r['first_dead']}), capped boundary "
                      f"{r['capped_b_end']}, V(r) slopes {r['slopes0']}, audit {r['audit']} "
                      f"{r['audit_d']:.2f}")


def scan6(jobs: int) -> None:
    out = f"results/sheet_variants_w6_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[Dict] = []
    with ProcessPoolExecutor(jobs) as ex, open(out, 'w', newline='') as f:
        w = None
        for r in ex.map(grow_one, W6_SPECS):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                w.writeheader()
            w.writerow(r)
            f.flush()
            rows.append(r)
            print(f"{r['variant']:>6} beta {r['beta']} seed {r['seed']}: N {r['n_end']} B/sqrtN "
                  f"{r['B_sqrtN_end']:.2f} ecc exp {r['ecc_exp_last']:.2f} dead {r['dead_end']} "
                  f"deg7 {r['deg7_end']} [{r['seconds']} s]", flush=True)
    print(f"saved: {out}")
    read6(rows)


def read8(rows: List[Dict]) -> None:
    """Wave 8 frozen readings: relief against the hex-corrected hyperbolic-disc law."""
    ok_g = [r for r in rows if not r['stalled'] and r['dead_end'] == 0 and r['max_deg'] <= CAP + 1]
    for r in rows:
        if r not in ok_g:
            print(f"    GATE FAIL beta {r['beta']} seed {r['seed']}: stalled {r['stalled']} "
                  f"dead {r['dead_end']} max_deg {r['max_deg']}")
    print(f"  gate: {len(ok_g)}/{len(rows)} growths pass")
    allp, steep_h, steep_r = [], [], []
    per_beta: Dict[float, List[float]] = {}
    for r in ok_g:
        lam = r['deg7_end'] / r['n_end']
        sk = math.sqrt(2 * math.pi * lam / (3 * math.sqrt(3)))
        out = []
        for t in r['slopes_fill'].split():
            rr, sl, fr = t.split(':')
            rr, sl, fr = int(rr), float(sl), float(fr)
            if fr > 0.5:
                continue
            fs = lattice_slope(rr) / 2
            h = sl / (hyperbolic_slope_hex(rr, lam) * fs)
            g = sl / (hyperbolic_slope(rr, lam) * fs)
            xh = HEX_AREA * rr * sk
            allp.append(abs(h - 1) <= 0.10)
            per_beta.setdefault(r['beta'], []).append(h)
            if xh >= 4:
                steep_h.append(h)
                steep_r.append(g)
            out.append(f"{rr}:x{xh:.2f}:{h:.3f}/{g:.3f}")
        print(f"    beta {r['beta']} seed {r['seed']}: lambda {lam:.2e} (radius:x:hex/registered) "
              + ' '.join(out))
    frac = sum(allp) / max(len(allp), 1)
    print(f"  pairs within 10% (hex law): {sum(allp)}/{len(allp)} = {frac:.3f}")
    means_ok = True
    for b, v in sorted(per_beta.items()):
        mu = sum(v) / len(v)
        means_ok &= 0.97 <= mu <= 1.03
        print(f"  beta {b}: mean hex ratio {mu:.3f} over {len(v)} pairs")
    mh = sum(steep_h) / len(steep_h) if steep_h else float('nan')
    mr = sum(steep_r) / len(steep_r) if steep_r else float('nan')
    print(f"  steep part (x_hex >= 4): {len(steep_h)} pairs, mean hex ratio {mh:.3f}, "
          f"registered-law ratio {mr:.3f}")
    base = frac >= 0.90 and means_ok and len(per_beta) > 0
    if base and len(steep_h) >= 10 and 0.95 <= mh <= 1.05:
        v = 'HEX LAW HOLDS'
    elif base and len(steep_h) < 10:
        v = 'HEX LAW HOLDS (steep part untested)'
    else:
        v = 'HEX LAW MISSED'
    print(f"  W8-a: {v}")


def scan8(jobs: int) -> None:
    out = f"results/sheet_variants_w8_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[Dict] = []
    with ProcessPoolExecutor(jobs) as ex, open(out, 'w', newline='') as f:
        w = None
        for r in ex.map(grow_one, W8_SPECS):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                w.writeheader()
            w.writerow(r)
            f.flush()
            rows.append(r)
            print(f"relief beta {r['beta']} seed {r['seed']}: N {r['n_end']} dead {r['dead_end']} "
                  f"max_deg {r['max_deg']} deg7 {r['deg7_end']} [{r['seconds']} s]", flush=True)
    print(f"saved: {out}")
    read8(rows)


def signed_slope_hex(r: float, lam_net: float) -> float:
    """Wave 9: the hex-corrected disc law with signed curvature. lam_net is
    (interior degree-7 minus degree-5 nodes) per node; negative is a spherical
    cap, A ~ 1 - cos x, slope x sin x / (1 - cos x)."""
    if lam_net >= 0:
        return hyperbolic_slope_hex(r, lam_net)
    x = HEX_AREA * r * math.sqrt(2 * math.pi * -lam_net / (3 * math.sqrt(3)))
    if x < 1e-6:
        return 2.0
    return x * math.sin(x) / (1 - math.cos(x))


def dead_set(adj, guard: int) -> set:
    """Capped boundary nodes that cannot close (ids as grown, detached nodes empty)."""
    return {x for x in range(len(adj)) if len(adj[x]) >= CAP
            and any(_is_boundary(adj, x, z) for z in adj[x])
            and _closable(adj, x, CAP, guard) is None}


def flat_or_crossed(r: Dict) -> str:
    """The wave 6 W6-b per-growth reading (B/sqrtN over the last four octaves and
    the centre's eccentricity exponent over the last one)."""
    prof = {int(t.split(':')[0]): float(t.split(':')[1].split('/')[0])
            for t in r['profile'].split()}
    if 2 ** 16 not in prof:
        return 'short'
    ratio = float(r['B_sqrtN_end']) / prof[2 ** 16]
    ecc = float(r['ecc_exp_last'])
    if 0.85 <= ratio <= 1.15 and 0.40 <= ecc <= 0.60:
        return 'flat'
    if ratio > 1.3 or ecc < 0.35:
        return 'crossed'
    return 'neither'


def grow_one9(spec) -> Dict:
    """One wave 9 growth: (variant, beta, seed, n_max, rate[, stop_net]). With stop_net the
    growth stops once interior degree-5 minus degree-7 points reach it (a spherical cap
    still covering itself once; interior degrees never change after a closure)."""
    from sheet_growth import compact
    variant, beta, seed, n_max, rate = spec[:5]
    stop_net = spec[5] if len(spec) > 5 else None
    net = [0]
    marks = [2 ** k for k in range(12, 30) if 2 ** k <= n_max]
    random.seed(seed)
    np.random.seed(seed)
    guard = CAP if variant in ('sheet', 'detach') else CAP + 1
    det = DeadDetector(CAP, guard=guard)

    def on_event(kind, nodes, adj):
        if kind == 'detach':
            det.interior.difference_update(nodes)   # a detachment can re-expose them
            nodes = nodes[1:]
        det(kind, nodes, adj)
        if kind == 'close' and stop_net is not None:
            net[0] += {CAP - 1: 1, CAP + 1: -1}.get(len(adj[nodes[0]]), 0)
            return net[0] >= stop_net
        return False

    rows: List[Dict] = []
    prev_dead = {'s': set()}
    heal = [0, 0]

    def on_ck(n, adj):
        ds = dead_set(adj, guard)
        heal[0] += len(prev_dead['s'])
        heal[1] += len(prev_dead['s'] - ds)
        prev_dead['s'] = ds
        rows.append(checkpoint_stats(compact(adj), guard))

    t0 = time.time()
    raw = grow_sheet(n_max, CAP, beta, seed, checkpoints=marks, on_checkpoint=on_ck,
                     on_event=on_event, variant=variant, rate=rate)
    adj = compact(raw)
    n = len(adj)
    on_b = np.zeros(n, dtype=bool)
    b_edges = 0
    for u in range(n):
        for v in adj[u]:
            if u < v and len(adj[u] & adj[v]) == 1:
                on_b[u] = on_b[v] = True
                b_edges += 1
    deg = np.fromiter((len(s) for s in adj), dtype=np.int64, count=n)
    i5 = np.nonzero(~on_b & (deg == CAP - 1))[0]
    i7 = set(np.nonzero(~on_b & (deg == CAP + 1))[0].tolist())
    other = int((~on_b & ((deg < CAP - 1) | (deg > CAP + 1))).sum())
    paired = sum(1 for f in i5.tolist()
                 if any(y in i7 or any(z in i7 for z in adj[y]) for y in adj[f]))
    A = to_csr(adj)
    d0 = dijkstra(A, indices=0, unweighted=True)
    V = np.cumsum(np.bincount(d0[np.isfinite(d0)].astype(int)))
    sf = []
    for r in RADII9:
        if int(r * 1.25) < len(V):
            sl = float(np.log(V[int(r * 1.25)] / V[int(r / 1.25)]) / np.log(1.5625))
            sf.append(f"{r}:{sl:.4f}:{V[int(r * 1.25)] / n:.4f}")
    bd = d0[on_b & np.isfinite(d0)]
    r_complete = int(bd.min()) if len(bd) else int(d0[np.isfinite(d0)].max()) + 1
    from curvature_census import disc_audit
    au = disc_audit(A, seed) if n >= 2000 else {'verdict': 'n/a', 'd_last': float('nan')}
    last = rows[-1] if rows else {'B_sqrtN': b_edges / math.sqrt(n), 'dead': -1, 'ecc0': 0}
    if len(rows) >= 2:
        prev = rows[-2]
        ecc_exp = float(np.log(last['ecc0'] / prev['ecc0']) / np.log(last['N'] / prev['N']))
    else:
        ecc_exp = float('nan')
    stopped = int(stop_net is not None and net[0] >= stop_net)
    return {'variant': variant, 'beta': beta, 'seed': seed, 'rate': rate,
            'stop_net': stop_net if stop_net is not None else -1, 'stopped': stopped, 'n_end': n,
            'ids': len(raw), 'stalled': int(n < n_max and not stopped), 'B_end': b_edges,
            'B_sqrtN_end': b_edges / math.sqrt(n), 'ecc_exp_last': ecc_exp,
            'i5': len(i5), 'i7': len(i7), 'int_other': other, 'paired5': paired,
            'max_deg': int(deg.max()), 'min_int_deg': int(deg[~on_b].min()) if (~on_b).any() else -1,
            'dead_end': len(dead_set(adj, guard)),
            'first_dead': det.records[0]['N'] if det.records else -1,
            'healed': f"{heal[1]}/{heal[0]}",
            'slopes_fill': ' '.join(sf), 'r_complete': r_complete,
            'audit': au['verdict'], 'audit_d': au['d_last'],
            'profile': ' '.join(f"{r['N']}:{r['B_sqrtN']:.2f}/{r['dead']}/{r['deg7_interior']}"
                                f"/{r['ecc0']:.0f}" for r in rows),
            'profile5': ' '.join(f"{r['N']}:{r['deg5_interior']}/{r['deg7_interior']}/{r['B']}"
                                 for r in rows),
            'seconds': round(time.time() - t0)}


def _cell_verdict(rs: List[Dict]) -> str:
    k = [flat_or_crossed(r) for r in rs]
    m = len(rs)
    if sum(r['stopped'] for r in rs) >= m / 2:
        return 'CAP STOP'
    if sum(r['stalled'] for r in rs) >= m / 2:
        return 'STALLED'
    if k.count('flat') >= math.ceil(7 / 8 * m):
        return 'FLAT'
    if k.count('crossed') >= math.ceil(6 / 8 * m):
        return 'CROSSOVER'
    return 'UNDECIDED'


def _law_pairs(r: Dict, gross: bool = False) -> List[tuple]:
    """(radius, measured / predicted, lambda) on eligible radii: the ball V(1.25 r) holds at
    most half the graph and no boundary node (a complete geodesic ball). gross: the
    prediction from the degree-7 points alone, ignoring the degree-5 ones."""
    lam = (r['i7'] - (0 if gross else r['i5'])) / r['n_end']
    out = []
    for t in r['slopes_fill'].split():
        rr, sl, fr = t.split(':')
        rr, sl, fr = int(rr), float(sl), float(fr)
        if fr > 0.5 or int(rr * 1.25) >= r['r_complete']:
            continue
        out.append((rr, sl / (signed_slope_hex(rr, lam) * lattice_slope(rr) / 2), lam))
    return out


def read9(rows: List[Dict]) -> None:
    """Wave 9 frozen readings (docs/superpowers/specs/2026-10-19-wave9-preregistration.md)."""
    def cells(v):
        out: Dict[tuple, List[Dict]] = {}
        for r in rows:
            if r['variant'] == v:
                out.setdefault((r['beta'], r['rate'], r['stop_net']), []).append(r)
        return dict(sorted(out.items()))

    def show(r, extra=''):
        print(f"      seed {r['seed']}: N {r['n_end']}, B/sqrtN {r['B_sqrtN_end']:.2f}, ecc exp "
              f"{r['ecc_exp_last']:.2f} -> {flat_or_crossed(r)}; i5 {r['i5']} i7 {r['i7']}, dead "
              f"{r['dead_end']} (first {r['first_dead']}){extra}")

    # W9-a disloc and its relief controls
    dl = cells('disloc')
    if dl:
        print("W9-a disloc (every cone point paired with a degree-5 point):")
        gate_fail, law = 0, []
        for (b, _, _), rs in dl.items():
            for r in rs:
                ok = (not r['stalled'] and r['dead_end'] == 0 and r['max_deg'] <= CAP + 1
                      and r['int_other'] == 0 and r['paired5'] >= 0.95 * r['i5'])
                gate_fail += int(not ok)
                law += _law_pairs(r)
            print(f"  beta {b}: {_cell_verdict(rs)}")
            for r in rs:
                show(r, f", 5s with a 7 within two hops {r['paired5']}/{r['i5']}")
        within = sum(abs(h - 1) <= 0.10 for _, h, _ in law)
        print(f"  gate: {'PASS' if gate_fail == 0 else f'FAIL ({gate_fail} growths)'}")
        print(f"  net-curvature law: {within}/{len(law)} pairs within 10% "
              f"({'HOLDS' if law and within >= 0.9 * len(law) else 'MISSED'}); mean ratio "
              f"{sum(h for _, h, _ in law) / max(len(law), 1):.3f}")
        ctl = [r for r in rows if r['variant'] == 'relief' or
               (r['variant'] == 'eager' and r['rate'] == 1.0)]
        if ctl:
            print("  relief controls (relief, and eager at rate 1, which is relief):")
            for b in sorted({r['beta'] for r in ctl}):
                rb = [r for r in ctl if r['beta'] == b]
                k = [flat_or_crossed(r) for r in rb]
                print(f"    beta {b}: crossed {k.count('crossed')}/{len(rb)}; B/sqrtN "
                      f"{min(r['B_sqrtN_end'] for r in rb):.2f}-{max(r['B_sqrtN_end'] for r in rb):.2f}")
    # W9-b pent: the net-curvature law against the gross (degree-7-only) law
    pc = cells('pent')
    if pc:
        print("W9-b pent (uncompensated degree-5 closures): net against gross curvature")
        law, disc_net, disc_gross, sph = [], [], [], []
        for (b, rate, stop), rs in pc.items():
            reg = rate <= 1e-5                  # 3e-5 and 1e-4 are descriptive (spherical side)
            print(f"  beta {b}, rate {rate:g}: {_cell_verdict(rs)}"
                  f"{'' if reg else ' (descriptive)'}")
            for r in rs:
                p = _law_pairs(r)
                g = _law_pairs(r, gross=True)
                if reg:
                    law += p
                    for (rr, h, _), (_, hg, _) in zip(p, g):
                        if abs(hg / h - 1) > 0.10:       # the two predictions differ by > 10%
                            disc_net.append(h)
                            disc_gross.append(hg)
                else:
                    sph += [h for _, h, lam in p if lam < 0]
                closed = ' CLOSED' if r['B_end'] == 0 else ''
                show(r, f", lambda_net {(r['i7'] - r['i5']) / r['n_end']:+.2e}, B {r['B_end']}"
                        f"{closed}, complete ball to r {r['r_complete']}; net/gross ratios " +
                        ' '.join(f"{rr}:{h:.3f}/{hg:.3f}" for (rr, h, _), (_, hg, _) in zip(p, g)))
        within = sum(abs(h - 1) <= 0.10 for _, h, _ in law)
        mn = sum(disc_net) / len(disc_net) if disc_net else float('nan')
        mg = sum(disc_gross) / len(disc_gross) if disc_gross else float('nan')
        print(f"  net law: {within}/{len(law)} pairs within 10%; discriminating pairs "
              f"{len(disc_net)}: net mean {mn:.3f}, gross mean {mg:.3f}")
        if len(disc_net) < 10:
            v = 'UNDERPOWERED'
        elif within >= 0.9 * len(law) and 0.95 <= mn <= 1.05 and mg < 0.90:
            v = 'NET LAW HOLDS'
        else:
            v = 'NET LAW MISSED'
        print(f"  W9-b: {v}")
        if sph:
            print(f"  spherical side (descriptive): {len(sph)} complete-ball pairs, mean ratio "
                  f"{sum(sph) / len(sph):.3f}")
    # W9-c eager: the race law
    ec = cells('eager')
    if ec:
        print("W9-c eager (closure rate rho): cone density against rho")
        ok_all = True
        for b in sorted({k[0] for k in ec}):
            xs, ys = [], []
            for (bb, rate, _), rs in ec.items():
                if bb != b:
                    continue
                lam = [r['i7'] / r['n_end'] for r in rs]
                print(f"  beta {b} rho {rate:g}: lambda {np.mean(lam):.3e} (cones "
                      f"{' '.join(str(r['i7']) for r in rs)}), " +
                      ', '.join(flat_or_crossed(r) for r in rs))
                for r in rs:
                    xs.append(math.log(rate))
                    ys.append(math.log(max(r['i7'], 0.5) / r['n_end']))
            sl = float(np.polyfit(xs, ys, 1)[0])
            ok = -1.25 <= sl <= -0.75
            ok_all &= ok
            print(f"  beta {b}: d ln lambda / d ln rho = {sl:.3f} ({'in' if ok else 'outside'} "
                  f"[-1.25, -0.75])")
        print(f"  W9-c: {'RACE LAW' if ok_all else 'RACE LAW MISSED'}")
    # W9-d detach (C4)
    dc = cells('detach')
    if dc:
        print("W9-d detach (C4, reversible attachment on sheet):")
        for (b, rate, _), rs in dc.items():
            mu = math.log(rate) / b
            hp = [tuple(map(int, r['healed'].split('/'))) for r in rs]
            print(f"  beta {b}, mu {mu:.1f}: {_cell_verdict(rs)}; detached "
                  f"{sum(r['ids'] - r['n_end'] for r in rs) / sum(r['ids'] for r in rs):.3f} of "
                  f"all nodes ever grown; dead corners healed between checkpoints "
                  f"{sum(h for h, _ in hp)}/{sum(t for _, t in hp)}")
            for r in rs:
                show(r, f", healed {r['healed']}")


def scan9(jobs: int) -> None:
    out = f"results/sheet_variants_w9_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[Dict] = []
    with ProcessPoolExecutor(jobs) as ex, open(out, 'w', newline='') as f:
        w = None
        for r in ex.map(grow_one9, W9_SPECS):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                w.writeheader()
            w.writerow(r)
            f.flush()
            rows.append(r)
            print(f"{r['variant']} beta {r['beta']} rate {r['rate']:g} seed {r['seed']}: N "
                  f"{r['n_end']} B/sqrtN {r['B_sqrtN_end']:.2f} i5 {r['i5']} i7 {r['i7']} dead "
                  f"{r['dead_end']} [{r['seconds']} s]", flush=True)
    print(f"saved: {out}")
    read9(rows)


def read10(rows: List[Dict]) -> None:
    """Wave 10 frozen readings (docs/superpowers/specs/2026-10-19-wave10-preregistration.md):
    `disloc` at beta 0.5 and 0.75 to 2^22. Per growth, the crossing reading over the last four
    octaves (base N_end / 16), the octave factors of B/sqrtN, and the remainder n7 - n5 against
    the boundary; per cell, the feedback verdict, the remainder per boundary edge at N_end
    against N_end / 4, and the net disc law."""
    cells: Dict[float, List[Dict]] = {}
    for r in rows:
        cells.setdefault(r['beta'], []).append(r)
    for b, rs in sorted(cells.items()):
        print(f"W10 disloc beta {b}:")
        kinds, acc, sett, law = [], 0, 0, []
        q = {}   # checkpoint N -> [sum of n7 - n5, sum of boundary edges] over the cell
        for r in rs:
            prof = {int(t.split(':')[0]): float(t.split(':')[1].split('/')[0])
                    for t in r['profile'].split()}
            n_end = r['n_end']
            base = n_end // 16
            if r['stalled'] or base not in prof:
                kinds.append('short')
                print(f"    seed {r['seed']}: short / stalled (N {n_end})")
                continue
            ratio = r['B_sqrtN_end'] / prof[base]
            ecc = r['ecc_exp_last']
            kind = ('flat' if 0.85 <= ratio <= 1.15 and 0.40 <= ecc <= 0.60 else
                    'crossed' if ratio > 1.3 or ecc < 0.35 else 'neither')
            kinds.append(kind)
            octs = [base * 2 ** k for k in range(5)]
            g = [prof[octs[k + 1]] / prof[octs[k]] for k in range(4)]
            if g[-1] > g[0] + 0.01:
                acc += 1
            elif g[-1] < g[0] - 0.01:
                sett += 1
            rem = []
            for t in r['profile5'].split():
                nn, rest = t.split(':')
                d5, d7, bb = (int(x) for x in rest.split('/'))
                if int(nn) >= 2 ** 16:
                    rem.append((int(nn), d7 - d5, bb))
                    q.setdefault(int(nn), [0, 0])
                    q[int(nn)][0] += d7 - d5
                    q[int(nn)][1] += bb
            pairs = _law_pairs(r)
            law += pairs
            print(f"    seed {r['seed']}: B/sqrtN {r['B_sqrtN_end']:.2f} (x{ratio:.3f} over 4 "
                  f"octaves), ecc exp {ecc:.2f} -> {kind}; octave factors "
                  f"{' '.join(f'{x:.3f}' for x in g)}; remainder/boundary "
                  f"{' '.join(f'{d}/{bb}' for _, d, bb in rem)}; law "
                  f"{sum(abs(h - 1) <= 0.10 for _, h, _ in pairs)}/{len(pairs)}; dead {r['dead_end']}")
        m = len(rs)
        if kinds.count('short') >= m / 2:
            cv = 'STALLED'
        elif kinds.count('flat') >= math.ceil(7 / 8 * m):
            cv = 'FLAT'
        elif kinds.count('crossed') >= math.ceil(6 / 8 * m):
            cv = 'CROSSOVER'
        else:
            cv = 'UNDECIDED'
        fb = ('ACCELERATING' if acc >= math.ceil(6 / 8 * m) else
              'SETTLING' if sett >= math.ceil(6 / 8 * m) else 'STEADY')
        n_hi = max(r['n_end'] for r in rs)
        n_lo = n_hi // 4
        if n_hi in q and n_lo in q and q[n_lo][0] > 0:
            ratio_q = (q[n_hi][0] / q[n_hi][1]) / (q[n_lo][0] / q[n_lo][1])
            rv = ('SHRINKING' if ratio_q < 0.7 else 'BOUNDARY-PROPORTIONAL' if ratio_q <= 1.4 else
                  'BETWEEN' if ratio_q < 1.6 else 'AREA-LIKE')
            rtxt = (f"q = sum(n7 - n5) / sum(B): {q[n_lo][0]}/{q[n_lo][1]} at {n_lo}, "
                    f"{q[n_hi][0]}/{q[n_hi][1]} at {n_hi}, ratio {ratio_q:.2f} -> {rv}")
        else:
            rtxt = 'no remainder at N_end / 4 -> UNREAD'
        within = sum(abs(h - 1) <= 0.10 for _, h, _ in law)
        lv = 'HOLDS' if law and within >= 0.90 * len(law) else 'MISSED'
        print(f"  cell: {cv}; feedback {fb} (accelerating {acc}/{m}, settling {sett}/{m}); "
              f"remainder {rtxt}; net disc law {within}/{len(law)} "
              f"-> {lv}")


def scan10(jobs: int) -> None:
    out = f"results/sheet_variants_w10_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[Dict] = []
    with ProcessPoolExecutor(jobs) as ex, open(out, 'w', newline='') as f:
        w = None
        for r in ex.map(grow_one9, W10_SPECS):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                w.writeheader()
            w.writerow(r)
            f.flush()
            rows.append(r)
            print(f"disloc beta {r['beta']} seed {r['seed']}: N {r['n_end']} B/sqrtN "
                  f"{r['B_sqrtN_end']:.2f} i5 {r['i5']} i7 {r['i7']} [{r['seconds']} s]", flush=True)
    print(f"saved: {out}")
    read10(rows)


def _load10(path: str) -> List[Dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        for k in ('beta', 'rate', 'B_sqrtN_end', 'ecc_exp_last'):
            r[k] = float(r[k])
        for k in ('seed', 'n_end', 'stalled', 'i5', 'i7', 'dead_end', 'r_complete'):
            r[k] = int(r[k])
        rows.append(r)
    return rows


def read7(rows: List[Dict]) -> None:
    """Wave 7 frozen readings: per growth, frozen corners vs crossing."""
    a_n = a_flat = b_n = b_cross = 0
    tot_p = tot_d = 0
    for r in rows:
        fp = {int(t.split(':')[0]): int(t.split(':')[1]) for t in r['frozen_profile'].split()}
        prof = {int(t.split(':')[0]): float(t.split(':')[1].split('/')[0])
                for t in r['profile'].split()}
        ratio = r['B_sqrtN_end'] / prof[2 ** 16]
        flat = 0.85 <= ratio <= 1.15 and 0.40 <= r['ecc_exp_last'] <= 0.60
        crossed = ratio > 1.3 or r['ecc_exp_last'] < 0.35
        first = min([n for n, k in fp.items() if k > 0], default=-1)
        if fp.get(2 ** 19, 0) == 0:
            a_n += 1
            a_flat += int(flat)
        if 0 < first <= 2 ** 18:
            b_n += 1
            b_cross += int(crossed)
        p, d = map(int, r['frozen_persist'].split('/'))
        tot_p += p
        tot_d += d
        print(f"    beta {r['beta']} seed {r['seed']}: first frozen at <= {first}, frozen at 2^19 "
              f"{fp.get(2 ** 19, 0)}, at 2^20 {fp.get(2 ** 20, 0)}; B/sqrtN {r['B_sqrtN_end']:.2f} "
              f"(x{ratio:.2f}), ecc exp {r['ecc_exp_last']:.2f} -> "
              f"{'FLAT' if flat else 'CROSSED' if crossed else 'neither'}; dead {r['dead_end']}")
    print(f"  P1 (no frozen corner by 2^19 -> flat at 2^20): {a_flat}/{a_n}")
    print(f"  P2 (first frozen corner by 2^18 -> crossed at 2^20): {b_cross}/{b_n}")
    print(f"  persistence (frozen at one checkpoint, still frozen at the next): {tot_d}/{tot_p}")
    ok1 = a_n >= 4 and a_flat >= 0.9 * a_n
    ok2 = b_n >= 4 and b_cross >= 0.9 * b_n
    gate = all(r['dead_end'] == 0 and r['first_dead'] < 0 for r in rows)
    print(f"  gate (no dead corner): {'PASS' if gate else 'FAIL'}")
    if ok1 and ok2:
        v = 'FROZEN CORNERS DECIDE'
    elif a_n < 4 or b_n < 4:
        v = 'UNDERPOWERED'
    else:
        v = 'NOT DECISIVE'
    print(f"  W7: {v}")


def scan7(jobs: int) -> None:
    out = f"results/sheet_variants_w7_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[Dict] = []
    with ProcessPoolExecutor(jobs) as ex, open(out, 'w', newline='') as f:
        w = None
        for r in ex.map(grow_one, W7_SPECS):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                w.writeheader()
            w.writerow(r)
            f.flush()
            rows.append(r)
            print(f"strict beta {r['beta']} seed {r['seed']}: B/sqrtN {r['B_sqrtN_end']:.2f} "
                  f"frozen {r['frozen_profile'].split()[-1]} [{r['seconds']} s]", flush=True)
    print(f"saved: {out}")
    read7(rows)


def cap7() -> None:
    """C6: growth rate of cap-7 sheets against ln 1.5 (Keller-Peyerimhoff)."""
    bound = float(np.log(1.5))
    ok = True
    for s in (100, 101, 102):
        A = load_csr(f'results/sheet_c7_b2_s{s}_N512000.npz')
        n = A.shape[0]
        deg = np.diff(A.indptr)
        rng = np.random.default_rng(s)
        # interior sources: nodes among the oldest tenth (the core)
        src = rng.choice(np.arange(n // 10), 30, replace=False)
        D = dijkstra(A, indices=src, unweighted=True)
        mus = []
        for row in D:
            h = np.bincount(row[np.isfinite(row)].astype(int))
            V = np.cumsum(h)
            r = np.arange(len(V))
            m = (V < 0.1 * n) & (r >= 2)
            if m.sum() >= 4:
                mus.append(np.polyfit(r[m], np.log(V[m]), 1)[0])
        mu = float(np.median(mus))
        frac7 = float(np.mean(deg == 7))
        ok &= mu >= bound
        print(f"cap 7 seed {s}: N {n}, fraction of degree-7 nodes {frac7:.3f}, max degree "
              f"{deg.max()}, growth rate mu = {mu:.3f} per hop (bound ln 1.5 = {bound:.3f}), "
              f"radii used {int(m.sum())}")
    print(f"C6: {'BOUND RESPECTED' if ok else 'BELOW THE BOUND'}")


def _validate() -> bool:
    ok = True
    a = grow_sheet(20000, CAP, 2.0, 3)
    b = grow_sheet(20000, CAP, 2.0, 3, variant='sheet')
    good = len(a) == len(b) and all(x == y for x, y in zip(a, b))
    print(f"[{'PASS' if good else 'FAIL'}] default variant is bit-identical to the generator")
    ok &= good
    # wave 6: the flat limit of the hyperbolic-disc slope is 2
    good = (abs(hyperbolic_slope(100, 0.0) - 2) < 1e-9 and
            abs(hyperbolic_slope(1e-3, 3e-4) - 2) < 1e-3)
    print(f"[{'PASS' if good else 'FAIL'}] hyperbolic slope -> 2 in the flat limit")
    ok &= good
    det = DeadDetector(CAP)
    adj = grow_sheet(40000, CAP, 0.75, 12, on_event=det, variant='strict')
    good = not det.records
    print(f"[{'PASS' if good else 'FAIL'}] strict: no dead corner ever appears (beta 0.75, "
          f"{len(adj)} nodes; detector saw {len(det.records)})")
    ok &= good
    for v, cap_max in (('relief', CAP + 1), ('veto', CAP), ('strict', CAP)):
        adj = grow_sheet(40000, CAP, 1.0, 11, variant=v)
        st = checkpoint_stats(adj, CAP + 1 if v == 'relief' else CAP)
        # Gauss-Bonnet with interior degree-7 cone points:
        # sum_bdry (4 - d) = 6 + (number of interior degree-7 nodes).
        n = len(adj)
        bdry = [u for u in range(n) if any(len(adj[u] & adj[w]) == 1 for w in adj[u])]
        bset = set(bdry)
        gb = sum(4 - len(adj[u]) for u in bdry)
        int7 = sum(1 for u in range(n) if u not in bset and len(adj[u]) == 7)
        int_other = sum(1 for u in range(n) if u not in bset and len(adj[u]) not in (6, 7))
        good = st['max_deg'] <= cap_max and gb == 6 + int7 and int_other == 0
        print(f"[{'PASS' if good else 'FAIL'}] {v}: max degree {st['max_deg']} (<= {cap_max}); "
              f"Gauss-Bonnet sum_bdry(4 - d) = {gb} = 6 + {int7} interior degree-7 nodes; "
              f"dead corners {st['dead']}")
        ok &= good
    # wave 9: eager at rate 1 is relief; every variant stays a manifold disc with
    # sum_bdry(4 - d) = 6 + (interior 7s) - (interior 5s)
    from sheet_growth import compact
    a = grow_sheet(20000, CAP, 1.0, 4, variant='relief')
    b = grow_sheet(20000, CAP, 1.0, 4, variant='eager', rate=1.0)
    good = a == b
    print(f"[{'PASS' if good else 'FAIL'}] eager at rate 1 is bit-identical to relief")
    ok &= good
    for v, beta, rate in (('disloc', 0.75, 1.0), ('pent', 1.0, 3e-5), ('eager', 0.75, 16.0),
                          ('detach', 1.0, math.exp(3.0))):
        raw = grow_sheet(40000, CAP, beta, 11, variant=v, rate=rate)
        adj = compact(raw)
        n = len(adj)
        E = sum(len(s) for s in adj) // 2
        F = sum(len(adj[u] & adj[w]) for u in range(n) for w in adj[u] if u < w) // 3
        bad = sum(1 for u in range(n) for w in adj[u] if u < w and len(adj[u] & adj[w]) not in (1, 2))
        bset = {u for u in range(n) for w in adj[u] if len(adj[u] & adj[w]) == 1}
        gb = sum(4 - len(adj[u]) for u in bset)
        i5 = [u for u in range(n) if u not in bset and len(adj[u]) == 5]
        i7 = {u for u in range(n) if u not in bset and len(adj[u]) == 7}
        io = sum(1 for u in range(n) if u not in bset and len(adj[u]) not in (5, 6, 7))
        good = n - E + F == 1 and bad == 0 and io == 0 and gb == 6 + len(i7) - len(i5)
        extra = ''
        if v == 'disloc':
            near = sum(1 for f in i5 if any(y in i7 or adj[y] & i7 for y in adj[f]))
            good &= len(i5) <= len(i7) and near == len(i5)
            extra = f"; every 5 has a 7 within two hops ({near}/{len(i5)})"
        if v == 'detach':
            good &= len(raw) > n and all(raw[i] for i in range(3))
            extra = f"; {len(raw) - n} nodes detached"
        print(f"[{'PASS' if good else 'FAIL'}] {v}: Euler characteristic {n - E + F}, "
              f"non-manifold edges {bad}, Gauss-Bonnet {gb} = 6 + {len(i7)} - {len(i5)}{extra}")
        ok &= good
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--validate', action='store_true')
    m.add_argument('--scan', action='store_true')
    m.add_argument('--cap7', action='store_true')
    m.add_argument('--scan6', action='store_true')
    m.add_argument('--read6', type=str, metavar='CSV')
    m.add_argument('--scan7', action='store_true')
    m.add_argument('--read7', type=str, metavar='CSV')
    m.add_argument('--scan8', action='store_true')
    m.add_argument('--read8', type=str, metavar='CSV')
    m.add_argument('--scan9', action='store_true')
    m.add_argument('--read9', type=str, metavar='CSV')
    m.add_argument('--scan10', action='store_true')
    m.add_argument('--read10', type=str, metavar='CSV')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.scan:
        scan(args.jobs)
    elif args.scan6:
        scan6(args.jobs)
    elif args.scan7:
        scan7(args.jobs)
    elif args.scan8:
        scan8(args.jobs)
    elif args.scan9:
        scan9(args.jobs)
    elif args.scan10:
        scan10(args.jobs)
    elif args.read10:
        read10(_load10(args.read10))
    elif args.read9:
        rows = []
        for r in csv.DictReader(open(args.read9)):
            for k in ('beta', 'rate', 'B_sqrtN_end', 'ecc_exp_last', 'audit_d'):
                r[k] = float(r[k])
            for k in ('seed', 'stop_net', 'stopped', 'n_end', 'ids', 'stalled', 'B_end', 'i5', 'i7', 'int_other',
                      'paired5', 'max_deg', 'min_int_deg', 'dead_end', 'first_dead',
                      'r_complete'):
                r[k] = int(r[k])
            rows.append(r)
        read9(rows)
    elif args.read8:
        rows = []
        for r in csv.DictReader(open(args.read8)):
            r['beta'] = float(r['beta'])
            for k in ('seed', 'n_end', 'stalled', 'dead_end', 'deg7_end', 'max_deg'):
                r[k] = int(r[k])
            rows.append(r)
        read8(rows)
    elif args.read7:
        rows = []
        for r in csv.DictReader(open(args.read7)):
            for k in ('beta', 'B_sqrtN_end', 'ecc_exp_last'):
                r[k] = float(r[k])
            for k in ('seed', 'first_dead', 'dead_end'):
                r[k] = int(r[k])
            rows.append(r)
        read7(rows)
    elif args.read6:
        rows = []
        for r in csv.DictReader(open(args.read6)):
            for k in ('beta', 'B_sqrtN_end', 'ecc_exp_last', 'audit_d'):
                r[k] = float(r[k])
            for k in ('seed', 'n_end', 'stalled', 'first_dead', 'dead_end', 'deg7_end',
                      'capped_b_end'):
                r[k] = int(r[k])
            rows.append(r)
        read6(rows)
    else:
        cap7()


if __name__ == '__main__':
    main()
