"""
Roughness spectrum of a `sheet`'s edge, and the onset of roughening in one seed.

Pre-registration: docs/superpowers/specs/2026-10-13-edge-spectrum-preregistration.md.

The boundary of the prefix graph (edges in exactly one triangle) is walked into a
sequence; the height profile h(s) = rho(s) - R (root distance minus its median)
is Fourier-analysed along it. A shape instability at a fixed wavelength shows as
a spectral peak whose wavelength does not move with N; a scale-free rough surface
shows as a straight power law.

Usage:
    python edge_spectrum.py --validate
    python edge_spectrum.py results/sheet_cross_b2_s3*_N4194304.npz
    python edge_spectrum.py --onset results/sheet_cross_b2_s308_N4194304.npz
"""

import argparse
import csv
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from sheet_growth import grow_sheet, load_csr
from arm_birth import adjacency_sets, sets_to_csr, base_census

ROUGH_B = 4.5          # B / sqrt(N) at or above which the edge counts as rough
EXCESS = 3.0           # bin excess over the power-law fit that counts as a peak
LAMBDA_MIN = 40        # shortest wavelength (boundary nodes) entering the fit
K_MIN = 4              # k = 1..3 are the disc's global shape modes (offset, ellipticity, ...)
BINS_PER_DECADE = 8
MIN_MODES = 8
COVER_MIN = 0.9
OCTAVES = [2 ** k for k in range(15, 23)]


def boundary_sequence(adj: List[set]) -> Tuple[np.ndarray, float, int]:
    """Longest non-revisiting walk through the boundary-edge graph.

    Returns (sequence, covered fraction of boundary nodes, boundary edges).
    """
    bg: Dict[int, List[int]] = {}
    n_edges = 0
    for u, nb in enumerate(adj):
        for v in nb:
            if u < v and len(nb & adj[v]) == 1:
                n_edges += 1
                bg.setdefault(u, []).append(v)
                bg.setdefault(v, []).append(u)
    if not bg:
        return np.array([], dtype=int), 0.0, 0
    seen: set = set()
    best: List[int] = []
    for s in bg:
        if s in seen:
            continue
        seq = [s]
        seen.add(s)
        cur = s
        while True:
            nxt = [w for w in bg[cur] if w not in seen]
            if not nxt:
                break
            cur = nxt[0]
            seen.add(cur)
            seq.append(cur)
        if len(seq) > len(best):
            best = seq
    return np.array(best, dtype=int), len(best) / len(bg), n_edges


def binned_spectrum(h: np.ndarray, lambda_min: int = LAMBDA_MIN
                    ) -> Optional[Dict[str, np.ndarray]]:
    """Periodogram of h, log-binned with >= MIN_MODES modes per bin."""
    L = len(h)
    if L < 4 * lambda_min:
        return None
    hh = h - h.mean()
    P = np.abs(np.fft.rfft(hh)) ** 2 / L
    k = np.arange(len(P))
    kmax = L // lambda_min
    sel = (k >= K_MIN) & (k <= kmax)
    if sel.sum() < 2 * MIN_MODES:
        return None
    ks, Ps = k[sel], P[sel]
    step = 10 ** (1 / BINS_PER_DECADE)
    edges = [ks[0]]
    while edges[-1] <= ks[-1]:
        nxt = max(edges[-1] * step, edges[-1] + MIN_MODES)
        edges.append(nxt)
    bk, bP, bn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ks >= lo) & (ks < hi)
        if m.sum() >= MIN_MODES:
            bk.append(np.exp(np.mean(np.log(ks[m]))))
            bP.append(Ps[m].mean())
            bn.append(int(m.sum()))
    if len(bk) < 4:
        return None
    return {'k': np.array(bk), 'P': np.array(bP), 'n': np.array(bn), 'L': L}


def peak_analysis(h: np.ndarray, lambda_min: int = LAMBDA_MIN
                  ) -> Optional[Dict[str, float]]:
    """Power-law fit (two-pass), slope, and the strongest excess bin."""
    b = binned_spectrum(h, lambda_min)
    if b is None:
        return None
    lk, lP = np.log10(b['k']), np.log10(b['P'] + 1e-300)
    slope, icpt = np.polyfit(lk, lP, 1)
    resid = lP - (slope * lk + icpt)
    keep = np.ones(len(lk), dtype=bool)
    if len(lk) >= 6:
        keep[int(np.argmax(resid))] = False
    slope, icpt = np.polyfit(lk[keep], lP[keep], 1)
    excess = 10 ** (lP - (slope * lk + icpt))
    j = int(np.argmax(excess))
    return {'slope': float(slope), 'peak_lambda': float(b['L'] / b['k'][j]),
            'peak_excess': float(excess[j]), 'peak': bool(excess[j] >= EXCESS),
            'n_bins': int(len(lk)), 'L': int(b['L'])}


def analyze_prefix(A: sp.csr_array, n: int, census: bool = False
                   ) -> Dict[str, object]:
    P = A[:n, :n].tocsr()
    adj = adjacency_sets(P)
    seq, cover, n_bedges = boundary_sequence(adj)
    rho = dijkstra(P, indices=0, unweighted=True)
    R = float(np.median(rho[seq])) if seq.size else float('nan')
    h = rho[seq] - R
    out: Dict[str, object] = {
        'N': n, 'B': n_bedges, 'B_over_sqrtN': n_bedges / np.sqrt(n),
        'L': int(seq.size), 'cover': cover, 'R': R,
        'sigma_over_R': float(h.std() / R) if seq.size else float('nan'),
        'feature_2N_over_B': 2 * n / max(n_bedges, 1),
        'rough': bool(n_bedges / np.sqrt(n) >= ROUGH_B)}
    pa = peak_analysis(h) if seq.size else None
    if pa is None:
        out.update({'slope': float('nan'), 'peak_lambda': float('nan'),
                    'peak_excess': float('nan'), 'peak': False, 'n_bins': 0})
    else:
        out.update(pa)
    if census and seq.size:
        j = int(np.argmax(np.abs(h)))
        v = int(seq[j])
        found = {'adj': adj, 'bnd': np.unique(seq), 'base': v, 'P': P}
        cs = base_census(found)
        out.update({'ext_pos': j / seq.size, 'ext_h_over_R': float(h[j] / R),
                    'ext_sign': 'bump' if h[j] > 0 else 'fjord',
                    'ext_node': v, 'ext_blocked_ratio': cs['ratio'],
                    'ext_blocked_near': cs['blocked_density_near'],
                    'ext_blocked_rest': cs['blocked_density_rest'],
                    'ext_mean_deg_near': cs['mean_deg_near'],
                    'ext_mean_deg_rest': cs['mean_deg_rest']})
    return out


def reading(rows: List[Dict[str, object]]) -> str:
    seeds = sorted({r['seed'] for r in rows})
    counting, fixed, free = 0, 0, 0
    for s in seeds:
        rough = [r for r in rows if r['seed'] == s and r['rough']
                 and r['cover'] >= COVER_MIN and r['n_bins'] > 0]
        if len(rough) < 2:
            continue
        counting += 1
        if all(r['peak'] for r in rough):
            lam = [r['peak_lambda'] for r in rough]
            if max(lam) / min(lam) <= 2.0:
                fixed += 1
        if (not any(r['peak'] for r in rough)
                and all(-2.5 <= r['slope'] <= -1.0 for r in rough)):
            free += 1
    if counting == 0:
        return 'UNDECIDED (no seed with >= 2 rough sizes)'
    if fixed >= np.ceil(2 * counting / 3):
        return f'FIXED WAVELENGTH ({fixed}/{counting} counting seeds)'
    if free >= np.ceil(2 * counting / 3):
        return f'SCALE-FREE ({free}/{counting} counting seeds)'
    return (f'UNDECIDED (fixed {fixed}, scale-free {free} of {counting} '
            f'counting seeds)')


def seed_of(path: str) -> int:
    base = os.path.basename(path)
    i = base.index('_s') + 2
    return int(base[i:i + 3])


HEADER = ('seed', 'N', 'B', 'B_over_sqrtN', 'L', 'cover', 'R', 'sigma_over_R',
          'feature_2N_over_B', 'rough', 'slope', 'peak_lambda', 'peak_excess',
          'peak', 'n_bins')


def run(paths: Sequence[str], sizes: Sequence[int], out_csv: str) -> List[Dict]:
    rows: List[Dict[str, object]] = []
    print(f"{'seed':>4} {'N':>9} {'B/sqrtN':>7} {'cover':>5} {'sig/R':>6} "
          f"{'2N/B':>5} {'slope':>6} {'lam*':>7} {'excess':>6} {'peak':>5}")
    for path in paths:
        A = load_csr(path)
        s = seed_of(path)
        for n in sizes:
            if n > A.shape[0]:
                continue
            t0 = time.time()
            r = analyze_prefix(A, n)
            r['seed'] = s
            rows.append(r)
            print(f"{s:4d} {n:9,d} {r['B_over_sqrtN']:7.2f} {r['cover']:5.2f} "
                  f"{r['sigma_over_R']:6.3f} {r['feature_2N_over_B']:5.0f} "
                  f"{r['slope']:6.2f} {r['peak_lambda']:7.0f} "
                  f"{r['peak_excess']:6.2f} {str(r['peak']):>5}"
                  f"   [{time.time() - t0:.0f} s]", flush=True)
        del A
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out_csv}")
    print(f"VERDICT (frozen rules): {reading(rows)}")
    compact = [r['slope'] for r in rows if not r['rough'] and r['n_bins'] > 0]
    if compact:
        print(f"compact-edge slopes: mean {np.mean(compact):.2f}, "
              f"range {min(compact):.2f} .. {max(compact):.2f}")
    peaks = [r for r in rows if r['peak'] and r['rough']]
    if peaks:
        ratio = [r['peak_lambda'] / r['feature_2N_over_B'] for r in peaks]
        print(f"rough peaks: lambda* / (2N/B) median {np.median(ratio):.2f} "
              f"(range {min(ratio):.2f} .. {max(ratio):.2f})")
    return rows


def onset(path: str, n_min: int, n_max: int, per_octave: int,
          out_csv: str) -> List[Dict]:
    A = load_csr(path)
    s = seed_of(path)
    sizes = []
    n = float(n_min)
    while n <= min(n_max, A.shape[0]) + 0.5:
        sizes.append(int(round(n)))
        n *= 2 ** (1 / per_octave)
    rows = []
    print(f"{'N':>8} {'B/sqrtN':>7} {'cover':>5} {'sig/R':>6} {'slope':>6} "
          f"{'lam*':>6} {'excess':>6} {'ext h/R':>8} {'kind':>5} {'pos':>5} "
          f"{'blocked ratio':>13} {'deg near/rest':>13}")
    for n in sizes:
        r = analyze_prefix(A, n, census=True)
        r['seed'] = s
        rows.append(r)
        print(f"{n:8,d} {r['B_over_sqrtN']:7.2f} {r['cover']:5.2f} "
              f"{r['sigma_over_R']:6.3f} {r['slope']:6.2f} "
              f"{r['peak_lambda']:6.0f} {r['peak_excess']:6.2f} "
              f"{r['ext_h_over_R']:+8.3f} {r['ext_sign']:>5} {r['ext_pos']:5.2f} "
              f"{r['ext_blocked_ratio']:13.2f} "
              f"{r['ext_mean_deg_near']:6.2f}/{r['ext_mean_deg_rest']:.2f}",
              flush=True)
    keys = list(HEADER) + ['ext_pos', 'ext_h_over_R', 'ext_sign', 'ext_node',
                           'ext_blocked_ratio', 'ext_blocked_near',
                           'ext_blocked_rest', 'ext_mean_deg_near',
                           'ext_mean_deg_rest']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out_csv}")
    return rows


def _validate() -> bool:
    ok = True
    rng = np.random.default_rng(0)
    print("[1] planted wavelength on a noisy circle")
    L = 6000
    s = np.arange(L)
    h = 3.0 * np.sin(2 * np.pi * s / 300) + rng.normal(0, 0.5, L)
    pa = peak_analysis(h)
    good = pa is not None and pa['peak'] and 240 <= pa['peak_lambda'] <= 375
    ok &= good
    print(f"  planted 300: read lambda* {pa['peak_lambda']:.0f}, excess "
          f"{pa['peak_excess']:.1f}, peak {pa['peak']}  {'OK' if good else 'FAIL'}")
    print("[2] periodic random walks: slope near -2, no peak (<= 1 of 10)")
    slopes, npk = [], 0
    for _ in range(10):
        w = np.cumsum(rng.normal(0, 1, L))
        w -= np.linspace(0, w[-1] - w[0], L) + w[0]       # bridge
        pa = peak_analysis(w)
        slopes.append(pa['slope'])
        npk += int(pa['peak'])
    good = all(-2.6 <= sl <= -1.4 for sl in slopes) and npk <= 1
    ok &= good
    print(f"  slopes {min(slopes):.2f} .. {max(slopes):.2f}, peaks {npk}/10  "
          f"{'OK' if good else 'FAIL'}")
    print("[3] boundary sequence of a 40 k-node compact sheet is one cycle")
    adj = grow_sheet(40000, 6, 3.0, seed=11)
    seq, cover, nb = boundary_sequence(adj)
    A = sets_to_csr(adj)
    r = analyze_prefix(A, len(adj))
    good = (cover >= 0.98 and abs(seq.size - nb) <= 2 and r['B_over_sqrtN'] < ROUGH_B
            and not r['rough'] and r['sigma_over_R'] < 0.1)
    # the compact edge's peak flag is printed, not asserted: L ~ 750 gives few modes
    ok &= good
    print(f"  cover {cover:.3f}, {seq.size} nodes / {nb} boundary edges, "
          f"B/sqrtN {r['B_over_sqrtN']:.2f}, sigma/R {r['sigma_over_R']:.3f}, "
          f"slope {r['slope']:.2f}, peak {r['peak']}  {'OK' if good else 'FAIL'}")
    print("[4] readings")
    mk = lambda seed, peaks, lams, slopes: [
        {'seed': seed, 'rough': True, 'cover': 1.0, 'n_bins': 10, 'peak': p,
         'peak_lambda': l, 'slope': sl} for p, l, sl in zip(peaks, lams, slopes)]
    fixed = mk(1, [True, True, True], [300, 350, 400], [-1.5] * 3) \
        + mk(2, [True, True], [200, 380], [-1.5] * 2) \
        + mk(3, [False, True], [0, 300], [-2, -2])
    free = mk(1, [False] * 3, [0] * 3, [-1.8, -1.9, -2.0]) \
        + mk(2, [False] * 2, [0] * 2, [-1.6, -2.1]) \
        + mk(3, [True, True], [300, 310], [-1.5, -1.5])
    for name, rows, want in (('fixed', fixed, 'FIXED WAVELENGTH'),
                             ('free', free, 'SCALE-FREE'),
                             ('mixed', fixed[:3] + free[3:5], 'UNDECIDED')):
        got = reading(rows)
        good = got.startswith(want)
        ok &= good
        print(f"  {name:>5}: {got}  {'OK' if good else 'FAIL'}")
    print("\nPASS: edge spectrum" if ok else "\nFAIL: edge spectrum")
    return ok


def main() -> None:
    global K_MIN
    ap = argparse.ArgumentParser(description="Roughness spectrum of a sheet's edge")
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--onset', action='store_true',
                    help='fine onset scan of one graph (descriptive)')
    ap.add_argument('--n-min', type=int, default=2 ** 14)
    ap.add_argument('--n-max', type=int, default=2 ** 17)
    ap.add_argument('--per-octave', type=int, default=4)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--k-min', type=int, default=K_MIN,
                    help='lowest wave number entering the fit (frozen: 4; a post-hoc '
                         're-read may raise it to skip the disc\'s six-fold shape mode)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.k_min != K_MIN:
        print(f"NOTE: k_min = {args.k_min} (frozen value 4): exploratory re-read")
        K_MIN = args.k_min
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if not args.paths:
        ap.error('give saved sheet graphs (.npz)')
    os.makedirs('results', exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    if args.onset:
        out = args.out or f'results/edge_onset_s{seed_of(args.paths[0])}_{stamp}.csv'
        onset(args.paths[0], args.n_min, args.n_max, args.per_octave, out)
    else:
        out = args.out or f'results/edge_spectrum_{stamp}.csv'
        run(args.paths, OCTAVES, out)


if __name__ == '__main__':
    main()
