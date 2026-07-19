"""
Causal-set instruments and their known-answer calibration.

This is **step 1** of the staged plan in LORENTZIAN_SPIKE.md: build the
dimension estimators that will run on the causal event DAG, and calibrate them
against sprinklings into flat spacetime, where the answer is known. It contains
no model dynamics at all -- deliberately. Until the instruments recover a known
dimension, nothing measured on an event DAG can be trusted.

**Kill criterion (from the spike): if the two estimators disagree on
known-answer sprinklings, stop.** There is no instrument, and every downstream
stage is unsupported.

TWO PRIMARY INSTRUMENTS, chosen so that neither inherits the other's
assumptions -- MM counts related PAIRS (a two-point statistic), midpoint
scaling bisects VOLUME (a one-point extremum). They share only the underlying
causal set, which is what makes their agreement evidence rather than tautology.

  MYRHEIM-MEYER -- the ordering fraction r = R / C(N,2) within an interval,
      where R counts causally related pairs, inverted through the known
      function r(d) for Poisson sprinklings into d-dimensional Minkowski space:

          r(d) = Gamma(d+1) Gamma(d/2) / ( 2 Gamma(3d/2) )

      LORENTZIAN_SPIKE.md deliberately declined to state this constant from
      memory. It is reconstructed here from the 2-chain abundance and pinned
      against two independently derived anchors -- r(1) = 1 (a 1D causal set is
      a total order, so every pair is related) and r(2) = 1/2 (in light-cone
      coordinates an interval is a square and two uniform points are related
      iff both coordinates agree in order) -- then confirmed numerically to
      about 1% at d = 2, 3, 4 by `--validate`.

  MIDPOINT SCALING -- bisect an interval by volume; the midpoint halves the
      proper time and so divides the volume by 2^d.

A THIRD, DEMOTED: interval scaling, |I(p,q)| ~ l(p,q)^d with l the longest
chain. This was the spike's proposed primary estimator, on the grounds that it
is the direct port of `dimension.py`'s ball growth (geodesic radius -> causal
depth) and self-calibrating, since it fits an exponent and needs no constant.
**Calibration refuted that.** It is systematically biased low -- 1.92 / 2.79 /
3.53 against a true 2 / 3 / 4 -- while reporting R^2 > 0.99, i.e. it fits a
clean power law with the wrong exponent. Two rounds of fixing narrowed but did
not remove the bias:

  1. Binning by chain length and averaging log|I| was regression dilution:
     l is the noisy variable, and noise in the independent variable biases the
     slope toward zero. Regressing the other way (exact cardinality on the
     x-axis, l on the y) moved 1.75/2.21/1.86 to 1.91/2.75/3.54.
  2. The residual bias is NOT finite-size -- it is flat in N (3.70, 3.71, 3.59
     at N = 1500/3000/4500). A large-|I| regime gate helps at d=2 but is
     non-monotonic and never converges at d=3, so it would have to be tuned
     per-dimension. An estimator whose gate must be tuned against the known
     answer is not an instrument, since on a real event DAG there is no known
     answer to tune against.

It is retained as a reported diagnostic because its bias is reproducible, but
nothing should rest on it. The wider lesson is the one FINDINGS keeps
relearning: a high R^2 certifies that a power law was fitted, not that the
right exponent was recovered.

Usage:
    python causal_sets.py --validate
    python causal_sets.py --validate --points 4000 --reps 5
"""

import argparse
import random
from typing import Dict, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln
from tqdm import tqdm


# --------------------------------------------------------------------------
# Myrheim-Meyer ordering fraction
# --------------------------------------------------------------------------

def mm_ordering_fraction(d: float) -> float:
    """
    Expected ordering fraction of a Poisson sprinkling into an Alexandrov
    interval of d-dimensional Minkowski space (large-N limit).

    r(d) = Gamma(d+1) Gamma(d/2) / (2 Gamma(3d/2)), monotonically decreasing,
    so it inverts uniquely. See the module docstring on provenance.
    """
    log_r = gammaln(d + 1.0) + gammaln(d / 2.0) - np.log(2.0) - gammaln(1.5 * d)
    return float(np.exp(log_r))


def mm_dimension(r: float, lo: float = 0.5, hi: float = 12.0) -> float:
    """
    Invert `mm_ordering_fraction`. Returns nan when r falls outside the
    bracket (r > r(lo) means "less structured than 0.5-dimensional", which is
    a diagnostic that the causal set is not manifold-like, not a dimension).
    """
    if not np.isfinite(r) or r <= 0.0:
        return float('nan')
    f_lo, f_hi = mm_ordering_fraction(lo) - r, mm_ordering_fraction(hi) - r
    if f_lo * f_hi > 0:
        return float('nan')
    return float(brentq(lambda d: mm_ordering_fraction(d) - r, lo, hi))


# --------------------------------------------------------------------------
# Sprinkling into flat spacetime
# --------------------------------------------------------------------------

def sprinkle_interval(d: int, n_points: int,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Poisson-sprinkle `n_points` uniformly into the Alexandrov interval between
    p = origin and q = (1, 0, ..., 0) in d-dimensional Minkowski space.

    Returns an (n_points, d) array whose column 0 is time. Uniform sampling is
    by rejection from the bounding cylinder t in [0,1], |x| <= 1/2; a point is
    inside the interval iff |x| < min(t, 1 - t) (future of p and past of q).
    """
    kept = []
    got = 0
    while got < n_points:
        batch = max(n_points * 4, 1024)
        t = rng.random(batch)
        # Uniform in the (d-1)-ball of radius 1/2: direction x radius^(1/(d-1)).
        if d == 1:
            x = np.zeros((batch, 0))
            radius = np.zeros(batch)
        else:
            direction = rng.normal(size=(batch, d - 1))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            radius = 0.5 * rng.random(batch) ** (1.0 / (d - 1))
            x = direction * radius[:, None]
        inside = radius < np.minimum(t, 1.0 - t)
        pts = np.column_stack([t[inside], x[inside]])
        kept.append(pts)
        got += len(pts)
    return np.vstack(kept)[:n_points]


def causal_relations(coords: np.ndarray) -> np.ndarray:
    """
    Boolean relation matrix R[i, j] = (i precedes j), i.e. j is in the causal
    future of i: dt > 0 and dt^2 - |dx|^2 > 0.
    """
    t = coords[:, 0]
    x = coords[:, 1:]
    dt = t[None, :] - t[:, None]                  # dt[i, j] = t_j - t_i
    sq = (x ** 2).sum(axis=1)
    dx2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(dx2, 0.0, out=dx2)                 # kill round-off negatives
    return (dt > 0) & (dt ** 2 > dx2)


# --------------------------------------------------------------------------
# Estimator (b): ordering fraction
# --------------------------------------------------------------------------

def ordering_fraction(R: np.ndarray) -> float:
    """Fraction of pairs that are causally related."""
    n = R.shape[0]
    if n < 2:
        return float('nan')
    return float(np.count_nonzero(R)) / (n * (n - 1) / 2.0)


# --------------------------------------------------------------------------
# Estimator (a): interval cardinality vs longest chain
# --------------------------------------------------------------------------

def longest_chain_from(R: np.ndarray, src: int,
                       time_order: np.ndarray) -> np.ndarray:
    """
    Longest chain length from `src` to every element (-1 where unreachable).

    Dynamic program over a topological order; since R[i, j] implies t_i < t_j,
    ascending time is a valid topological order.
    """
    n = R.shape[0]
    L = np.full(n, -1, dtype=np.int32)
    L[src] = 0
    for j in time_order:
        if L[j] >= 0:
            continue
        preds = R[:, j] & (L >= 0)
        if preds.any():
            L[j] = L[preds].max() + 1
    return L


def interval_scaling_dimension(R: np.ndarray, rng: np.random.Generator,
                               n_sources: int = 12,
                               min_cardinality: int = 8,
                               n_bins: int = 8,
                               min_bins: int = 4) -> Dict[str, float]:
    """
    Fit |I(p,q)| ~ l(p,q)^d over sampled sub-intervals, by regressing
    log l on log |I| and inverting the slope (d = 1 / slope).

    REGRESSION ORIENTATION IS LOAD-BEARING. The naive form -- bin by chain
    length l, average log|I| -- is biased badly low (measured 1.75 / 2.21 /
    1.86 against true 2 / 3 / 4) while still reporting R^2 > 0.96, because l
    is the noisy quantity and noise in the independent variable dilutes the
    slope toward zero. `dimension.py` never hits this: its radius r is exact.
    Here the exact quantity is the cardinality |I| (a count), so it goes on
    the x-axis and l on the y-axis, where its noise costs variance instead of
    bias.

    Note this estimator is COORDINATE-FREE: it needs only the relation
    matrix. Descending out-degree is a valid topological order on any DAG,
    because i < j implies future(i) is a strict superset of future(j) -- so
    no time coordinate is required. That matters for the event DAG, where
    there is no global time to sort by.

    Returns d_eff, r_squared, n_bins, n_samples (d_eff is nan if gated out).
    """
    n = R.shape[0]
    out_degree = R.sum(axis=1)
    time_order = np.ascontiguousarray(np.argsort(out_degree)[::-1])

    sources = rng.choice(n, size=min(n_sources, n), replace=False)
    ells, cards = [], []
    for src in sources:
        L = longest_chain_from(R, int(src), time_order)
        future = R[src, :]
        targets = np.nonzero(future & (L > 0))[0]
        if len(targets) == 0:
            continue
        # |I(src, q)| = |future(src) & past(q)|
        card = (future[None, :] & R[:, targets].T).sum(axis=1)
        keep = card >= min_cardinality
        ells.extend(L[targets][keep].tolist())
        cards.extend(card[keep].tolist())

    if not ells:
        return {'d_eff': float('nan'), 'r_squared': float('nan'),
                'n_bins': 0, 'n_samples': 0}

    ells = np.asarray(ells, dtype=float)
    cards = np.asarray(cards, dtype=float)

    # Log-spaced bins in cardinality (the exact variable); mean log l per bin.
    edges = np.geomspace(cards.min(), cards.max() + 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (cards >= lo) & (cards < hi) & (ells >= 2)
        if sel.sum() < 5:
            continue
        xs.append(np.log(cards[sel]).mean())
        ys.append(np.log(ells[sel]).mean())

    if len(xs) < min_bins:
        return {'d_eff': float('nan'), 'r_squared': float('nan'),
                'n_bins': len(xs), 'n_samples': len(ells)}

    xs, ys = np.asarray(xs), np.asarray(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    d_eff = 1.0 / slope if slope > 1e-9 else float('nan')
    return {'d_eff': float(d_eff), 'r_squared': r2,
            'n_bins': len(xs), 'n_samples': len(ells)}


# --------------------------------------------------------------------------
# Estimator (c): midpoint scaling
# --------------------------------------------------------------------------

def midpoint_dimension(R: np.ndarray, rng: np.random.Generator,
                       n_pairs: int = 40,
                       min_cardinality: int = 64,
                       min_intervals: int = 20) -> Dict[str, float]:
    """
    Bombelli-style midpoint scaling: bisect an interval by volume.

    For I(p,q) of cardinality N, find the element r maximising
    min(|I(p,r)|, |I(r,q)|) -- the "midpoint". It halves the proper time, so
    in d dimensions it halves the volume by 2^d:

        N_half ~ N / 2^d   =>   d = log2( N / N_half )

    This is genuinely independent of Myrheim-Meyer: MM counts related PAIRS
    (a two-point statistic), this bisects VOLUME (a one-point extremum). They
    share only the underlying causal set, which is what makes their agreement
    evidence rather than tautology.

    `min_cardinality` is kept modest on purpose. Raising it does NOT buy
    accuracy: demanding |I| >= 200 at d=4, N=2500 left so few qualifying
    intervals that the estimate went to 4.47 on sampling noise, while |I| >= 64
    with a full sample gave 3.94. The meaningful gate is therefore the SAMPLE
    COUNT (`min_intervals`), which is diagnosable without knowing the true
    dimension -- unlike a cardinality threshold, which would have to be tuned
    per-dimension against the answer.

    Accuracy still degrades with d at fixed N, because the midpoint divides |I|
    by 2^d and the halves get too discrete to bisect cleanly: d=4 wants
    N >~ 2500. Refuses (nan) rather than guessing when the sample is too thin.

    Returns d_eff, its spread across sampled intervals, and n_intervals.
    """
    n = R.shape[0]
    ds = []
    tries = 0
    while len(ds) < n_pairs and tries < n_pairs * 40:
        tries += 1
        p = int(rng.integers(n))
        future_p = R[p, :]
        cand = np.nonzero(future_p)[0]
        if len(cand) == 0:
            continue
        q = int(cand[rng.integers(len(cand))])
        in_interval = future_p & R[:, q]
        S = np.nonzero(in_interval)[0]
        if len(S) < min_cardinality:
            continue

        # n1[k] = |I(p, S[k])|,  n2[k] = |I(S[k], q)|
        n1 = (future_p[:, None] & R[:, S]).sum(axis=0)
        n2 = (R[S, :] & R[:, q][None, :]).sum(axis=1)
        k = int(np.argmax(np.minimum(n1, n2)))
        n_half = 0.5 * (n1[k] + n2[k])
        if n_half < 1:
            continue
        ds.append(np.log2(len(S) / n_half))

    if len(ds) < min_intervals:
        # Refuse rather than report a number off a starved sample.
        return {'d_eff': float('nan'), 'd_std': float('nan'),
                'n_intervals': len(ds)}
    return {'d_eff': float(np.mean(ds)), 'd_std': float(np.std(ds)),
            'n_intervals': len(ds)}


# --------------------------------------------------------------------------
# Known-answer calibration
# --------------------------------------------------------------------------

def _validate(dims=(2, 3, 4), n_points: int = 3000, reps: int = 3,
              seed: int = 0, tol_mm: float = 0.15,
              tol_agree: float = 0.35) -> bool:
    """
    Calibrate both estimators against flat-space sprinklings of known
    dimension, and apply the spike's kill criterion.
    """
    rng = np.random.default_rng(seed)
    print("Analytic anchors for r(d) (see module docstring):")
    for d, expect in ((1.0, 1.0), (2.0, 0.5)):
        got = mm_ordering_fraction(d)
        flag = 'OK' if abs(got - expect) < 1e-12 else 'FAIL'
        print(f"  r({d:.0f}) = {got:.10f}  expected {expect:.10f}  {flag}")
        if flag == 'FAIL':
            print("\nFAIL: the ordering-fraction formula is wrong at an anchor.")
            return False

    print(f"\nSprinkling into flat Minkowski, N={n_points}, {reps} rep(s):")
    print("Primary instruments: Myrheim-Meyer and midpoint scaling. "
          "'interval' is reported\nas a diagnostic only -- see the module "
          "docstring for why it is not trusted.")
    header = (f"{'d':>3s} {'r meas':>8s} {'r pred':>8s} {'d_MM':>7s} "
              f"{'d_midpt':>9s} {'+-':>6s} {'d_interval':>11s}")
    print(header)
    print('-' * len(header))

    ok = True
    for d in dims:
        r_meas, d_mm, d_mid, mid_sd, d_int = [], [], [], [], []
        for _ in tqdm(range(reps), desc=f'd={d}', leave=False):
            coords = sprinkle_interval(d, n_points, rng)
            R = causal_relations(coords)
            r = ordering_fraction(R)
            r_meas.append(r)
            d_mm.append(mm_dimension(r))
            mid = midpoint_dimension(R, rng)
            d_mid.append(mid['d_eff'])
            mid_sd.append(mid['d_std'])
            d_int.append(interval_scaling_dimension(R, rng)['d_eff'])

        r_bar = float(np.mean(r_meas))
        r_pred = mm_ordering_fraction(d)
        dmm = float(np.mean(d_mm))
        dmid = float(np.nanmean(d_mid))
        print(f"{d:3d} {r_bar:8.4f} {r_pred:8.4f} {dmm:7.3f} "
              f"{dmid:9.3f} {float(np.nanmean(mid_sd)):6.3f} "
              f"{float(np.nanmean(d_int)):11.3f}")

        # The two estimators have different JOBS, so they get different gates.
        #
        # Gate 1 -- MM is the primary instrument and must recover the truth.
        if abs(dmm - d) > tol_mm:
            print(f"     FAIL: Myrheim-Meyer recovered {dmm:.3f}, expected {d}")
            ok = False
        # Gate 2 (the spike's kill criterion) -- midpoint is the CROSS-CHECK.
        # Its job is to catch a gross error in MM, so it is gated on agreeing
        # with MM, not on independent precision. Its own deviation from truth
        # is characterised below rather than gated: it carries a documented
        # +4% bias at d=4 (4.15/4.16 at N=4000/2500 -- stable in N, mechanism
        # understood as the discreteness of a extremum over few elements).
        # Gating it at MM's tolerance would fail a working instrument for a
        # known systematic; loosening MM's tolerance to cover it would hide a
        # real bias. Separate roles, separate gates.
        if not np.isfinite(dmid) or abs(dmid - dmm) > tol_agree:
            print(f"     FAIL: primary estimators disagree "
                  f"(midpoint {dmid:.3f} vs MM {dmm:.3f})")
            ok = False
        elif abs(dmid - d) > tol_mm:
            print(f"     note: midpoint {dmid:+.3f} vs truth {d} "
                  f"(known systematic, within cross-check tolerance)")

    print(f"\n{'PASS' if ok else 'FAIL'}: causal-set instruments "
          f"(kill criterion: the two primary estimators must agree "
          f"on known answers)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Causal-set dimension estimators and their calibration.")
    parser.add_argument('--validate', action='store_true',
                        help='Known-answer calibration; the step-1 gate.')
    parser.add_argument('--dims', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--points', type=int, default=3000)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(
            dims=tuple(args.dims), n_points=args.points,
            reps=args.reps, seed=args.seed) else 1)

    parser.print_help()


if __name__ == '__main__':
    main()
