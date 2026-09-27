"""
Exact per-radius decomposition of the pruned-WS d(p) fine structure (round 3).

Pre-registration:
docs/superpowers/specs/2026-09-27-dp-round3-decomposition-preregistration.md

The ball-growth fit  log|B(r)| = d*log r + c + a/r  is linear in the log
counts, so the fitted slope is an exact weighted sum

    d = sum_r w_r * ln|B(r)|,     w = first row of pinv(X),

and any waveform in d(p) splits, with no modelling, into one contribution
per radius. This driver rebuilds the banked dense grid through the same code
path as `prune_dimension.measure_pruned`, checks it reproduces the banked
CSV, and reports where in radius the waveform lives.

Usage:
    python dp_decomposition.py --validate
    python dp_decomposition.py --jobs 10
    python dp_decomposition.py --analyze results/dp_decomposition_TIMESTAMP.csv
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from dimension import fast_dimension_field, dimension_stats, local_dimension
from prune_dimension import prune_to_convergence

BANKED_CSV = "results/prune_dimension_dense32k_20260811.csv"
N_NODES, K, MAX_RADIUS, SAMPLES, SEEDS = 32000, 6, 10, 400, 6
PS = np.geomspace(0.02, 0.5, 30)   # nominal; the run uses banked_ps()
DETREND_ORDER = 3


def banked_ps() -> List[float]:
    """The p values the banked grid actually used.

    They are `geomspace(0.02, 0.5, 30)` ROUNDED to 3 significant digits
    (they were passed on a command line), so rebuilding from the unrounded
    grid builds different graphs. Read them from the banked CSV instead.
    """
    with open(BANKED_CSV) as f:
        return sorted({float(r["p"]) for r in csv.DictReader(f)})


def fit_weights(max_radius: int) -> np.ndarray:
    """w such that the fitted slope is sum_r w_r ln|B(r)|, r = 1..max_radius."""
    r = np.arange(1, max_radius + 1, dtype=float)
    X = np.column_stack([np.log(r), np.ones_like(r), 1.0 / r])
    return np.linalg.pinv(X)[0]


def measure(spec: Tuple[float, int, int]) -> Dict:
    """One (p, seed): `measure_pruned`'s path, keeping the ball counts."""
    p, seed, n = spec
    random.seed(seed)
    np.random.seed(seed)
    G = create_initial_graph(n, topology="small_world", k=K, p=p, seed=seed)
    G = prune_to_convergence(G)
    n_now = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, weight=None, format="csr",
                                 dtype=np.float32)
    field = fast_dimension_field(A, max_radius=MAX_RADIUS, n_samples=SAMPLES)
    stats = dimension_stats(field, n_now)
    logB = np.array([[np.log(c) for _, c in balls]
                     for (_, _, balls) in field.values()])
    row = {"p": p, "seed": seed, "N": n,
           "d_eff_median": stats["d_eff_median"]}
    for r in range(MAX_RADIUS):
        row[f"L{r + 1}"] = float(logB[:, r].mean())
    return row


def detrend(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - np.polyval(np.polyfit(x, y, DETREND_ORDER), x)


def analyze(rows: List[Dict], check_banked: bool = True) -> Dict:
    ps = np.array(sorted({float(r["p"]) for r in rows}))
    seeds = sorted({int(r["seed"]) for r in rows})
    R = MAX_RADIUS
    key = {(round(float(r["p"]), 12), int(r["seed"])): r for r in rows}

    ok = True
    if check_banked:
        worst, n_matched = 0.0, 0
        with open(BANKED_CSV) as f:
            for b in csv.DictReader(f):
                k = (round(float(b["p"]), 12), int(b["seed"]))
                if k in key:
                    n_matched += 1
                    worst = max(worst, abs(float(b["d_eff_median"])
                                           - float(key[k]["d_eff_median"])))
        print(f"  ({n_matched} of {len(rows)} rows matched a banked row)")
        gate_r = worst < 1e-9 and n_matched == len(rows)
        ok &= gate_r
        print(f"Gate R -- reproduces the banked dense grid: worst |diff| = "
              f"{worst:.2e}  {'PASS' if gate_r else 'FAIL'}")
        if not gate_r:
            return {"ok": False}

    d_med = np.array([np.mean([float(key[(round(p, 12), s)]["d_eff_median"])
                               for s in seeds]) for p in ps])
    L_seed = np.array([[[float(key[(round(p, 12), s)][f"L{r + 1}"])
                         for r in range(R)] for s in seeds] for p in ps])
    L = L_seed.mean(axis=1)                       # (n_p, R)
    w = fit_weights(R)
    d_lin = L @ w
    x = np.log(ps)

    res_med, res_lin = detrend(x, d_med), detrend(x, d_lin)
    res_L = np.array([detrend(x, L[:, r]) for r in range(R)]).T
    c = res_L * w[None, :]
    assert np.allclose(c.sum(axis=1), res_lin, atol=1e-9)

    corr = float(np.corrcoef(res_med, res_lin)[0, 1])
    gate_m1 = corr >= 0.8
    print(f"Gate M1 -- corr(resid d_med, resid d_lin) = {corr:+.3f} "
          f"(required >= 0.8)  {'PASS' if gate_m1 else 'FAIL'}")
    print(f"  peak-to-peak: resid d_med {np.ptp(res_med):.3f}, "
          f"resid d_lin {np.ptp(res_lin):.3f}")
    if not gate_m1:
        return {"ok": False, "corr": corr}

    var = float(np.var(res_lin))
    share = np.array([float(np.mean((c[:, r] - c[:, r].mean())
                                    * (res_lin - res_lin.mean()))) / var
                      for r in range(R)])
    # seed-to-seed noise of each detrended log count
    res_L_seed = np.array([[detrend(x, L_seed[:, s, r]) for r in range(R)]
                           for s in range(len(seeds))])   # (seed, R, n_p)
    noise = res_L_seed.std(axis=0, ddof=1).mean(axis=1) / np.sqrt(len(seeds))

    print("\n   r    weight   ptp resid ln|B|   seed SE   ptp contribution"
          "   variance share")
    for r in range(R):
        print(f"  {r + 1:2d}  {w[r]:+8.3f}   {np.ptp(res_L[:, r]):12.4f}   "
              f"{noise[r]:7.4f}   {np.ptp(c[:, r]):14.4f}   {share[r]:+10.3f}")
    local, meso = float(share[:3].sum()), float(share[3:].sum())
    if local >= 0.7:
        reading = "LOCAL"
    elif meso >= 0.7:
        reading = "MESOSCALE"
    else:
        reading = "DISTRIBUTED"
    s = float(max(np.ptp(res_L[:, r]) for r in range(R)))
    amplified = np.ptp(res_lin) > 3.0 * s
    print(f"\n  share r<=3: {local:+.3f}   share r>=4: {meso:+.3f}   "
          f"->  {reading}")
    print(f"  amplification: ptp resid d_lin {np.ptp(res_lin):.4f} vs 3 x "
          f"largest per-radius ptp {3 * s:.4f}  ->  "
          f"{'AMPLIFIED' if amplified else 'not amplified'}")
    print("\n  waveform (p : resid d_med, resid d_lin, and resid ln|B(r)| at "
          "r = 1, 2, 3, 6, 10):")
    for i, p in enumerate(ps):
        print(f"   {p:.4f}: {res_med[i]:+.3f} {res_lin[i]:+.3f}   "
              + " ".join(f"{res_L[i, r]:+.4f}" for r in (0, 1, 2, 5, 9)))
    return {"ok": True, "corr": corr, "share": share, "reading": reading,
            "amplified": bool(amplified)}


def _validate() -> bool:
    ok = True
    print("[1] the weights reproduce the project's estimator")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        counts = np.cumsum(rng.integers(1, 20, size=MAX_RADIUS)) + 1
        balls = [(r + 1, int(c)) for r, c in enumerate(counts)]
        d, _ = local_dimension(balls, 10 ** 9, r2_threshold=-np.inf)
        worst = max(worst, abs(d - float(fit_weights(MAX_RADIUS)
                                         @ np.log(counts))))
    good = worst < 1e-9
    ok &= good
    print(f"  worst |local_dimension - w . ln B| = {worst:.2e}  "
          f"{'OK' if good else 'FAIL'}")

    print("[2] a bump planted at one radius is attributed to that radius")
    ps = np.geomspace(0.02, 0.5, 30)
    x = np.log(ps)
    rows = []
    for s in range(3):
        for p in ps:
            row = {"p": p, "seed": s, "N": 0}
            # background exactly cubic in ln p, so detrending removes it
            # and the planted bump is the only residual anywhere
            lp = np.log(p)
            Lr = [np.log(6.0 * r) + 0.1 * r * lp + 0.01 * r * lp ** 2
                  + 0.001 * lp ** 3 for r in range(1, 11)]
            Lr[4] += 0.02 * np.sin(4.0 * np.log(p))       # planted at r = 5
            for r in range(10):
                row[f"L{r + 1}"] = Lr[r]
            row["d_eff_median"] = float(fit_weights(10) @ np.array(Lr))
            rows.append(row)
    out = analyze(rows, check_banked=False)
    good = out["ok"] and int(np.argmax(out["share"])) == 4 \
        and out["share"][4] > 0.95
    ok &= good
    print(f"  -> planted radius 5 carries share {out['share'][4]:.3f}  "
          f"{'OK' if good else 'FAIL'}")
    print("\nPASS: d(p) decomposition instrument" if ok
          else "\nFAIL: d(p) decomposition instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Per-radius decomposition of the pruned-WS d(p) waveform",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--analyze', type=str, default=None)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds global RNGs; the grid uses the banked '
                         'seeds 0..5, so this is a formality')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.analyze:
        with open(args.analyze) as f:
            analyze(list(csv.DictReader(f)))
        return

    specs = [(float(p), s, N_NODES) for p in banked_ps()
             for s in range(SEEDS)]
    if args.jobs <= 1:
        rows = [measure(sp) for sp in tqdm(specs, desc='grid')]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(tqdm(ex.map(measure, specs), total=len(specs),
                             desc='grid'))
    Path("results").mkdir(exist_ok=True)
    path = f"results/dp_decomposition_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Saved {path}")
    analyze(rows)


if __name__ == '__main__':
    main()
