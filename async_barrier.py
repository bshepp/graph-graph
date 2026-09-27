"""
The bootstrapping barrier under ASYNCHRONOUS updates -- step 5 of the
Lorentzian ladder (LORENTZIAN_SPIKE.md sec 5-6; pre-registration
docs/superpowers/specs/2026-09-27-step5-async-barrier-preregistration.md).

Re-runs the banked flagship negative (FINDINGS.md "Quantified barrier";
barrier_scaling.py: extent ~ N^alpha, rewiring alpha ~ 0.13 vs lattice 0.515)
with event-driven Poisson-clock dynamics, paired against the synchronous rule
on the identical start graph, and asks whether alpha survives emergent time.

  Gate 0 (reproduction): the sync arm must reproduce the banked CSV rows.
  Gate 1 (dynamic positive control): `prune` on small_world DOES grow extent;
    both schedules must show it (alpha > 0.5) and agree (|diff| < 0.1), so a
    low async alpha for `triadic` is not an instrument floor.
  Gate 2 (the question): paired per-seed alpha, async vs sync, `triadic` from
    a random graph. Verdict rules are frozen in the pre-registration.

State-graph only: the step-3-retired causal DAG is not involved.

Usage:
    python async_barrier.py --validate
    python async_barrier.py --gate0
    python async_barrier.py --gate1 --jobs 8
    python async_barrier.py --gate1a --jobs 8
    python async_barrier.py --gate2 --jobs 16
    python async_barrier.py --long --jobs 8
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from rules import get_rule
from barrier_scaling import estimate_extent, fit_exponent
from async_engine import run_sequential_multi

T_CRIT_95 = {3: 3.182, 7: 2.365}      # two-sided 95% t, by degrees of freedom
BANKED_CSV = "results/barrier_scaling_20260530_094041.csv"

# Frozen by the pre-registration.
GATE2_NODES = [1000, 2000, 4000, 8000, 16000, 32000]
GATE2_SEEDS = 8
GATE1_NODES = [1000, 2000, 4000, 8000]
GATE1_SEEDS = 4
SWEEPS = 200
INTERVAL = 50
SURVIVE_BOUND = 0.25
INVARIANCE_BAND = 0.05


def start_graph(n: int, topology: str, seed: int) -> nx.Graph:
    """The banked start graph: global RNGs seeded, then `create_initial_graph`."""
    random.seed(seed)
    np.random.seed(seed)
    return create_initial_graph(n, topology=topology, k=6, seed=seed)


def measure(G: nx.Graph, seed: int, t: float) -> Dict[str, float]:
    """
    Extent of `G`, with the estimator's source draws pinned.

    `estimate_extent` picks BFS sources from the global numpy RNG. Re-seeding
    from (seed, t) immediately before the call makes the draws identical for
    the two schedules, so the estimator cannot contribute a schedule effect.
    """
    np.random.seed((seed * 1_000_003 + int(round(t))) % (2 ** 32))
    diam, mean_ecc, lcc = estimate_extent(G)
    return {'diam': diam, 'mean_ecc': mean_ecc, 'lcc_frac': lcc}


def run_sync(rule: str, n: int, topology: str, seed: int, sweeps: int,
             interval: int, keep: Optional[List[nx.Graph]] = None
             ) -> List[Dict]:
    """Synchronous arm: the rule applied as a global sweep, `sweeps` times.

    If `keep` is a list, the final graph is appended to it.
    """
    G = start_graph(n, topology, seed)
    rows = [{'t': 0, **measure(G, seed, 0)}]
    random.seed(seed)
    np.random.seed(seed)
    f = get_rule(rule)
    for step in range(1, sweeps + 1):
        f(G)
        if step % interval == 0:
            state = np.random.get_state()     # measuring must not perturb
            rows.append({'t': step, **measure(G, seed, step)})
            np.random.set_state(state)        # the rule's own RNG stream
    if keep is not None:
        keep.append(G)
    return rows


def run_async(rule: str, n: int, topology: str, seed: int, sweeps: int,
              interval: int, keep: Optional[List[nx.Graph]] = None
              ) -> List[Dict]:
    """
    Asynchronous arm: independent rate-1 Poisson clocks, one event at a time.

    Absolute Poisson time `t` equals sweep-equivalents (mean events per node),
    the step-4 unit. Extent is recorded the first time the clock passes each
    multiple of `interval`, and at the end.
    """
    G0 = start_graph(n, topology, seed)
    rows = [{'t': 0, **measure(G0, seed, 0)}]
    marks = list(range(interval, sweeps, interval))
    n_events = [0]

    def on_event(event_id: int, node: int, rule_idx: int, t: float,
                 G: nx.Graph) -> None:
        n_events[0] = event_id + 1
        if marks and t >= marks[0]:
            m = marks.pop(0)
            rows.append({'t': m, **measure(G, seed, m)})

    G, times, _ = run_sequential_multi(
        G0, [rule], [1.0], max_time=float(sweeps), seed=seed,
        on_event=on_event)
    rows.append({'t': sweeps, **measure(G, seed, sweeps)})
    for r in rows:
        r['events_per_node'] = len(times) / n
    if keep is not None:
        keep.append(G)
    return rows


def _job(spec: Tuple) -> List[Dict]:
    schedule, rule, topology, n, seed, sweeps, interval = spec
    run = run_sync if schedule == 'sync' else run_async
    rows = run(rule, n, topology, seed, sweeps, interval)
    return [{'schedule': schedule, 'rule': rule, 'topology': topology,
             'N': n, 'seed': seed, **r} for r in rows]


def run_grid(rule: str, topology: str, nodes: Sequence[int], seeds: int,
             sweeps: int, interval: int, jobs: int) -> List[Dict]:
    specs = [(sch, rule, topology, n, s, sweeps, interval)
             for n in sorted(nodes, reverse=True) for s in range(seeds)
             for sch in ('sync', 'async')]
    rows: List[Dict] = []
    if jobs <= 1:
        for sp in tqdm(specs, desc=f"{rule}/{topology}"):
            rows.extend(_job(sp))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for out in tqdm(ex.map(_job, specs), total=len(specs),
                            desc=f"{rule}/{topology}"):
                rows.extend(out)
    return rows


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------

def final_extents(rows: List[Dict], schedule: str, t: int
                  ) -> Dict[int, Dict[int, float]]:
    """{seed: {N: diameter at time t}} for one schedule."""
    out: Dict[int, Dict[int, float]] = {}
    for r in rows:
        if r['schedule'] == schedule and int(r['t']) == t:
            out.setdefault(int(r['seed']), {})[int(r['N'])] = float(r['diam'])
    return out


def per_seed_alpha(ext: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    out = {}
    for s, byn in ext.items():
        Ns = np.array(sorted(byn), dtype=float)
        E = np.array([byn[int(n)] for n in Ns], dtype=float)
        out[s] = fit_exponent(Ns, E)[0]
    return out


def pooled_alpha(ext: Dict[int, Dict[int, float]]) -> Tuple[float, float]:
    """The banked statistic: slope of seed-MEAN extent vs N. (alpha, R^2)."""
    Ns = sorted({n for byn in ext.values() for n in byn})
    E = [np.mean([byn[n] for byn in ext.values() if n in byn]) for n in Ns]
    a, r2, _, _ = fit_exponent(np.array(Ns, float), np.array(E, float))
    return a, r2


def mean_ci(x: Sequence[float]) -> Tuple[float, float, float]:
    """(mean, standard error, 95% half-width) by the t-interval."""
    x = np.asarray(x, dtype=float)
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return float(x.mean()), se, T_CRIT_95[len(x) - 1] * se


def verdict_gate2(a_sync: Dict[int, float], a_async: Dict[int, float]
                  ) -> Dict[str, object]:
    """The frozen Gate-2 rules, applied mechanically."""
    seeds = sorted(set(a_sync) & set(a_async))
    d = [a_async[s] - a_sync[s] for s in seeds]
    ma, sea, ha = mean_ci([a_async[s] for s in seeds])
    ms, ses, hs = mean_ci([a_sync[s] for s in seeds])
    md, sed, hd = mean_ci(d)
    survives = (ma + ha) < SURVIVE_BOUND
    lo, hi = md - hd, md + hd
    if -INVARIANCE_BAND <= lo and hi <= INVARIANCE_BAND:
        inv = 'SCHEDULE-INVARIANT'
    elif (lo > 0 or hi < 0) and abs(md) >= INVARIANCE_BAND:
        inv = 'SHIFT'
    else:
        inv = 'UNDERPOWERED'
    return {'alpha_async': (ma, sea, ha), 'alpha_sync': (ms, ses, hs),
            'delta': (md, sed, hd), 'survives': survives, 'invariance': inv,
            'n_seeds': len(seeds)}


def save(rows: List[Dict], tag: str) -> str:
    Path("results").mkdir(exist_ok=True)
    path = f"results/async_barrier_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")
    return path


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def gate0(jobs: int) -> bool:
    """Sync arm vs the banked CSV (seeds 0..2, N = 1000..16000)."""
    from barrier_scaling import run_trajectory
    banked: Dict[Tuple[int, int, int], float] = {}
    with open(BANKED_CSV) as f:
        for r in csv.DictReader(f):
            if r['rule'] == 'triadic':
                banked[(int(r['N']), int(r['seed']), int(r['step']))] = \
                    float(r['diam'])
    print("Gate 0 -- sync arm reproduces the banked barrier rows")
    ok = True
    worst0, worst_final = 0.0, 0.0
    for n in (1000, 2000, 4000, 8000, 16000):
        for s in range(3):
            traj = run_trajectory('triadic', n, s, SWEEPS, INTERVAL,
                                  'random', 6)
            d0 = abs(traj[0]['diam'] - banked[(n, s, 0)])
            df = abs(traj[-1]['diam'] - banked[(n, s, SWEEPS)])
            worst0, worst_final = max(worst0, d0), max(worst_final, df)
            mine = run_sync('triadic', n, 'random', s, SWEEPS, INTERVAL)
            dm = abs(mine[-1]['diam'] - banked[(n, s, SWEEPS)])
            flag = 'OK' if (d0 == 0 and df <= 2 and dm <= 2) else 'FAIL'
            ok &= flag == 'OK'
            print(f"  N={n:6d} seed={s}: banked {banked[(n, s, 0)]:.0f}->"
                  f"{banked[(n, s, SWEEPS)]:.0f}  run_trajectory "
                  f"{traj[0]['diam']:.0f}->{traj[-1]['diam']:.0f}  "
                  f"run_sync final {mine[-1]['diam']:.0f}  {flag}")
    print(f"  worst |step-0 diff| = {worst0:.0f} (required 0); "
          f"worst |final diff| = {worst_final:.0f} (required <= 2)")
    print("  Gate 0:", "PASS" if ok else "FAIL")
    return ok


def gate1(jobs: int) -> bool:
    rows = run_grid('prune', 'small_world', GATE1_NODES, GATE1_SEEDS,
                    SWEEPS, INTERVAL, jobs)
    save(rows, 'gate1')
    print("\nGate 1 -- dynamic positive control: prune on small_world")
    alphas = {}
    for sch in ('sync', 'async'):
        ext = final_extents(rows, sch, SWEEPS)
        a, r2 = pooled_alpha(ext)
        alphas[sch] = a
        means = {n: np.mean([ext[s][n] for s in ext]) for n in GATE1_NODES}
        e0 = final_extents(rows, sch, 0)
        m0 = {n: np.mean([e0[s][n] for s in e0]) for n in GATE1_NODES}
        print(f"  {sch:>5}: alpha = {a:.3f} (R2={r2:.3f})   diameter "
              + "  ".join(f"N={n}: {m0[n]:.0f}->{means[n]:.0f}"
                          for n in GATE1_NODES))
    ok = (alphas['sync'] > 0.5 and alphas['async'] > 0.5
          and abs(alphas['sync'] - alphas['async']) < 0.1)
    print(f"  |alpha_sync - alpha_async| = "
          f"{abs(alphas['sync'] - alphas['async']):.3f} (required < 0.1); "
          f"both > 0.5 required")
    print("  Gate 1:", "PASS" if ok else "FAIL")
    return ok


def _gate1a_job(spec: Tuple[int, int]) -> Dict:
    n, seed = spec
    ks: List[nx.Graph] = []
    ka: List[nx.Graph] = []
    rs = run_sync('prune', n, 'small_world', seed, SWEEPS, SWEEPS, keep=ks)
    ra = run_async('prune', n, 'small_world', seed, SWEEPS, SWEEPS, keep=ka)
    es = {frozenset(e) for e in ks[0].edges()}
    ea = {frozenset(e) for e in ka[0].edges()}
    return {'N': n, 'seed': seed,
            'jaccard_dist': 1.0 - len(es & ea) / len(es | ea),
            'removed_sync': 1.0 - len(es) / (3 * n),
            'removed_async': 1.0 - len(ea) / (3 * n),
            'growth_sync': rs[-1]['diam'] / rs[0]['diam'],
            'growth_async': ra[-1]['diam'] / ra[0]['diam']}


def gate1a(jobs: int) -> bool:
    """
    AMENDED Gate 1 (post hoc -- see the pre-registration's amendment).

    The frozen Gate 1 compared diameters, but pruning a Watts-Strogatz ring
    FRAGMENTS it (lcc_frac 0.3-1.0), and the largest fragment's diameter
    hinges on a handful of edges. The amended control compares what the two
    schedules actually DO -- the final edge sets -- and keeps the requirement
    that both produce large extent growth in every run.
    """
    specs = [(n, s) for n in GATE1_NODES for s in range(GATE1_SEEDS)]
    if jobs <= 1:
        out = [_gate1a_job(sp) for sp in tqdm(specs, desc='gate1a')]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            out = list(tqdm(ex.map(_gate1a_job, specs), total=len(specs),
                            desc='gate1a'))
    save(out, 'gate1a')
    print("\nGate 1A (amended) -- prune on small_world, edge-set agreement")
    for r in out:
        print(f"  N={r['N']:5d} seed={r['seed']}: jaccard distance "
              f"{r['jaccard_dist']:.4f}  edges removed sync/async "
              f"{r['removed_sync']:.4f}/{r['removed_async']:.4f}  diameter "
              f"growth x{r['growth_sync']:.0f} / x{r['growth_async']:.0f}")
    worst_j = max(r['jaccard_dist'] for r in out)
    min_g = min(min(r['growth_sync'], r['growth_async']) for r in out)
    ok = worst_j <= 0.01 and min_g >= 5.0
    print(f"  worst jaccard distance = {worst_j:.4f} (required <= 0.01); "
          f"smallest diameter growth = x{min_g:.1f} (required >= 5)")
    print("  Gate 1A:", "PASS" if ok else "FAIL")
    return ok


def report_gate2(rows: List[Dict]) -> Dict[str, object]:
    print("\nGate 2 -- triadic from a random graph, async vs sync (paired)")
    nodes = sorted({int(r['N']) for r in rows})
    exts = {sch: final_extents(rows, sch, SWEEPS) for sch in ('sync', 'async')}
    for sch in ('sync', 'async'):
        ext = exts[sch]
        a, r2 = pooled_alpha(ext)
        print(f"  {sch:>5} pooled alpha = {a:.3f} (R2={r2:.3f});  mean "
              "diameter " + "  ".join(
                  f"{n}:{np.mean([ext[s][n] for s in ext]):.1f}"
                  for n in nodes))
        lcc = np.mean([float(r['lcc_frac']) for r in rows
                       if r['schedule'] == sch and int(r['t']) == SWEEPS])
        print(f"        mean lcc_frac at t={SWEEPS}: {lcc:.3f}")
    epn = [float(r['events_per_node']) for r in rows
           if r['schedule'] == 'async' and int(r['t']) == SWEEPS]
    print(f"  async events/node = {np.mean(epn):.1f} (target {SWEEPS})")

    a_s, a_a = per_seed_alpha(exts['sync']), per_seed_alpha(exts['async'])
    print("  per-seed alpha (sync, async): " + "  ".join(
        f"s{s}:({a_s[s]:.3f},{a_a[s]:.3f})" for s in sorted(a_s)))
    v = verdict_gate2(a_s, a_a)
    for key in ('alpha_sync', 'alpha_async', 'delta'):
        m, se, h = v[key]
        print(f"  {key:>11}: {m:+.3f} +- {se:.3f} (SE)   95% CI "
              f"[{m - h:+.3f}, {m + h:+.3f}]")
    print(f"  VERDICT (frozen rules, {v['n_seeds']} paired seeds): barrier "
          f"{'SURVIVES' if v['survives'] else 'NOT ESTABLISHED'} under async "
          f"(bound {SURVIVE_BOUND}); schedule dependence: {v['invariance']}")
    print("  trajectory (mean diameter by time):")
    for sch in ('sync', 'async'):
        for n in (nodes[0], nodes[-1]):
            ts = sorted({int(r['t']) for r in rows})
            line = "  ".join(
                f"t={t}:" + "{:.1f}".format(np.mean(
                    [float(r['diam']) for r in rows if r['schedule'] == sch
                     and int(r['N']) == n and int(r['t']) == t]))
                for t in ts)
            print(f"    {sch:>5} N={n:6d}  {line}")
    return v


def _validate(quick: bool) -> bool:
    """Known-answer checks on the reduction + a small end-to-end run."""
    ok = True
    print("[1] exponent recovery on synthetic extents")
    Ns = np.array([1000, 2000, 4000, 8000], float)
    for true in (0.5, 0.13):
        ext = {s: {int(n): 3.0 * n ** true for n in Ns} for s in range(8)}
        a = per_seed_alpha(ext)
        good = all(abs(v - true) < 1e-9 for v in a.values())
        ok &= good
        print(f"  alpha={true}: recovered {np.mean(list(a.values())):.4f} "
              f"{'OK' if good else 'FAIL'}")

    print("[2] frozen verdict rules on constructed cases")
    rng = np.random.default_rng(0)
    base = {s: 0.13 + 0.01 * rng.standard_normal() for s in range(8)}
    same = {s: base[s] + 0.005 * rng.standard_normal() for s in range(8)}
    moved = {s: base[s] + 0.30 + 0.005 * rng.standard_normal()
             for s in range(8)}
    noisy = {s: base[s] + 0.2 * rng.standard_normal() for s in range(8)}
    cases = (("identical arms", same, True, 'SCHEDULE-INVARIANT'),
             ("async +0.30", moved, False, 'SHIFT'),
             ("noisy async", noisy, None, 'UNDERPOWERED'))
    for name, arm, want_surv, want_inv in cases:
        v = verdict_gate2(base, arm)
        good = v['invariance'] == want_inv and (
            want_surv is None or v['survives'] == want_surv)
        ok &= good
        print(f"  {name:>15}: survives={v['survives']} "
              f"invariance={v['invariance']}  {'OK' if good else 'FAIL'}")

    print("[3] measurement does not perturb the sync rule's RNG stream")
    a = run_sync('triadic', 300, 'random', 0, 20, 5)
    b = run_sync('triadic', 300, 'random', 0, 20, 20)
    G1 = start_graph(300, 'random', 0)
    random.seed(0)
    np.random.seed(0)
    f = get_rule('triadic')
    for _ in range(20):
        f(G1)
    ref = measure(G1, 0, 20)
    good = a[-1] == b[-1] and a[-1]['diam'] == ref['diam'] \
        and a[-1]['lcc_frac'] == ref['lcc_frac']
    ok &= good
    print(f"  interval 5 vs 20 vs unmeasured: {'OK' if good else 'FAIL'}")

    print("[4] async arm: matched budget and both arms share the start graph")
    n = 300 if quick else 600
    s_rows = run_sync('triadic', n, 'random', 1, 40, 20)
    a_rows = run_async('triadic', n, 'random', 1, 40, 20)
    epn = a_rows[-1]['events_per_node']
    good = (s_rows[0]['diam'] == a_rows[0]['diam']
            and abs(epn - 40) < 2.0
            and [r['t'] for r in a_rows] == [0, 20, 40])
    ok &= good
    print(f"  events/node = {epn:.1f} (target 40); t-grid "
          f"{[r['t'] for r in a_rows]}  {'OK' if good else 'FAIL'}")

    print("[5] the instrument can see extent growth (prune on small_world)")
    s_rows = run_sync('prune', n, 'small_world', 0, 200, 200)
    a_rows = run_async('prune', n, 'small_world', 0, 200, 200)
    good = (s_rows[-1]['diam'] > 3 * s_rows[0]['diam']
            and a_rows[-1]['diam'] > 3 * a_rows[0]['diam'])
    ok &= good
    print(f"  sync {s_rows[0]['diam']:.0f}->{s_rows[-1]['diam']:.0f}, "
          f"async {a_rows[0]['diam']:.0f}->{a_rows[-1]['diam']:.0f}  "
          f"{'OK' if good else 'FAIL'}")

    print("\nPASS: async barrier instrument" if ok
          else "\nFAIL: async barrier instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Bootstrapping barrier under asynchronous updates (step 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--gate0', action='store_true')
    ap.add_argument('--gate1', action='store_true')
    ap.add_argument('--gate1a', action='store_true',
                    help='amended Gate 1 (edge-set agreement)')
    ap.add_argument('--gate2', action='store_true')
    ap.add_argument('--long', action='store_true',
                    help='descriptive rider: 800 sweeps at N=1000, 4000')
    ap.add_argument('--report', type=str, default=None,
                    help='re-run the Gate-2 reduction on a saved CSV')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds global RNGs; the gates use frozen internal '
                         'seeds (0..7), so this is a formality')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(args.quick) else 1)
    if args.report:
        with open(args.report) as f:
            report_gate2(list(csv.DictReader(f)))
        return
    if args.gate0:
        raise SystemExit(0 if gate0(args.jobs) else 1)
    if args.gate1:
        raise SystemExit(0 if gate1(args.jobs) else 1)
    if args.gate1a:
        raise SystemExit(0 if gate1a(args.jobs) else 1)
    if args.gate2:
        rows = run_grid('triadic', 'random', GATE2_NODES, GATE2_SEEDS,
                        SWEEPS, INTERVAL, args.jobs)
        save(rows, 'gate2')
        report_gate2(rows)
        return
    if args.long:
        rows = run_grid('triadic', 'random', [1000, 4000], 4, 800, 100,
                        args.jobs)
        save(rows, 'long')
        print("\nLong-budget rider (mean diameter by time):")
        for sch in ('sync', 'async'):
            for n in (1000, 4000):
                ts = sorted({int(r['t']) for r in rows})
                print(f"  {sch:>5} N={n}: " + "  ".join(
                    f"t={t}:" + "{:.1f}".format(np.mean(
                        [float(r['diam']) for r in rows
                         if r['schedule'] == sch and int(r['N']) == n
                         and int(r['t']) == t])) for t in ts))
        return
    ap.print_help()


if __name__ == '__main__':
    main()
