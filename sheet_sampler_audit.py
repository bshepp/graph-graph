"""
Brute-force audit of `grow_sheet`'s rejection-free sampler (wave 9).

A stale bucket entry is harmless (it is rejected and dropped when drawn), but a valid
event with no live entry can never fire and biases the growth. This instruments a copy
of `grow_sheet` to check, before every draw, that every valid event (attachment,
closure, degree-5 closure, detachment) has a live entry. Small graphs only.

Usage:
    python sheet_sampler_audit.py [--nodes 1500] [--seed 7]
"""
import argparse
import inspect
import math
import random

import numpy as np

import sheet_growth as sg


def audited_grower(counter):
    src = inspect.getsource(sg.grow_sheet)
    loop = "    while n_live < n_nodes:\n"
    assert loop in src
    src = src.replace(loop, loop + "        _CHECK(adj, buckets, token, kmax, P5, DET, five, detach, "
                                   "five_ok, guard, c)\n")

    def check(adj, buckets, token, kmax, P5, DET, five, detach, five_ok, guard, c):
        live = {(k, e) for k, b in enumerate(buckets) for e, t in b if token.get(e) == t}
        for u in range(len(adj)):
            for v in adj[u]:
                if u < v and sg._is_boundary(adj, u, v):
                    du, dv = len(adj[u]), len(adj[v])
                    if du >= c or dv >= c:
                        if any(len(adj[x]) >= c and sg._closable(adj, x, c, guard) for x in (u, v)):
                            counter['missing'] += (kmax, (u, v)) not in live
                    else:
                        counter['missing'] += (du + dv, (u, v)) not in live
                    if five and (five_ok(u) or five_ok(v)):
                        counter['missing'] += (P5, (u, v)) not in live
            if detach and u > 2 and sg._removable(adj, u):
                counter['missing'] += (DET + len(adj[u]), (u, -1)) not in live
        counter['checks'] += 1

    ns = dict(vars(sg))
    ns['_CHECK'] = check
    exec(src, ns)
    return ns['grow_sheet']


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nodes', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    ok = True
    for v, beta, rate in (('sheet', 1.0, 1.0), ('relief', 0.5, 1.0), ('strict', 0.75, 1.0),
                          ('disloc', 0.5, 1.0), ('pent', 1.0, 1e-3), ('eager', 0.5, 4.0),
                          ('detach', 1.0, math.exp(3.0)), ('detach', 0.75, math.exp(0.75 * 3.8))):
        counter = {'missing': 0, 'checks': 0}
        audited_grower(counter)(a.nodes, 6, beta, a.seed, variant=v, rate=rate)
        good = counter['missing'] == 0
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {v} beta {beta} rate {rate:.3g}: "
              f"{counter['checks']} draws audited, {counter['missing']} valid events missing")
    print('\nAUDIT', 'PASSED' if ok else 'FAILED')


if __name__ == '__main__':
    main()
