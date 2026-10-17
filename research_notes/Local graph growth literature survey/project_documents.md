# graph-graph: inventory of the project's own claims, measurements, retractions and open items

Corpus: local files only, read in full as of 2026-10-16 (HEAD = 6a8ca11, 2026-10-14). Every
"source" below is `file:line` (or `file:line-line`) in `F:\science-projects\graph-graph`.
Line numbers refer to the files as they stand at that commit. Specs live in
`docs/superpowers/specs/` and are cited by their file name only.

Naming used throughout: FINDINGS = `FINDINGS.md`; BRANCHES = `BRANCHES.md`; SPIKE =
`LORENTZIAN_SPIKE.md`; DC = `DIMENSIONAL_COHERENCE.md`; spec `X` = `docs/superpowers/specs/X`.

Status vocabulary for section 2: **HELD** (stands as stated), **REFUTED** (a pre-registered
prediction failed), **SUPERSEDED** (a measurement that stands as data but whose
interpretation was replaced), **WITHDRAWN** (the authors retracted the statement
themselves), **UNDECIDED** (frozen reading could not be made).

---

## 1. Exact definitions of the generators, rules and instruments as implemented

### Takeaway

The project has two growth generators (`grown`, a degree-capped random 2-tree that is
tree-like at every scale; `sheet`, a manifold-respecting, wheel-closing frontier growth with
coordination-dependent Poisson rates that assembles a patch of the triangular lattice), one
hand-built flat reference (`triangular`), eight synchronous local update rules (three
state/weight rules, five topology rules), an asynchronous Poisson-clock engine for a subset
of them, and a ball-growth dimension estimator that since 2026-09-28 gates on window
stability. Everything is coordinate-free; node identity is growth order.

### Cited Findings

**1.1 `grown` (degree-capped frontier growth)** -- `simulation.py:21-57`

- Seed: a triangle on nodes 0,1,2 (`simulation.py:39`). Loop: choose a uniformly random
  entry of a *frontier list* of edges; if either endpoint has degree >= `degree_cap`
  (default 6; the CLI's `k`) the entry is swap-removed and the step is retried; otherwise a
  new node `w` is added adjacent to both endpoints `u, v`, and `(w,u)`, `(w,v)` are appended
  to the frontier (`simulation.py:41-55`). The chosen edge `(u,v)` is **not** removed, so one
  edge can receive several attachments; the audit notes this "puts three triangles on an
  edge and branches the surface" (`sheet_growth.py:13-14`).
- Structural consequence stated by the project: every step glues a triangle onto an edge,
  so the graph is a partial 2-tree (treewidth <= 2), "tree-like at every scale above a few
  hops, however triangle-rich it is locally. Nothing in the rule makes two growing arms meet,
  so the frontier stays proportional to the volume" -- FINDINGS:764-768. Live frontier is a
  constant 37% of N at every size (FINDINGS:761-762). Mean degree ~4 regardless of cap; the
  cap bounds the tail, not the mean (SPIKE:238-239).
- Growth extinction: stops when every frontier edge has an endpoint at the cap;
  `_grow_dimensional` returns the small graph silently and `create_initial_graph` now warns
  (`simulation.py:93-99`). Extinction probability: cap 4: 0.474 +- 0.009 (sizes 5-6); cap 5:
  0.279 +- 0.008 (survival plateaus 0.718 beyond ~600 nodes, heavy tail 6-537); cap 6:
  0.0082 +- 0.0006 (164/20000, sizes 7-20); caps 7, 8: 0.0002 (FINDINGS:814-822). Decided in
  the first few dozen attachments (FINDINGS:822-823).
- The docstring still says "Effective dimension rises with the cap (cap 6 -> ~2D ... cap 8 ->
  ~3D)" (`simulation.py:33-35`) -- see section 5.

**1.2 `sheet` (wheel-closing frontier growth with coordination-dependent rates)** --
`sheet_growth.py`

- State: adjacency only (list of sets). Seed triangle `{0,1,2}` (`sheet_growth.py:88`).
- **Boundary edge**: an edge `(u,v)` whose endpoints have exactly one common neighbour, i.e.
  the edge lies in exactly one triangle (`_is_boundary`, `sheet_growth.py:52-53`). Interior
  edges lie in two triangles; `census` counts any edge in 0 or >= 3 triangles as a
  "non-manifold edge" (`sheet_growth.py:236-241`), and the validator requires zero of them
  (`sheet_growth.py:349-355`).
- **Attachment event** at boundary edge `(u,v)` with `deg u < c` and `deg v < c`: add a new
  node `w` adjacent to `u` and `v` (`sheet_growth.py:148-154`). Rate class = `deg u + deg v`;
  weight `exp(beta * (deg u + deg v - 2c))` (`sheet_growth.py:89-90`; reference
  implementation `sheet_growth.py:183`). Up to the common factor `exp(-2 beta c)` this is the
  docstring's "attach at (u,v): exp(beta * (deg u + deg v))" (`sheet_growth.py:20-24`).
- **Closure event**: a boundary edge with an endpoint of degree >= `c` can *only* fire a
  closure (`sheet_growth.py:102-104`, `137-147`), and only if some endpoint `x` with
  `deg x >= c` is **closable**. `_closable(adj, x, c)` (`sheet_growth.py:56-66`) returns the
  link ends `(a, b)` iff all of: (i) `x` has exactly two boundary neighbours `a, b` (the ends
  of its link path); (ii) `a` and `b` are not adjacent; (iii) `adj[a] & adj[b] == {x}`
  (otherwise "would pinch the surface"); (iv) `deg a < c` and `deg b < c` ("keeps every degree
  <= c"). The closure adds edge `a--b`, completing `x`'s wheel; `x` becomes interior. Rate
  class `kmax = 2c`, weight 1 = `exp(beta * (2c - 2c))`, i.e. the docstring's "close at x:
  exp(beta * 2c)" -- a closure fires at the rate of the maximal-degree attachment class.
- **Sampler**: rejection-free bucketed draw (class chosen with probability proportional to
  rate x population, then uniform within the class; stale entries invalidated by per-edge
  tokens) -- `sheet_growth.py:76-82`, `121-135`. Validated against an O(E)-per-event
  recompute-everything reference at N = 300, 40 seeds, |z| < 3 on boundary fraction
  (`sheet_growth.py:162-200`, `358-369`). This is a continuous-time Markov process with
  independent exponential clocks, i.e. a kinetic Monte Carlo / Gillespie growth model.
- **Invariants**: max degree <= c; manifold; deterministic per seed (`sheet_growth.py:346-356`).
  Interior vertices have degree exactly 6 at c = 6 in every run (fraction 1.000,
  FINDINGS:1086-1087): "a patch of the triangular lattice, assembled without coordinates"
  (FINDINGS:1087-1089).
- **Stall**: if no live event exists the returned graph is smaller than requested
  (`sheet_growth.py:84-86`, `126-127`). c = 5 stalls at exactly 12 nodes in 5 of 5 seeds --
  the icosahedron (FINDINGS:1075-1076; validator `sheet_growth.py:371-375`).
- **Curvature knob**: "`c` sets the curvature: 6 is flat, 7 is negatively curved, 5 closes
  up" (`sheet_growth.py:18-19`). c = 7 grows exponentially "whatever the rates do"
  (FINDINGS:1076-1077).
- **beta**: "beta = 0 is uniform growth; large beta is compact growth" (`sheet_growth.py:24-25`).
  At beta = 0 the sheet is tree-like like `grown` (FINDINGS:1062).
- **Dead corner / blocked closure** (the crossover trigger, 2026-10-14): a boundary node of
  degree >= c for which `_closable` returns None (spec
  `2026-10-13-blocked-closure-preregistration.md:32-35`; FINDINGS:1755-1761). "Degrees never
  fall and adjacency never shrinks, so a degree-c boundary node that is not closable is dead
  forever: no event will ever fire on its edges, it stays on the boundary while the disc grows
  past it, and it becomes the bottom of a fjord whose walls keep growing" (FINDINGS:1758-1761).
  In every crossed seed the first dead corner is a *pair of adjacent nodes* born within a few
  thousand nodes of each other and within ~1% of N_first -- "the two link ends reach the cap
  together" (FINDINGS:1802-1804).
- **Registered topology**: `create_initial_graph(..., 'sheet')` uses `grow_sheet(n, c=k,
  beta=3.0)` with beta hard-wired to 3 (`simulation.py:82-89`). Generation cost ~10 min per
  512,000 nodes (BRANCHES:47); 4 M nodes ~1.5 h and ~5 GB per seed (FINDINGS:1405-1406).

**1.3 `triangular` and the other hand-built topologies** -- `simulation.py:60-101`

- `triangular`: `nx.triangular_lattice_graph(side, 2*side)`, `side = floor(sqrt(n)) - 1`;
  "degree 6, every edge in two triangles" (`simulation.py:75-81`).
- `lattice`: 2D square grid, `n` rounded down to a perfect square (`simulation.py:69-72`).
- `small_world`: Watts-Strogatz(n, k=6, p=0.1) (`simulation.py:65-66`); `scale_free`:
  Barabasi-Albert m = k//2 = 3 (`:67-68`); `random`: Erdos-Renyi with p = k/n, mean degree 6
  (`:73-74`).
- Initial state: each node `active` with prob 0.1, `state` uniform in {0,1}; every edge
  `weight = 0.5` (`simulation.py:103-110`).

**1.4 The eight synchronous local rules** -- `rules.py`; applied in sequence once per step
(`simulation.py:145-148`); registry `RULES` at `rules.py:405-414`.

| key | mechanism (defaults) | source |
|---|---|---|
| `activation` | SIS: an active node stays active with prob 0.9 (decay 0.1); an inactive node with m active neighbours activates with prob 1 - 0.7^m | `rules.py:13-46` |
| `reinforcement` | Hebbian: every edge weight -= 0.01, += 0.1 if both endpoints active, clamped to [0.01, 1.0]. The weight is read by no other rule | `rules.py:49-83` |
| `majority` | node adopts `argmax` of neighbour state counts (2 states); with prob 0.01 a random state. `argmax` tie-break is deterministic toward state 0, so the rule is **not Z2-symmetric** | `rules.py:86-120`; FINDINGS:2088-2098 |
| `rewire` | each edge with prob 0.01: move the v-end to a uniformly random node (not u, not already adjacent, <= 10 tries). Target is global, so this rule is the deliberate *non-local* control | `rules.py:123-190` |
| `prune` | each edge with prob 0.05: if neither endpoint has degree <= 2 (`min_degree`) and the endpoints share < 1 common neighbour (`min_overlap`), remove it. Self-terminating | `rules.py:193-232` |
| `triadic` | each edge (u,v) with prob 0.05, if deg v > 2: pick a random neighbour x of u and a random neighbour w of x; if w != u and not adjacent, rewire (u,v) -> (u,w). Edge count conserved | `rules.py:235-279` |
| `geometrize` | each node with prob 0.2: if degree > 6 shed the edge to the neighbour with the fewest common neighbours (among neighbours with degree > 3); if degree < 6 add a friend-of-friend edge. NetworkX backend only | `rules.py:282-343` |
| `ricci` | Forman-Ricci F = 4 - deg u - deg v + 3 x (#triangles on the edge); each edge with prob 0.05, if it has zero common neighbours and deg v > 2, rewire its v-end to a random node two hops from u. Positive-curvature edges untouched, "so curvature only increases". NetworkX only | `rules.py:346-401` |

- Locality invariant: "every function in rules.py may only read a node's own state and its
  immediate neighbors (and incident edge weights). No global reductions, no distance-r>1
  lookups" (`AGENTS.md:19-21`). Note `triadic`/`ricci`/`geometrize` *write* at distance 2
  (friend-of-friend), which is why their async conflict radius is 3 (SPIKE:249-258).

**1.5 Asynchronous engine** -- `async_engine.py` (described in SPIKE)

- Each node holds an independent exponential clock (Harris construction); `argmin` selects
  the next event (spec `2026-07-29-causal-dag-static-calibration-design.md:79-88`). Multi-rule
  runs use one clock per (node, rule), merged by `argmin` (spec
  `2026-08-02-async-censorship-checkpoint-design.md:65-76`).
- Conflict-free batching: radius 1 for state rules, 2 for `prune`, 3 for
  `triadic`/`ricci`/`geometrize`; rounds per sweep 7 / 24 / 58 after degree thinning
  (acceptance `(c(v)+1)/(c_max+1)`, needed because naive batching under-samples high-degree
  nodes and shifted steady-state activation 11%, z = 5.9) -- SPIKE:241-283. `rewire` is not
  an async event (SPIKE:350-352); `geometrize`/`ricci` are not either (FINDINGS:922).
- Time unit: absolute Poisson time with every clock at rate 1 = one sweep-equivalent
  (spec `2026-08-02-...:160-179`).

**1.6 The dimension estimator** -- `dimension.py`

- `ball_sizes`: cumulative |B(v,r)| for r = 1..max_radius by BFS (`dimension.py:27-47`).
- `local_dimension` (`dimension.py:87-203`): trim to radii with |B| < 0.1 N
  (`saturation_frac`, `:167-168`); **Gate 1**: >= 6 unsaturated radii (`min_radii`, `:171`);
  fit `log|B| = d log r + c + a/r` by least squares (`:183-185`); **Gate 2**: R^2 >= 0.95
  (`:193`); **Gate 3** (added 2026-09-28, owner-approved): if >= 8 unsaturated radii, the slope
  over radii 1..n and over 1..k, k = max(5, ceil(n/2)), must differ by <= 0.25 (`DRIFT_TOL`,
  `:50`, `:197-201`). `drift_tol=None` restores the old behaviour (`:147`).
- Field level (`dimension_stats`, `dimension.py:368-468`): `median_drift` over all testable
  nodes; `window_stable` = True iff |median_drift| <= 0.10 (`FIELD_DRIFT_TOL`, `:52`), None if
  fewer than half the sampled nodes could be tested (`:414-425`). "A d_eff from a field that
  is not window-stable is a reading of one fit window, not a dimension" (`:395-396`).
- Known blind spot: "At max_radius 10 it catches exponential growth of base >~ 1.2 ...
  growth of base ~1.1 is indistinguishable from a power law inside that window and passes"
  (`dimension.py:149-153`; FINDINGS:862-866).
- `window_stability.py` audit: median d_eff over fit windows R6..R40; verdict DIMENSION
  DEFINED iff drift (max - min) <= 0.15 **and** the power-law + 1/r fit beats an
  exponential + 1/r fit on RSS; EXPONENTIAL iff drift > 0.15 and the exponential wins; else
  MIXED (`window_stability.py:19-26`, `128-138`). Known-answer controls: 2D/3D tori, and a
  subdivided random 3-regular graph (exponential by construction, base 2^(1/6) = 1.1225)
  (FINDINGS:737-742).
- Finite-size correction rationale: a plain log-log fit reads d ~ 1.6 for a 2D grid and 2.3
  for 3D at reachable radii; the 1/r regressor recovers truth to ~0.05 at radius 6 (DC:132).
- Spectral dimension (`spectral_flow.py`, frozen in spec
  `2026-09-27-spectral-flow-preregistration.md:24-51`): lazy walk `S = (I + D^-1/2 A
  D^-1/2)/2`; node-averaged return probability by Hutchinson trace with Rademacher probes,
  `P(2k) = E||S^k z||^2 / N`; `P~(t) = P(t) - 1/N`; grid `t_j = 16 * 2^(j/2)`; `D(t_j) = -2 x`
  slope of `ln P~` vs `ln t` over `[t_j/sqrt2, t_j sqrt2]`; admissible if `P~ >= 20/N` and
  `t_j sqrt2 <= 20000`; plateau = >= 5 consecutive points with range <= 0.2; verdicts
  INSUFFICIENT RANGE / FLAT / PLATEAU-TO-PLATEAU FLOW (two plateaus differing by > 0.3) /
  RUNAWAY (total change > 0.5, Spearman > 0.9, no plateau in last factor 4) / UNCLASSIFIED.
- Causal-set instruments (`causal_sets.py`, `causal_dag.py`): Myrheim-Meyer ordering
  fraction `r(d) = Gamma(d+1) Gamma(d/2) / (2 Gamma(3d/2))` (SPIKE:104); midpoint scaling
  (Bombelli): `N_half ~ N / 2^d` (SPIKE:142-147); interval scaling `|I| ~ l^d` demoted to a
  diagnostic (SPIKE:126-139). Event DAG: event `(v, n)` reads closed neighbourhood; parents =
  latest event at each read node (SPIKE:26-36; spec `2026-07-29-...:77-98`).

**1.7 Sheet-specific observables** (all coordinate-free)

- Boundary length B(N) = number of boundary edges; local boundary exponent
  `e(N_i) = ln(B(N_{i+1})/B(N_i)) / ln 2` at factor-2 checkpoints, compact 0.5, tree-like 1.0
  (spec `2026-09-28-sheet-beta-transition-preregistration.md:29-33`). Compact disc reads
  `B / sqrt(N) = 3.75-3.85` at every size (FINDINGS:1373-1374).
- Crossover size N*: where `e` crosses 0.75 going up and stays above, linearly interpolated
  in ln N between bracketing midpoints; censored if never (spec `...beta-transition:33-36`).
  N*_D: diameter exponent falls below 0.3 and stays; N*_A: arm fraction (nodes within 10 hops
  of a boundary edge) slope rises above -0.25 and stays (spec
  `2026-09-28-sheet-crossover-4M-preregistration.md:29-35`).
- Arm (item 1): connected component of boundary nodes with root distance > 1.5 R(N),
  R = median root distance of the boundary; counts if >= 50 nodes (spec
  `2026-10-13-round2-preregistrations.md:33-39`).
- Edge height profile `h(s) = rho(s) - R` along the boundary cycle; periodogram in log bins
  of >= 8 modes; peak = bin excess >= 3.0 over a power-law fit; k = 1-3 excluded as global
  shape modes (spec `2026-10-13-edge-spectrum-preregistration.md:24-46`).
- Throat ensemble (portal/critical-collapse program): two balls `B(c1, r)`, `B(c2, r)` with
  `dist(c1, c2) >= 2r + 6`; A strands drawn uniformly from `B1 x B2`; capacity `|B1||B2|`;
  density `a = A / capacity`; peeling fixed point = iteratively strip strands with < 1 common
  neighbour (spec `2026-08-17-throat-criticality-design.md:31-62`).

### Inferences

- In statistical-physics terms `sheet` is a kinetic Monte Carlo growth of a triangulated
  disc with bond-counting (coordination) energetics: an attachment at `(u,v)` has rate
  `exp(beta * (deg u + deg v))`, so beta plays the role of an inverse temperature multiplying
  a "number of existing bonds at the site" energy. The closure step is a *constrained* move
  (needs both link ends below the cap); the dead corner is a kinetic trap of that
  constraint. This is the model class the literature agents should match against
  (Eden / solid-on-solid / KMC crystal growth with a hard coordination cap, growth of
  triangulations by local moves).
- `grown` is a random 2-tree with a degree cap and uniform edge choice -- in the literature's
  terms a stacked/Apollonian-type triangle-gluing process without planarity, which the
  project itself characterises as treewidth <= 2 and logarithmic diameter.
- `rewire` is the one rule that violates locality (uniform global target); the project uses
  it as the destructive control, not as a candidate for emergence.

### Gaps

- The project does not derive or test which configurations of degrees actually produce a
  dead corner (condition (iv) vs (ii)/(iii)); FINDINGS states the degree guard is the
  operative one ("the rule's own degree guard then forbids the closure", FINDINGS:1843-1844)
  but no census distinguishes the four `_closable` failure modes.
- No document states the effective "energy" units of beta beyond the rate formula; the
  mapping to exp(5.4 beta) is explicitly unexplained (FINDINGS:1388, 1851-1853).

---

## 2. Inventory of banked results (chronological), with status and governing pre-registration

### Takeaway

Across ~40 banked items, the durable positives are: the window-stable 2D `sheet` (two knobs,
finite size range), the exact peeling-fixed-point mechanism for throat survival, the
schedule-invariance of censorship and the barrier under async updates, and the measured
obstruction exponent alpha ~ 0.13-0.17 for fixed-N rewiring. The major retractions all trace
to one instrument fault (a radius-10 power-law fit on slowly exponential ball growth): every
`grown`-is-2D statement, the 1->2 `prune` knob, grown's coherence, portal "dimension
inflation", and the quantum "structural portal" result. The 2026-10-14 dead-corner trigger is
the most recent mechanism result and has a sharp, untested prediction attached.

### Cited Findings

Pre-pre-registration era (no spec; dated by git log `c22dfce 2026-06-27` and earlier):

1. **Instrument: finite-size-corrected, gated estimator** -- claim: `log|B| = d log r + c +
   a/r` recovers 1D/2D/3D within ~0.05; undefined on expanders (FINDINGS:14-34). **HELD**,
   extended by Gate 3 on 2026-09-28 (FINDINGS:831-839). Validator now 15 checks (FINDINGS:839).
2. **Rule-effect table** (FINDINGS:36-45): `rewire` destroys lattice d~2 -> undefined in ~30
   steps; `prune` on small-world -> d~1, defined_frac 0 -> 100%; `triadic` raises clustering
   0.002 -> 0.625 but dimension stays undefined. **HELD** for those three rows. Row "grown:
   tunable emergent d" **SUPERSEDED** 2026-09-26 (FINDINGS:44, 62).
3. **No global attractor; bistability under `geometrize(target=6)`** (FINDINGS:70-92):
   lattice stable fixed point (d~2); random / scale-free stuck at defined_frac ~0; small-world
   geometrizes but d drifts; `grown` cap 10 fragments (lcc 100% -> ~13%); target degree does
   not tune (4 -> 2.0, 6 -> 2.2, 8 -> destroyed). **HELD** (never re-run under the gate; the
   lattice row is window-stable by construction).
4. **Bootstrapping barrier** (FINDINGS:94-167): "fixed-N local rewiring cannot grow extent
   (diameter) from an expander" (FINDINGS:133). `ricci` on a random graph: clustering 0.003 ->
   0.43, mean Forman curvature -9.8 -> -6.6, eccentricity 6 -> 7 -> 8 at steps 0/80/480
   (FINDINGS:115-119). Scope limited to fixed-N local edge moves (FINDINGS:147-155). **HELD**;
   the audit *extends* it: frontier growth (`grown`) also fails to make polynomial extent
   (FINDINGS:792-796).
5. **Quantified barrier** (step 2; FINDINGS:2020-2055; `barrier_scaling.py`, N 1000-16000,
   200 steps, 3 seeds): alpha = 0.515 lattice (R^2 1.00), 0.083 expander baseline, 0.127
   triadic (R^2 0.53), 0.136 geometrize, 0.147 ricci, 0.194 grown. **HELD** for the rewiring
   rows. Grown row's reading ("polynomial but globally compressed", "diameter ~ N^0.19")
   **SUPERSEDED**: diameter is logarithmic (FINDINGS:785-786, 2057-2065). Rider 2026-09-27:
   extent under `triadic` peaks near 300 sweeps and then falls below start (N = 1000: 8.5 ->
   11.5 -> 4.0), so alpha ~ 0.13 is "a statement about a 200-step budget" and "should not be
   quoted as an asymptotic quantity" (FINDINGS:914-920).
6. **cap -> d finite-size scaling** (step 1; FINDINGS:1985-2018): fixed-radius d_eff plateaus
   at 2.2 / 3.0 / 3.6 for caps 6/7/8 by N ~ 1e4 / 2e4 / 1e5. **SUPERSEDED** 2026-09-26: "the
   convergence of a radius-10 reading" (FINDINGS:777-779, 1987-1988).
7. **`prune` d(p) crossover, not transition** (step 3, 2026-06-27; FINDINGS:2100-2169):
   median d_eff 1.00 / 0.98 / 1.04 / 1.52 / 1.91 / 2.22 / 2.26 / 2.16 / 2.07 at p = 0.05 ...
   0.80 (N = 3.2e4); across-N spread 0.063; peak slope 4.80 / 4.97 / 4.79 at N = 2e3/8e3/3.2e4;
   ER control at matched degree undefined. "Crossover, not critical point" **HELD** and
   strengthened (FINDINGS:783-784). "Third continuum dimension knob 1 -> 2" **SUPERSEDED**:
   d = 1 is real only for p <~ 0.1-0.15; above p ~ 0.2 the pruned graph is a small world
   (FINDINGS:780-784; gated table FINDINGS:1463-1484: window-stable 3/3 to p = 0.15, 2/3 at
   0.20, 0/3 from 0.30; defined_frac 0.99 -> 0.56 across p = 0.05-0.2 while still d = 1).
8. **Preservation on `grown`** (FINDINGS:169-225; N = 1e4, 200 steps): prune 99.6% -> 99.3%,
   diameter 42 -> 42; triadic -> ~46%, diameter 22, lcc 0.48; rewire -> ~2%, diameter 14, lcc
   0.95. **SUPERSEDED** 2026-09-29 (notice FINDINGS:171-174); qualitative pattern **HELD** on
   real substrates (item 33).
9. **Spatial coherence on `grown`** (FINDINGS:227-278): Moran's I = 0.881, z 86.6, defined
   0.99, edge_frac 0.99 (N = 5000, cap 6); lattice I = 0.958. **SUPERSEDED** 2026-10-13:
   gated `grown` reads defined 0.27, edge_frac 0.14, "field fragmented (N/A)"; lattice I =
   0.932 unchanged (FINDINGS:1451-1461). Validator (gradient +0.97, checkerboard -1.00,
   random ~0) **HELD**.
10. **FSS pipeline validated on majority-vote/Ising** (FINDINGS:2067-2098): Binder crossing
    q_c ~ 0.08 vs literature ~0.075; collapse with beta/nu = 1/8, 1/nu = 1. **HELD**. Side
    finding: project `majority` is Z2-biased (argmax -> state 0), shows no clean transition
    (FINDINGS:2088-2098).
11. **Log-periodicity / DSS scan** (2026-08-11; FINDINGS:472-516; method pre-registered in
    the module): global null; bounds ~10% amplitude on grown extent scaling, ~10-20% on ball
    growth; d(p) at n = 10 "a90 = inf, no claim". **HELD**. Methodology lesson: permutation
    test on max periodogram power fires on any smooth misfit (FINDINGS:507-513).

Pre-registered era (spec named per item):

12. **Portal tolerance** (2026-07-16; FINDINGS:2175-2211; module-docstring predictions):
    on `grown` cap 6, defined_frac stays 0.97-1.00 to k = 400 portals; d_eff 2.21 -> 3.19;
    Moran I 0.881 -> 0.668; mean pair distance 20.3 -> 12.9; placement saturates near k ~ 200
    at N = 2000 (FINDINGS:2187-2211). "Dimension inflation" **SUPERSEDED** 2026-09-29 (FINDINGS:
    800-802, 1257-1263): on real 2D substrates the window audit turns unstable at **k* = 20**
    (N = 5000, r0 = 10) and defined_frac falls 0.96 -> 0.3; far pairs exhausted at ~125
    portals. Note: this subsection carries no supersession notice (see section 5).
13. **Censorship P1/P2, synchronous** (2026-07-16; FINDINGS:2213-2242): long-portal survival
    0.03 (prune) / 0.03 (ricci) / 0.12 (triadic+prune); detour-2 survival 1.00 / 1.00 / 0.22;
    mean removal step 18.1 (geometric ~1/0.05 = 20); rank corr(adv, t) -0.07; collateral 0 / 0 /
    0.584; ~2 of 40 portals woven in. **HELD**; reproduced on triangular and sheet 2026-09-29
    (spec `2026-09-28-portals-on-geometry-preregistration.md`): P1 PASS everywhere (long
    0.010 / 0.015 / 0.020; detour-2 1.00; collateral 0); P2 PASS but weaker on 2D (long
    0.065 / 0.040 vs 0.14; collateral 0.855 / 0.853 vs 0.583) (FINDINGS:1223-1239).
14. **Walkers** (2026-07-16 -> 3b 2026-07-18 -> 3c 2026-07-18; FINDINGS:2310-2455):
    original peak-gain headline x2,534 **WITHDRAWN** as a quantity (t_max-dependent baseline;
    RNG coupling of placement to `expm_multiply`, FINDINGS:2365-2387). Horizon-free `Pbar`:
    offset portal gain 47.9x (H = A) / 131.2x (H = L); classical stationary occupancy 1.00x;
    hitting time 6.9x (FINDINGS:2415-2419). Reading "kinetic classically, structural
    quantum-mechanically" **SUPERSEDED** 2026-09-29: on triangular / sheet the offset portal
    gives quantum Pbar gain 1.0x and classical 1.3-1.4x; `grown` 122x / 166x (FINDINGS:
    1267-1291). Cause: eigenvector localization -- N x median IPR 39.7 / 42.1 / 45.0 (H = A)
    and 68.0 / 102.0 / 154.3 (H = L) for grown caps 6/7/8 vs 2.6-3.2 on the planes, 3.4 on a
    random k = 6 graph, 26.8 on pruned WS p = 0.05 (FINDINGS:1293-1312). The `Pbar` method
    (degeneracy-grouped, Krylov-validated, K_1,5 blind sum 0.180 vs true 0.060) **HELD**
    (FINDINGS:2403-2410).
15. **De-toying ladder** (2026-07-16; FINDINGS:319-401): rung 1 Lorentzian/causal DAG; rung 2
    causal-set calibration; rung 3 entanglement edges; rung 4 action principle + universality.
    Rung 1 steps 1-5 done; rung 2 flat-space done, curved CONDITIONAL; rungs 3-4 CONDITIONAL
    (BRANCHES:30-32).
16. **Causal-set instruments, flat-space calibration** (step 1, 2026-07-19; SPIKE:98-176;
    FINDINGS:369-383): MM 2.000 / 2.988 / 3.976; midpoint 2.010 / 3.078 / 4.151 (+4% bias at
    d = 4, stable in N); interval scaling **failed** (1.92 / 2.79 / 3.53 at R^2 > 0.99, bias
    flat in N) and the spike's claim that `dimension.py`'s scaffolding ports is **WITHDRAWN**
    (SPIKE:138-140). **HELD**.
17. **Async engine** (step 2, 2026-07-19; SPIKE:241-296): conflict radii 1/2/3; batch
    fractions 0.143 / 0.041 / 0.017; rounds per sweep 7 / 24 / 58; degree-thinning correction.
    **HELD**. Open: batch application still a Python loop (SPIKE:353-355; BRANCHES:65).
18. **Causal calibration KEY NEGATIVE** (step 3, 2026-07-29; spec
    `2026-07-29-causal-dag-static-calibration-design.md`; FINDINGS:403-470): null `d_causal =
    d_H + 1` **REJECTED**. MM on 1D path / 2D / 3D lattice: 1.9 / 2.5 / 3.15 vs 2 / 3 / 4
    (deficits 0.1 / 0.5 / 0.85); midpoint 1.6 / 2.0 / 2.37; grown cap 6 d_mm 2.32 vs target
    2.85; cap 8 2.65 vs 3.67 (offset +0.0). Same deficit on an isotropic 2D random-geometric
    graph. Positive control: artificial fixed-speed cone recovers d ~ 3 on the same lattice.
    Mechanism: no fixed light-cone speed; longest chain ~6-7x worldline separation; ordering
    fraction 0.34 vs Minkowski 0.229 (last-passage-percolation effect). Causal-set dimension
    retired as an *absolute* observable; `r_graph(D)` = 0.54 / 0.31 / 0.20 at D = 2/3/4 is a
    clean monotone family usable only relatively (FINDINGS:463-470). **HELD** as a negative.
    The grown rows' targets were built on radius-10 readings -- **SUPERSEDED** (FINDINGS:787-789).
19. **Censorship under async** (step 4, 2026-08-03; spec
    `2026-08-02-async-censorship-checkpoint-design.md`; FINDINGS:2244-2308): P1
    schedule-invariant (long survival 0.020 vs 0.025, z 0.34; detour-2 1.00; removal 20.6 vs
    21.3; collateral 0). P2 survives: 40-seed paired (N = 1200) long survival 0.126 +- 0.009
    sync vs 0.129 +- 0.008 async (t = -0.28); woven 2.98 vs 3.00 (t = -0.08). Preliminary
    "~2x attenuation" **WITHDRAWN** as small-sample noise (FINDINGS:2293-2302). **HELD**.
20. **d(p) fine structure, dense grid** (2026-08-11; FINDINGS:518-560): 30 log-spaced p in
    [0.02, 0.5], N = 32000, 6 seeds; perm-p 0.0005; period ~1.6 in ln p; peaks ~0.02 / 0.10 /
    0.36, troughs ~0.033 / 0.23; amplitude +-0.05-0.08; window-invariant (r +0.96-0.99 across
    max_radius 8/10/12). "Log-periodic" **WITHDRAWN** next day; the structure's reality
    **HELD** (later reframed, item 24).
21. **Overnight d(p) tests** (2026-08-12; spec `2026-08-12-dp-overnight-preregistration.md`;
    FINDINGS:562-602): P1 log-periodic third cycle **REFUTED** (fitted period 1.58 -> 2.71 on
    the extended range); P2 N-invariance **HELD** (waveform r +0.963 at N = 8000, +0.945 at
    128000; d(p) mean 1.173 / 1.174 / 1.173); P3 protection-hierarchy unsupported (cluster
    onsets fall in the 0.15-0.23 trough); secondary DSS bound not tightened (a90 worsened
    0.2-0.4 at N = 1e5).
22. **d(p) mechanism round 2** (2026-08-17; spec
    `2026-08-17-dp-mechanism-round2-preregistration.md`; FINDINGS:604-631): integer-r_c
    crossing **REFUTED/KILLED** -- r_c undefined at 22 of 30 points; M2 R^2 = 0.099 (< 0.2
    kill); convergence rounds drift smoothly 7 -> 10.
23. **Throat criticality stage 1** (2026-08-17; spec
    `2026-08-17-throat-criticality-design.md`; FINDINGS:633-715): Gate 1 exact -- survivors ==
    peeling core in 40/40 runs; per-draw threshold exists 2000/2000; frozen width w = 0.0716 ->
    0.0397 -> 0.0256 (capacities 134 / 425 / 1043) fires "sharp" but is a 1/capacity rescaling:
    relative width w/a50 rises 1.473 -> 1.574 -> 1.655, absolute width 7 -> 13 -> 22 strands,
    **onset core 2.61 / 2.59 / 2.65 strands** at every capacity. Verdict: local-motif onset,
    no critical collapse -- **HELD**; stages 2-3 CLOSED (BRANCHES:86). Gate 2: triadic
    demolishes cores (19-28 -> 0-4 strands, retention ~0.1) -- **HELD**. Critical slowing down:
    7-39 sweeps deep sub-threshold vs 99-105 near threshold, read as cascade depth
    (FINDINGS:672-675). "Persistent expander phase" (19/2000 at all FSS geometries) --
    **resolved** 2026-09-26 as growth extinction: same 19 seeds, 7-17 node graphs
    (FINDINGS:806-813).
24. **Audit of 2026-09-26/27** (FINDINGS:717-871; `window_stability.py`): `grown` cap 6/7/8
    EXPONENTIAL 3/3 seeds, bases 1.30 / 1.47 / 1.60; d_eff by window R6 1.74 ... R10 2.20 ...
    R32 3.95 (cap 6); |B(r+1)|/|B(r)| flat 1.26-1.31 for r = 9..32; diameter 31 / 43 / 52 /
    61 / 71 at N = 2e3 / 8e3 / 3.2e4 / 1.28e5 / 5.12e5 (~+10 per factor 4, logarithmic);
    pruned WS p <= 0.1 DIMENSION DEFINED d = 1; p = 0.2 MIXED; p >= 0.3 EXPONENTIAL (bases
    1.15 / 1.24 / 1.30). Retracted: grown's dimension, cap->d plateau, prune knob 1->2, the
    N^0.19 diameter, step-3 grown rows (FINDINGS:772-789). Stands: barrier; everything
    measured *on* grown as dynamics on a tree-like fabric (FINDINGS:790-804). Gate 3 added
    2026-09-28; gated defined_frac table (FINDINGS:843-853): lattice 1.00 / drift +0.02;
    grown cap 6 0.99 -> 0.21 / +0.51; cap 8 -> 0.08 / +1.06. **HELD**. Also the estimator
    blind spot (base ~1.12 passes at radius 10, FINDINGS:862-866).
25. **Step 5: barrier under async** (2026-09-27; spec
    `2026-09-27-step5-async-barrier-preregistration.md`; FINDINGS:873-925): alpha sync 0.158 +-
    0.022 (CI [0.106, 0.211]), async 0.168 +- 0.013 (CI [0.136, 0.200]); paired diff +0.010 +-
    0.032; BARRIER SURVIVES (upper bound 0.200 < 0.25) -- **HELD**; schedule invariance
    UNDERPOWERED. Gate 0 (reproduce banked CSV within +-2) **failed as frozen** (realization
    scatter up to 4); Gate 1 (`prune` control exponents within 0.1) **failed as frozen**
    (0.825 vs 0.576; ring fragments); post-hoc Gate 1A Jaccard distance <= 0.0052 PASS
    (FINDINGS:900-912; spec amendment lines 92-112).
26. **d(p) round 3** (2026-09-27; spec `2026-09-27-dp-round3-decomposition-preregistration.md`;
    FINDINGS:927-976): Gate R (bit-reproduce) **failed as frozen** (55/180 rows; sample
    differs, cause not found); amended R' correlation +0.991; M1 +0.873. Reading **MESOSCALE**:
    radii >= 4 carry 109% of variance, r <= 3 carry -9%; not amplified (peak-to-peak 0.125 <
    raw 0.162); r = 10 signal 0.16 vs seed SE 0.008. **HELD**.
27. **Spectral-dimension flow on `grown`** (2026-09-27; spec
    `2026-09-27-spectral-flow-preregistration.md`; FINDINGS:978-1038): Gate A anchors ring
    0.998, 2D torus 2.027, 3D torus 3.041, Sierpinski gasket 1.368 (truth 1.365; Hausdorff
    1.585), random 3-regular not FLAT. Gate B: grown D(t) 1.15 -> 1.5 -> 1.8-2.05 (cap 6), 1.23
    -> 1.7 -> 1.9-2.0 (cap 7), 1.30 -> 1.7 -> 1.9-2.05 (cap 8), N-robust to 0.045. P1 RUNAWAY
    **REFUTED** (1 of 18 runs); P2 **HELD**; P3 ordering fails only at last point. Reading: a
    `d_s`/`d_H` split "of the kind trees are known for", not CDT-like (FINDINGS:1021-1029).
    Pruned WS p = 0.3, 0.5 RUNAWAY 4/4. `sheet` beta = 0: 1.40 -> 1.92; `sheet` c = 7: 1.57 ->
    2.03 (FINDINGS:1031-1032). **HELD**.
28. **`sheet` confirmatory run** (2026-09-27; spec
    `2026-09-27-sheet-growth-preregistration.md`; FINDINGS:1040-1106): N = 512000, seeds
    100-102. c = 6, beta = 2: diameter exponents 0.496, 0.500; boundary 0.499, 0.503; window
    audit DEFINED d = 1.99 3/3; spectral FLAT 1.977 / 1.991 / 1.985 -> **GEOMETRY**. beta = 3:
    0.501, 0.505 / 0.509, 0.502 / d = 1.99 / FLAT 1.985 / 1.999 / 1.941 -> **GEOMETRY**. beta
    = 0: TREE-LIKE (boundary exponent 1.00); beta = 1: diameter 0.248 -> 0.174, 2D to N ~ 8000
    then tree-like; c = 7, beta = 2: exponential. Diameter at beta = 2: 56 / 111 / 226 / 450 /
    899 at N = 2e3 ... 5.12e5; boundary 0.5% of volume, within ~15% of a perfect disc's
    perimeter (FINDINGS:1071-1073). Interior degree 6 fraction 1.000. Limit found: the window
    audit labelled beta = 1 DEFINED (drift 0.09) because it probes radii <= 40
    (FINDINGS:1098-1102). **HELD**, with the later size qualifier (item 32).
29. **`sheet` rate-strength scan** (2026-09-28; spec
    `2026-09-28-sheet-beta-transition-preregistration.md`; FINDINGS:1108-1166): N*(beta) for
    beta = 1.0 ... 1.7: 5,600 / 16,200 / 9,300 / 17,900 / 32,700 / 151,000 / 164,000 / 320,000;
    1.8-2.0 censored (> 512000). Exponential `ln N* = 2.68 + 5.82 beta` (RSS 1.34, AICc -7.88)
    vs power law `ln N* = 10.23 - 5.49 ln(2.325 - beta)` (RSS 1.13, AICc -3.68): CROSSOVER
    SUPPORTED, "thinly" (consistency clause passed by 0.6%; non-monotone N*). Predicted
    N*(2.0) = 1.6 M. Law **SUPERSEDED** by the 12-seed refit (item 32); verdict **HELD**.
30. **d(p) round 4** (2026-09-28; spec `2026-09-28-dp-round4-crossover-preregistration.md`;
    FINDINGS:1168-1207; fresh seeds 6-11): single crossover form `a + h ln(1 + (p/p0)^g)`
    leaves RMS 5.8 seed-SE at r = 10 vs cubic 4.8 -> **H REFUTED** (split-half r +0.92). Two
    crossovers (7 par.) leave 0.7-0.9 SE at every radius 4-10 where a sextic leaves 1.2-1.7:
    "the structure survived a sextic detrend because polynomials are the wrong family"
    (FINDINGS:1188-1193). Remaining waveform in d 0.050 vs noise ~0.03. Thread now reads:
    fixed-window reading of ball counts at r >= 4 following a smooth two-scale crossover;
    scales unidentified (FINDINGS:1195-1205). **HELD**.
31. **Portals on substrates with a dimension** (2026-09-28/29; spec
    `2026-09-28-portals-on-geometry-preregistration.md`; FINDINGS:1209-1329): E1 P1 PASS x3,
    P2 PASS (weaker on 2D); E2 k* = 20 on both planes (predicted 10-50), grown unstable at k =
    0; E3 quantum Pbar gain > 10x **FAILED** (1.0x), classical gain larger on 2D **FAILED**
    (1.3-1.4x vs 6.7x); baseline Pbar x N ~ 0.97-0.98 on planes (equipartition) vs 0.03 on
    grown (FINDINGS:1267-1287). `triangular` and `sheet` agree to two decimals (consistency
    check, FINDINGS:1324-1326). **HELD**.
32. **`sheet` beta = 2 at 4 M nodes** (2026-09-29; spec
    `2026-09-28-sheet-crossover-4M-preregistration.md`; FINDINGS:1331-1406): Part B 12-seed
    rescan beta = 1.0-1.7 N* = 7.3k / 11k / 14k / 25k / 34k / 117k / 122k / 324k (monotone);
    `ln N* = 3.30 + 5.38 beta` (AICc -16.3) vs `10.00 - 4.76 ln(2.270 - beta)` (AICc -14.4);
    N*(2) = 1.26 M, 95% PI [0.47 M, 3.4 M]. Part A seeds 300-303 to 4,194,304: N*_B 782,000,
    N*_A 777,000, N*_D 2,340,000 -> **CROSSOVER CONFIRMED, quantitatively**; transition at
    beta_c ~ 2.3 excluded. Per seed B/sqrt(N) at 512k/1M/2M/4M: 3.85/3.99/7.05/11.66 (300),
    3.86/6.17/9.16/12.16 (301), 3.84/3.85/7.09/12.75 (302), 3.83/3.84/5.22/11.09 (303)
    (FINDINGS:1376-1381). Reading "nucleation: one protrusion escapes ... grows like a tree"
    (FINDINGS:1383-1388) **WITHDRAWN** 2026-10-13 (no arm; item 35). "0.8 M at beta = 2"
    headline **WITHDRAWN** 2026-10-13 (revision note FINDINGS:1390). M4/M5 on the 4 M graph:
    d 1.94-1.99 at every window radius 6-160, 1.94 / 1.84 at 250 / 400 (MIXED, drift 0.15);
    d_s FLAT 1.96, declining 2.00 -> 1.85 (FINDINGS:1397-1402). **HELD** (crossover; law).
33. **Preservation of a real dimension** (2026-09-29; spec
    `2026-09-29-preservation-on-geometry-preregistration.md`; FINDINGS:1408-1444): on
    triangular and sheet (N = 10000, 200 steps, seeds 0-2) all four PASS: `majority` 0.98 ->
    0.98, d 1.96; `prune` exactly inert (29601 -> 29601 edges); `triadic` 0.98 -> 0.02, median
    d 6.8 / 5.5, unstable from step 20, drift turning negative first (-0.13, -0.25, -0.33);
    `rewire` destroyed within 40 steps, untestable from step 20. **HELD**.
34. **Round 2026-10-13 item 3** (spec `2026-10-13-round2-preregistrations.md` item 3;
    FINDINGS:1446-1486): coherence and d(p) tables re-run under the gate; predictions held
    (numbers in items 7 and 9). **HELD**.
35. **Item 1: what nucleates the arm** (same spec, item 1; `arm_birth.py`;
    FINDINGS:1488-1532): by the frozen definition no arm exists on any of the 12 graphs ->
    **UNDECIDED by construction**. Exploratory: boundary edges 2,660 / 3,883 / 9,717 / 22,684
    at 512k / 1M / 2M / 4M (seed 300); 2N/B ~ 394 / 540 / 432 / 370 ("finger width"); boundary
    root distance q5/q95 0.88/1.08 -> 0.75/1.22 R; max depth from edge 365 -> 675
    (FINDINGS:1502-1509). Mullins-Sekerka shape-instability reading **WITHDRAWN** 2026-10-14
    (FINDINGS:1522-1525); measured facts stand.
36. **Item 4: crossover-size distribution** (same spec, item 4; `crossover_cv.py`;
    FINDINGS:1534-1610): 12 seeds; 7 finite N*_B (475k ... 1,901k), 3 exactly compact at 4.19 M
    (B/sqrt(N) 3.85-3.86), one turning (4.57), seed 308 left at ~41,000 (B/sqrt(N) 9.00 at 512k)
    but censored by the frozen rule. Frozen label IN BETWEEN (CV 0.44, bootstrap SE 0.10,
    censored 0.42) -- label **HELD**, predicted mechanism (rising hazard) **not supported**.
    Right-censored Weibull: shape k = 1.02 [0.49, 1.78] / 0.91 / 0.80 depending on treatment of
    308; LR vs k = 1: p = 0.96 / 0.75 / 0.45; median N* 2.1-3.0 M; P(no crossover by 4.19 M)
    0.29-0.37 (FINDINGS:1578-1582). "One crossover event per ~3-4 M nodes grown at beta = 2"
    (FINDINGS:1584). N* spans 4e4 to > 4.2e6: "two decades across twelve seeds. A sharp
    threshold is excluded outright" (FINDINGS:1572-1573).
37. **Item 2: throat criticality on planes** (same spec, item 2; `throat_compare.py`;
    FINDINGS:1612-1687): capacities triangular 329 / 1264 / 3448, sheet 330 / 1265 / 3458 (vs
    grown 134 / 425 / 1043); A*50 8 / 14 / 22 (A*50 ~ capacity^0.4); relative width 1.315 /
    1.312 / 1.416 (tri), 1.315 / 1.339 / 1.354 (sheet) vs 1.473-1.655 grown; **onset core 2.48
    / 2.68 / 2.66 (tri), 2.50 / 2.61 / 2.77 (sheet)**; Gate 1 40/40 exact on both; Gate 2 (b)
    core retention 0.08 / 0.02; 0 regenerations. **MOTIF ONSET HOLDS** on both -- **HELD**.
38. **Edge roughness spectrum** (2026-10-13; spec
    `2026-10-13-edge-spectrum-preregistration.md`; FINDINGS:1720-1744): frozen verdict
    **UNDECIDED** (fixed 2, scale-free 0 of 5). Instrument fault: the three never-crossing
    seeds (305, 309, 310) show "peaks" at lambda = 389 / 551 / 781 / 1110 boundary nodes at 512k
    / 1M / 2M / 4M, i.e. k = L/lambda ~ 7 at every size, excess 3.7-8.6 -- the disc's six-fold
    anisotropy on a triangular lattice, a global shape mode the frozen rule (k >= 4) did not
    exclude (FINDINGS:1725-1731). Post hoc k >= 16: compact seeds lose all peaks (slopes -3.4
    to -4.1); rough edges slopes -2.1 to -3.1 with modest peaks (excess 3.2-5.9, lambda 183-685,
    no shared wavelength across sizes except 308) -> UNDECIDED again (fixed 1 of 5). Compact
    edges smooth: slopes -3.4 to -4.7; rough -2.0 to -2.8 (FINDINGS:1723-1724). Seed 308: a
    genuine peak lambda 247-301 (262k-2M, frozen) / 217 / 353 / 428 (1M-4M, k >= 16), slope
    -2.0 to -2.5, sigma_h/R 0.19 -> 0.26, lambda/(2N/B) = 1.1-1.8 (FINDINGS:1737-1743).
39. **Dead-corner trigger** (2026-10-13/14; spec
    `2026-10-13-blocked-closure-preregistration.md`; FINDINGS:1745-1865): seed 308 onset: zero
    degree->= 6 boundary nodes to 25,500; two adjacent blocked nodes (24862, 25107) at 26,000;
    24862 oldest boundary node from 27,000; deepest fjord point from 32,768; blocked count 2 ->
    3 -> 5 -> 7 -> 12 -> 24 at 26k -> 32k -> 40k -> 65k -> 131k -> 524k, all strung up the fjord
    wall; B/sqrt(N) 3.79 -> 8.99 (FINDINGS:1763-1779). Twelve saved beta = 2 graphs: every
    crossed seed's first dead corner lies inside the octave in which B/sqrt(N) leaves 3.85,
    never before; 3 compact seeds have none; seed 311 (turning at 4 M) got its first at 3.34 M;
    14-81 dead corners in 1-4 clusters at 4.19 M; oldest boundary node is a dead corner in 9/9
    crossed seeds -- **12/12** (FINDINGS:1781-1809). Fresh test, 8 seeds at beta = 1.5 to
    524,288: N*_B / N_first = 2.42, 1.35, 1.02, 1.13, 1.13, 0.96, 1.01, 0.91; blocked nodes at
    524k 36-100; **TRIGGER CONFIRMED (6/8 by the frozen rule; 8/8 within a factor 2.5)**; no
    uncrossed control seed existed (all eight crossed) (FINDINGS:1811-1838). Reframing:
    exp(5.4 beta) "reads as the beta-dependence of the probability per boundary site of
    producing a dead corner ... a hypothesis, not yet a measurement" (FINDINGS:1849-1853).
    Sharp prediction: a variant that cannot leave a dead corner should have no crossover at
    any beta (FINDINGS:1855-1858). **HELD**.
40. **LORENTZIAN_SPIKE.md** (2026-07-19, updated through 2026-08-03): a decision document,
    "nothing is committed" (`README.md:84`). Its own corrections: the "~4-5 rounds per sweep"
    estimate was ~12x optimistic for geometry rules (SPIKE:241-283); `dimension.py`
    scaffolding "ports" claim retracted (SPIKE:138-140); causal measurement capped ~1e4 nodes x
    1e3 steps by DAG memory, dynamics not capped (SPIKE:298-311). Steps 1-4 marked DONE
    (SPIKE:337-387); step 5 (barrier + causal future growth) was the remaining rung-1 work and
    the barrier half is now done (item 25); causal-future growth deferred.

### Inferences

- The chain `grown has d ~ 2.2` -> coherence -> tolerance "inflation" -> walker "structural
  portal" was one instrument fault propagating through four banked results; each has since
  been re-measured on `triangular`/`sheet` and the rule-level statements (censorship P1/P2,
  motif onset, preservation pattern) survived while every statement *about `grown`'s geometry*
  did not.
- The project's own two-slope picture of `sheet` -- boundary ~ sqrt(N) then ~ N -- is a
  compact-to-ramified morphology change triggered by a frozen defect, with a per-added-node
  constant hazard at fixed beta; the exp(5.4 beta) dependence is a mean-N* fit, now
  reinterpreted as a rate.
- Everything substrate-dependent in the throat program (capacity, A* in strands, relative
  width) moved in the direction "a plane is a bigger, more regular container for the same
  motif" (FINDINGS:1680-1682); the motif size ~2.6 strands is the rule-level invariant.

### Gaps

- No banked result yet tests the dead-corner-free variant (BRANCHES:44); the whole causal
  story for the sheet's crossover rests on correlation (12/12, 8/8) plus the mechanism
  argument in the rule's own terms.
- The exp(5.4 beta) form has never been re-tested after being reinterpreted as a rate
  (FINDINGS:1599-1601).
- No banked result re-ran the `geometrize` bistability table (item 3) under the gate; its
  non-lattice rows depend on `defined_frac`.

---

## 3. Every OPEN item, verbatim-ish, plus open questions in FINDINGS not in BRANCHES

### Takeaway

BRANCHES carries 14 OPEN rows, 4 CONDITIONAL, 3 PARKED and 4 ENGINEERING items. The most
consequential open items are the dead-corner-free `sheet` variant (a falsifiable prediction)
and the beta-dependence of the dead-corner rate; FINDINGS additionally raises roughly a dozen
questions that have no BRANCHES row.

### Cited Findings -- BRANCHES OPEN (BRANCHES:16-51)

- **Triangulated-base `prune` transition** -- origin step-3 fork 2026-06-27; "the un-taken
  alternative was a genuine-transition hunt on a 2D-triangulated substrate; never invalidated.
  Note 2026-09-27: `grown` is not such a substrate (it has no dimension); the triangular
  lattice is, and `sheet` may be"; cost medium (`prune_dimension.py` variant) -- BRANCHES:20.
- **`d(p)` multi-bump fine structure -- mechanism hunt (round 3)** -- rounds 1-2 killed;
  round 3 (MESOSCALE) and round 4 (two crossovers to seed noise) done; "Remaining question is
  only what the two scales are (candidates: shortcuts within reach; backbone thinning) -- a
  low-priority identification, no longer a mystery"; cost low -- BRANCHES:21.
- **Causal ordering-fraction as a *relative* comparator** -- "the monotone family r_graph(D)
  is intact for async-vs-sync or rule-vs-rule *comparisons*; offered post-step-3, never picked
  up"; cost medium -- BRANCHES:22.
- **Quasi-1D fragmentation scaling** -- same fork as the triangulated base; "never
  invalidated; lowest-value survivor of that fork"; cost low-medium -- BRANCHES:23.
- **Throat-motif <-> d(p) mechanism cross-pollination** -- "the d(p) fine-structure mechanism
  hunt (round 3) and the throat onset concern the same object -- protection motifs in random
  shortcut ensembles -- in different ensembles; the exact peeling machinery now exists to
  count motifs directly in the pruned-WS ensemble"; cost low-medium -- BRANCHES:38.
- **A `sheet` rule without dead corners: does the crossover disappear?** -- "a variant whose
  closure step cannot be blocked (e.g. allow the closure when a link end is at the cap,
  raising it to c + 1; or re-rate dead edges as growth events) should be a plane at **every**
  beta and N. If it still crosses over, the dead-corner reading is wrong. Cheap:
  `sheet_growth` flag + `sheet_transition`-style scan at beta = 1.0-1.5 where N* is 1e4-1e5";
  cost low-medium -- BRANCHES:44.
- **Why exp(5.4 beta)? The per-site rate of dead corners vs beta** -- "measure directly:
  count dead corners per boundary site (or per node grown) at beta = 1.0-2.0 on short
  growths, fit ln(rate) vs beta, compare the slope to -5.4. Also the degree histogram of the
  boundary vs beta (high beta should keep link ends below the cap)"; cost low -- BRANCHES:45.
- **Roughness spectrum beyond seed 308** -- "with the trigger known, the spectrum describes
  how a stalled pit roughens the edge (fjord walls as random surfaces). Revive if someone
  wants exponents: a proper width-vs-N and spectrum-vs-k analysis per pit, with the k = 6
  mode excluded from the start"; cost low -- BRANCHES:46.
- **`sheet` as a registered topology, and the rules run on it** -- "wiring it into
  `create_initial_graph` lets preservation, portals, censorship and `prune` run on a substrate
  with a real dimension. Generation is slow (~10 min per 512000 nodes); whether it belongs in
  the core is an owner call"; cost low-medium -- BRANCHES:47. (Note: `sheet` *is* already in
  `create_initial_graph` at beta = 3, `simulation.py:82-89`; the row predates that and was not
  updated -- see section 5.)
- **Schedule-invariance of the barrier exponent, at power** -- "difference +0.010 +- 0.032,
  CI wider than the +-0.05 band ... Needs a continuous extent observable (mean eccentricity or
  mean pair distance) or ~4x the seeds; a new pre-registration either way"; cost low-medium --
  BRANCHES:48.
- **Long-run collapse of extent under `triadic`** -- "extent peaks near 300 sweeps then falls
  below its starting value (N=1000: 8.5 -> 11.5 -> 4.0) with the largest component intact ...
  What is the long-run state -- a hub? does the peak time scale with N?"; cost low --
  BRANCHES:49.
- **Async events for `geometrize` and `ricci`** -- "the three-rule barrier claim is tested
  under async for one rule"; cost low-medium -- BRANCHES:50.
- **Re-run the banked `defined_frac` results under the window-stability gate** -- "coherence
  (done: grown reads 'fragmented') and the `prune` d(p) curve remain; preservation and
  tolerance were re-done on real substrates instead (2026-09-29)"; cost low -- BRANCHES:51.
  (Both named remainders were done 2026-10-13, item 3; the row text is stale -- section 5.)

### Cited Findings -- BRANCHES CONDITIONAL / PARKED / ENGINEERING (BRANCHES:25-68)

- CONDITIONAL: **`ricci` under async** (gate: any async-vs-sync discrepancy) -- BRANCHES:29.
  **Ladder rung 2 remainder: curved-spacetime sprinkling** (gate: causal-set instruments
  regain a consumer) -- BRANCHES:30. **Ladder rung 3: entanglement edges (stabilizer states,
  ER=EPR toys)** (owner prioritization) -- BRANCHES:31. **Ladder rung 4: action principle +
  universality (Metropolis at T, FSS across microrules)** (owner prioritization; "carries the
  continuum-limit requirement the whole ladder points at") -- BRANCHES:32.
- PARKED: **Density-healing rule design** ("a strictly-local density-regulating rule is the
  missing 'restoring force' for lump-type seeds ... the closest thing to gravity in the rule
  set") -- BRANCHES:57. **Large-cap (1/cap) analytic limit** ("expander limit is the tractable
  end -- inverse of the paper's large-D trick") -- BRANCHES:58. **Horizon-free quantum
  hitting-time analogue** ("needs a measurement model first") -- BRANCHES:59; also listed
  CLOSED as moot at BRANCHES:82 (section 5).
- ENGINEERING: **Vectorised batch application** (blocks async at >= 1e5) -- BRANCHES:65;
  **`rewire` under async** (collision detection + deferral) -- BRANCHES:66; **`adv_corr` guard
  backport** -- BRANCHES:67; **`throat_criticality.py --fss` width label prints unguarded
  value** -- BRANCHES:68.

### Cited Findings -- open questions raised in FINDINGS / SPIKE / DC with no BRANCHES row

- Ollivier-Ricci flow untested ("expected to behave the same and would add little") --
  FINDINGS:284-286, 160-161.
- Does `grown`'s `d_s(t)` level at 2? "cannot be decided inside the admissible range" --
  FINDINGS:1018-1019.
- The window audit's radius ceiling: beta = 1 labelled DEFINED because it only probes r <= 40
  -- FINDINGS:1098-1102; the estimator's base ~1.12 blind spot -- FINDINGS:862-866.
- The residual after the two-crossover form (peak-to-peak 0.050 vs noise ~0.03): "a small
  remainder is not excluded" -- FINDINGS:1200-1201.
- Why the first dead corner is always an *adjacent pair* born within ~1% of N_first --
  FINDINGS:1802-1804 (stated, not asked).
- Whether later pits are seeded by the first ("the first pit seeds further ones along its
  walls, and a second or third pit forms elsewhere later") -- FINDINGS:1805-1806.
- The step-2 N*_B estimator lands 4-50% before N_first ("interpolates the 0.75-crossing
  between octave midpoints") -- FINDINGS:1806-1809, 1831-1835: a half-octave resolution limit.
- Seed 308's drifting preferred wavelength 220-490 nodes (BRANCHES:46 mentions it; the
  mechanism of a *drifting* wavelength is not posed).
- Causal-future growth `|J+(e, n)|` as the causal restatement of the barrier: deferred, "at
  best a *relative* comparator" -- SPIKE:194-208, 388-391; FINDINGS:925.
- DC Phase 3 (correlate dimension with dynamics: activation propagation within vs across
  phases; reinforcement within phases; rewiring moving boundaries; d_eff vs local
  computational density) and Phase 4 ("the gravitational test": local update rate vs
  d_eff/density) -- DC:158-175 -- never run and not registered anywhere.
- Phase-transition route "remains open only for rules/observables not yet tried" --
  DC:181; SCALING:13.
- "Does anything converge as N grows, and is it rule-independent?" -- the standing
  universality requirement -- FINDINGS:398-401; SPIKE:402-414.
- Finite-size constraint on portals: a 2D substrate of 5000 nodes exhausts far pairs at ~125
  portals -- FINDINGS:1262-1263.
- `grown` mean degree ~4 regardless of cap (SPIKE:238-239) -- an unexplained generator fact.

### Inferences

- Two of the OPEN rows (BRANCHES:47 and :51) describe work that has since been done; the
  registry's own anti-rot rule ("a branch leaves this file only by being promoted to work or
  explicitly CLOSED", BRANCHES:13-14) has not yet been applied to them.
- The most literature-sensitive open items are the two sheet rows (BRANCHES:44-45): whether a
  blocked-closure defect is a known kinetic trap, and whether an exp(-a beta) defect rate is
  the expected Arrhenius-type form.

### Gaps

- BRANCHES has no row for the DC Phase 3/4 experiments, so their status (abandoned vs
  deferred) is unstated anywhere.

---

## 4. Questions the literature could answer (derived from the above)

### Takeaway

The project's results map onto five literature areas: kinetic growth / self-assembly of
triangulated surfaces (sheet, dead corners, roughness), random tree-like graph growth (grown),
dimension estimation on graphs (Hausdorff vs spectral, estimator pitfalls), local rewiring and
discrete curvature flows (the barrier), and bootstrap-percolation / k-core thresholds (throat
motif, pruned WS). Each question below names the project claim it would bear on.

### Cited Findings (the project statements each question is anchored to)

**A. The `sheet` growth model and its boundary**

1. Is a locally grown planar triangulation with a hard degree cap `c` and attachment rate
   `exp(beta * (deg u + deg v))` at boundary edges a known model class (kinetic Monte Carlo
   solid-on-solid growth with bond-counting energetics; Eden model on the triangular lattice
   with coordination-dependent rates; "kinetic self-assembly" of a 2D crystal)? Which name
   does the literature give to beta (inverse temperature times bond energy)? -- anchored to
   `sheet_growth.py:10-25`, FINDINGS:1047-1055.
2. Is boundary length ~ sqrt(N) (compact) with a crossover to boundary ~ N (ramified) known
   for Eden/ballistic/DLA-type growth as a surface-tension or noise-reduction parameter
   varies, and does any model show the compact range growing *exponentially* in the coupling
   (`ln N* = 3.3 + 5.4 beta`, FINDINGS:1344)? -- FINDINGS:1108-1166, 1339-1349.
3. What are the known roughness exponents and height-spectrum slopes of Eden-type cluster
   boundaries (KPZ/EW: 1D interface with roughness exponent alpha has spectrum ~ k^-(1+2alpha),
   i.e. k^-2 for alpha = 1/2)? Do the project's rough-edge slopes -2.0 to -3.1 and compact-edge
   slopes -3.4 to -4.7 fall into a known class, and does "fjord walls as random surfaces"
   (BRANCHES:46) correspond to a KPZ-like interface? -- FINDINGS:1723-1724, 1732-1734.
4. Is the six-fold anisotropy of Eden clusters on a triangular lattice (the k = 7 "hexagon"
   mode that contaminated the frozen spectrum reading, FINDINGS:1727-1731) a documented
   finite-size shape effect, and is the ~15% perimeter excess over a perfect disc
   (FINDINGS:1073) consistent with the known Eden-cluster shape?
5. Is a permanent, un-repairable local defect that stalls growth at one site ("dead corner"
   -> fjord) a known failure mode in kinetic growth of triangulations / in self-assembly
   (kinetic traps, poisoned sites, frozen boundary vertices, "blocked" moves in constrained
   triangulation growth)? Are there growth models of discs/triangulations whose boundary
   vertices can become frozen under a maximum-coordination constraint? -- FINDINGS:1755-1761,
   1840-1853.
6. Is `exp(-a beta)` for the per-site probability of a specific kinetic defect the expected
   Arrhenius-type form in KMC growth, and can `a = 5.4` be related to the bond deficit of the
   dead-corner configuration (two adjacent link ends reaching degree c before the closure
   fires, with closure rate `exp(2 beta c)` vs attach rates `exp(beta * (deg u + deg v))`)? --
   BRANCHES:45; FINDINGS:1849-1853; `sheet_growth.py:89-90`.
7. Is a constant hazard per added node (Weibull shape k ~ 1, FINDINGS:1578-1589) what a
   per-attachment defect probability predicts, versus a hazard rising with boundary length
   (k ~ 1.5 if the probability is per boundary site per unit time)? Which clock is the right
   one for first-defect statistics in a growing boundary? -- spec
   `2026-10-13-round2-preregistrations.md:112-128`.
8. Is the Mullins-Sekerka picture (growth under surface tension unstable beyond a critical
   radius) applicable at all to a lattice growth with coordination-dependent rates, and did
   the project withdraw it for the right reason (near-scale-free spectrum, no fixed
   wavelength)? -- FINDINGS:1516-1525.
9. Does the `{3, c}` classification (c = 5 icosahedron, 6 flat, 7 hyperbolic) have a known
   stochastic growth analogue, and is exponential ball growth with a *finite* spectral
   dimension flow (sheet c = 7: 1.57 -> 2.03, FINDINGS:1032) known for hyperbolic
   triangulations grown by local rules?
10. Are "dynamical triangulations" (DT/CDT) and "random planar maps" related to `sheet`? Those
    *sample* triangulations; `sheet` *grows* one by local moves with no detailed balance. Is
    there literature on growth-generated (non-equilibrium) triangulations and their Hausdorff
    dimension?

**B. `grown` and tree-like graph growth**

11. Are random 2-trees / Apollonian-type triangle-gluing processes with a degree cap known to
    have logarithmic diameter and exponential ball growth (the project: treewidth <= 2,
    diameter +10 per factor 4, frontier 37% of N, FINDINGS:757-768)? Random Apollonian
    networks have logarithmic diameter in the literature; does the degree cap change that?
12. What is the extinction probability of a degree-capped triangle-gluing process (project:
    0.474 / 0.279 / 0.0082 / 0.0002 for caps 4/5/6/7-8, decided within the first dozens of
    attachments, FINDINGS:814-823)? Is this a branching-process early-extinction result?
13. Why would mean degree be ~4 regardless of cap (SPIKE:238-239)? (Each attachment adds one
    node and two edges, so mean degree -> 4 exactly as N grows -- a trivial identity the
    project did not state.)
14. Spectral vs Hausdorff dimension on trees and tree-like bounded-degree graphs: is a slow
    monotone rise of `d_s(t)` from ~1.2 toward ~2 over three decades (FINDINGS:1015-1029)
    known for graphs with exponential volume growth but strong bottlenecks? The project's
    literature stance was that `d_s != d_H` on fractals is a known feature (FINDINGS:1959-1962).
    Does the random-walk literature on supercritical trees (non-amenable, spectral gap) predict
    a *defined* spectral dimension at all here, or a slow approach to exponential decay?
15. Eigenvector localization on trees and sparse graphs with dangling ends (the IPR finding,
    FINDINGS:1293-1312): is "27-70% of eigenvectors on < 20 nodes" expected for bounded-degree
    tree-like graphs, and is it known that a single added long-range edge then delocalizes the
    walker (the 122-166x Pbar gain)?

**C. Dimension estimation**

16. Is the pitfall "a radius-10 power-law fit of slowly exponential ball growth returns a
    confident window-dependent number at R^2 > 0.97" (FINDINGS:726-731) documented in the
    network-dimension literature, and is a window-drift / power-vs-exponential model
    comparison a standard check?
17. Does the literature on scale-dependent spectral dimension (CDT flow, Millan et al.
    tunable-d_s networks, cited FINDINGS:1932-1943) include any *non-causal, locally grown*
    graph with a CDT-like plateau-to-plateau flow? (Project: predicted null; grown gave a
    slow rise, sheet gave FLAT 2.)
18. For Myrheim-Meyer on a graph x Poisson-time causal set: is the FPP light-cone-shape
    sensitivity (last-passage percolation; longest chain ~6-7x worldline separation; ordering
    fraction 0.34 vs 0.229) a documented estimator bias, and are shape/frame corrections
    known? -- FINDINGS:446-454; spec `2026-07-29-...:137-154`.

**D. Rewiring, curvature flow and the barrier**

19. Is it known that zero-temperature local rewiring (triadic closure, degree homeostasis,
    Forman-Ricci surgery) on a sparse random graph cannot increase diameter beyond ~log N
    ("crumples, not unfolds"; alpha ~ 0.13-0.17 vs 0.5; FINDINGS:2037-2055, 881-889)? How does
    this sit against combinatorial-quantum-gravity results where a curvature *action* at low
    temperature (Metropolis, global energy) does condense random graphs into lattices -- is
    the barrier specific to greedy local moves without an accept/reject temperature?
20. Do triadic-closure network models report the diameter collapse / clique crumpling and
    the long-run extent collapse (peak at ~300 sweeps then fall, FINDINGS:914-920)?
21. Do discrete Ricci-flow-by-rewiring studies (Forman or Ollivier) report that the flow
    pinches off / fragments communities rather than producing extended geometry, consistent
    with `ricci` plateauing at a clumped fixed point (FINDINGS:109-124) and `geometrize`
    fragmenting `grown` (FINDINGS:88)?
22. Is "every edge of a triangulation lies in a triangle, so overlap-based pruning is exactly
    inert" (FINDINGS:1422, 1437-1438) and the converse "shortcut edges have zero common
    neighbours by definition" (FINDINGS:2219-2221) used in the backbone-extraction / edge-
    filtering literature?

**E. Pruned small worlds and protection motifs**

23. For Watts-Strogatz with shortcuts pruned unless they lie in a triangle: is the surviving
    structure (ring for p <~ 0.15, small world again above p ~ 0.25, FINDINGS:1477-1484) a
    known "triangle-protected shortcut" percolation, and are the two characteristic densities
    of the round-4 crossover form (FINDINGS:1195-1199) identifiable with known WS crossover
    scales?
24. Is the ~2.6-strand "mutually protecting motif" onset (FINDINGS:656-662, 1646-1652)
    the first cycle / smallest 2-core in a sparse random bipartite overlap graph, and is it
    known that such an onset is a local event with no finite-size sharpening (relative width
    flat-to-rising with capacity)? Does bootstrap-percolation theory predict A*50 ~
    capacity^0.4 (FINDINGS:1647)?
25. Is "exact peeling fixed point = stochastic prune dynamics up to timing" (40/40, three
    substrates) a known equivalence between k-core peeling and a stochastic removal process
    with no edge creation?

**F. Portals and transport**

26. Is it known that a single added long-range edge is invisible to the infinite-time-average
    transfer of a continuous-time quantum walk on a 2D lattice because the no-portal baseline
    already sits at equipartition (Pbar x N ~ 1, FINDINGS:1279-1283)? What does the perfect-
    state-transfer / CTQW-on-lattices literature say about one defect edge?
27. Is "a long-range edge changes classical hitting time but not stationary occupancy" merely
    the degree-determined stationary distribution (the project says so, FINDINGS:2425-2430)
    and is the hitting-time gain 6.7x on a tree vs 1.3-1.4x on a plane consistent with known
    first-passage results?

**G. Methodology**

28. Over 1.5 decades of N*, how reliably can an exponential law be distinguished from a
    power-law divergence by AICc (the project's margins: -7.88 vs -3.68, then -16.3 vs -14.4;
    FINDINGS:1132-1135, 1343-1345)? Is there a standard treatment for right-censored crossover
    sizes in growth models (the project used censored Weibull MLE, FINDINGS:1576-1589)?
29. Is a permutation test on maximum periodogram power known to fire on smooth misfit
    (FINDINGS:507-513), and are the project's DSS discriminators (>= 2-3 cycles, detrend-order
    ladder, staircase control) the standard ones in the log-periodicity literature?
30. Is the Z2-breaking of `argmax` majority vote with even degree (constant 2-2 ties -> state
    0, FINDINGS:2088-2098) a documented pitfall in majority-vote / voter-model simulations?

### Inferences

- Questions A5-A7 are the ones whose answers would most change the project's current
  description of `sheet`: a known Arrhenius form with a derivable energy would turn the
  exp(5.4 beta) fit from a fitted law into a predicted one.
- Question B13 can likely be answered by arithmetic alone (two edges per added node ->
  mean degree 4), which would retire a stated puzzle.

### Gaps

- The project cites only three literature items by name (Coumbe et al. arXiv:1711.02685;
  Millan et al. PRR 3, 023015; Ecker/Ecker/Grumiller arXiv:2601.14358 -- FINDINGS:474-475,
  1936-1943) plus Mullins-Sekerka, Choptuik, Cox-Durrett, Myrheim-Meyer, Bombelli,
  Alexander-Orbach-type names in passing. No growth-model (Eden/KPZ/KMC) literature is cited
  anywhere in the corpus; that is the survey's largest blank.

---

## 5. Internal tensions, stale statements, and inter-document disagreements

### Takeaway

The audit of 2026-09-26 was propagated carefully through FINDINGS (supersession notices) but
not through README, SCALING, DIMENSIONAL_COHERENCE, CLAUDE.md, BRANCHES' OPEN rows, or the
code docstrings, several of which still state retracted claims as current. There are also two
places where FINDINGS itself lacks a supersession notice that its own later text requires.

### Cited Findings

**README.md (project tour) -- stale after the audit and the 2026-09-29/10-13 rounds**

- `README.md:50` -- `rules.py` described as "Local update rules (activation, reinforcement,
  majority, rewire)"; eight rules exist (`rules.py:405-414`).
- `README.md:60` -- `cap_dimension_scaling.py`: "does d_eff plateau on a stable value as N
  grows? (it does -- at non-integer values)" with no "not a dimension" qualifier in the table
  (the qualifier appears only later at `README.md:241-245`).
- `README.md:62` -- `coherence.py`: "(`grown` is coherent, I~0.88)"; superseded 2026-10-13
  (FINDINGS:1455: "field fragmented (N/A)").
- `README.md:63` -- `prune_dimension.py`: "showing d slides 1->2 as a crossover"; the 1->2
  knob is retracted (FINDINGS:780-784).
- `README.md:67` -- `shortcut_walkers.py`: "a portal turns out to be *kinetic* classically
  (speed only) and *structural* quantum-mechanically (~50-130x long-run transfer)"; superseded
  2026-09-29 (FINDINGS:1283-1285: "a statement about localization on a tree-like graph").
- `README.md:76` -- `async_censorship.py`: "12-seed paired estimate"; FINDINGS:2277-2291 makes
  the 40-seed run authoritative.
- `README.md:282-299` (section "prune as a tunable-dimension knob") -- "`p` tunes the pruned
  dimension **continuously** from ~1 (ring) to ~2 (mesh) ... So `prune` is a third continuum
  dimension knob alongside the `grown` cap" -- both retracted (FINDINGS:780-784).
- `README.md:301-315` (section "Spatial coherence") -- "on `grown` the field is defined
  almost everywhere and **strongly coherent** (I~0.88, z~87)" -- superseded.
- `README.md:376-382` -- "Four initial topologies are available" followed by five entries;
  `triangular` and `sheet` (added 2026-09-28, `simulation.py:75-89`, commit bf6987d) are
  missing from the list.
- `README.md:155` -- "Seven pre-tuned animations, one for each rule (epidemic spreading,
  Hebbian reinforcement, majority-vote domains, small-world rewiring)" -- four rules named as
  if they were all.

**CLAUDE.md**

- `CLAUDE.md:45` -- "Topologies (`--topology`): `small_world` (default), `scale_free`,
  `lattice`, `random`, `grown`" -- `triangular` and `sheet` missing.

**SCALING.md**

- `SCALING.md:11` -- cap -> d finite-size scaling: "Does the cap->dimension law sharpen to
  clean integers as N grows, or drift? Prerequisite for trusting anything at 1e8" -- presented
  as live; `grown` has no dimension.
- `SCALING.md:12` -- "Spectral dimension via walks on `grown` ... Does `d_s` match ball-growth
  `d_eff`?" -- done 2026-09-27; there is no `d_eff` to match (FINDINGS:978-1038).
- `SCALING.md:40-43` -- "All four rules (activation, reinforcement, majority, rewire) are
  vectorized"; CLAUDE.md:43-44 says six are vectorized in both backends.
- `SCALING.md:17-19` -- "the open, compute-worthy question is whether a *minimal local growth
  rule* reproduces a CDT-like spectral-dimension flow" -- answered (no: slow tree-like rise on
  grown; FLAT 2 on sheet).

**DIMENSIONAL_COHERENCE.md**

- `DIMENSIONAL_COHERENCE.md:144-147` -- "Result: on `grown` the d_eff field is strongly
  coherent (I~0.88, z~87) where the field is whole" -- superseded 2026-10-13; the partial
  retraction note at DC:95-104 covers only the cap -> d claim.
- `DIMENSIONAL_COHERENCE.md:149-150` -- temporal tracking pointer to FINDINGS "Preservation"
  (the `grown` table, itself superseded) rather than to "Preservation of a real dimension".

**LORENTZIAN_SPIKE.md**

- `SPIKE:383-386` -- step 4 described with the "12-seed paired estimate ... woven ratio 0.81,
  t=1.77 n.s."; FINDINGS:2288-2291 later shows the 40-seed run found even that hint was
  noise (woven ratio 1.01, t = -0.08). SPIKE not updated.
- `SPIKE:237-239` -- "a full sweep costs ~4-5 rounds" immediately contradicted by the step-2
  correction box at SPIKE:241-283 (7/24/58); kept deliberately as a record, but a reader of
  the first paragraph alone gets the wrong number.

**BRANCHES.md**

- `BRANCHES:47` -- OPEN row "`sheet` as a registered topology ... `sheet_growth.py` is a
  standalone module" -- `sheet` *is* registered in `create_initial_graph` (`simulation.py:82-89`,
  commit bf6987d 2026-09-28), with beta hard-wired to 3.0; the row should either close or be
  narrowed to "beta as a CLI parameter / rules run on it".
- `BRANCHES:51` -- OPEN row says "coherence (done ...) and the `prune` d(p) curve remain";
  both were done 2026-10-13 (FINDINGS:1446-1486). Stale.
- `BRANCHES:59` vs `BRANCHES:82` -- "Horizon-free quantum hitting-time analogue" is listed
  both as PARKED and as CLOSED ("moot 2026-09-29").
- `BRANCHES:21` -- the row header still reads "mechanism hunt (round 3)" while the body
  describes round 4 as done.

**FINDINGS.md internal**

- `FINDINGS:2175-2211` (Portal "1. Tolerance ... it inflates") carries **no supersession
  notice**, although FINDINGS:800-802 ("'dimension inflation 2.21 -> 3.19 under portals' ...
  are statements about the radius-10 reading") and FINDINGS:1257-1263 (portals destroy, k* =
  20) supersede it. Sections 3 and 3c of the same chapter do carry notices (FINDINGS:2312-2315,
  2391-2395).
- `FINDINGS:280-310` ("Open threads") is a pre-audit status list: "Spatial coherence: done --
  Moran's I confirms `grown`'s d_eff field is spatially coherent" (FINDINGS:287-289),
  "Preservation ... `grown`'s dimension is a stable fixed point" (FINDINGS:292-295), and the
  walker "48x / 131x" headline (FINDINGS:303-310) are presented without the later
  retractions; only "Robustness / scale" (FINDINGS:290-291) is marked superseded.
- `FINDINGS:1108` (section header) states the law as exp(5.8 beta); FINDINGS:1344 refits to
  5.38; FINDINGS:1390 and :1591 quote 5.4; `simulation.py:84` quotes "exp(2.7 + 5.8*beta)".
  Three coefficients for one law across documents (historically ordered, but un-reconciled in
  the code comment).
- `FINDINGS:1160-1164` ("What `sheet` is, restated": "two-dimensional ... about exp(2.7 +
  5.8 beta) nodes: 5e8 at beta = 3") and `FINDINGS:1390-1395` ("final form ... exp(3.3 + 5.4
  beta) ... 0.2 G at beta = 3") and `FINDINGS:1661` ("~exp(3.3 + 5.4 x 3) ~ 2e8") -- the
  revision note at FINDINGS:1390 says "Quote the distribution, not a size", yet the section
  continues to quote sizes, and the beta = 3 margin "a factor 200" (FINDINGS:1395) is a
  mean-curve statement that item 4 says should not be quoted (FINDINGS:1588-1590).
- `FINDINGS:1383-1388` (step 2: "the signature of a nucleation: one protrusion escapes ...
  once it is an arm, grows like a tree") stands in the text without a bracketed withdrawal,
  although item 1 (FINDINGS:1493-1497) found no arm and "the step-2 'arm' picture is
  withdrawn" (FINDINGS:1696).
- `FINDINGS:1049-1055` describes `sheet`'s closure as reading "the same radius as `triadic`";
  `_closable` reads the adjacency of `x`'s two boundary neighbours and their common neighbours
  (`sheet_growth.py:58-63`), i.e. distance 2 from `x` -- consistent, but the attach rate reads
  only endpoint degrees (distance 1); the two event types have different footprints and the
  document does not say so.
- `FINDINGS:1828-1829` -- "TRIGGER CONFIRMED (6 / 8 crossed seeds ...)" vs BRANCHES:44 "12/12
  saved graphs and 8/8 fresh seeds" and BRANCHES:77 "6/8 by the frozen rule, 8/8 within a
  factor 2.5": consistent once the qualifier is read, but the BRANCHES:44 row drops it.

**Code docstrings that state retracted claims**

- `simulation.py:33-36` -- `_grow_dimensional`: "Effective dimension rises with the cap (cap
  6 -> ~2D, the degree of a triangular lattice; cap 8 -> ~3D) ... any emergent dimension is a
  genuine consequence of the local rule" -- retracted 2026-09-26.
- `simulation.py:84` -- `sheet` comment: "a two-dimensional sheet up to ~exp(2.7 +
  5.8*beta) nodes" -- superseded twice (12-seed law; distributional reading).
- `rules.py:293-295` -- `geometrize`: "Bounded degree + locality are the ingredients that
  produce dimension (cf. the 'grown' generator)" -- `grown` produces no dimension.
- `rules.py:32-33` and `README.md:95` describe `rewire` as a local rule in the "All rules are
  local" table (`README.md:88`), while its target is uniform over all nodes (`rules.py:146,
  174`) -- the one rule that is not local by the project's own invariant (`AGENTS.md:19-21`).

**Pre-registration vs result disagreements recorded by the project itself** (not errors; listed
because a reader of the spec alone would be misled)

- Spec `2026-08-03-step5-barrier-async-readiness.md:14-17, 82-84` calls `grown` the alpha ~
  0.5 positive control and says the banked fit reached ~1e5; corrected in spec
  `2026-09-27-step5-async-barrier-preregistration.md:17-29` (lattice is the control, grown's
  banked alpha is 0.194, the grid was N = 1000..16000, sequential async is fast enough).
- Spec `2026-09-27-spectral-flow-preregistration.md:19-21` predicts RUNAWAY on `grown`;
  refuted (FINDINGS:1006-1008).
- Spec `2026-10-13-round2-preregistrations.md:4-15` -- "jaga is not required"; the status
  addendum on the same page records that the desktop ran out of memory and most of the round
  moved to jaga.
- Spec `2026-10-13-edge-spectrum-preregistration.md:34-38` excluded k = 1-3 only; the result
  needed k <= ~7 excluded (FINDINGS:1727-1731).
- Spec `2026-10-13-round2-preregistrations.md:33-39` (item 1) defines the object "arm"; the
  object does not exist in the data (FINDINGS:1493-1497).

### Inferences

- The README is the document most likely to mislead a literature reviewer: three of its
  module blurbs (coherence, prune knob, walkers) and two of its sections state results that
  FINDINGS retracted weeks earlier, and its topology list omits the two substrates on which
  every surviving portal/throat result now rests.
- Within FINDINGS the supersession discipline is uneven: notices were added where the
  authors re-ran a table, but not where a *reading* was replaced by a later section (portal
  tolerance; the step-2 nucleation paragraph; the "Open threads" list).

### Gaps

- I did not check `simulation_fast.py`, `measure.py`, `visualize.py`, `showcase.py` or
  `sweep.py` docstrings for stale claims; only the four modules the brief named were skimmed.
- Memory files outside the repository (`plan-after-audit.md`, `emergence-research-state.md`)
  were not read, per the "repository documents" scope; they may hold owner decisions on the
  BRANCHES rows flagged stale above.
