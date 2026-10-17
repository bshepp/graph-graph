# Kinetic errors, frozen defects and rare-event nucleation: literature map for the `sheet` dead-corner crossover

Scope note. These notes map five literatures onto one project fact: the `sheet` generator
(degree cap c = 6, closure bias beta) stays a flat triangular disc up to a size N* that is
exponentially distributed across seeds (Weibull shape 0.8-1.0, CI covering 1), with
mean ln N* = 3.3 + 5.4 beta, and the failure is nucleated by one permanent "dead corner"
(a boundary node at the cap whose two boundary neighbours are also at the cap).

Verification legend used throughout:
- **[verified]** — read from the source (full text or abstract) during this survey.
- **[abstract only]** — bibliographic data and abstract verified; theorem/formula details not read.
- **[unverified]** — recalled from training knowledge, not checked against the source in this
  session. Treat citations marked this way as "look up before quoting".
- **[derived]** — my own calculation from verified definitions; not a published claim.

Mapping strength labels: *direct* (same mathematical object), *analogous* (same structure,
different mechanism), *speculative* (suggestive only).

---

## Q1. kTAM error kinetics: per-site error rate vs G_se, proofreading, frozen errors, stalling

### Takeaway
In Winfree's kinetic Tile Assembly Model the per-tile error rate at the fastest useful
operating point scales as eps ~ e^{-G_se} (up to O(1) factors) and the growth speed as
r ~ beta_k eps^2; k x k proofreading tile sets push this to r ~ beta_k eps (and, in the
idealised kinetic-trap model, block error rates ~ eps^k). An error is "frozen" when the
mismatched tile is buried by further growth before it detaches. The project's beta plays
the role of G_se only *analogously*: both appear as log-odds biases, but the kTAM
exponent comes from a Boltzmann detachment rate while the project's 5.4 beta comes from
the composition of several independent closure decisions.

### Cited Findings
- Winfree & Bekbolatov, "Proofreading tile sets: Error correction for algorithmic
  self-assembly", DNA Computing 9 (DNA9, Madison 2003), Springer LNCS 2943, pp. 126-144
  (2004), DOI 10.1007/978-3-540-24628-2_13. Abstract **[verified]**: "it has been shown
  that controlling temperature and tile concentration can push the error rate arbitrarily
  low, but at the cost of slower assembly. For tile sets that directly implement blocked
  cellular automata, r ≈ β ε² was optimal. ... proofreading tile sets ... exploit the
  cooperativity of tile assembly reactions to dramatically improve the scaling behavior to
  r ≈ β ε and better." — [CaltechAUTHORS record](https://authors.library.caltech.edu/22768)
  Mapping: **analogous** — the project's rate-per-node ~ e^{-5.4 beta} is a
  speed/fidelity trade-off of the same type (larger beta = more closure attempts per
  accepted attachment = slower effective "growth per decision").
- kTAM rate definitions (as restated in a 2024 arXiv paper that uses the kTAM): G_se is
  "the unitless, sign-reversed free energy of a strength-one interaction, with detachment
  rate r_off = k_f e^{-b G_se}, where b is the total interaction strength between a tile
  and its neighbours." **[verified via secondary source]** —
  [arXiv:2405.10911](https://arxiv.org/pdf/2405.10911). The forward (attachment) rate is
  r_f = k_f e^{-G_mc} with G_mc = -ln([tile]/u_0) **[unverified; standard, Winfree 1998
  thesis Ch. 6]**.
- Kinetic-trap error estimate **[unverified; Winfree 1998 PhD thesis, Caltech, Ch. 6; also
  Sec. 2 of Winfree & Bekbolatov 2004]**: near the melting point G_mc = 2 G_se − ε'
  (ε' > 0 small), a correct tile (b = 2) is held with off-rate r_{r,2} = k_f e^{-2G_se},
  a mismatched tile (b = 1) with r_{r,1} = k_f e^{-G_se}. The chance a mismatched tile is
  buried before it detaches is
  eps ≈ r_f / (r_f + r_{r,1}) ≈ e^{-(G_mc − G_se)} = e^{-(G_se − ε')}, while the net growth
  rate is r ≈ r_f − r_{r,2} = k_f e^{-2G_se}(e^{ε'} − 1) ≈ k_f ε' e^{-2G_se}. Eliminating
  G_se gives r ∝ eps² up to logarithms — the r ≈ β ε² in the verified abstract.
- Proofreading scaling **[abstract only + unverified detail]**: replacing each tile by a
  k x k block of tiles such that a single mismatch cannot be followed by any correct
  attachment until k tiles have attached in error gives, in the idealised model, block
  error rate ~ eps^k; Winfree & Bekbolatov analysed k = 2 and 3 **[unverified detail]**.
- Chen & Goel, "Error free self-assembly using error prone tiles", DNA Computing 10,
  Springer LNCS 3384, pp. 62-75 (2005), DOI 10.1007/11493785_6 **[abstract only]**:
  "snaked proof-reading ... can correct both growth and nucleation errors"; later analyses
  state k must be Ω(log n) for a shape of size n. —
  [unpaywall record](https://unpaywall.org/10.1007%2F11493785_6);
  [Stanford copy, 3D criteria paper](https://web.stanford.edu/~ashishg/papers/criteria3d.pdf)
  Mapping: **analogous** — snaked proofreading works by making the *wrong* growth path
  require several improbable events in series; the project's dead corner already needs
  several consecutive anti-closure decisions, which is why the exponent is 5.4 rather
  than ~1.
- Chen, Schulman, Goel, Winfree, "Reducing facet nucleation during algorithmic
  self-assembly", Nano Letters 7(9), 2913-2919 (2007) **[abstract verified via Caltech
  record; author list unverified]**: "snaked proofreading tile set is also designed to
  reduce nucleation errors on crystal facets ... the rate of facet nucleation was 4-fold
  smaller for snaked proofreading tile sets" than for uniform proofreading. —
  [CaltechAUTHORS](https://resolver.caltech.edu/CaltechAUTHORS:20110309-104159746)
  Mapping: **analogous** — "facet nucleation" is an error that starts on an otherwise
  stalled boundary; the project's dead corner is the opposite failure (a boundary site
  that can never advance), but both are boundary-local rare events.
- Schulman & Winfree, "Synthesis of crystals with a programmable kinetic barrier to
  nucleation", PNAS 104(39), 15236-15241 (2007) **[abstract verified]**: zig-zag ribbons of
  fixed width; "nucleation rates decrease for wider ribbons. It is therefore possible to
  program the nucleation rate by choosing a ribbon width"; a seed removes the barrier. —
  [CaltechAUTHORS](https://resolver.caltech.edu/CaltechAUTHORS:SCHUpnas07)
- Schulman & Winfree, "Programmable control of nucleation for algorithmic self-assembly",
  SIAM J. Comput. 39(4), 1581-1616 (2009), arXiv:cond-mat/0607317 **[abstract verified]**:
  "under suitable physical conditions, enlarging the tile set reduces the rate of spurious
  nucleation exponentially"; simulations show < 1% spurious nucleation for large seeded
  crystals. — [arXiv](https://arxiv.org/abs/cond-mat/0607317v2);
  [Caltech PDF](https://authors.library.caltech.edu/18675/1/Schulman2009p9781Siam_J_Comput.pdf)
  Mapping: **analogous** — this is the cleanest published statement that a rare-event
  rate is exponential in a design-controlled "barrier size"; the project's exp(−5.4 beta)
  is the same shape with beta as the barrier height per decision.
- Evans & Winfree, "Physical principles for DNA tile self-assembly", Chem. Soc. Rev. 46,
  3808-3829 (2017), DOI 10.1039/C6CS00745G **[bibliographic data verified; body not
  readable in fetch]**. Review of kTAM physics, error types (growth/mismatch errors,
  facet errors, spurious nucleation), and the experimental status of proofreading. —
  [RSC](https://pubs.rsc.org/en/content/articlelanding/2017/cs/c6cs00745g);
  [Caltech PDF](https://www.dna.caltech.edu/Papers/principles2017.pdf)
- Error taxonomy (from a kTAM concentration-optimisation paper) **[verified snippet]**:
  "facet error: an incorrect tile attaching where no tile is supposed to attach;
  nucleation error: single tiles attaching to each other to form a lattice without the seed
  structure." — [arXiv:1210.5786](https://arxiv.org/pdf/1210.5786)
- Malformed-tile fragility of snaked proofreading **[verified snippet]**: Meng (Natural
  Computing 2010) found the Chen-Goel snaked set "fails to form even moderately sized tile
  assemblies when malformed tiles are present" — a published example of an
  error-correction scheme that *creates* permanent growth stalls. —
  [Meng & Kashyap, malformed tiles](https://ece.iisc.ac.in/~nkashyap/Papers/malformed_tiles.pdf);
  [Natural Computing record](https://unpaywall.org/10.1007%2FS11047-010-9234-9)
  Mapping: **analogous** — stalls caused by a rule that *forbids* some attachments are
  exactly the project's failure class.
- Reliability definition **[verified snippet]**: "reliability of a tile set [is] the
  probability that the assembly process ... completes without any incorrect tiles being
  present in the terminal configuration" (Winfree-style kinetic trapping analysis). —
  [arXiv:1412.7219](https://arxiv.org/pdf/1412.7219)

### Inferences
- Is beta analogous to G_se? Partially. In kTAM the *same* parameter G_se sets both the
  correct-bond lifetime and the error-bond lifetime, so eps ~ e^{-G_se} with exponent 1
  (per bond). In the project beta is a bias between two *moves* (closure vs. plain
  attachment) and the dead corner needs several moves to go the unlikely way; the
  measured slope a = 5.4 is then naturally read as "the number of independent
  anti-closure decisions, weighted by how each depends on beta". If the acceptance
  probability of closure is logistic in beta (p_close/p_attach = e^{beta}), a dead corner
  that requires m closures to be *skipped* has probability ~ e^{-m beta}, predicting
  a ≈ m. Testing whether 5.4 is close to an integer count of skipped closures in the
  minimal dead-corner motif is a direct check (see Q6/new paths).
- kTAM has no exact analogue of "permanent" stalls: every kTAM configuration has a
  nonzero detachment rate, so errors are frozen only in the limit of burial. The project's
  rule is the r_{r,b} = 0 limit (irreversible aTAM-like growth with a *forbidden*
  attachment). This makes the project closer to aTAM "deadlock"/blocked configurations
  than to kTAM errors — a point that matters for which remedies transfer: proofreading
  (which is purely combinatorial) transfers; annealing (which needs r_{r,b} > 0) requires
  adding a detachment move.
- kTAM proofreading suppresses the *per-block* error rate to eps^k but costs a factor k² in
  tiles; in the project the analogue cost is extra local checks per attachment (not extra
  nodes), so the "resolution" price of proofreading does not apply directly.

### Gaps
- I could not read the Winfree-Bekbolatov or Winfree-1998 full text in this session; the
  explicit eps ≈ e^{-(G_se−ε')} and the k-dependence of block error rates are reconstructed
  from memory and should be checked against Winfree 1998, Ch. 6 and W&B 2004 Sec. 2-3.
- I did not find a published kTAM/aTAM result specifically on *growth stalling at a
  mismatched site* that quantifies stall duration; aTAM "deadlock" (no tile fits) is a
  known concept in the theory literature but I have no verified citation for it here.
- Experimental per-tile error rates with proofreading (Fujibayashi et al. 2008 Nano Lett;
  Barish et al. 2009 PNAS; Woods et al. 2019 Nature, ~0.03% per tile) are **[unverified]**
  and not cited above.

---

## Q2. Irreversible growth with a local rule that can fail: RSA/CSA jamming, kinetically constrained models, pinned interfaces

### Takeaway
Random/cooperative sequential adsorption and kinetically constrained models are the
canonical physics of *irreversible local rules that leave permanently blocked sites*;
the jammed state is non-Gibbsian and the frozen-defect density depends on rate ratios.
The closest interface analogue is quenched pinning (directed-percolation depinning), where
a sparse density of blocked cells pins the front locally and the front flows around them —
leaving overhangs/fjords — until the blocked-cell density crosses a percolation threshold.
I found no verified source stating an exponential-in-bias blocked-site density for CSA
specifically; that scaling is an inference.

### Cited Findings
- Evans, "Random and cooperative sequential adsorption", Rev. Mod. Phys. 65, 1281 (1993),
  DOI 10.1103/RevModPhys.65.1281 **[citation verified; content not read]**. Review of RSA
  (irreversible, exclusion only) and CSA (rates depend on neighbourhood); jamming
  coverages; the statement, in a snippet citing it, that "the distribution of filled or
  transformed sites in RSA and CSA is not described by an equilibrium Gibbs measure, even
  for the saturation 'jammed' state". — [citing paper](https://arxiv.org/pdf/0806.0131);
  [Evans & Nord 1992 ladder/strip jamming](https://commons.emich.edu/faculty_sch/644)
  Mapping: **analogous** — the project's growth is a sequential irreversible process whose
  dead corners are jammed sites; the sheet's final state is likewise not a Gibbs measure
  over triangulations.
- CSA with nearest-neighbour exclusion and next-nearest-neighbour interaction
  **[abstract-level]**: jamming coverage and structure depend continuously on the
  interaction ratio. — [arXiv:0806.0131](https://arxiv.org/pdf/0806.0131)
- Jamming fluctuations scale as a power of system size with a universal exponent for
  RSA on homogeneous and heterogeneous media **[abstract-level]**. —
  [arXiv:cond-mat/0312170](https://arxiv.org/pdf/cond-mat/0312170);
  [arXiv:cond-mat/0312171](https://arxiv.org/pdf/cond-mat/0312171)
- Kinetically constrained models (Fredrickson-Andersen, Kob-Andersen; review Ritort &
  Sollich, Adv. Phys. 52, 219 (2003)) **[unverified]**: a local constraint forbids moves
  unless the neighbourhood satisfies a condition; "blocked" configurations are
  permanent; glassy slowdown is Arrhenius/super-Arrhenius in the facilitation parameter.
- Quenched-disorder interface depinning (directed-percolation depinning, DPD):
  Tang & Leschhorn, Phys. Rev. A 45, R8309 (1992); Buldyrev et al., Phys. Rev. A 45,
  R8313 (1992) **[unverified]**: an interface advancing through a lattice with a density
  p of blocked cells is pinned globally when the blocked cells form a directed-percolating
  path; below threshold the front advances leaving the blocked cells behind.
  Mapping: **analogous** — a dead corner is a blocked cell of infinite residency *created
  by the dynamics itself*; at small e^{-5.4 beta} density the front is never globally
  pinned but each blocked cell leaves a permanent fjord/overhang in the boundary.
- Kibble-Zurek contrast **[unverified]**: in finite-rate quenches through a continuous
  transition the frozen defect density scales as a *power* of the quench rate. The
  project's exponential dependence on beta therefore does not look like a quench
  mechanism; it looks like a Boltzmann/Arrhenius activated rate (see Q4).

### Inferences
- Exponential-in-bias blocked-site densities arise generically whenever a blocked motif
  requires m independent "unfavourable" local events each with probability ∝ e^{-beta}:
  density ~ e^{-m beta}. This is a counting statement, not a theorem from the RSA
  literature; CSA papers typically present jamming coverages as smooth functions of rate
  ratios rather than in this activated form.
- Unlike RSA, in which blocked sites are *bulk* (interior) and the jammed state is reached
  at finite coverage, the project's blocked sites are *boundary* objects that redirect
  growth — hence the structural (dimension) consequence rather than a density consequence.

### Gaps
- No verified source in this session gives a blocked-site density of the form e^{-a beta}
  in a CSA model; this remains an inference.
- The DPD/quenched-pinning references and threshold values are unverified here.

---

## Q3. Crystal growth: step pinning by immobile impurities, dislocations, notches and the Gibbs-Thomson criterion

### Takeaway
Cabrera-Vermilyea (1958) step pinning is the closest crystal-growth analogue of a dead
corner: an immobile "stopper" on a step blocks advance locally; the step bows around it,
and whether the step can pass depends on whether the gap between stoppers exceeds the
critical 2D-nucleus diameter 2 r_c set by the Gibbs-Thomson relation. In the project the
"gap" the boundary must bridge is always a fixed one-hop closure, and the rule forbids it
at a dead corner, so the fjord can never heal — the stopper has infinite residency and
r_c is effectively infinite. Screw dislocations (BCF) are the standard *permanent*
growth-shaping defect but they accelerate rather than block growth, so that mapping is
weaker.

### Cited Findings
- Cabrera & Vermilyea, "The growth of crystals from solution", in *Growth and Perfection of
  Crystals* (Wiley, 1958) **[cited through secondary sources; original not fetched]**.
  Secondary description **[verified]**: steps are pinned by a periodic array of immobile
  impurities; "Cabrera and Vermilyea assumed as an approximation that although the
  curvature κ of the moving front varied with time, it was constant in space"; "If the
  stopper separation d is below the diameter of the critical nucleus, 2 r_c, the step
  cannot percola[te]". — [Lee-Thorp, Shtukenberg & Kohn, step anisotropy
  paper](https://math.nyu.edu/~kohn/papers/leethorp-shtukenberg-kohn-step-anisotropy.pdf)
  Mapping: **analogous** — dead corner = stopper of infinite lifetime at the growth front.
- Gibbs-Thomson critical radius **[unverified; standard, e.g. Pimpinelli & Villain,
  *Physics of Crystal Growth*, CUP 1998]**: for a 2D island/step, r_c = γ Ω / (k_B T ln S),
  with γ the step free energy per unit length, Ω the area per unit, S the supersaturation.
  A notch or gap of radius < r_c cannot be bridged by nucleation; a step pinned at two
  stoppers with spacing d stops when d < 2 r_c (the CV "dead zone" at low S).
- Thermal-fluctuation correction to CV **[verified via summary]**: a PRL finds that
  piercing of an impurity fence "is not solely determined by the Gibbs-Thomson effect, as
  assumed by Cabrera-Vermilyea" and that "step retardation is dominated by the formation of
  critically sized fluctuations" near growth cessation. —
  [Phys. Rev. Lett. 114, 245501 (2015)](https://link.aps.org/doi/10.1103/PhysRevLett.114.245501)
  (authors not confirmed in this session).
- Point vs large impurities **[verified via summary]**: kinetic Monte Carlo (CSIC/UB
  group) finds "the CV model only applies in the case of effectively large impurities with
  long surface residency times"; for point impurities "much higher impurity densities are
  required to induce step blocking because thermal fluctuations drive the system past the
  limit implied by the Gibbs Thomson effect"; multilayer steps can "boost" escape from the
  "death zone". — [UB seminar summary](https://www.ub.edu/portal/web/ciencies-terra/detall/-/detall/mineral-growth-close-to-equilibrium);
  [Radboud 1996 experiments on K2Cr2O7](https://vsc.science.ru.nl/1996/96-15.html)
  Mapping: **analogous** — the project rule has no thermal fluctuation that could "pierce
  the fence"; the dead corner is a large, infinitely resident stopper, i.e. exactly the
  regime where CV logic applies.
- Burton, Cabrera & Frank, "The growth of crystals and the equilibrium structure of their
  surfaces", Phil. Trans. R. Soc. A 243, 299-358 (1951) **[unverified]**: screw dislocation
  provides a permanent step source (spiral growth); step kinetics via adatom diffusion.
  Mapping: **speculative** — a permanent defect that organises growth, but it enhances
  rather than blocks growth.
- Faceting at pinned steps **[verified snippet]**: Lee-Thorp, Shtukenberg & Kohn note the
  CV analysis "involved uncontrolled approximations and did not consider what happens when
  step anisotropy induces faceting". — [same PDF](https://math.nyu.edu/~kohn/papers/leethorp-shtukenberg-kohn-step-anisotropy.pdf)
  Mapping: **analogous** — the project's boundary is intrinsically faceted (triangular
  lattice directions), so the fjord walls will be facets; this paper's framing of faceted
  step-pinning is the right one to compare notch shapes against.

### Inferences
- Heal vs deepen: in CV the notch heals when the front's local curvature radius exceeds
  r_c. In the project, closure only ever links a new node to neighbours of its attachment
  node, so the "bridgeable gap" is one lattice spacing; a fjord of mouth width ≥ 2 hops
  can only be closed by filling it from the inside, and the dead corner at its root cannot
  accept the required edge. So the project is the r_c → ∞ limit of CV: every notch rooted
  at a dead corner deepens. This explains why the single dead corner is sufficient — no
  density threshold is needed, unlike DPD/CV where sparse stoppers are passed.
- Grain-boundary analogue: when the two boundary arcs flanking a fjord grow past each
  other they cannot register (no closure across the fjord), so the fjord is a permanent
  "open grain boundary". No published source found for this specific analogy; speculative.

### Gaps
- Original CV 1958 text and BCF 1951 not fetched; the r_c formula is textbook-standard but
  unverified here.
- No source found that treats a *single* permanent stopper in 2D step-flow and asks
  whether the resulting notch deepens indefinitely; the CV literature works with arrays
  of stoppers.

---

## Q4. Nucleation statistics: exponential first-defect size, constant hazard, Arrhenius-in-bias, censored estimation

### Takeaway
A constant per-site hazard gives an exponential (Weibull shape 1) distribution of the
first-defect size — the standard Poisson/weakest-link result used in nucleation
experiments at constant supersaturation (Sear 2014). The project's Weibull shape
0.8-1.0 with CI covering 1 is consistent with this. The proper estimator for a constant
hazard with right-censored runs is the exposure-based MLE lambda_hat = d / (total
exposure), with chi-square CIs; a Nelson-Aalen cumulative-hazard plot tests constancy.
The ln N* ∝ beta dependence is Arrhenius-like (barrier linear in beta), not the
CNT form (barrier ∝ 1/ln S or 1/(ln S)^2).

### Cited Findings
- Sear, "Quantitative studies of crystal nucleation at constant supersaturation:
  experimental data and models", CrystEngComm 16, 6506-6522 (2014), DOI 10.1039/C4CE00344F
  **[abstract/summary verified]**: for a constant nucleation rate the fraction of samples
  not yet nucleated decays exponentially in time; when "it deviates from exponential
  behaviour ... it may often be possible to fit ... the Weibull distribution of
  extreme-value statistics ... this fit can be used to determine the scaling of the median
  time to nucleation with volume"; "in many experiments the dynamic range of nucleation
  times is too large to be measured. This is presumably due to highly variable impurities".
  — [RSC](https://pubs.rsc.org/en/content/articlehtml/2014/ce/c4ce00344f);
  [DOI](https://dx.doi.org/10.1039/C4CE00344F)
  Mapping: **direct** — replace "time" by "nodes added" and "volume" by "boundary
  length"; the project's N* survival curve is exactly Sear's P(not nucleated) curve.
- Weibull fits to droplet nucleation data **[verified snippet]**: Weibull distributions
  fitted to supersaturations at nucleation in sessile microdroplets. —
  [arXiv:2111.11775](https://arxiv.org/pdf/2111.11775)
- Schulman & Winfree 2009 (see Q1): spurious nucleation rate exponential in barrier size
  — [arXiv:cond-mat/0607317](https://arxiv.org/abs/cond-mat/0607317v2). Mapping:
  **analogous** — a published model where ln(rate) is linear in a design parameter.
- Classical nucleation theory **[unverified; Kashchiev, *Nucleation: Basic Theory with
  Applications*, Butterworth-Heinemann 2000]**: J = A exp(−ΔG*/k_B T), with
  ΔG* = 16πγ³Ω²/(3 (k_B T ln S)²) in 3D and ΔG* = πγ²Ω/(k_B T ln S) in 2D. First-nucleation
  time in volume V at constant J is exponential with mean 1/(J V).
- Censored-data estimation **[unverified; standard survival analysis, e.g. Lawless,
  *Statistical Models and Methods for Lifetime Data*, Wiley 2003]**: for exponential
  lifetimes with right censoring, lambda_hat = d / T where d = number of observed events and
  T = Σ_i min(N*_i, N_cens,i) is the total exposure (censored runs contribute their full
  exposure); an exact two-sided (1−α) CI is
  [χ²_{2d, α/2} / (2T), χ²_{2d+2, 1−α/2} / (2T)]. The Nelson-Aalen estimator
  Λ_hat(N) = Σ_{N_i ≤ N} d_i / n_i of the cumulative hazard should be a straight line
  through the origin if the hazard is constant; a Weibull fit's shape CI covering 1 is
  the parametric version of the same test.

### Inferences
- Because the project's hazard is per *added node* while dead corners can only form at
  the *boundary*, a constant per-node hazard implies that the number of "eligible" boundary
  sites per added node is roughly constant — plausible if the dead-corner motif can only
  be completed by the most recently attached nodes (a fixed-size active zone), not by
  arbitrary boundary nodes. If instead eligibility scaled with boundary length ∝ √N,
  the hazard would grow as N^{1/2} and Weibull shape would be 1.5. The observed shape
  0.8-1.0 therefore says something concrete: the nucleation site is the growth tip, not the
  whole boundary. This is worth confirming by recording *where* (how many steps old) the
  three nodes of the first dead corner are.
- Arrhenius vs CNT: ln N* linear in beta with slope 5.4 indicates a fixed-size critical
  motif whose "cost" is 5.4 units of beta, not a critical size that itself depends on
  beta. In CNT terms this is the regime where the critical nucleus is a fixed small
  cluster (a few units) and only the Boltzmann factor varies — the "atomistic" or
  small-cluster limit of nucleation theory (Kashchiev's atomistic nucleation,
  **[unverified]**).

### Gaps
- No source found that treats right-censored nucleation runs explicitly in a survival
  framework (Sear 2014 notes dynamic-range problems but, from what I could read, does not
  give censored estimators). The estimator above is textbook survival analysis, not from
  the nucleation literature.

---

## Q5. Degree-constrained graph growth and combinatorial curvature: when is a degree-≤6 triangulation flat, and what does a defect do?

### Takeaway
Combinatorial curvature κ(v) = 1 − |v|/2 + Σ_{f∋v} 1/|f| makes the degree cap exactly
meaningful: an interior vertex of a triangulation has κ = (6 − |v|)/6, so cap 6 is the
unique choice giving κ ≥ 0 with a flat (κ = 0) bulk. Hua-Jost-Liu prove that a
tessellation with κ ≥ 0 everywhere has at most quadratic volume growth (|B_r| ≤ C r²), and
DeVos-Mohar that κ > 0 everywhere forces finiteness (≤ 3444 vertices, later ≤ 208). The
converse direction is weaker than the question presumes: Keller-Peyerimhoff's exponential
growth and Cheeger bounds need κ ≤ −C < 0 at *every* vertex, so a *single* degree-7
vertex or a single un-closable boundary vertex does not by itself force exponential
growth; a positive *density* of negative curvature does. For the project this means
the dead corner's role is to seed a growing region of negative-curvature boundary (the
fjord), not to be negative curvature "enough" on its own.

### Cited Findings
- Keller & Peyerimhoff, "Cheeger constants, growth and spectrum of locally tessellating
  planar graphs", Math. Z. 268, 871-886 (2011), DOI 10.1007/s00209-010-0699-0,
  arXiv:0903.4793 **[verified, full text]**:
  - corner curvature κ_C(v,f) = 1/|v| + 1/|f| − 1/2 (1/|f| = 0 for an infinigon);
  - vertex curvature κ(v) = Σ_{f∋v} κ_C(v,f) = 1 − |v|/2 + Σ_{f∋v} 1/|f|;
  - Cheeger constants α(G) = inf |∂_E W|/|W| and α̃(G) = inf |∂_E W|/vol(W) over finite W;
  - exponential growth μ(G, v_0) = limsup_n (log σ_n)/n with σ_n = |S_n(v_0)|;
  - Theorem 1 (|v| ≤ p, |f| ≤ q): if C := inf_v (−κ(v)) > 0 then α(G) ≥ 2 C_{p,q} C; if
    c := inf_v (−κ(v)/|v|) > 0 then α̃(G) ≥ 2 C_{p,q} c, with C_{p,q} = 1 (q = ∞),
    1 + 2/(q−2) (q < ∞, p = ∞), (1 + 2/(q−2))(1 + 2/((p−2)(q−2)−2)) (p, q < ∞); "sharp in
    the case of regular trees";
  - Proposition 1.2: for regular G_{p,q} with −κ = C ≥ 0, μ ≥ log(1 + 2q C/(q−1));
  - Theorem 4: no cut locus and |v| ≤ p ⇒ μ(G) ≤ log(p−1) (the tree bound);
  - Corollary 1.3: c > 0 ⇒ λ_0(G) ≥ 1 − √(1 − (2C_{p,q}c)²).
  - The paper notes flat tessellations G_{3,6}, G_{4,4}, G_{6,3} have κ = 0 and that
    strictly positive curvature implies finiteness (DeVos-Mohar); it states no Cheeger or
    growth conclusion for κ ≥ 0. — [ar5iv full text](https://ar5iv.arxiv.org/html/0903.4793);
    [arXiv](https://arxiv.org/pdf/0903.4793)
  Mapping: **direct** for the curvature bookkeeping; **analogous** for the growth theorems
  (they are for infinite tessellations, the project has a finite growing disc).
- Keller & Peyerimhoff, "Geometric and spectral properties of locally tessellating planar
  graphs", arXiv:0805.1683 (2008) **[abstract only]**: bounds on Cheeger constant and
  exponential growth in terms of combinatorial curvature. —
  [ar5iv](https://ar5iv.labs.arxiv.org/html/0805.1683)
- Baues & Peyerimhoff, "Curvature and geometry of tessellating plane graphs", Discrete
  Comput. Geom. 25, 141-159 (2001), DOI 10.1007/s004540010076 **[abstract only]**: "growth
  of plane tessellations ... can be bounded from below using upper bounds on combinatorial
  curvature"; explicit growth and isoperimetric estimates for negatively curved
  tessellations when geodesics extend to infinity (true for (p,q) ∈ {(3,6),(4,4),(6,3)}
  and beyond). Companion: "Geodesics in non-positively curved plane tessellations", Adv.
  Geom. (2006): non-positive corner curvature ⇒ every geodesic extends to infinity. —
  [Durham repository](https://durham-repository.worktribe.com/output/1591334);
  [Springer](https://link.springer.com/doi/10.1007/s004540010076)
- Higuchi, "Combinatorial curvature for planar graphs", J. Graph Theory 38, 220-229 (2001)
  **[unverified]**: introduces κ(v) in the above form and conjectures κ > 0 everywhere ⇒
  finite.
- DeVos & Mohar, "An analogue of the Descartes-Euler formula for infinite graphs and
  Higuchi's conjecture", Trans. AMS 359, 3287-3300 (2007) **[abstract verified]**: for a
  finite polygonal surface R, Σ_v κ(v) = 2π χ(R) (angle form); "If f(v) is everywhere
  positive, then G is finite. Furthermore, if G is neither a prism, nor an antiprism, then
  G has at most 3444 vertices"; later sharpened to 208 (optimal) per a secondary source. —
  [AMS](https://www.ams.org/tran/2007-359-07/S0002-9947-07-04125-6/);
  [SFU PDF](https://www.sfu.ca/~mdevos/papers/higuchi.pdf);
  [Open Problem Garden](https://openproblemgarden.org/op/what_is_the_largest_graph_of_positive_curvature)
  Mapping: **direct** — a cap of 5 would force a finite closed shell (icosahedron-like);
  cap 6 is the minimal cap allowing unbounded flat growth.
- Hua, Jost & Liu, "Geometric analysis aspects of infinite semiplanar graphs with
  nonnegative curvature", J. reine angew. Math. 700, 1-36 (2015), arXiv:1107.2826
  **[abstract verified; theorem restated in a secondary source]**: for a tessellation with
  κ ≥ 0 (Higuchi's sense) there is C depending only on the maximal face degree with
  |B_r| ≤ C r² for all r; also volume doubling, Poincaré inequality, parabolicity, and a
  polynomial-growth harmonic function theorem. — [arXiv](https://arxiv.org/abs/1107.2826v2);
  [MPI MiS preprint](https://www.mis.mpg.de/publications/preprints/2011/prepr2011-44.html);
  [restatement in arXiv:1612.08676](https://arxiv.org/pdf/1612.08676)
  Mapping: **direct** — this is the theorem behind "bounded-degree-6 planar
  triangulations are at most quadratic-growth"; combined with the κ = 0 bulk it is why the
  sheet reads d_eff ≈ 2 before the crossover.
- Hua & Jost sequel, arXiv:1112.6282 **[abstract-level]**: optimal (linear in growth rate)
  dimension estimate for polynomial-growth harmonic functions. —
  [arXiv](https://arxiv.org/abs/1112.6282v3)
- Random Apollonian networks **[verified summary]**: start from a triangle; repeatedly
  pick a bounded face uniformly and join a new vertex to its three corners; after n − 3
  steps a random triangulation on n vertices results; degree distribution is scale-free,
  degrees unbounded; "asymptotically almost surely, every path in a RAN has length o(n)"
  (Ebrahimzadeh, Farczadi, Gao, Mehrabian, Sato, Wormald, Zung, arXiv:1303.5213). —
  [arXiv](https://arxiv.org/pdf/1303.5213)
  Mapping: **analogous** (contrast case) — RAN is also "attach and close two triangles"
  but with *no* degree cap and with interior insertion; it is small-world (log diameter),
  i.e. the uncapped version of the project's move set has no dimension at all. The cap
  is what buys flatness.
- Haslett, Bullock & Brede, "Planar growth generates scale-free networks", arXiv:1602.03774
  (2016) **[abstract verified]**: nodes placed at random in the unit square, edges added
  without crossing; RAN and deterministic Apollonian are extreme cases; "allowing edge
  crossings with some probability produces a smooth crossover from power-law to exponential
  degree distributions". — [arXiv](https://arxiv.org/pdf/1602.03774v1)
  Mapping: **speculative** — a planar-growth model with a tunable crossover, but the
  control parameter and the order parameter (degree distribution) differ from the
  project's.
- No degree-bounded RAN or "degree-capped growing triangulation" paper surfaced in
  this survey (searched; nothing found). **[gap]**

### Inferences (all **[derived]** from the verified K-P definitions and Euler's formula)
- Interior vertex of a triangulation: κ(v) = 1 − |v|/2 + |v|/3 = (6 − |v|)/6. So |v| = 5
  gives +1/6 (a 60° positive disclination), |v| = 6 gives 0, |v| = 7 gives −1/6.
- Boundary vertex of a triangulated disc with the outer face treated as an infinigon
  (1/|f| = 0) and degree d (so d − 1 triangles): κ(v) = 1 − d/2 + (d−1)/3 = (4 − d)/6.
  Degree 2: +1/3; 3: +1/6; 4: 0; 5: −1/6; 6 (at the cap): **−1/3**. A dead corner is a
  boundary vertex with κ = −1/3 that can never change.
- Gauss-Bonnet for the disc: with the outer face as an infinigon, Σ_v κ(v) = V − E + (F−1)
  = 1 (Euler). Interior vertices contribute 0 when all have degree 6, so the boundary
  carries total curvature exactly +1 (the discrete "exterior angles sum to 2π"). A
  permanent −1/3 at a dead corner must be balanced by extra +1/6 / +1/3 convex boundary
  vertices elsewhere; this forces a longer boundary for the same area — a lower
  isoperimetric ratio, i.e. the beginning of a positive Cheeger constant. Each additional
  cap-saturated boundary vertex that the fjord creates adds another −1/3 that must be
  compensated, so the boundary length grows faster than √N. This is the combinatorial
  content of "the disc deforms around the dead corner".
- The project's crossover is *not* an interior-disclination story (cap 6 forbids degree
  7), so KTHNY-style 5/7 defect language does not apply literally (see Q6). It is an
  *infinigon* story in K-P's sense: an un-closable boundary region behaves like a face of
  infinite degree. K-P Theorem 1 then says: once a positive fraction of vertices sit on
  infinigon boundaries at κ ≤ −C, the Cheeger constant is bounded below and growth is
  exponential — i.e. once fjords proliferate, the structure is tree-like, which matches
  the reported "stops being 2D at large scale" and the earlier `grown` audit finding.
- The question "does a single degree-7 interior vertex force exponential growth" has the
  answer *no* by these theorems: a single −1/6 is a cone point; a cone has quadratic
  growth with a smaller constant. Exponential growth needs a density of defects. For the
  project the relevant statement is instead: *the rule converts one permanent boundary
  defect into a growing density of boundary defects* (the fjord walls are themselves
  cap-saturated), and that is what the single-nucleation → global-failure phenomenology
  requires. Measuring the fraction of boundary vertices with κ ≤ −1/6 vs N after the first
  dead corner would test this.

### Gaps
- No published treatment found of growth of *finite* planar triangulations with a hard
  degree cap and local attach/close moves; the cleanest neighbours are RAN (no cap) and
  the infinite-tessellation curvature theorems (no growth process).
- Dynamical triangulations / flip-grown random surfaces (Ambjørn-Durhuus-Jonsson,
  *Quantum Geometry*, CUP 1997; random planar maps with Hausdorff dimension 4)
  **[unverified]** were not searched for in this session; they are the relevant
  "random surfaces grown by local moves" literature but have unbounded degree.
- Higuchi 2001 bibliographic details are from memory.

---

## Q6. Defect-mediated compact → branched transitions; KTHNY / hexatic analogies

### Takeaway
I found no published 2D growth model in which a *single* rare frozen defect, Poisson-
nucleated at a rate exponential in a bias, drives a compact-to-branched transition with no
bulk instability; the closest families are (a) quenched-pinning interface models, (b)
nucleation-limited failure in self-assembly (Schulman-Winfree; Whitelam-Jack kinetic
trapping), and (c) KTHNY defect unbinding, which is an equilibrium, thermally driven
transition and only loosely analogous. The project's phenomenology appears to be a
genuinely under-described combination: aTAM-style irreversible rule + hard degree cap +
boundary-only nucleation.

### Cited Findings
- Whitelam & Jack, "The statistical mechanics of dynamic pathways to self-assembly",
  Annu. Rev. Phys. Chem. 66, 143-163 (2015), arXiv:1407.2505 **[abstract verified]**:
  review of assembly pathways; flags far-from-equilibrium pathways as a growth area. —
  [arXiv](https://arxiv.org/pdf/1407.2505)
- Grant, Jack & Whitelam, "Analyzing mechanisms and microscopic reversibility of
  self-assembly", J. Chem. Phys. 135, 214505 (2011), DOI 10.1063/1.3662140,
  arXiv:1108.4542 **[abstract verified]**: compares kinetic trapping in model chaperonins
  and an Ising lattice gas "paying particular attention to the number of ways that
  particles can misbind"; a search summary of the paper adds that "effective dynamical
  assembly typically requires bond-making and bond-breaking events" (that sentence is from
  a search snippet, not the abstract). — [arXiv](https://arxiv.org/abs/1108.4542)
  Mapping: **analogous** — the project's rule has bond-making only; the paper's thesis is
  that this is exactly the regime in which misbinding is permanent.
- Hagan, Elrad & Jack, "Mechanisms of kinetic trapping in self-assembly and phase
  transformation", J. Chem. Phys. 135, 104115 (2011), arXiv:1105.2830 **[abstract
  verified]**: "two distinct mechanisms of kinetic trapping", one captured by rate
  equations, one involving breakdown of cluster-size-as-reaction-coordinate. —
  [arXiv](https://arxiv.org/pdf/1105.2830)
  Mapping: **analogous** — the dead corner is a trap not describable by "size" alone
  (it is a shape/degree configuration), matching their second mechanism.
- KTHNY **[unverified]**: Kosterlitz & Thouless, J. Phys. C 6, 1181 (1973); Halperin &
  Nelson, Phys. Rev. Lett. 41, 121 (1978); Nelson & Halperin, Phys. Rev. B 19, 2457
  (1979); Young, Phys. Rev. B 19, 1855 (1979). Dislocation = bound 5-7 disclination pair;
  unbinding of dislocations gives the hexatic phase; unbinding of disclinations gives the
  liquid. Nelson, *Defects and Geometry in Condensed Matter Physics* (CUP 2002) for the
  geometric view (disclination charge = integrated Gaussian curvature).
  Mapping: **speculative** — in the project, interior 5/7 defects are either forbidden
  (7) or transient boundary states (5); the permanent defects are boundary κ = −1/3
  charges. KTHNY's lesson that transfers is only the bookkeeping: total disclination
  charge is topologically fixed (Gauss-Bonnet), so a frozen negative charge must be
  compensated by positive charge elsewhere (here, extra convex boundary).
- Directed-percolation depinning (see Q2) **[unverified]** is the closest "sparse frozen
  blockers deform an otherwise flat front" model.

### Inferences
- The project's transition is better described as *nucleation-limited* than
  *defect-mediated* in the KTHNY sense: there is no defect density that tunes continuously
  through a critical value; a single event is sufficient, and the "transition size" N* is
  a first-passage variable. The appropriate theoretical frame is therefore first-passage /
  Poisson nucleation (Q4) plus combinatorial Gauss-Bonnet (Q5), not a KT renormalisation
  flow.
- The hexatic analogy does carry one testable prediction: if dead corners act as
  disclination charges of −1/3 on the boundary, then boundary length L(N) after the first
  dead corner should grow faster than the √N of a disc by an amount proportional to the
  number of cap-saturated boundary vertices, since each needs +2 extra degree-3 vertices
  (or one degree-2) to compensate.

### Gaps
- No model found matching all three features (irreversible, hard degree cap, single-defect
  nucleation of a geometric crossover). This is a genuine gap in what I could locate, not
  just a search failure of this session, but a dedicated search on "degree-constrained
  Eden growth" or "triangular-lattice Eden model with coordination cap" would be a
  reasonable follow-up.

---

## New paths (rule variants and measurements), each grounded in a specific source

### Takeaway
Six concrete follow-ups; items 1-3 are rule variants, 4-6 are measurements. Each names
the paper it borrows from and the prediction it tests.

### Cited Findings
1. **Proofreading analogue (lookahead veto).** Winfree & Bekbolatov 2004 / Chen & Goel
   2005 **[abstract only]**: make the error path require k improbable events in series.
   Rule variant: when a node attaches at boundary node u, refuse (or down-weight by e^{-β})
   an attachment that would leave u at the cap with both of u's boundary neighbours at the
   cap — a 1-hop read of neighbours' degrees, so locality is preserved. Prediction: the
   per-node hazard exponent rises from a ≈ 5.4 toward a ≈ 5.4 + (number of extra vetoed
   decisions) × 1, i.e. the slope of ln N* vs β increases by an integer-like amount, and
   Weibull shape stays ≈ 1. If instead the hazard becomes *sub*-exponential in β, the
   dead-corner motif has a second channel the veto does not cover.
   — [W&B record](https://authors.library.caltech.edu/22768); [Chen & Goel record](https://unpaywall.org/10.1007%2F11493785_6)
2. **Reversible attachment (kTAM detachment).** kTAM r_{r,b} = k_f e^{-b G_se}
   **[verified via arXiv:2405.10911]**; Grant-Jack-Whitelam 2011 **[abstract verified]**.
   Rule variant: a boundary node of degree b may detach with probability ∝ e^{-b β} per
   step. Since dead-corner nodes are at the cap (b = 6) they essentially never detach, so
   this remedy acts only *before* the motif completes, by removing singly-bonded
   attachments that are the precursors. Prediction: the N* distribution stays exponential
   (constant hazard) but the slope a increases by ≈ 1 per precursor bond that must now
   survive detachment; if a is unchanged, the precursor is not a low-degree node and the
   remedy is useless — a cheap discriminating experiment.
   — [arXiv:2405.10911](https://arxiv.org/pdf/2405.10911); [arXiv:1108.4542](https://arxiv.org/abs/1108.4542)
3. **Local cap relaxation (one disclination allowed).** Hua-Jost-Liu 2015 (κ ≥ 0 ⇒ |B_r| ≤
   C r²) **[abstract verified]**; K-P curvature formula **[verified]**. Rule variant:
   allow a dead corner to accept one extra edge (degree 7, κ = −1/6 if interior, or −1/2
   if it stays on the boundary) with probability e^{-γ}. Prediction from curvature:
   isolated −1/6 cone points leave growth quadratic (d_eff ≈ 2 with a reduced prefactor),
   so the crossover N* should move to much larger N or disappear if the fjord never forms;
   if N* is unchanged, the fjord is driven by the *boundary* κ = −1/3 and not by the
   interior degree constraint — which decides whether the cap or the boundary pinch rule
   is the real culprit.
   — [arXiv:1107.2826](https://arxiv.org/abs/1107.2826v2); [ar5iv 0903.4793](https://ar5iv.arxiv.org/html/0903.4793)
4. **Curvature accounting through growth.** K-P definitions **[verified]**; Gauss-Bonnet
   Σκ = 1 for a disc with infinigon outer face **[derived]**. Measurement: at each snapshot
   compute κ(v) for every vertex, verify Σκ = 1, histogram boundary κ ∈ {+1/3, +1/6, 0,
   −1/6, −1/3}, and track (i) the first appearance of a permanent −1/3 vertex (should
   coincide with the pre-registered dead-corner detector), (ii) the count of −1/3 boundary
   vertices vs N after nucleation, (iii) boundary length L(N) vs √N. Prediction: L grows
   super-√N by ~2 extra vertices per −1/3 charge; the fraction of boundary vertices with
   κ ≤ −1/6 rises toward a plateau, at which point K-P Theorem 1 predicts a positive
   Cheeger constant — compare against the d_eff crossover already measured.
   — [ar5iv 0903.4793](https://ar5iv.arxiv.org/html/0903.4793); [DeVos-Mohar](https://www.ams.org/tran/2007-359-07/S0002-9947-07-04125-6/)
5. **Censoring-correct hazard estimation and constancy test.** Sear 2014 **[verified
   summary]** for the exponential-survival framing; exposure MLE and Nelson-Aalen
   **[unverified, standard]**. Measurement: for each β pool seeds, compute
   λ_hat = d / Σ_i min(N*_i, N_max) with the χ² CI; plot Nelson-Aalen Λ_hat(N) vs N (should be
   linear through origin); refit Weibull shape with censored likelihood. Prediction: shape
   CI covers 1 at every β and ln λ_hat is linear in β with slope −5.4 ± CI; a shape
   significantly > 1 at large N would signal boundary-length-dependent eligibility
   (hazard ∝ √N), overturning the "tip-only nucleation" inference in Q4.
   — [Sear 2014](https://pubs.rsc.org/en/content/articlehtml/2014/ce/c4ce00344f)
6. **Barrier-size programming (motif-size test of a = 5.4).** Schulman & Winfree 2009
   (spurious nucleation rate exponential in tile-set/ribbon size) **[abstract verified]**.
   Measurement + variant: enumerate the minimal dead-corner motif and count the number m
   of closure decisions that must go the unlikely way; check whether a ≈ m × (d ln
   p_skip/dβ). Then alter the closure precondition so the motif needs one more skipped
   closure (e.g. require two cap-saturated boundary neighbours *and* a cap-saturated
   second neighbour before a node is declared un-closable). Prediction: a increases by one
   unit of (d ln p_skip/dβ), and the Weibull shape remains 1; a failure of the slope to
   move by the predicted amount means the 5.4 is not a decision count and the Arrhenius
   reading in Q4 is wrong.
   — [arXiv:cond-mat/0607317](https://arxiv.org/abs/cond-mat/0607317v2)

### Inferences
- Items 1, 2 and 6 all predict the *same observable* (a shift of the slope a with the
  Weibull shape pinned at 1) and so can share one sweep harness; item 3 predicts a shift
  in *whether* a crossover exists; items 4-5 are pure measurements on existing pickles.

### Gaps
- Items 2 and 3 introduce moves (edge/node removal, local cap violation) that the project
  currently forbids by invariant; they should be run as explicitly labelled variants of
  the generator, not as changes to `sheet`.
