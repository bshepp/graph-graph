# Readiness assessment — spectral-dimension flow `d_s(scale)` on `grown`

**Date:** 2026-08-17
**Status:** PRE-BRAINSTORM readiness note / proposal. **Not an approved spec.** Exists so
the brainstorm starts grounded; owner picks every fork. (Same stance as the step-5 and
2026-08-03 readiness notes.)

## The question (the registry's flagship OPEN branch)

FINDINGS ("Scaling directions" #1, gate satisfied since May): does a **minimal local
growth rule** — `grown`, with no baked-in causal or geometric structure — spontaneously
show a CDT-like **scale-dependent spectral dimension flow** `d_s(t)`, and does `d_s`
agree with or split from the ball-growth `d_H`? Pre-committed null already on record:
`d_s = d_H` (no flow) is the *expected* outcome and is publishable as-is ("minimal local
growth gives a manifold-like graph with no anomalous spectral flow, isolating what extra
ingredient CDT's reduction requires").

## What ports / what exists

- `d_s` comes from random-walk return probability: `P0(t) ~ t^(-d_s/2)`, so
  `d_s(t) = -2 dlnP0/dlnt` is the flow curve. Mechanically: sparse matvec iteration of
  the walk operator from sampled sources, averaging the return probability — the same
  `A @ v` pattern `dimension.py --fast` already uses; no new dependency class.
- Known-answer anchors are free: 2D lattice → flat `d_s = 2`; ring → 1; an expander →
  no power law (control). `grown` cap 6 (`d_H ≈ 2.2`) is the application target.
- Substrates exist to N = 2×10⁵ locally (`cap_dimension_scaling` precedent); jaga's
  256 GiB fits far larger CSRs if the brainstorm wants decades of `t` at 10⁶⁺.
- Literature framing is already banked in FINDINGS (CDT: arXiv:1711.02685; Millán et
  al. PRR 3, 023015) — the novelty bar and the "between the two literatures" positioning
  are written down.

## Forks for the owner (the brainstorm's agenda)

1. **Walk convention.** Lazy walk (stay-probability ½) kills parity/bipartite
   oscillations in `P0(t)` and is the standard fix; non-lazy + even-`t`-only is the
   alternative. Recommend lazy; it changes only the time normalization.
2. **Estimator path.** Deterministic sparse matvec per source (exact per-source, cost
   `t_max` matvecs × sources) vs Monte-Carlo walkers (cheap memory, `1/sqrt(W)` noise).
   Recommend deterministic matvec at N ≤ 10⁶; it reuses the validated sparse pattern.
3. **Scale + platform.** First pass local at N = 2×10⁵ (minutes) vs jaga at 10⁶–10⁷.
   Recommend local first-cut, jaga for the production curve. (jaga note: next wake
   needs the power button.)
4. **What counts as "flow" (pre-commitment shape).** A `d_s(t)` curve needs a frozen
   definition of flat-vs-flowing before the run — e.g. window-fit `d_s` at short/long
   scales with an agreed difference threshold against the anchors' measured spread.
   This is the fork most worth care; the rest is machinery.
5. **Gate structure.** Two-tier in the house style (lattice/ring anchors must read
   flat at the known values before `grown` is interpreted) — presumably uncontroversial.

## Cost

Small module (`spectral_flow.py`) + `--validate` anchors; first physics-relevant curve
within a session; jaga production run well under an overnight. The expensive ingredient
is the fork-4 pre-commitment, which is thinking, not compute.

## Recommendation

Ready to brainstorm as soon as the owner wants a second live thread (it is independent
of the critical-collapse program). Suggested entry point: fork 4 first — freeze what
"flow" means — then the machinery forks are quick.
