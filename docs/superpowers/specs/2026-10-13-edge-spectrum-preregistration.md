# Pre-registration — the roughness spectrum of the sheet's edge, and the onset in seed 308

**Date:** 2026-10-13, late evening. **Status:** PRE-REGISTRATION, committed before the tool
touched a real graph (its `--validate` runs on synthetic profiles and a 40 k-node sheet only).
Follow-up to round-2 items 1 and 4 (BRANCHES row "Is the sheet's crossover a shape
instability -- and what triggers it?"). Owner's standing instruction: sidequests approved;
pre-register before data. Runs on the desktop on the twelve saved graphs.

## Question

Item 1 read the post-crossover edge as a shape instability with a fixed feature size
(2N / B ~ 350-400 nodes at 2-4 M). Item 4 found the *onset* is a rare event at a constant
rate. If the edge is unstable at a fixed wavelength, the roughness spectrum along the edge
should carry a peak at that wavelength from the first roughening on, and the wavelength
should not move with N. If the rough edge is instead a scale-free random surface (the
Edwards-Wilkinson / KPZ kind), the spectrum is a straight power law with no peak.

## Data

`results/sheet_cross_b2_s3{00..11}_N4194304.npz` (beta = 2). Prefix subgraphs on nodes < N
as in item 1 (node index is growth order; closure edges added later between early nodes are
included; boundary biased low by a few percent, disclosed).

## Frozen definitions (`edge_spectrum.py`)

- Boundary edge: an edge of the prefix graph in exactly one triangle. Boundary-edge graph:
  boundary nodes joined by boundary edges. The **edge sequence** is the longest walk through
  the boundary-edge graph that never revisits a node (preferring unvisited neighbours; on a
  simple cycle this is the cycle). The covered fraction of boundary nodes is reported; a
  reading is only made when it is >= 0.9.
- Height profile: `h(s) = rho(s) - R` along the sequence, `rho` = BFS distance from node 0,
  `R` = median `rho` of the sequence. Roughness amplitude `sigma_h / R`.
- Spectrum: periodogram of `h` over the sequence (`|FFT|^2 / L`), wave number k = 4 ..
  L / 40 (wavelengths of at least 40 boundary nodes, above the lattice scale; k = 1-3 are
  the disc's global shape modes -- offset of node 0 from the centre, ellipticity,
  triangularity -- and are not roughness), averaged in
  log-spaced bins of 8 per decade merged so that every bin holds >= 8 Fourier modes. Fit a
  power law to the binned log-spectrum, drop the highest-residual bin, refit; the
  **excess** of a bin is its mean over the refitted power law. A **peak** is a bin with
  excess >= 3.0 (for a bin of 8 exponential modes that is a ~1e-4 event; the self-check
  demands <= 1 false peak in 10 periodic random walks). `lambda_peak = L / k_bin`. (The
  first draft used 4 modes per bin and dropped two bins; it produced 2 false peaks in 10
  random walks in the self-check and was tightened before any real graph was read.)
- Rough: `B / sqrt(N) >= 4.5` (a compact disc reads 3.85; item 4's table).
- Sizes: N = 2^15 .. 2^22 (octaves) for every seed; for seed 308 also 2^14 .. 2^17 at
  2^(1/4) spacing (the onset scan).

## Frozen readings (on the rough (seed, N) pairs; a seed counts if it has >= 2 rough sizes)

- **FIXED WAVELENGTH**: in >= 2/3 of the counting seeds, every rough size has a peak and the
  seed's `max / min lambda_peak` across its rough sizes is <= 2.
- **SCALE-FREE**: in >= 2/3 of the counting seeds, no rough size has a peak, and the fitted
  slope lies in [-2.5, -1.0] at every rough size.
- **UNDECIDED**: otherwise (reported with the per-seed peaks and slopes).

Also reported, no verdict attached: the slope of the compact (pre-crossover) spectra; the
ratio `lambda_peak / (2N / B)`; and `sigma_h / R` against `B / sqrt(N)`.

## Onset scan of seed 308 (descriptive, no verdict rule)

At each size of the fine scan: `B / sqrt(N)`, `sigma_h / R`, the covered fraction, the
position along the sequence of the largest |h| and its sign (outward bump or inward fjord),
and the blocked-closure density of the boundary within 20 hops of that node against the rest
of the boundary (item 1's census). The point is to see the first excursion of the earliest
trigger of the twelve, nothing more.

## Prediction

FIXED WAVELENGTH at about 50%, with `lambda_peak` of a few hundred boundary nodes
(comparable to 2N / B ~ 370); SCALE-FREE 30%; UNDECIDED 20%. The compact spectra should be
steep (slope near -2 or steeper: a smooth disc's edge).
