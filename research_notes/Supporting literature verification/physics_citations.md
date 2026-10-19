# Verification of random-geometry, growth and nucleation citations (checked 2026-10-18)

Method: I pulled verbatim abstracts and journal-refs from the arXiv API (export.arxiv.org). Where possible I downloaded full-text PDFs and extracted them with pdftotext or PyMuPDF. Venues come from arXiv journal-ref, DOI prefix, Crossref or the publisher/repository record. Anything not read from a primary source is marked **[unverified]**.

## 1. Ambjørn, Jain, Thorleifsson 1993 (hep-th/9303149)

### Takeaway
The venue is right: Phys. Lett. B 307 (1993) 34-39. The project's paraphrase is **not supported**, for two reasons. First, the paper never mentions Hausdorff dimension; it relates the distribution of minimal-neck baby universes to the string susceptibility exponent gamma. Second, d_H of 2D pure-gravity dynamical triangulations is 4, not 2, so baby universes do not "destroy d_H = 2".

### Cited Findings
- Title "Baby universes in 2d quantum gravity". Authors: J. Ambjørn, S. Jain, G. Thorleifsson. Journal-ref: Phys. Lett. B307 (1993) 34-39, DOI 10.1016/0370-2693(93)90188-N — [arXiv API record](https://arxiv.org/abs/hep-th/9303149)
- Abstract, verbatim: "We investigate the fractal structure of 2d quantum gravity ... The roughness of the surfaces is described in terms of baby universes and using numerical simulations we measure their distribution which is related to the string susceptibility exponent γ_string." — [arXiv](https://arxiv.org/abs/hep-th/9303149)
- Definition from the full text, verbatim: "a closed, non-intersecting loop along the links will separate the surface in two parts. The smallest such loop will be of length 3. ... If the smallest part is different from a single triangle we call it (following the notation of [1]) a 'minimum neck baby universe', abbreviated 'mimbu'." Ref [1] is S. Jain and S.D. Mathur, Phys. Lett. 286B (1992) 239, which is where the concept originates. — [full text PDF](https://arxiv.org/pdf/hep-th/9303149)
- Key formula (eq. 1.3): the average number of mimbus of area B on a spherical surface of area N_T is n̄_NT(B) ~ (N_T − B)^{γ−2} B^{γ−2}, with γ = −1/2 for pure gravity. — [full text](https://arxiv.org/pdf/hep-th/9303149)
- The paper's only "fractal structure" statement links more baby universes to larger gamma: "the number of baby universes of a given size will increase, i.e. the fractal structure will be more pronounced with increasing c." The words "Hausdorff" and "d_H" do not appear anywhere in the extracted text. — [full text](https://arxiv.org/pdf/hep-th/9303149)
- Independent evidence that 2D pure-gravity triangulations are 4-dimensional, not 2-dimensional: Angel proves the UIPT has ball volume ~ r^4 (item 2). — [Angel, arXiv math/0208123](https://arxiv.org/abs/math/0208123)

### Inferences
- A defensible rewording: "in 2D dynamical triangulations, surfaces are dominated by minimal-neck (3-link loop) baby-universe outgrowths, whose size distribution is controlled by gamma_string (Jain-Mathur 1992; Ambjørn-Jain-Thorleifsson 1993)". The minbus are what make the geometry fractal (d_H = 4) rather than smooth. Any analogy to the project's "dead corner" should not claim that AJT tie mimbus to the loss of d_H = 2.
- AJT's mimbus are equilibrium (entropic) features. The project's dead corners are kinetic and irreversible. That is a structural difference, not just a terminological one.

### Gaps
- I did not read the Jain-Mathur 1992 paper.
- The d_H = 4 result for DT (Kawai-Kawamoto-Mogami-Watabiki 1993; Ambjørn-Watabiki 1995) is well known, but I did not check those primary sources here **[unverified]**. Angel's rigorous r^4 result is the verified support.

## 2. Angel 2003 (math/0208123)

### Takeaway
Confirmed. The paper is in Geom. Funct. Anal. (GAFA) 13(5):935-974 (2003); the volume and pages come from citing papers, not the publisher page. The paraphrase "ball boundary ~ r^2, volume ~ r^4" is correct, but both hold only up to polylogarithmic factors, and the boundary meant is specifically the boundary component that separates the ball from infinity.

### Cited Findings
- Abstract, verbatim: "a.s. the UIPT has growth rate r^4 up to polylogarithmic factors, confirming heuristic results from the physics literature. Additionally, the boundary component of the ball of radius r separating it from infinity a.s. has growth rate r^2 up to polylogarithmic factors." The abstract also proves site-percolation p_c = 1/2 on the UIPT. — [arXiv](https://arxiv.org/abs/math/0208123)
- Venue: "Geom. funct. anal., 13(5):935–974, Oct. 2003", as cited in later literature. The arXiv record has no journal-ref. — [search result citing papers, e.g. arXiv 1704.01561](https://arxiv.org/pdf/1704.01561)

### Inferences
- Cite it as "|∂B_r| ≈ r^2 and |B_r| ≈ r^4 up to polylog factors (hull boundary)".

### Gaps
- I did not see the GAFA publisher page; the volume and pages come from secondary citation.

## 3. López, Rodríguez, Cuerno 1997 (cond-mat/9703024)

### Takeaway
The venue is Phys. Rev. E 56, 3993 (1997) (from the DOI); the title is "Super-roughening versus intrinsic anomalous scaling of surfaces". The paraphrase "spectral exponent differing from local one = anomalous scaling" is **imprecise**. The paper names two independent causes of anomalous scaling. One is super-roughening (global α ≥ 1), which occurs even with a normal spectrum. The other is an "intrinsically anomalous spectrum" carrying an extra independent exponent. Anomalous scaling means local properties scale differently from global ones; it is not defined as spectral ≠ local.

### Cited Findings
- DOI 10.1103/PhysRevE.56.3993; the arXiv comment says it was submitted to PRE. — [arXiv](https://arxiv.org/abs/cond-mat/9703024)
- Abstract, verbatim: "we distinguish two independent causes for anomalous scaling. One is super-roughening (global roughness exponent larger than or equal to one), even if the spectrum behaves non anomalously. Another cause is what we term an intrinsically anomalous spectrum, in whose scaling an independent exponent exists, which induces different scaling properties for small and large length scales (that is, the surface is not self-affine)." — [arXiv](https://arxiv.org/abs/cond-mat/9703024)

### Inferences
- A safer paraphrase: "anomalous scaling, meaning local roughness exponent α_loc ≠ global α, arises either from super-roughening (α ≥ 1, spectrum normal) or from an intrinsically anomalous spectrum with an independent spectral exponent α_s (López-Rodríguez-Cuerno 1997)."

### Gaps
- I did not read the full text, so the exact notation (α_s, α_loc) and the relations α_loc = min(1, α_s) **[unverified]** are not checked against the paper.

## 4. Sisko, Yambartsev, Zohren 2013 (arXiv 1203.2869)

### Takeaway
Confirmed: J. Stat. Phys. 150 (2013) 353-374. The boundary length M_n is an exact birth-death Markov chain with P(ξ_n = ±1 | M_n = m) = (1/2)(m ± 1)/m. Rescaled as M_[un]/√n, it converges to dM_u = (1/M_u) du + dB_u, which is a Bessel-3 type diffusion (Theorem 4.1). So "boundary length performs a diffusion" holds, but the diffusion has a 1/M drift and is not a free diffusion.

### Cited Findings
- Journal-ref "Journal of Statistical Physics 150 (2013) 353-374", DOI 10.1007/s10955-012-0665-9. Abstract, verbatim: "convergence of the boundary process to a diffusion process is shown leading to an interesting duality relation between the growth process and a corresponding branching process." — [arXiv](https://arxiv.org/abs/1203.2869)
- Moves, verbatim from the full text: "in any move one adds one triangle ... the (+)-move increases the boundary by one, while the (−)-move decreases the boundary by one." Equation (2.4): P(ξ_n = ±1 | M_n = m) = ½·(m ± 1)/m. The paper calls the process "basically the time reversal of the so-called 'peeling' process". — [full text](https://arxiv.org/pdf/1203.2869v2)
- Theorem 4.1, verbatim: "M^(n)_u = M_[un]/√n ⇒ M_u ... solves the following Itô's equation dM_u = (1/M_u) du + dB_u, M_0 = l". — [full text](https://arxiv.org/pdf/1203.2869v2)
- Theorem 4.2 (geodesic-distance time): L_s = M_{n[st]}/t ⇒ L_s with dL_s = 2 ds + √(2L_s) dB_s, which is the Feller/branching diffusion. Theorem 4.3 links them by the random time change T_u = ∫_0^u 1/(2M_s) ds. — [full text](https://arxiv.org/pdf/1203.2869v2)
- Theorem 3.1: d_h = 2 a.s., from c1 t²/log² t ≤ n_t ≤ c2 t² log² t. — [full text](https://arxiv.org/pdf/1203.2869v2)

### Inferences
- In this causal model every added triangle changes the boundary length by exactly ±1, with a slight outward bias of 1/m. That gives √n boundary growth and d_h = 2. It is a useful exactly solved baseline for a degree-capped disc: flatness there coincides with a boundary length that is a recurrent-like chain biased by only O(1/m).

### Gaps
- None for the stated claims.

## 5. Gralka and Hallatschek 2019 (PMC6513619)

### Takeaway
Confirmed: eLife 8:e44359 (2019), "Environmental heterogeneity can tip the population genetics of range expansions". It is a generalized Eden lattice model with obstacle sites; impassable obstacles have k = 0. The expansion speed vanishes above ρ_c ≈ 0.4. Without obstacles the model is KPZ (α = 1/2, β = 1/3). At the critical density the measured exponents agree with quenched Edwards-Wilkinson (QEW).

### Cited Findings
- Bibliographic data: eLife 8, e44359, DOI 10.7554/eLife.44359, published 12 Apr 2019. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6513619/)
- Verbatim: "Our model is based on the classical Eden lattice model." Figure 4 caption: "the expansion speed vanishes above a critical obstacle density ρc≈0.4." Another passage: "At the transition, the resulting colonies are (site) percolation clusters on the square lattice". — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6513619/)
- Exponents at criticality, as reported to me by the fetch tool's summary rather than read in a quoted table: α_loc ≈ 0.9, α_G ≈ 1.15, β ≈ 0.78, "in excellent agreement with the QEW universality class". The lineage exponent ξ rises from 0.66 to 0.86. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6513619/)

### Inferences
- ρ_c ≈ 0.4 matches 1 − p_c(site, square) ≈ 1 − 0.593 = 0.407, as expected if the colony at threshold is a percolation cluster of free sites (my arithmetic).

### Gaps
- I read only the first 100k characters of the PMC page. The exponent values come from a summarizer, so their table placement should be spot-checked **[partially unverified]**.

## 6. Sear 2004 (cond-mat/0406019) and Sear 2014 (CrystEngComm)

### Takeaway
Sear 2004 is Phys. Rev. E 70, 021605 (2004), but it is about non-self-averaging nucleation rates caused by random impurities, mapped onto Derrida's Random Energy Model. It does not discuss Weibull fits, and exponential survival appears only implicitly. Sear 2014 is CrystEngComm 16, 6506-6522 (DOI 10.1039/C4CE00344F), "Quantitative studies of crystal nucleation at constant supersaturation: experimental data and models". I could not read its full text (HTTP 403). The explicit "exponential = well-defined rate; deviations → Weibull" statement is verified in a 2013 Sear conference abstract (ICCGE-17), not in either paper.

### Cited Findings
- Sear 2004 abstract, verbatim: "The rate then has the same form as the partition function of Derrida's Random Energy Model ... Non-self-averaging nucleation is nucleation with a rate that varies significantly from one realisation of the random variables to another." Journal-ref: Physical Review E v70, 021605 (2004). — [arXiv](https://arxiv.org/abs/cond-mat/0406019)
- A full-text grep of Sear 2004 found no occurrence of "Weibull". — [full text](https://arxiv.org/pdf/cond-mat/0406019)
- Sear 2014 bibliographic data: CrystEngComm 2014, 16, 6506–6522, open-access Highlight. — [RSC landing page](https://pubs.rsc.org/en/content/articlelanding/2014/ce/c4ce00344f); [DOI](https://dx.doi.org/10.1039/C4CE00344F)
- R. Sear, "Nucleation of crystals when the nucleation rate does not exist", ICCGE-17 oral abstract (submitted 2013-04-13). Verbatim: "deviations from an exponential function of time indicate an undefined rate"; "there is no well defined rate for a set of droplets at a given temperature and supersaturation"; "it may often be possible to fit to the data what is called the Weibull distribution of extreme-value statistics." — [science24](https://science24.com/paper/29581)

### Inferences
- For "constant hazard → exponential survival; Weibull shape ≠ 1 signals a heterogeneous or undefined rate", cite Sear 2014 only after checking its text. The ICCGE abstract supports the logic. Cite Sear 2004 for sample-to-sample rate heterogeneity (non-self-averaging), not for Weibull.
- The project's Weibull shape 1 means a homogeneous constant per-node hazard, which is the "well-defined rate" case in Sear's sense.

### Gaps
- Sear 2014 full text was blocked (RSC Cloudflare). Whether the review itself discusses exponential survival or Weibull is **[unverified]**.

## 7. Schulman and Winfree 2009; Winfree and Bekbolatov 2004; kTAM error rate

### Takeaway
Schulman and Winfree is confirmed: SIAM J. Comput. 39(4):1581-1616 (2009). Winfree and Bekbolatov 2004 is "Proofreading tile sets: Error correction for algorithmic self-assembly", DNA Computing 9 (eds. Chen and Reif, Springer LNCS 2943; the LNCS number is **[unverified]**). The kTAM error-rate scaling is verified from the Winfree-Bekbolatov text. The equilibrium error rate is ε ≈ 2N e^{−G_se}, independent of G_mc. With kinetic trapping near the melting line G_mc = 2G_se − δ, the result is still ε ≈ 2N e^{−G_se} times a kinetic factor, and the speed-error tradeoff is r ≈ βε² for the original tiles. Citing "Winfree 1998 thesis" is slightly off: the paper credits the kinetic-trap analysis and phase diagram to Winfree 1998, Caltech Tech. Report CSTR:1998.22, "Simulations of computing by self-assembly".

### Cited Findings
- Schulman and Winfree, "Programmable Control of Nucleation for Algorithmic Self-Assembly", SIAM J. Comput. 39(4):1581-1616 (2009), DOI 10.1137/070680266. Verbatim: "linearly increasing the size of the tile set exponentially reduces the rate of spurious nucleation." — [arXiv](https://arxiv.org/abs/cond-mat/0607317)
- Winfree-Bekbolatov rate constants: "k_r,1 = k_f e^{−G_se}"; "k_r,2 = k_f e^{−2G_se}"; [monomer]/M = e^{−G_mc}. — [Caltech preprint](https://authors.library.caltech.edu/records/jcvj4-7gn26)
- Equilibrium error, verbatim: "and thus ε ≈ 2N e^{−Gse}, where ε is the per-tile error rate. It is mildly surprising that Gmc has no effect on the equilibrium error rates." — [Caltech preprint](https://authors.library.caltech.edu/records/jcvj4-7gn26)
- Kinetic trapping, verbatim: "tiles that associate with a mismatch on the growth front don't have time to dissociate, and are frozen in place by further growth; thus equilibrium error rates are observed only near the melting transition." The trap model (Fig. 3b) has states E, C, M, FC, FM, and "The error rate is taken to be the probability that, starting in E, the system reaches FM." — [Caltech preprint](https://authors.library.caltech.edu/records/jcvj4-7gn26)
- Tradeoff: "all points lie above r ≈ βε²"; "decreasing error rates by a factor of 10 entails slowing down the self-assembly process by a factor of 100." 2×2 and 3×3 proofreading tiles give r ≈ βε^1.0 and r ≈ βε^0.89. — [Caltech preprint](https://authors.library.caltech.edu/records/jcvj4-7gn26)
- Ref [28] in that paper: "E. Winfree. Simulations of computing by self-assembly. Technical Report CSTR:1998.22, Caltech, 1998." — [Caltech preprint](https://authors.library.caltech.edu/records/jcvj4-7gn26)

### Inferences
- The kTAM trap model has the same structure as the project's dead corner. A wrong local configuration at the growth front is "frozen in place by further growth" before it can be undone. The error probability per site is Arrhenius in one bond energy (e^{−G_se}), and that matches the project's per-node hazard ∝ e^{−6.44 β}. This is the closest prior art for the "lost race at the front → permanent defect" step.

### Gaps
- I did not open the 1998 thesis ("Algorithmic Self-Assembly of DNA", Caltech PhD) or CSTR:1998.22 directly.

## 8. Grant, Jack, Whitelam 2011 and Hagan, Elrad, Jack 2011

### Takeaway
Venues confirmed. Grant, Jack and Whitelam is J. Chem. Phys. 135, 214505 (2011), "Analyzing mechanisms and microscopic reversibility of self-assembly". Hagan, Elrad and Jack is J. Chem. Phys. 135, 104115 (2011), "Mechanisms of kinetic trapping in self-assembly and phase transformation". The "bond-making and bond-breaking" wording is not in either abstract **[unverified at full-text level]**.

### Cited Findings
- Grant et al., verbatim: "we use measurements of dynamical activity and assembly progress to compare their propensities for kinetic trapping ... paying particular attention to the number of ways that particles can misbind." — [arXiv 1108.4542](https://arxiv.org/abs/1108.4542)
- Hagan et al., verbatim: "we show how simulations in a self-assembling steady state can be used to identify two distinct mechanisms of kinetic trapping. We argue that one of these mechanisms can be adequately captured by kinetic rate equations, while the other involves a breakdown of theories that rely on cluster size as a reaction coordinate." The article number 104115 comes from Crossref. — [arXiv 1105.2830](https://arxiv.org/abs/1105.2830); [DOI 10.1063/1.3635775](https://doi.org/10.1063/1.3635775)

### Inferences
- Both papers support the general statement that "efficient assembly requires microscopic reversibility; traps arise when misbound units cannot be undone before burial". The project's dead corner is the irreversible limit of this.

### Gaps
- Full texts were not read, so specific "bond-making/bond-breaking" activity measures are unverified.

## 9. Gärtner and Zehmakan 2018 (1711.10920)

### Takeaway
The arXiv record lists **no venue** (cs.FL, submitted 24 Nov 2017). A conference version, "Color War: Cellular Automata with Majority-Rule", appeared at LATA 2017, pp. 393-404, per the ANU research portal; I did not verify that it is the same text. The year "2018" is not supported by anything I found. The paraphrase holds only on the **two-dimensional torus T_{n,n} with von Neumann neighbourhood**. With biased majority (ties go to blue), p_b ≫ 1/log n gives blue consensus w.h.p.; n^{−1} ≪ p_b ≪ 1/log n gives coexistence; p_b ≪ n^{−1} gives red. "Initial density near zero" therefore means p_b → 0 more slowly than 1/log n. The authors credit the 1/log n blue result to Schonmann and prove its tightness.

### Cited Findings
- Abstract, verbatim: "If in case of a tie a vertex always chooses blue color, it is called biased majority model ... for a two-dimensional torus T_{n,n}, there are two thresholds ... result in monochromatic configuration by red, stable coexistence of both colors, and monochromatic configuration by blue, respectively in O(n²) number of steps". — [arXiv](https://arxiv.org/abs/1711.10920)
- Full text, verbatim: "by just changing the tie breaking rule in favor of blue, the process ends up in a blue monochromatic configuration with high probability even for initial concentration very close to 0." Also: "Schonmann[22] proved that in biased majority cellular automata with Neumann neighborhood p_b ≫ 1/log n outputs final complete occupancy by blue with high probability. We show that this bound is tight". — [full text](https://arxiv.org/pdf/1711.10920)
- The thresholds are as stated above. For contrast, the unbiased majority model on the same torus needs p_b ≫ 1 − n^{−1/2}, i.e. very close to 1, for blue to take over. — [full text](https://arxiv.org/pdf/1711.10920)
- LATA 2017 "Color War" record: [ANU portal](https://researchportalplus.anu.edu.au/en/publications/color-war-cellular-automata-with-majority-rule/)

### Inferences
- The mechanism is bootstrap-percolation-like nucleation of blue on a 4-regular lattice, which is why the scale is 1/log n. On other graphs (random regular, G(n,p)) the companion papers show different behaviour (1711.07423; ISAAC 2018). Do not generalize beyond the torus.

### Gaps
- I did not confirm whether 1711.10920 was later published, for example in a journal in 2018 or after.

## 10. Bollobás–Thomason hitting time

### Takeaway
B. Bollobás and A. Thomason, "Random graphs of small order", in Random Graphs '83, North-Holland Mathematics Studies 118 (= Annals of Discrete Mathematics 28; that equivalence is **[unverified]**), pp. 47-97, 1985, DOI 10.1016/S0304-0208(08)73612-0. The result is that in the random graph process, the hitting time of minimum degree ≥ k equals the hitting time of k-connectivity w.h.p. I verified this attribution only through citing papers.

### Cited Findings
- Bibliographic data: North-Holland Mathematics Studies vol. 118, pp. 47–97, 1985, DOI 10.1016/S0304-0208(08)73612-0. — [Memphis Digital Commons](https://digitalcommons.memphis.edu/facpubs/5564)
- Secondary attribution: Bollobás and Thomason (1985) showed that in the random graph process, minimum degree ≥ k and k-connectedness arrive at the same time w.h.p. This is restated and extended to d-regular host graphs in Diskin & Geisler (arXiv 2409.14398) and to hypergraphs by Poole. — [arXiv 2409.14398](https://arxiv.org/abs/2409.14398); [Poole, EJC](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v22i1p69)

### Gaps
- I did not read the original theorem statement, including whether it covers fixed k only or k = k(n) **[unverified]**.

## 11. McKay 1981

### Takeaway
Venue confirmed via Crossref: B. D. McKay, "The expected eigenvalue distribution of a large regular graph", Linear Algebra Appl. 40 (1981) 203-216, DOI 10.1016/0024-3795(81)90150-6. The Kesten-McKay density for k-regular graphs is f(x) = k√(4(k−1) − x²) / (2π(k² − x²)) for |x| ≤ 2√(k−1). I confirmed it from secondary sources only.

### Cited Findings
- Crossref metadata: title, LAA vol. 40, pp. 203-216, Oct 1981, author McKay. — [DOI](https://doi.org/10.1016/0024-3795(81)90150-6)
- Density as quoted by citing literature: k√(4(k−1) − λ²) / (2π(k² − λ²)) on |λ| ≤ 2√(k−1), described as the limit for k-regular graphs that look like trees at most vertices, i.e. with few short cycles. — [arXiv 1306.6714](https://arxiv.org/abs/1306.6714)

### Gaps
- The original paper was paywalled, so the formula is not checked against McKay's own notation **[secondary only]**.

## 12. Holroyd 2003 bootstrap percolation

### Takeaway
Confirmed: A. E. Holroyd, "Sharp metastability threshold for two-dimensional bootstrap percolation", Probab. Theory Related Fields 125(2):195-224 (2003); λ = π²/18 ≈ 0.5483. The simulation figure I could verify is Adler-Stauffer-Aharony 1989: λ = 0.245 ± 0.015, about half the limit, from squares up to L = 28800. **The "~0.47" figure is unverified.** I found it in no source checked, including De Gregorio et al. 2004 and Hartarsky-Teixeira 2024/2025. Gravner-Holroyd prove a discrepancy of at least Ω((log L)^{−1/2}). Gravner-Holroyd-Morris give the second-order term −(log n)^{−3/2+o(1)}.

### Cited Findings
- Abstract, verbatim: "I(L,p) → 1 if liminf p log L > λ, and I(L,p) → 0 if limsup p log L < λ, where λ = π²/18 ... the determination of their values corrects numerical predictions of Adler, Stauffer and Aharony." Modified model: λ' = π²/6. — [arXiv math/0206132](https://arxiv.org/abs/math/0206132)
- Venue "Probability Theory and Related Fields, 125(2):195-224, 2003". Simulated value, verbatim: "λ = 0.245 ± 0.015" (Adler, Stauffer, Aharony, J. Phys. A 22:L297-L301, 1989). — [Holroyd's bootstrap page](https://personal.math.ubc.ca/~holroyd/boot)
- The L = 28800 size comes from a search-result snippet of a review. — [arXiv 1504.02898](https://arxiv.org/pdf/1504.02898)
- Gravner & Holroyd, "Slow Convergence in Bootstrap Percolation". Verbatim: "the discrepancy between the critical parameter and its limit λ is at least Ω((log L)^(-1/2)). In contrast, the critical window has width only Θ((log L)^(-1)). For the so-called modified model ... the relative discrepancy is at least 1% even when L = 10^3000." — [arXiv 0705.1347](https://arxiv.org/abs/0705.1347)
- Gravner, Holroyd, Morris, "A sharper threshold for bootstrap percolation in two dimensions", PTRF 153 (2012) (venue from a search snippet). Verbatim: "the second term in the expansion is −(log n)^{−3/2+o(1)} ... The exponent −3/2 corrects numerical predictions from the physics literature." — [arXiv 1002.3881](https://arxiv.org/abs/1002.3881)
- De Gregorio, Lawlor, Bradley, Dawson, "Clarification of the Bootstrap Percolation Paradox", PRL 93, 025501 (2004) (from DOI 10.1103/PhysRevLett.93.025501). They use importance sampling to reach larger bootstrap lengths and add systematic corrections. — [arXiv cond-mat/0403691](https://arxiv.org/abs/cond-mat/0403691)
- Hartarsky & Teixeira, "Locality approach to the bootstrap percolation paradox", PRL 134, 117102 (2025). They report numerics matching theory "up to the third-order expansion". — [arXiv 2410.01619](https://arxiv.org/abs/2410.01619)

### Gaps
- Source of "~0.47 at accessible sizes": not found. Either drop it or replace it with the verified 0.245 ± 0.015 (Adler et al. 1989) plus the Gravner-Holroyd slow-convergence bound.

## 13. Crochet hyperbolic planes, edge-growth sheets, and novelty check

### Takeaway
Henderson & Taimina, "Crocheting the Hyperbolic Plane", Math. Intelligencer 23(2):17-28 (2001), state that the increase ratio sets the radius r (curvature −1/r²). The ratio must stay constant, and in their annular construction the spacing of radial geodesics changes by r/(r+d) per annulus of width d. From that it follows that one extra stitch per N stitches with row height h gives r ≈ N·h; this is my derivation, not their stated formula. Sharon-Marder-Swinney 2004 and Klein-Efrati-Sharon 2007 are confirmed, but I verified them only at abstract level. I found **no prior art** for the specific chain "lost race at growth front → permanent −1/3 boundary defect → hyperbolic crossover". The closest analogues are the kTAM kinetic-trap errors (item 7), edge-growth metric prescription, and the unimodular-triangulation result that mean degree > 6 implies hyperbolic.

### Cited Findings
- Henderson & Taimina, verbatim: "the ratio determines the radius (the r in the annular hyperbolic plane) of the hyperbolic plane." "You will get a hyperbolic plane ONLY if you will be increasing the number of stitches in the same ratio all the time." "the Gaussian Curvature of the hyperbolic plane is −1/r²." "The distance between these radial geodesics changes by r/(r + d) every time they cross one annulus." — [Taimina, Cornell page](https://pi.math.cornell.edu/~dtaimina/crochet/hplane.htm)
- Venue: Mathematical Intelligencer 23(2):17-28, Spring 2001 (from search results). A recommended classroom ratio of 12:13 is reported on Taimina's site (search snippet). — [Cornell hypplanes page](https://pi.math.cornell.edu/~dtaimina/hypplanes.htm); [Wikipedia: Daina Taimiņa](https://en.wikipedia.org/wiki/Daina_Taimi%C5%86a)
- Kekkonen, "Crocheting Mathematics" (arXiv 2508.10597), Sec. 4.1: C = 2πS sinh(R/S), with S the radius of the hyperbolic plane. — [arXiv](https://arxiv.org/html/2508.10597v1)
- Sharon, Marder, Swinney, "Leaves, Flowers and Garbage Bags: Making Waves", American Scientist 92:254-261 (May-June 2004). Theme: rippled, fractal edge patterns in torn plastic sheets and leaves arise because the edge has excess length relative to the interior, which breaks symmetry spontaneously. — [UT Austin CNLD record](https://chaos.utexas.edu/publications/leaves_flowers_and_garbage_bags_making_waves); [Semantic Scholar](https://www.semanticscholar.org/paper/Leaves,-flowers-and-garbage-bags:-Making-waves-Sharon-Marder/bd1ae24b7b091e55ace473af1616d1925f6f3237)
- Klein, Efrati, Sharon, "Shaping of elastic sheets by prescription of non-Euclidean metrics", Science 315(5815):1116-1120 (2007). Abstract: "thin gel sheets that undergo laterally nonuniform shrinkage. This differential shrinkage prescribes non-Euclidean metrics on the sheets"; "the free sheets form three-dimensional structures that follow the imposed metric", showing "both large-scale buckling and multiscale wrinkling". — [HUJI CRIS](https://cris.huji.ac.il/en/publications/shaping-of-elastic-sheets-by-prescription-of-non-euclidean-metric/)
- Angel, Hutchcroft, Nachmias, Ray, "Unimodular Hyperbolic Triangulations: Circle Packing and Random Walk", Invent. Math. (from DOI prefix 10.1007/s00222-016-0653-9; volume and year **[unverified]**). Verbatim: "the circle packing type of a unimodular random plane triangulation is parabolic if and only if the expected degree of the root is six, if and only if the triangulation is amenable". — [arXiv 1501.04677](https://arxiv.org/abs/1501.04677)
- Angel, Benjamini, Horesh, "An isoperimetric inequality for planar triangulations": hexagons maximize volume for a given boundary in any triangulation with minimal degree 6, a discrete Weil theorem for non-positive curvature. — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863)
- Curien, "Planar stochastic hyperbolic infinite triangulations": spatial-Markov random triangulations with "sharp exponential volume growth". — [arXiv 1401.3297](https://arxiv.org/abs/1401.3297)
- Novelty searches found nothing on kinetic, defect-rate-driven hyperbolic growth of discrete triangulations. Queries covered growth disclinations deposited at the front, degree-bounded triangulation growth turning hyperbolic, and frozen-boundary kinetic emergent geometry. Results were limited to static hyperbolic lattices ({7,3}, {3,7}), curved-surface crystallization, and disclination mechanics. — [search: e.g. arXiv 0807.2089](https://arxiv.org/pdf/0807.2089); [arXiv 2501.08894](https://arxiv.org/pdf/2501.08894)

### Inferences
- Quantitative crochet analogue, derived from H&T's r/(r+d) rule: adding one stitch per N in each row of height h multiplies the circumference by (N+1)/N = 1 + h/r per row, so r ≈ N·h and K = −1/(N h)². Equivalently, the radius in stitch units equals the number of stitches per increase.
- Mapped to the project, a fixed per-added-node probability q of depositing curvature −1/3 is a crochet with "one increase per ~1/q stitches". This predicts a hyperbolic radius that scales like 1/q in lattice units. With q ≈ 0.15 e^{−6.44 β}, that gives r_hyp ∝ e^{+6.44 β}, and r_hyp would set the crossover size. This is a prediction to test, not a sourced result.
- Through Gauss-Bonnet, the AHNR result frames dead corners as raising the average degree above 6 (curvature deficit < 0). Mean degree > 6 in the unimodular setting implies hyperbolic, so a nonzero density of −1/3 defects is enough in principle for a hyperbolic large-scale geometry.
- The novelty claim appears defensible as "not found in a targeted search". Prior art covers the separate parts: kinetic trapping of front errors (Winfree kTAM), constant-rate nucleation statistics (Sear), and edge-growth metric prescription (Sharon). I found none that joins them into one front-trapping → negative-curvature → hyperbolic-crossover chain.

### Gaps
- I did not read the full text of Henderson-Taimina 2001 or their book "Crocheting Adventures with Hyperbolic Planes" (A K Peters 2009; 2nd ed. CRC 2018 — **[unverified]**). An explicit stitch-ratio-to-radius formula may appear there.
- I did not read the full texts of Sharon et al. 2004 (AMS digest returned 403) or Klein et al. 2007.
- The novelty search was limited to about five queries, which is not exhaustive. Literature on growing hyperbolic graphene, carbon schwarzites and kirigami growth was not searched.
