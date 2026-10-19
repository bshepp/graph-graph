# Discrete-geometry citations: checked against the full papers (2026-10-18)

Method: I downloaded every arXiv/author PDF below and extracted its text locally (pypdf). Quotes are verbatim from that extraction, so some math layout was lost: fractions are flattened and subscripts are inlined. Anything I did not read in the primary source is marked [unverified].

---

## 1. Hua, Jost, Liu, arXiv 1107.2826: does nonnegative curvature force |B_r| <= C r^2?

### Takeaway
The paraphrase is right in substance but leaves out conditions. Theorem 1.1 (Section 1, p. 4) gives relative volume comparison, volume doubling and |B_R| <= C(D) R^2. All three hold only for **infinite** semiplanar graphs embedded in a surface **without boundary**. "Volume" here means the **degree-weighted** count. The constant depends on the maximum face degree D. The theorem does **not** cover a finite disc, or any surface with an outer boundary or an infinite face.

### Cited Findings
- Statement, Theorem 1.1, p. 4: "Let G be a semiplanar graph with Sec(G) ≥ 0. Then for any p ∈ G, 0 < r < R, we have (1.4) |B_R(p)|/|B_r(p)| ≤ C(D)(R/r)^2, (1.5) |B_2R(p)| ≤ C(D)|B_R(p)|, (1.6) |B_R(p)| ≤ C(D)R^2, (R ≥ 1) where C(D) is a constant only depending on D which is the maximal facial degree of G." — [arXiv 1107.2826v3](https://arxiv.org/abs/1107.2826)
- How volume is measured, p. 3: "The volume of B_R(p) is defined as |B_R(p)| = Σ_{x∈B_R(p)} d_x." In other words, a sum of vertex degrees, not a vertex count. — [arXiv 1107.2826](https://arxiv.org/abs/1107.2826)
- Definition 2.1 and standing assumptions, Section 2, p. 6: "A graph G = (V, E) is called semiplanar if it can be embedded into a connected 2-manifold S without self-intersection of edges and each face is homeomorphic to the closed disk with finite edges as the boundary." Also: "we shall always assume that the surface S has no boundary except in Remark 3.7 and G is a simple graph", and "we assume that 3 ≤ d_x < ∞ and 3 ≤ deg(σ) < ∞ for each vertex x and face σ". — [arXiv 1107.2826](https://arxiv.org/abs/1107.2826)
- The paper studies infinite graphs (p. 2: "In this paper, we are interested in infinite graphs."). Through Chen it notes that the possible topologies are "R2, the cylinder without boundary, and the projective plane minus one point." — [arXiv 1107.2826](https://arxiv.org/abs/1107.2826)
- Curvature (p. 2): Φ(x) = 1 − d_x/2 + Σ_{σ∋x} 1/deg(σ), with 2πΦ(x) = 2π − Σ_x. Sec(G) ≥ 0 means Φ ≥ 0 everywhere, which is equivalent to S(G) being an Alexandrov space of nonnegative curvature. — [arXiv 1107.2826](https://arxiv.org/abs/1107.2826)
- Remark 3.7 (p. 13) is the only place boundaries appear. It extends **only the splitting result (Theorem 3.6)**, not Theorem 1.1. The quote is: "we may formulate the above theorem in the case of regular polygonal surfaces with boundary... the polygonal surface S(G) is isometric to either the cylinder without boundary or the cylinder with boundary, i.e. [a, b] × R." — [arXiv 1107.2826](https://arxiv.org/abs/1107.2826)
- Consequences in the paper: Theorem 1.2 (Poincaré inequality) and Theorem 1.3, "Any semiplanar graph G with Sec(G) ≥ 0 is parabolic" (obtained from the quadratic growth (1.6)). — [arXiv 1107.2826](https://arxiv.org/abs/1107.2826)

### Inferences
- In a triangulation every face has degree 3, so Φ(x) = 1 − d_x/6. Nonnegative curvature then means 3 ≤ d_x ≤ 6. For an **infinite**, boundaryless triangulation with every degree in [3, 6], Theorem 1.1 applies with D = 3. Because 3 ≤ d_x ≤ 6, the degree-weighted volume is within a factor of 6 of the vertex count, so |B_r| <= C r^2 also holds for vertex counts.
- So the result is "volume doubling plus an upper bound of order R^2". That is a **ceiling**, not a statement that growth is exactly quadratic. (1.4)/(1.6) do not give a matching lower bound of order R^2. Cones with positive curvature, and the cylinder (linear growth), are also allowed.
- **Applying it to the project's disc model is an extrapolation.** A finite triangulated disc with an outer face is outside the theorem's hypotheses. The fair wording is: "the infinite boundaryless analogue (degree <= 6 everywhere) has |B_r| <= C r^2 by HJL Thm 1.1". Boundary vertices of a capped disc are not covered by the paper.

### Gaps
- I found no published version of Theorem 1.1 for surfaces with boundary.

---

## 2. Angel, Benjamini, Horesh, "An isoperimetric inequality for planar triangulations", arXiv 1604.05863

### Takeaway
Both statements are quoted correctly. In Theorem 2, **V counts all vertices, boundary vertices included.** Equality is reached only by particular convex hexagons in the triangular lattice, not by every lattice region. Proposition 6 is about an **infinite plane triangulation with minimum degree 6** (not a disc). It gives a uniform isoperimetric constant α(R) > 0.

### Cited Findings
- Theorem 2, p. 1: "Any disc triangulation with V vertices and n boundary vertices, and with all internal degrees at least 6 has V ≤ ⌊(n+3)^2/12⌋. Equality is achieved in the Euclidean triangular lattice." — [arXiv 1604.05863v1](https://arxiv.org/abs/1604.05863)
- V counts all vertices: the proof uses Euler's formula over every vertex, "V − E + F = 1 (since the external face is not counted)", and "V − n is the number of internal vertices" (p. 2). Note that the proof is headed "Proof of Theorem 1", which is a typo for Theorem 2. — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863)
- Mechanism (p. 4): "at distance k from the boundary there are at most n − 6k vertices", which gives V ≤ Σ_{k≥0}(n − 6k)_+ = ⌊(n+3)^2/12⌋. — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863)
- Equality cases (p. 4): "equality is achieved by convex hexagons in the triangular lattice with all boundary segments of suitable lengths in {k, k+1, k+2}. For example, if n = 6k+7 the boundary segments have lengths (k, k+2, k+1, k+1, k+1, k+2)". — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863)
- Corollary 3 extends the bound to any simple map in a disc with internal degrees ≥ 6. Theorem 4 gives an edge-boundary bound |∂A| ≥ √(48V) when all degrees are ≥ 6. — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863)
- Theorem 5 (hyperbolic analogue, p. 5): "If T is a disc triangulation with minimal degree δ and V ≥ V_R vertices. Then n ≥ S_R." (V_R and S_R are the ball and sphere sizes in the δ-regular triangulation.) — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863)
- Proposition 6, p. 5: "For every R > 0 there exists α > 0 such that the following holds. Let T be an infinite plane triangulation with minimal degree 6, and such that for some R, every ball of radius R contains a vertex of degree at least 7. Then T is non-amenable, with isoperimetric constant at least α." The proof begins by reducing to bounded degree (≤ 9), and the authors remark "the proof above gives α > e^{−c/ε}". — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863)

### Inferences
- Theorem 2 needs **internal degree ≥ 6**, which is nonpositive curvature. A cap-6 model enforces **degree ≤ 6**. Both hold only when every interior vertex has degree exactly 6. Any interior vertex below the cap (degree < 6, positive curvature) breaks the hypothesis, and the bound need not hold. Check that the project uses the bound only on regions whose interior vertices all have degree 6, or uses it as a flat-case reference.
- Proposition 6 needs degree-≥7 vertices, so it cannot apply to a cap-6 model. It is relevant only to variants with a cap of at least 7.

### Gaps
- None for these two statements.

---

## 3. Keller and Peyerimhoff, "Cheeger constants, growth and spectrum of locally tessellating planar graphs", arXiv 0903.4793

### Takeaway
The curvature formula is correct. The growth bound μ ≥ log(1 + 2qC/(q−1)) is real, but it is **Proposition 1.2** and it is proved **only for the regular (p,q)-tessellations G_{p,q}**, not for every face-regular tessellation with curvature ≤ −C. The paraphrase is **too strong**. For general q-face-regular tessellations, the paper offers a **comparison** theorem instead (Theorem 3).

### Cited Findings
- Curvature (p. 1): the corner curvature is "κC(v, f) = 1/|v| + 1/|f| − 1/2" and "The curvature at a vertex v ∈ V is given by κ(v) = Σ_{f:v∈f} κC(v, f) = 1 − |v|/2 + Σ_{f:v∈f} 1/|f|." For infinigons (faces with infinitely many edges), 1/|f| = 0. — [arXiv 0903.4793v3](https://arxiv.org/abs/0903.4793)
- Exponential growth is defined from spheres (p. 4): "µ(G, v0) = lim sup_{n→∞} log σ_n / n", where σ_n = |S_n(v0)|. — [arXiv 0903.4793](https://arxiv.org/abs/0903.4793)
- Proposition 1.2, p. 6: "Let G_{p,q} be non-positively curved, i.e., C = −κ(v) = p(1/p + 1/q − 1/2) ≥ 0 ∀v ∈ V, then we have µ(G_{p,q}) = log x_{p,q} ≥ log(1 + 2q/(q−1) · C)." The paper describes it as "A direct consequence of the isoperimetric inequality in [BP1, Cor. 5.2]". — [arXiv 0903.4793](https://arxiv.org/abs/0903.4793)
  - Sign caveat: the extracted text reads "C = −κ(v) = p(1/p + 1/q − 1/2)". Since κ(v) = p(1/p + 1/q − 1/2), the consistent reading is C = p(1/2 − 1/p − 1/q). Either the source has a typo or a minus sign was lost in extraction; I did not check this visually. For {3,7} (p = 7, q = 3): C = 1/6, and the bound gives μ ≥ log 1.5 ≈ 0.405. The true value is log 2.618 ≈ 0.962.
- Theorem 2 (growth recursion, p. 4–5) applies to "a q-face regular plane tessellation without cut locus". Theorem 3 (curvature/growth comparison, p. 6): for "two q-face regular plane tessellations with non-positive vertex curvature" whose spherical average curvatures satisfy κ(S̃_n) ≤ κ(S_n) ≤ 0 ∀n, "the difference sequence σ̃_n − σ_n ≥ 0 is monotone non-decreasing and, in particular, we have µ(G̃) ≥ µ(G)." — [arXiv 0903.4793](https://arxiv.org/abs/0903.4793)
- Theorem 1 (Cheeger constant, p. 3): when |v| ≤ p and |f| ≤ q everywhere and C := inf_v −κ(v) > 0, α(G) ≥ 2 C_{p,q} C. This is a statement about the isoperimetric constant, not about growth. Theorem 4 (p. 6) gives μ(G) ≤ log(p−1) for graphs without cut locus and with |v| ≤ p. — [arXiv 0903.4793](https://arxiv.org/abs/0903.4793)

### Inferences
- A correct citation would be: "for the regular tessellation G_{p,q}, μ ≥ log(1 + 2qC/(q−1)) with C = −κ (KP Prop. 1.2). For non-regular q-face-regular tessellations without cut locus, growth is monotone in spherical-average curvature (KP Thm. 3)."
- For a triangulation (q = 3) where degree ≥ 6 gives a nonpositive curvature sum, Theorem 3 lets you compare growth against G_{7,3} only if the spherical-average curvature is at most that of G_{7,3} on every sphere. Sparse degree-7 defects give **less** negative curvature, so the inequality goes the other way: G_{7,3} is an upper comparison, not a lower one.

### Gaps
- The sign in Proposition 1.2 should be checked visually in the PDF.

---

## 4. DeVos and Mohar (Trans. AMS 359, 2007), and the sharpening to 208

### Takeaway
The paraphrase is correct. The required hypotheses are: 2-cell embedding, every vertex **and every face** of degree ≥ 3, and φ > 0 everywhere. The paper (Theorem 1.7) gives finiteness, a surface that is the sphere or projective plane, and |V| ≤ 3444 except for prisms, antiprisms and their projective analogues. Higuchi 2001 (J. Graph Theory 38, 220–229) is the source of the conjecture. **The sharp bound of 208 was proved by Luca Ghidelli** (arXiv 1708.08502, Theorem 3.1). Oh 2017 proved 380, and Nicholson and Sneddon 2011 found the 208-vertex examples.

### Cited Findings
- Theorem 1.7 (p. 5): "Let G be a graph which is 2-cell embedded in a surface S so that every vertex and face has degree ≥ 3. If φ is everywhere positive, then S is homeomorphic to either the 2-sphere or the projective plane and G is finite. Furthermore, if G is not a prism, antiprism, or the projective planar analogue of one of these, then |V(G)| ≤ 3444." — [DeVos & Mohar, author PDF](https://www.sfu.ca/~mdevos/papers/higuchi.pdf)
- Conjecture 1.6 (Higuchi), p. 5: "Let G be a graph which is 2-cell embedded in a surface S so that every vertex and face has degree ≥ 3. If S is homeomorphic to a subset of the 2-sphere and φ is everywhere positive, then G is finite." The paper cites "[3] Y. Higuchi, Combinatorial curvature for planar graphs, J. Graph Theory 38 (2001) 220–229." — [DeVos & Mohar PDF](https://www.sfu.ca/~mdevos/papers/higuchi.pdf)
- Main tool, Theorem 1.3 (p. 3): this is the Gauss–Bonnet-type inequality for **tame** (ν > 0) infinite polygonal surfaces, Σκ(v) ≤ 2πχ(S) − 2πt. — [DeVos & Mohar PDF](https://www.sfu.ca/~mdevos/papers/higuchi.pdf)
- The journal is Trans. Amer. Math. Soc. 359 (2007), 3287–3300 (from the AMS listing in search results; the AMS page returned 403 when fetched). Ghidelli's reference list misprints the volume as "369". — [AMS listing](https://www.ams.org/tran/2007-359-07/S0002-9947-07-04125-6/); [Ghidelli](https://arxiv.org/abs/1708.08502)
- Ghidelli, Theorem 3.1: "Let G be a planar PCC graph. Then #V ≤ 208." The abstract adds that the bound is sharp, faces have at most 41 sides (also sharp), and projective PCC graphs have at most 104 vertices. — [arXiv 1708.08502v2](https://arxiv.org/abs/1708.08502)
- History, from Ghidelli's introduction (p. 2): "DeVos and Mohar [6] proved that all planar PCC graphs have at most 3444 vertices... The first conjectured answer was 120... lower bound was improved to 138 in [8] and to 208 in [9, 10]... The paper [11] lowers it to 579, but unfortunately it contains a mistake... Oldridge [10] lowered the bound to 244, conditionally... an unconditional upper bound of 380 vertices was recently given by Oh [12]." In that list, [8] is Réti–Bitay–Kosztolányi 2005, [9] Nicholson–Sneddon, NZ J. Math 41 (2011), [10] Oldridge PhD thesis 2017, [11] Zhang, Discrete Math 2008, and [12] B.-G. Oh, Discrete Math 340 (2017) 1300–1310. — [arXiv 1708.08502](https://arxiv.org/abs/1708.08502)

### Inferences
- In a triangulation φ = 1 − d/6, so positive curvature everywhere means every degree is ≤ 5. The theorem then forces the graph to be finite and to live on the sphere or the projective plane. This is a closed-surface statement. It says nothing directly about discs with boundary.

### Gaps
- I did not read the published version of Ghidelli (journal print [24]), Oh 2017, or Higuchi 2001 directly [unverified beyond the citations quoted above].

---

## 5. Wu, Menichetti, Rahmede, Bianconi, "Emergent complex network geometry", arXiv 1412.3405 (Sci. Rep. 2015): the algorithm

### Takeaway
The model has two parameters: m, the maximum number of triangles per link, and p. It starts from one triangle. Each step runs (a) always and (b) with probability p. (a) glues a new triangle with a new node onto a uniformly random unsaturated link. (b) adds one link between nodes at distance 2, chosen through a pair of adjacent unsaturated links, and closes every triangle it can without pushing any link past m. With m = 2 or p = 0 the graph is a planar triangulation (χ = 1). The authors report an **infinite Hausdorff dimension** (D ∝ log N) and a **finite spectral dimension**. The degree distribution is exponential for m = 2 and has a power law with γ = 3 for m = ∞. Curvature is R_i = 1 − k_i/2 + t_i/3.

### Cited Findings (algorithm, verbatim, pp. 4–5)
- Setup: "we start from a network formed by a single triangle... At each time we perform two processes". Saturation: "In the case in which m is finite we call the links to which we can still add at least one triangle unsaturated. All the other links we call saturated." — [arXiv 1412.3405v2](https://arxiv.org/abs/1412.3405)
- Process (a): "We add a triangle to an unsaturated link (i, j)... We choose this link randomly with probability Π[1]_(i,j) = a_ij ρ_ij / Σ_{r,s} a_rs ρ_rs (1)... ρ_ij = 1 if the number of triangles to which the link (i, j) belongs is less than m, otherwise it is zero... Having chosen the link (i, j) we add a node s, two links (i, s) and (j, s) and the new triangle linking node i, node j and node s." In other words, the link is chosen **uniformly among unsaturated links**. — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
- Process (b): "With probability p we add a single link between two nodes at hopping distance 2, and we add all the triangles that this link closes, without adding more than m triangles to each link. In order to do this, we choose an unsaturated link (i, j) with probability Π[1]_(i,j) given by Eq. (1), then we choose one random unsaturated link adjacent either to node i or node j as long as this link is not already part of a triangle including node i and node j." The choice uses Eq. (2), Π[2]_{r,s} = (1/N)[a_is ρ_is δ_{r,i} + a_rj ρ_rj δ_{j,s} − a_is ρ_is a_sj ρ_sj δ_{i,r} − a_ri ρ_ri a_rj ρ_rj δ_{j,s}]. Then: "Let us assume without loss of generality that the chosen link (r, s) = (r, j). Then we add a link (i, r) and all the triangles passing through node i and node r as long as this process is allowed (i.e. if by doing so we do not add more than m triangles to each link). Otherwise we do nothing." — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
- Figure 1 caption: "In process (b) with probability p two nodes at distance two in the simplicial complex are connected and all the possible triangles that can link these two nodes are added as long as this is allowed (no link acquires more than m triangles incident to it)." The caption also says the figure shows m = 2. The supplementary information contains MATLAB code (not read). — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)

#### Algorithm in steps (my reconstruction from the quotes above)
1. Start with nodes {0, 1, 2}, links {01, 12, 02}, one triangle. Every link has triangle count 1.
2. At each time step t:
   - (a) Pick an unsaturated link (i, j) uniformly at random, where unsaturated means triangle count < m. Add a new node s, links (i, s) and (j, s), and triangle (i, j, s). Increase the triangle counts: (i, j) goes up by 1, and (i, s) and (j, s) start at 1.
   - (b) With probability p: pick an unsaturated link (i, j) uniformly. Pick a random unsaturated link (r, j) adjacent to j, or (i, r) adjacent to i, that is not already in a triangle containing both i and j. Call the far endpoint r (at distance 2 from the other end). If adding link (i, r), together with **all** triangles it closes (every common neighbour w of i and r gives triangle (i, r, w)), would keep every link's triangle count ≤ m, add it. Otherwise do nothing.
   - Ambiguity: the text does not say whether (b) checks or adds the closed triangles one at a time or all at once. "as long as this process is allowed... Otherwise we do nothing" reads as all-or-nothing, but the Figure 1 caption ("as long as this is allowed") could also mean adding triangles until one would exceed m. The SI MATLAB code would settle it [unverified].
3. Process (a) adds exactly one node per step, so N = t + 3.

### Cited Findings (parameters and results)
- Parameters in the figures: Fig. 2 (visualizations) uses m = 2, p = 0.9; m = ∞, p = 0.9 in the caption (the main text says "m = ∞, p = 0."); and m = 4, p = 0.9. Fig. 3 uses N = 10^5, p ∈ {0, 0.9}, m ∈ {2, 3, 4, 5, ∞}. Fig. 4 uses m = 2, 3, 4, 5, ∞ with p = 0.9. Fig. 5 uses N = 10^5 over 10 realizations, and N = 10^4 for the K-core. Fig. 6 (spectral dimension) uses N = 10^4, p = 0.9, m = 2, 3, 4 in panel A, and in panel B an average over 40 realizations with p = 0.8, 0.9. — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
- Planarity (p. 7): "For m = 2 and any value of p, or for p = 0 and any value of m the networks are planar graphs... Therefore in these cases we have an Euler characteristic χ = 1 (in fact here we do not count the external face)." — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
- Freezing (p. 7): "for p = 0, each node will increase its connectivity indefinitely with time having always exactly two unsaturated links attached to it. On the contrary, if p > 0 there is a small probability that some nodes will have all adjacent links saturated, and a degree that is frozen". m = 2, p = 0.9 shows "an active boundary... and a frozen bulk region". — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
- Hausdorff and spectral dimension (p. 13): "The geometrical growing network is growing exponentially, so the Hausdorff dimension is infinite. Nevertheless, these networks develop a finite spectral dimension d_S as clearly shown in Figure 6, for m = 2, 3, 4 and p = 0.9." Also (p. 13): "asymptotically in time D ∝ log N, independently of the value of p and m." The text gives **no numerical d_S values for the model**; they appear only in Fig. 6B, which I could not read. Spectral dimension is defined (Methods) by g(λ) ∼ λ^{d_S/2−1} for the combinatorial Laplacian L = kδ − a. — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
- Degree distribution: "scale-free for m = ∞ and exponential for m = 2 for any value of p. Moreover for finite values of m > 2 the degree distribution can develop a tail that is broader for increasing values of m" (p. 7). Methods: for m = ∞, p = 0, P(k) = 12/[(k+2)(k+1)k] (γ = 3). For m = 2, p = 0, P(k) = (1/2)(2/3)^{k−1}, k ≥ 2 (eq. 13). — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
- Curvature (p. 9): "R_i = 1 − k_i/2 + t_i/3, (4) where k_i is the degree of node i, and t_i is the number of triangles passing through node i", with Gauss–Bonnet χ = Σ R_i. For bulk nodes (k_i = t_i), R_i = 1 − k_i/6 (eq. 6). For boundary nodes (k_i = t_i + 1), R_i = (4 − k_i)/6 (eq. 7). For m > 2, p > 0 the network is non-planar; the authors keep eq. (4) and identify it with Knill's curvature truncated at triangles. — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)
  - Internal inconsistency: Methods (p. 17) says "the curvature of the nodes is in this case R = 1 − k/4" (for m = ∞, p = 0, and again for m = 2, p = 0). That contradicts eq. (7), which gives (4 − k)/6 when k = t + 1. Use eq. (4) directly.
- Euler characteristic: when m = 2, process (b) "add[s] only one new link and one new triangle, therefore ∆χ = 0". When m > 2, p > 0, "χ grows linearly with N" (pp. 12, 17–18). — [arXiv 1412.3405](https://arxiv.org/abs/1412.3405)

### Inferences
- With m = 2, every link lies in at most 2 triangles, so the complex is a 2-manifold with boundary (a disc). A node with all links saturated is an **interior** vertex. Process (a) only attaches to boundary links, and (b) closes boundary angles. The model has **no degree cap**: interior degrees are whatever they are when a node freezes. That differs from a cap-6 model, and is a natural axis for comparison.
- With m = 2, p = 0, every node stays on the boundary (it always has two unsaturated links), so there are no interior vertices. This is the same object as NGF with s = −1, d = 2 (item 6).

### Gaps
- Numerical d_S values (only in Fig. 6B), the SI MATLAB code, and the exact semantics of the triangle-closing check in (b) [unverified].

---

## 6. Bianconi and Rahmede, arXiv 1607.05710: flavour and hyperbolicity

### Takeaway
Check the citation first. arXiv 1607.05710 is **"Emergent Hyperbolic Network Geometry"** (Bianconi and Rahmede; arXiv v2 dated 29 Dec 2016). "Network Geometry with Flavor" is a **different, earlier paper** (Bianconi & Rahmede, Phys. Rev. E 93, 032315 (2016), believed to be arXiv 1511.04539 [unverified]). On the claims themselves:
- Claim 1 is close but paraphrases two separate statements. The paper says the networks are **small-world** for every (s, d) except s = −1, d = 1 (a chain). It then argues that the natural embedding is hyperbolic Hⁿ, and adds that the embedding "works particularly well for flavor s = −1 while some caution is required" for s = 0, 1. "Hyperbolic for every flavour and dimension except the chain" is therefore slightly too strong.
- Claim 2 (s = −1, d = 2) is correct in substance. Each link lies in at most 2 triangles, so a link that already holds one triangle accepts exactly one more. The dual is a tree. The result is a triangulated disc whose dual graph is a tree.

### Cited Findings
- Dynamics (p. 4): "At each time we add a d-dimensional simplex to a (d − 1)-face α. The face α is chosen randomly with probability Π_α = (1 + s n_α)/Σ_{α'}(1 + s n_{α'}). (1)... For s = −1 it is possible to attach a simplex only to faces with n_α = 0... the resulting network is a manifold, with each (d − 1)-face incident at most to two d-dimensional simplicial complexes, i.e. n_α = 0, 1." Here n_α is the "number of d-dimensional simplices incident to it minus one". — [arXiv 1607.05710v2](https://arxiv.org/abs/1607.05710)
- Small-world and hyperbolic (p. 5): "The resulting networks are small world for every flavor s and any dimension d except from the special case s = −1, d = 1 in which the resulting network is a chain... the small-world property suggests that the natural embedding of these networks is hyperbolic. Nevertheless the small-world property might not be sufficient to guarantee an embedding in the hyperbolic space. Here we show that... the hidden geometry... are the hyperbolic spaces H^d". — [arXiv 1607.05710](https://arxiv.org/abs/1607.05710)
- Caveat (p. 7): "the natural hyperbolic embedding... works particularly well for flavor s = −1 while some caution is required when using this embedding for flavors s = 0 and s = 1." The hyperbolicity "is a consequence of the assumption that each link of the simplicial complex must have equal length". — [arXiv 1607.05710](https://arxiv.org/abs/1607.05710)
- Dual tree (p. 5): "Since in our simplicial complex evolution at each time we glue a new d-simplex to a (d − 1)-face, the resulting structure of the dual network is a tree." — [arXiv 1607.05710](https://arxiv.org/abs/1607.05710)
- Degree distributions (pp. 8–9): "scale-free for dimension d > (1 − s), (5) while for dimension d = 1 − s they have exponential degree distribution". For d + s = 1, P(k) = (d/(d+1))^{k−d} · 1/(d+1). For d + s > 1 the result is a Gamma-function form with γ = 2 + 1/(d + s − 1) ≤ 3. Also: "for d + s = 0 the simplicial complexes reduce to chains". — [arXiv 1607.05710](https://arxiv.org/abs/1607.05710)
- Table I (p. 8): modularity M and clustering C at N = 10^4 (20 realizations). For d = 2, M = 0.97 / 0.94 / 0.90 and C = 0.65 / 0.74 / 0.79 for s = −1 / 0 / 1. — [arXiv 1607.05710](https://arxiv.org/abs/1607.05710)

### Inferences
- s = −1, d = 2 gives d + s = 1, so the degree distribution is exponential, P(k) = (2/3)^{k−2}/3. Every new triangle attaches to a boundary edge (n_α = 0), so every vertex stays on the boundary (no interior vertices). The result is a maximal outerplanar triangulated polygon whose dual is a random tree. It is the same model as Wu et al. with m = 2, p = 0.
- "Planar tree of triangles" is fair wording if it means "a triangulated disc with a tree dual and all vertices on the boundary".
- The scale-free regime d > 1 − s is not one of the two cases quoted in the paper text (s = −1 with d = 1 and with d = 2). As a cross-check: s = 0 needs d > 1, s = 1 needs d > 0, and s = −1 needs d > 2.

### Gaps
- I did not read "Network Geometry with Flavor" (PRE 2016) itself. If the project means that paper, its flavour/dimension table should be checked separately [unverified].

---

## 7. Angle deficit, a curvature density from degree-7 defects, and the growth constant of {3,7}

### Takeaway
- The angle deficit 2π − d·π/3 = (6 − d)π/3 for an equilateral triangulation is standard Regge-calculus material. Loll 2023 (arXiv 2306.13782, Sec. 3.1, eq. 8) is a citable modern statement of the deficit-angle definition; it cites Regge 1961.
- K = −2πλ/(3√3) is a **derivation** (deficit −π/3 per degree-7 vertex divided by area √3/2 per vertex). I found no paper that states it in that form.
- I found **no paper** that tests ball growth of sparse-defect triangulations against the hyperbolic-disc area formula.
- The growth constant of {3,7} is **exactly (3+√5)/2 = φ² ≈ 2.618**, the largest root of x² − 3x + 1. This is verified from Keller–Peyerimhoff Prop. 1.1 and their recursion (Theorem 2). Sphere sizes are 1, 7, 21, 56, 147, …, with σ_{n+1} = 3σ_n − σ_{n−1} for n ≥ 2.

### Cited Findings
- Deficit angle (Loll, Sec. 3.1, p. 9): "the deficit angle δ_h associated with a (D−2)-dimensional subsimplex ('hinge') of a triangulation is defined by δ_h = 2π − Σ_{i|∆_i⊃h} α_i, (8)... In two dimensions, the deficit angle (8) is a direct measure of the Gaussian curvature associated in a distributional manner with the vertex h." On area (p. 10): "δ_h represents Gaussian curvature... integrated over a two-dimensional area element associated with the vertex h. A possible choice is given by the area of the polygon dual to h, whose corners are the barycentres of the triangles sharing h." Regge calculus is cited as "[11] T. Regge, General relativity without coordinates, Nuovo Cim. 19 (1961) 558-571". — [Loll, arXiv 2306.13782](https://arxiv.org/abs/2306.13782); Regge 1961 itself not read [unverified]
- Combinatorial version: HJL (p. 2) states "2πΦ(x) = 2π − Σ_x" with Φ = 1 − d/2 + Σ 1/deg(σ). For triangles that gives 2πΦ = 2π − dπ/3 = (6 − d)π/3. — [arXiv 1107.2826](https://arxiv.org/abs/1107.2826)
- Exact growth constant of G_{p,q} for q ∈ {3, 4, 6}, KP Proposition 1.1 (p. 5): "µ(G_{p,q}) = log(p/2 − 2/(q−2) + √((p/2 − 2/(q−2))² − 1))". The basis is that "g_{p,q} is divisible by z² − (p − 4/(q−2))z + 1 in the case q = 3, 4, 6". KP also say the growth function is the rational function h_{p,q}/g_{p,q}, which "agrees with results of Cannon and Wagreich [CaWa] and Floyd and Plotnick [FP, §3]", and that g_{p,q} is a Salem polynomial ([CaWa], [BaCS]). — [arXiv 0903.4793](https://arxiv.org/abs/0903.4793)
- Applying this to {3,7} (p = 7 vertex degree, q = 3 triangles) gives x = 3/2 + √(9/4 − 1) = (3 + √5)/2 ≈ 2.618, a root of z² − 3z + 1 (from KP's divisor with p − 4/(q−2) = 3). — derived from [KP Prop. 1.1](https://arxiv.org/abs/0903.4793)
- Sphere recursion: KP Theorem 2 with q = 3 gives N = 1. For G_{p,q} with q odd, "b_l − κ_k = p − 4" at l = (N−1)/2 = 0. The recursion is then σ_2 = 3σ_1 = 21 and σ_{n+1} = 3σ_n − σ_{n−1} for n > 1, so the sequence runs 1, 7, 21, 56, 147, 385, … These numbers are 7·F_{2n}, where F_{2n} are even-indexed Fibonacci numbers (my arithmetic). — derived from [KP Theorem 2](https://arxiv.org/abs/0903.4793)
- Ball-growth-vs-defect literature: a search found Angel–Benjamini–Horesh Prop. 6 (item 2, qualitative non-amenability) and hyperbolic random-triangulation papers, none of which make a quantitative comparison with A(r). — [arXiv 1604.05863](https://arxiv.org/abs/1604.05863); [arXiv 1804.07711](https://arxiv.org/abs/1804.07711) (search snippet only, [unverified])

### Inferences
- Defect curvature density: a unit equilateral triangulation has F ≈ 2V triangles of area √3/4 each, so the mean area per vertex is √3/2. This matches the barycentric dual cell of a degree-6 vertex, 6·(√3/4)/3. A degree-7 vertex carries deficit −π/3. A fraction λ of degree-7 vertices (all others degree 6) therefore averages to K = (−π/3)λ/(√3/2) = **−2πλ/(3√3)** ≈ −1.209λ. The arithmetic is right. It is a mean-field (coarse-grained) value, and only meaningful at scales much larger than λ^{−1/2}.
- The hyperbolic disc area should be written with |K|: A(r) = (2π/|K|)(cosh(r√|K|) − 1). The form "(2π/K)(cosh(r√K) − 1)" is ill-defined if K < 0. Graph distance r is also not Euclidean length: graph distance and geodesic distance on the piecewise-flat surface differ by a direction-dependent factor (in the flat triangular lattice, graph balls are hexagons). Any fit should allow a metric rescaling constant.
- Sanity check: for pure {3,7}, λ = 1 gives K ≈ −1.209 and √|K| ≈ 1.10, so the predicted e-folding per unit length is about 1.10. The exact combinatorial rate is log φ² ≈ 0.962 per graph step. The two differ by an O(1) metric factor, as expected, so expect a fitted conversion constant rather than an exact match.

### Gaps
- Regge 1961 was not read [unverified]. No primary paper states K = −2πλ/(3√3). No paper testing A(r) against sparse-defect triangulations was found. I did not open Cannon–Wagreich, Floyd–Plotnick or Bartholdi–Ceccherini-Silberstein, so the {3,7} constant is verified through KP's statement of their results.
