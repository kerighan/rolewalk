# RoleWalk — theory (Section 4 draft)

Notation. G = (V, E) with adjacency A (weighted, possibly directed);
lazy walk H = D̃⁻¹(I + A), D̃ = diag(row sums of I + A). For a node u and a
step w ≤ L, p_uj = (H^w)_uj. Θ = {θ_1, ..., θ_m} log-spaced in [1, 100].
Feature map f_θ(x) = e^{iθx} − 1, so f_θ(0) = 0.

Embedding of u at step w:

- characteristic part φ_u^w(θ) = Σ_j f_θ(p_uj), for θ ∈ Θ (real and
  imaginary parts);
- onion layers s_u^w(k) = Σ_{j : d(u,j) = k} p_uj, k = 0..w, scaled by ‖Θ‖;
- optional curvature moments: mean and std of κ_j under p_u·, scaled by ‖Θ‖,
  with κ_j the mean over edges j–z of Σ_x min(m_j(x), m_z(x)).

Directed graphs concatenate the same features for Aᵀ (in-walk).

## P1 — Isomorphism invariance and locality (why it transfers)

**Statement.** Let B_r(u) be the ball of radius r around u (out- and
in-neighbourhoods for directed graphs). If an isomorphism maps B_{L+1}(u) in
G onto B_{L+1}(u′) in G′ with u ↦ u′, then X_u(G) = X_{u′}(G′) (exactly
with tol = 0; with pruning, up to entries below tol, which are dropped
identically). In particular automorphic nodes get equal embeddings.

**Sketch.** (H^w)_u· only involves paths of length ≤ w from u, inside
B_w(u); the isomorphism maps them bijectively and preserves the lazy-walk
weights (degrees inside B_{w+1}). φ is a sum over j, so permutation
invariant; f_θ(0) = 0 makes it independent of nodes outside the support,
hence of |V|. Distances d(u, j) ≤ w are preserved. Curvature of a node
needs its 2-hop ball, so moments need B_{L+2} (state with L+2 if curvature).

**Consequence.** Coordinates mean the same thing in every graph: models
fitted on one graph apply to another (transfer, Tables 5-6), and a node
can be embedded from its neighbourhood only (online prediction, cost
independent of |V|).

## P2 — Squared distances are an MMD between transition distributions

**Statement.** Let μ_u^w = Σ_j δ_{p_uj} (counting measure of u's
transition probabilities, zeros included or not). Then
‖φ_u^w − φ_v^w‖² = MMD²_k(μ_u^w, μ_v^w) with the positive-definite kernel
k(a, b) = Σ_θ Re[f_θ(a) conj(f_θ(b))]
= Σ_θ [cos θ(a−b) − cos θa − cos θb + 1].

**Sketch.** φ_u^w is the kernel mean embedding of μ_u^w for the feature
map (f_θ)_θ; k is PD as an inner product of feature maps. Since
f_θ(0) = 0, atoms at 0 do not count: adding isolated or unreached nodes
changes nothing (contrast with P4).

## P3 — RWSE is contained in the onion layers

**Statement.** s_u^w(0) = (H^w)_uu, the return probability of the lazy
walk after w steps. The onion layers therefore contain the random-walk
structural encoding of the lazy walk, truncated at L steps, and refine it
with the mass at every other distance.

**Remark.** GPS's RWSE uses the simple walk P = D⁻¹A up to 16 steps; the
two coincide up to the lazy/non-lazy change of walk. Empirically,
RoleWalk with 3-5 steps separates as many 1-WL-equivalent pairs as RWSE
with 16 (Table 4c: 213 vs 214 BREC pairs).

## P4 — Relation to GraphWave, and why GraphWave does not transfer

**Statement.** GraphWave embeds u by the empirical characteristic
function of the heat-kernel column Ψ_u = U g_s(Λ) Uᵀ δ_u over all n nodes:
χ_u(t) = (1/n) Σ_j e^{itΨ_uj}. With RoleWalk's aggregation,
Σ_j f_t(Ψ_uj) = n (χ_u(t) − 1). Hence:

1. RoleWalk is GraphWave's construction with the heat kernel replaced by
   lazy-walk powers (sparse, local, directed, no eigendecomposition), and
   with the 1/n normalisation removed;
2. GraphWave's χ_u depends on n through the weight of the zero entries
   (1 − support/n) and on s, chosen per graph from the spectrum. Both
   change from one graph to another, so the same local structure gets
   different coordinates: this explains its chance-level transfer (0.249,
   Table 6).

## P5 — A limit: strongly regular graphs

**Statement.** Two strongly regular graphs with the same parameters
(n, k, λ, μ) get identical multisets of node embeddings.

**Sketch.** A² = kI + λA + μ(J − I − A), so every power of I + A is a
polynomial in I, A, J: (H^w)_uj only depends on whether j = u, j ~ u or
neither, with coefficients fixed by the parameters. Each row has the same
multiset of values and the same distance profile (diameter 2); the
curvature overlap only depends on λ. Matches Table 4c (Shrikhande vs rook
graph, beyond 3-WL, not separated). Consequently RoleWalk is bounded by
what walk statistics determine; it separates all 1-WL-equivalent pairs of
the atlas (≤ 7 nodes) and all 3,181 random regular pairs tested.

*TODO: is RoleWalk at least as expressive as 1-WL in general? Step 1
gives the degree exactly (d + 1 entries equal to 1/(d + 1)); step w
aggregates multisets of degrees along walks, but not the full WL tree.
Likely incomparable in theory, empirically stronger on every family tested.*

## P6 — Complexity

**Statement.** With pruning threshold tol, each row of H^w keeps at most
1/tol entries, so the cost is O(|V| · L · m / tol) in the worst case and
O(L · m · nnz(H^L)) in practice, linear in |V| for bounded-degree growth.
Onion layers reuse the supports of earlier powers (L element-wise
products); curvature costs O(Σ_u d_u²). Embedding a subset S of nodes
only touches B_L(S) (B_{L+2}(S) with curvature).

*Measured: 20,000-node Barabási-Albert graph in 6.5 s on one core;
100,000-node curvature in 0.5 s.*
