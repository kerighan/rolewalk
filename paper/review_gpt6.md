# Part 1 — Mathematical verification

## Overall assessment

Several central statements are correct, but the paper currently overstates what they establish. There are also substantive errors:

1. **The conjecture is false for the default, curvature-free RoleWalk.** Below I give an explicit 15-node counterexample at \(L=2\), and a modification giving a counterexample at \(L=3\).
2. **The stability proposition bounds only the characteristic-function block, not the embedding.** The onion-layer block can change by a constant after inserting an edge between arbitrarily high-degree endpoints.
3. **The MMD identity does not cover the reported curvature features.** Replacing the second moment by the standard deviation is nonlinear and destroys the stated atom-level kernel identity.
4. **Directed locality with symmetrized curvature is false for the stated out-/in-balls.** Weak, symmetrized neighborhoods are needed.
5. **Distance-regular graphs are a sufficient failure class, not an “exact limit.”**
6. **“Shrikhande / rook, beyond 3-WL” is wrong under the usual WL convention.**
7. **Pruning is insufficiently specified and can destroy later walk distributions even when the immediate characteristic-function approximation is accurate.**

I can verify the mathematics and arithmetic visible in the source. I cannot independently verify the CSVs, scripts, figure, dynamically loaded scalability table, or bibliography, because those files were not supplied.

---

## 1. Definitions and method

### 1.1 Walk

> “\(G=(V,E)\) with non-negative weights \(A\) … \(H=\widetilde D^{-1}(I+A)\).”

**Status: correct for a finite graph, with terminology and conventions to fix.**

Because
\[
\widetilde D_{uu}=1+\sum_v A_{uv}>0,
\]
\(H\) is well-defined, nonnegative, and row-stochastic. Thus
\[
p_u^w=e_u^\top H^w
\]
is a probability distribution.

Issues:

- This is a **self-loop-augmented random walk**, not necessarily a lazy walk in the common sense \(H_{uu}\ge 1/2\). On a loopless unweighted graph,
  \[
  H_{uu}=\frac1{d_u+1},
  \]
  which is below \(1/2\) when \(d_u>1\). It is aperiodic, but standard results specifically requiring a half-lazy chain should not be invoked.
- State whether input self-loops and parallel edges are allowed. Several later propositions require a **simple, loopless, undirected, unweighted** graph.
- Distances and support statements should use the graph of **positive-weight edges**. If \(E\) includes zero-weight edges, shortest-path distances need not be recoverable from walk supports.
- For directed graphs, the \(A^\top\) branch needs its **own row normalization**, as the definition appears to intend.

### 1.2 Characteristic function

> “\(\phi_u^w(\theta)=\sum_j(e^{i\theta p_{uj}^w}-1)\).”

**Status: mathematically well-defined; not literally a probability characteristic function.**

This is the centered Fourier transform of the finite counting measure
\[
\nu_u^w=\sum_{j:p_{uj}^w>0}\delta_{p_{uj}^w}.
\]
It is not the characteristic function of a probability measure, because it is unnormalized and equals zero at \(\theta=0\), rather than one.

That is not a defect in the construction. But the terminology should be precise: **“centered, unnormalized characteristic-function features of the probability multiset.”**

The support restriction is correct because \(f_\theta(0)=0\).

### 1.3 Onion layers

> “\(s_u^w(k)=\sum_{j:d(u,j)=k}p_{uj}^w\), \(k=0,\ldots,w\).”

**Status: correct with positive-edge distances and exact walks.**

For an exact walk,
\[
\sum_{k=0}^w s_u^w(k)=1.
\]

Because every vertex has a positive self-loop, the support of \(H^w\) is exactly the set of vertices reachable in at most \(w\) positive-edge steps. Thus earlier supports recover distance layers.

Two qualifications:

- The incoming branch must use distances in the reversed graph.
- **This support argument does not survive pruning.** A vertex can be removed when first reached and reappear later. Its first appearance in a pruned power need not equal its graph distance.

### 1.4 Curvature

> “For an edge \(xy\) of the symmetrised unweighted graph, \(\kappa(x,y)=\sum_z\min(m_x(z),m_y(z))\).”

**Status: correct if \(m_x\) is explicitly the walk on that symmetrized, unweighted graph. Currently ambiguous.**

The text could also be read as using the original directed/weighted \(m_x\). These are different definitions. Specify which one is implemented.

For the symmetrized, simple, unweighted graph,
\[
m_x=\frac{\mathbf 1_{N[x]}}{d_x+1},
\]
so
\[
\kappa(x,y)
=\frac{|N[x]\cap N[y]|}{\max(d_x+1,d_y+1)}.
\]
This useful closed form should be stated.

> “Since a unit of mass that does not overlap moves at least one step, \(W_1(m_x,m_y)\ge 1-\kappa(x,y)\).”

**Status: correct for shortest-path distance on the unweighted symmetrized graph.**

For every coupling, at most
\[
\sum_z\min(m_x(z),m_y(z))
\]
can remain on the diagonal. All off-diagonal transport costs at least one. Therefore
\[
W_1\ge 1-\kappa.
\]

For adjacent vertices, Ollivier curvature is \(1-W_1\), so the claimed upper bound follows.

However:

- This argument requires a metric whose distinct points have distance at least one. It is not automatically the appropriate statement for arbitrary weighted edge lengths.
- \(\kappa\) is **not Ollivier–Ricci curvature**. In particular, it is always nonnegative, whereas genuine Ollivier curvature can be negative.
- Define \(\kappa_x\) for isolated vertices. “Mean over the edges at \(x\)” is otherwise undefined.

### 1.5 Footprint and feature map

> “The embedding is the mean of a single feature map over the footprint, \(X_u=\int\Psi\,d\mu_u\).”

**Status: correct as an unnormalized integral, not a normalized mean.**

The footprint is a finite positive **counting measure**, with total mass
\[
\mu_u(\mathcal Z)=\sum_{w=1}^L|\operatorname{supp}p_u^w|.
\]
This varies across nodes. Calling the integral a “mean” risks confusion with probability kernel mean embeddings.

The curvature-free feature map correctly produces the characteristic-function and scaled layer-mass blocks.

For directed embeddings, a single-footprint formulation also needs an **orientation mark** or two orthogonal orientation blocks. As written, the footprint definition describes only one walk direction.

### 1.6 Curvature reparameterization

> “\(\Psi\) adds \(\|\Theta\|a(\kappa,\kappa^2)\); we report \(\|\Theta\|(\mu_u^w,\sigma_u^w)\), a fixed reparametrisation of these two moments.”

**Status: the reparameterization is information-preserving, but the subsequent kernel interpretation is not preserved.**

Let
\[
q_u^w=\sum_j p_{uj}^w\kappa_j^2.
\]
Then
\[
\sigma_u^w=\sqrt{q_u^w-(\mu_u^w)^2},
\qquad
q_u^w=(\sigma_u^w)^2+(\mu_u^w)^2.
\]
So the two representations contain the same information.

But the transformation is nonlinear. In general,
\[
\left\|(\mu,\sigma)-(\mu',\sigma')\right\|^2
\ne
\left\|(\mu,q)-(\mu',q')\right\|^2.
\]

Consequently, the reported curvature-augmented embedding is **not** the integral of the stated atom-level feature map, and its Euclidean distance is **not** the MMD of the stated kernel.

A simple general obstruction: standard deviation is not affine in a measure. Both \(\delta_0\) and \(\delta_1\) have standard deviation zero, but their equal mixture has standard deviation \(1/2\). Integrating a fixed atom feature is affine.

Fix this by either:

1. stating the MMD result only for the curvature-free embedding;
2. reporting raw first and second moments; or
3. defining a kernel on whole footprints after nonlinear summary extraction, without calling it the stated atom-level MMD.

### 1.7 Feature counts

> “\(2mL+L(L+3)/2\ (+2L)=69\ (75)\) features.”

**Status: correct for one walk direction.**

Indeed,
\[
\sum_{w=1}^L(w+1)=\frac{L(L+3)}2.
\]
For \(m=10,L=3\):
- characteristic-function coordinates: \(60\);
- layers: \(9\);
- curvature: \(6\).

Thus:

| Setting | \(L=3\) | \(L=5\) |
|---|---:|---:|
| One direction, no curvature | 69 | 120 |
| One direction, curvature | 75 | 130 |
| Two directions, no curvature | 138 | 240 |
| Two directions, curvature | 150 | 260 |

The paper should state the directed dimensions explicitly.

These are coordinate counts, not independent degrees of freedom: each layer vector sums to one.

### 1.8 Pruning

> “After each product, entries below \(\mathrm{tol}=10^{-4}\) are dropped; their mass is added back to \(\operatorname{Im}\phi\) to first order.”

**Status: the immediate approximation is valid; the iterative algorithm and its errors are not adequately specified.**

For discarded values \(a_j<\tau\), total discarded mass \(M=\sum_j a_j\),
\[
\left|\sum_j\bigl(e^{i\theta a_j}-1-i\theta a_j\bigr)\right|
\le \frac{\theta^2}{2}\sum_j a_j^2
\le \frac{\theta^2\tau M}{2}.
\]
Thus the stated first-order correction has a valid snapshot error bound.

But this does **not** justify dropping the entries before the next multiplication. Lost mass can subsequently concentrate.

**Counterexample: a large star.** Let \(u\) be the center of a star with \(d=10{,}000\) leaves. At step one every nonzero probability is
\[
\frac1{d+1}<10^{-4}.
\]
The entire row is pruned. Yet the exact step-two return probability is
\[
(H^2)_{uu}
=
\frac1{(d+1)^2}+\frac{d}{2(d+1)}
\approx \frac12.
\]
A scalar first-order correction at step one cannot reconstruct this later concentration.

Also specify:

- whether discarded mass is propagated at later steps;
- whether layer masses are corrected;
- whether curvature expectations are renormalized;
- how distances are recovered despite pruned supports;
- whether a minimum number of entries is retained per row.

The exact theory is explicitly restricted to \(\mathrm{tol}=0\), which is appropriate, but the paper needs an approximation analysis or at least a serious tolerance study.

### 1.9 Clustering and transfer

> “Fitted centroids label nodes of any graph.”

**Status: mechanically correct, not a transfer-performance guarantee.**

The feature coordinates have a fixed definition, so a trained predictor can accept another graph’s features. That does not imply distributional alignment or accurate transfer.

Ward clustering also has potentially quadratic memory/time costs. Embedding scalability should not be presented as scalability of the entire role-discovery pipeline unless clustering is benchmarked separately.

---

## 2. Proposition: locality and invariance

> “\(X_u\) is determined by … \(B_L(u)\), and … \(B_{L+2}(u)\) with curvature.”

### Without curvature

**Status: correct under weight- and direction-preserving rooted isomorphisms.**

The proof’s walk expansion is correct. For a length-\(w\) walk, every departure vertex lies within distance \(w-1\), and every outgoing neighbor needed to normalize its transition row lies within distance \(w\). Therefore the induced \(L\)-ball contains all required information.

The distance-layer argument is also correct.

### With curvature, undirected graphs

**Status: correct.**

A visited vertex \(j\) can lie at distance \(L\). Computing \(\kappa_j\) requires the neighborhoods of \(j\) and its neighbors, hence an \(L+2\) ball is sufficient.

### With curvature, directed graphs

**Status: wrong if \(B_r(u)\) means the union of the pure out-ball and pure in-ball, as the text suggests.**

Consider:

- graph 1: \(u\to v\);
- graph 2: \(u\to v\leftarrow z\).

For root \(u\), the union of its pure out- and in-balls is \(\{u,v\}\) in both graphs, for every radius. Their induced rooted directed graphs agree.

But their symmetrizations are respectively an edge and a three-node path. In graph 1,
\[
\kappa_u=\kappa_v=1.
\]
In graph 2,
\[
\kappa_u=\kappa_v=\frac23.
\]
Thus the curvature-augmented step-one embedding of \(u\) changes.

**Repair:** use the radius-\(L+2\) ball in the **underlying undirected graph**, or precisely define the mixed neighborhood required for the two directed walks and curvature.

> “In particular it does not depend on \(|V|\).”

Interpret this as invariance to changes outside the relevant neighborhood—not as independence from graph size when graph size changes the neighborhood itself.

> “Coordinates therefore mean the same thing in every graph, which is what transfer … requires.”

This establishes a fixed feature definition. It does not establish successful statistical transfer. Also, locality is sufficient for this kind of computation, not necessary for inductive features.

---

## 3. Proposition: injectivity

**Status: correct, with minor qualifications.**

### First claim

The function
\[
F(\theta)=\sum_{p\in P}e^{i\theta p}
-\sum_{q\in Q}e^{i\theta q}
-(|P|-|Q|)
\]
is entire. A set of zeros with a finite accumulation point forces \(F\equiv0\). Linear independence of exponentials with distinct exponents gives equality of all positive-atom multiplicities.

The restriction \(P,Q\subset(0,1]\) matters: zero atoms are deliberately invisible.

### Finite-grid claim

For \(P\ne Q\), \(F\) is nonzero entire, so its real zero set is discrete and countable. A countable union of these exceptional sets is null. For \(m\ge1\), almost every first coordinate alone separates the specified countable family; hence almost every grid does.

The argument is sound.

Qualifications:

- The result gives no numerical separation margin.
- The generic-grid theorem alone does not certify the particular fixed log-spaced grid.
- The countability restriction matters for arbitrary real-weighted graphs.
- Equality of probability multisets is not equality of rooted graphs or WL colors.

> “With \(\Theta\) an interval, \(\phi_u^w\) is exactly the multiset…”

Correct as an information-equivalence statement, not literally equality of the two objects.

---

## 4. Proposition: one kernel on the footprint

### Kernel identity

**Status: correct for the curvature-free embedding and unnormalized finite measures.**

For one frequency,
\[
\operatorname{Re}\left[
(e^{i\theta a}-1)\overline{(e^{i\theta b}-1)}
\right]
=
\cos(\theta(a-b))-\cos(\theta a)-\cos(\theta b)+1.
\]
The step and layer Kronecker factors follow from orthogonal blocks.

Thus
\[
\|X_u-X_v\|^2
=
\iint K(z,z')\,d(\mu_u-\mu_v)(z)\,d(\mu_u-\mu_v)(z').
\]

Two corrections:

- This is the kernel distance/MMD extended to **finite measures**, rather than the usual probability-distribution setting of Gretton et al.
- The kernel is positive semidefinite. It is finite-rank and generally not strictly positive definite or characteristic on arbitrary footprints.

The curvature issue discussed above prevents applying this identity to the reported \((\mu,\sigma)\) augmentation.

### Taylor bound

> \[
> |k_\Theta(a,b)-\|\Theta\|^2ab|
> \le \|\Theta\|^2ab\,t^2(7/12+t^2/36).
> \]

**Status: correct for \(a,b\ge0\) and the positive frequency grid. All constants check.**

For \(x=\theta a,y=\theta b\),
\[
|(\cos x-1)(\cos y-1)|\le \frac{x^2y^2}{4}.
\]
Also, writing \(\sin x=x+r_x\),
\[
|r_x|\le |x|^3/6,
\]
gives
\[
|\sin x\sin y-xy|
\le
\frac{|xy|}{6}(x^2+y^2)+\frac{|x|^3|y|^3}{36}.
\]
Since \(|x|,|y|\le t\),
\[
\le \theta^2ab\left(\frac{t^2}{3}+\frac{t^4}{36}\right).
\]
Adding the cosine-product bound yields
\[
\frac14+\frac13=\frac7{12}.
\]

If negative frequencies are allowed elsewhere, define
\[
t=\max_{\theta\in\Theta}|\theta|\max(a,b).
\]

### “Only factor”

> “\(\|\Theta\|\) is the only factor for which the two agree to first order.”

Correct only after fixing the unweighted frequency coordinates and requiring a nonnegative layer scale. Matching coefficients gives \(c^2=\sum_\theta\theta^2\).

This is a useful normalization convention, not a proof that it is statistically optimal.

### Hub interpretation

> “The first-order regime \(t\ll1\) holds for hubs, whose walks spread thin.”

**Status: false as a general statement.**

The star-center example above has arbitrarily large degree but a step-two return probability approaching \(1/2\). With \(\max\Theta=100\), \(t\) approaches \(50\), not zero.

High degree can make step-one probabilities small; it does not ensure diffuse later distributions.

### Full per-layer feature count

> “\(L(L+3)/2\) coordinates instead of \(2m(L+1)\) per step.”

**Status: inconsistent comparison.**

At step \(w\):
- layer masses use \(w+1\) coordinates;
- a full CF per layer uses \(2m(w+1)\).

Across all steps, the full per-layer CF uses
\[
2m\sum_{w=1}^L(w+1)=mL(L+3).
\]
For \(m=10,L=3\), that is \(180\), versus \(9\) mass coordinates. State separately whether the original mark-blind \(60\) coordinates are retained.

---

## 5. Proposition: stability

### Stated inequality

**Status: correct for the characteristic-function block on a simple, loopless, undirected, unweighted graph, with insertion of a previously absent edge.**

The proof steps check:

1. For positive \(\theta\),
   \[
   |e^{i\theta x}-e^{i\theta y}|\le\theta|x-y|.
   \]
2. Therefore
   \[
   \|\phi'_u-\phi_u\|_2
   \le \|\Theta\|_2\|p'_u-p_u\|_1.
   \]
3. The telescoping identity
   \[
   H'^w-H^w
   =
   \sum_{t=0}^{w-1}H'^t(H'-H)H^{w-1-t}
   \]
   is correct.
4. Stochastic matrices have induced infinity norm one, giving
   \[
   \|H'^w-H^w\|_\infty
   \le w\|H'-H\|_\infty.
   \]
5. The changed endpoint row has total variation in \(\ell_1\)
   \[
   (d_a+1)\left(\frac1{d_a+1}-\frac1{d_a+2}\right)
   +\frac1{d_a+2}
   =\frac2{d_a+2}.
   \]

The zero-effect condition is also correct: before traversing the inserted edge, a walk must first reach an endpoint.

Required qualifications:

- The paper’s general directed/weighted setting does not match the proof’s stated degree calculation.
- Directed out- and in-walks need separate endpoint strengths and a concatenated bound.
- The result is per step and for \(\phi\), not for all concatenated steps. For the concatenated CF block, the direct bound is
  \[
  \|\Delta X_{\rm CF}\|_2
  \le
  \frac{2\|\Theta\|_2}{d+2}
  \sqrt{\sum_{w=1}^Lw^2}.
  \]

### The full embedding is not covered—and has a counterexample

> “Insertions at high-degree nodes barely move embeddings…”

**Status: wrong for the full RoleWalk embedding.**

Let \(G=K_{2,d}\), with the two degree-\(d\) vertices called \(a,b\), and insert the edge \(ab\). Use root \(a\).

Before insertion, \(b\) is at distance two and
\[
p_{ab}^2=\frac{d}{3(d+1)}.
\]
It is the only distance-two vertex, so
\[
s_a^2(2)=\frac{d}{3(d+1)}\longrightarrow\frac13.
\]

After insertion, every vertex is at distance at most one from \(a\), hence
\[
s_a'^2(2)=0.
\]

Therefore the scaled layer block moves by at least
\[
\|\Theta\|\frac{d}{3(d+1)}
\longrightarrow \frac{\|\Theta\|}{3},
\]
even though both insertion endpoints have degree \(d\to\infty\).

This directly refutes the prose’s high-degree stability interpretation for the embedding.

Curvature introduces additional issues:

- its influence extends beyond the claimed \((L-1)\)-ball;
- standard deviation is not uniformly Lipschitz in the probability vector near zero variance.

The abstract should say **“a degree-dependent stability bound for the characteristic-function block”**, not imply a corresponding theorem for all embedding coordinates.

The reported “largest observed move is 0.52 of the bound” must specify exactly which block and norm were measured.

---

## 6. Proposition: degree and triangles

**Status: correct for simple, loopless, undirected, unweighted graphs. Too broadly worded otherwise.**

The first identity,
\[
s_u^1(0)=\frac1{d_u+1},
\]
requires no original self-loops and the stated uniform self-loop augmentation.

For a \(d\)-regular graph,
\[
H^3=\frac{I+3A+3A^2+A^3}{(d+1)^3}.
\]
The diagonal identities are correct:
\[
A_{uu}=0,\qquad (A^2)_{uu}=d,\qquad (A^3)_{uu}=2t_u.
\]
Hence
\[
(H^3)_{uu}=\frac{1+3d+2t_u}{(d+1)^3}.
\]

For the example:
- two triangles: \(d=2,t_u=1\), return probability \(1/3\);
- hexagon: \(d=2,t_u=0\), return probability \(7/27\).

Thus the example works.

Qualifications:

- For graph-level separation, define the graph representation as the multiset of node embeddings or specify an aggregation retaining this distinction.
- The standard 1-WL statement requires uniform initial labels and, for the usual same-color-histogram comparison, equal graph size.
- The formula does not count ordinary triangles on arbitrary irregular graphs.

Indeed, setting \(h_v=1/(d_v+1)\), on an irregular graph:
\[
(H^3)_{uu}
=
h_u^3
+2h_u^2\sum_{v\sim u}h_v
+h_u\sum_{v\sim u}h_v^2
+2h_u\sum_{\{u,v,z\}\text{ triangle}}h_vh_z.
\]
The triangle term is degree-weighted.

> “It counts triangles, which places it beyond 1-WL.”

Replace this with the precise claim: **it recovers node triangle counts on regular graphs and separates some pairs that 1-WL cannot.** That does not establish dominance over 1-WL.

---

## 7. Proposition: relation to RWSE

**Status: correct with \(d>0\) and the same simple undirected graph conventions.**

The distance-zero identity
\[
s_u^w(0)=(H^w)_{uu}
\]
is exact.

The expansion
\[
(H^w)_{uu}
=
(d+1)^{-w}
\sum_{k=0}^w\binom wk d^k(P^k)_{uu}
\]
is correct. The diagonal coefficient is nonzero for \(d>0\).

The inverse is also explicit:
\[
(P^k)_{uu}
=
d^{-k}
\sum_{r=0}^k
\binom kr(d+1)^r(-1)^{k-r}(H^r)_{uu}.
\]

For \(d=0\), \(P=A/d\) is undefined and needs a separate convention.

> “On irregular graphs … neither encoding contains the other.”

**Status: not established by the proof and too broad as written.**

Showing that the walk matrices differ does not prove incomparability of the resulting feature maps. You need examples in both directions, with the same step budget and a precise definition of “contains.”

As a blanket statement over all budgets it is already problematic at \(L=1\): simple-walk return probabilities on loopless graphs are identically zero and are trivially determined by RoleWalk.

Also distinguish:
- lazy-return coordinates versus simple-return coordinates;
- full RoleWalk versus RWSE;
- equal depth versus 3-step RoleWalk and 16-step RWSE.

---

## 8. Proposition: distance-regular graphs

**Status: correct under the standard definition of finite, connected, simple, undirected distance-regular graphs.**

The Bose–Mesner argument is sound.

The recurrence is correctly indexed:
\[
AA_i=b_{i-1}A_{i-1}+a_iA_i+c_{i+1}A_{i+1}.
\]
For example, \(i=1\) gives
\[
A^2=kI+a_1A+c_2A_2,
\]
as required.

It follows that
\[
(I+A)^w=\sum_i\alpha_{w,i}A_i,
\]
with coefficients fixed by the intersection array. The sphere sizes are also fixed:
\[
k_0=1,\qquad k_1=k,\qquad
k_{i+1}=\frac{k_i b_i}{c_{i+1}}.
\]
Therefore both the probability multiset and all layer masses agree.

The curvature calculation is correct:
\[
|N[x]\cap N[y]|=a_1+2,
\qquad
\kappa(x,y)=\frac{a_1+2}{k+1}.
\]
Thus the node curvature is constant, its walk expectation is that constant, and its standard deviation is zero.

Boundary conventions for \(A_i,b_i,c_i\) should be stated or left implicit under a standard reference.

### Overclaim

> “Distance-regular graphs … are its exact limit.”

**Status: not established and misleading.**

The theorem gives a **sufficient indistinguishability class**, not an if-and-only-if characterization. The paper itself gives another failure mechanism via equal local balls.

Use something like:

> “A provable failure class: distance-regular graphs with equal intersection arrays.”

### Examples and BREC

The named pairs are appropriate examples of the theorem, subject to correct construction in code.

The theorem explains a BREC pair only after confirming that its members have the same intersection array. The source claims such checks for some families, but those checks are not supplied.

### Wrong WL statement

> “Shrikhande / \(4\times4\) rook, beyond 3-WL.”

**Status: wrong under the usual WL convention.**

They are a standard failure case for 2-WL. Standard 3-WL distinguishes them.

One direct reason: the rook graph has 4-cliques, whereas the Shrikhande graph has clique number three. For a triangle triple, the 3-WL update can detect whether there is a vertex adjacent to all three vertices. Rook triangles have such an extension; Shrikhande triangles do not.

Correct the footnote and be explicit about any nonstandard WL convention.

---

## 9. Proposition: locality limits graph-level separation

**Status: the claim is correct; the written proof needs a rigorous 1-WL argument and a slightly more careful ball description.**

The two graphs are nonisomorphic because the cycle-arc lengths between the attachment vertices are:
\[
\{15,25\}\quad\text{versus}\quad\{20,20\}.
\]

With curvature and \(L=3\), the locality radius is five. The two pendant neighborhoods do not interact within that radius, and both graphs have the same multiset of rooted radius-five balls.

The sentence

> “every rooted ball … is fixed by \(u\)’s distance to [the pendant]”

is too compressed: distinguish a root on the cycle from a pendant root, and handle balls that contain no pendant. The multiset conclusion is nevertheless correct.

> “1-WL … eventually sees the distance … (checked with 40 rounds).”

A computational check is not a proof. Here is a short replacement.

Let \(h\) mark degree-three vertices. These marks are available after one WL round. Further WL rounds determine the quantities \(A^r h\), since these are recursively sums over neighbors.

In the first graph, from either degree-three vertex there is one length-15 walk to the other along the shorter cycle arc; there are no odd closed walks because the graph is bipartite. Thus \((A^{15}h)\) equals one at those marked vertices.

In the second graph, both degree-three vertices lie in the same bipartition class, so no odd-length walk connects them, and \((A^{15}h)\) equals zero there.

Hence 1-WL separates the graphs after sufficiently many rounds.

---

## 10. The conjecture is false

> “For every \(L\), the partition … induced by RoleWalk … refines … \(L\)-round 1-WL.”

**Status: false for curvature-free RoleWalk, even with an entire interval of frequencies.**

### Explicit 15-node counterexample at \(L=2\)

Start with a clique
\[
C=\{r,a,b,c,d\}\cong K_5,
\]
rooted at \(r\), and add ten outside vertices.

Construct two graphs:

- **Graph \(G\):** every outside vertex is adjacent to \(b,c,d\), and to no other vertex.
- **Graph \(G'\):** five outside vertices are adjacent to \(a,b\), and the other five to \(c,d\).

There are no edges between outside vertices.

The root has degree four in both graphs.

Its neighbor-degree multisets are:
\[
G:\quad \{4,14,14,14\},
\]
\[
G':\quad \{9,9,9,9\}.
\]
Therefore 2-round 1-WL distinguishes the roots: round one determines degrees, and round two sees these different neighbor-degree multisets.

But RoleWalk agrees exactly.

#### Step one

In both graphs, the five clique vertices each have probability \(1/5\). Hence
\[
\phi_r^1(\theta)=5f_\theta(1/5),
\qquad
s_r^1=(1/5,4/5).
\]

#### Step two

In \(G\), the reciprocal augmented degrees of the four root neighbors sum to
\[
\frac15+3\frac1{15}=\frac25.
\]
In \(G'\), they sum to
\[
4\frac1{10}=\frac25.
\]

Because the five core vertices form a clique, every core vertex has step-two probability
\[
\frac15\left(\frac15+\frac25\right)=\frac3{25}.
\]

Every outside vertex has probability:
\[
G:\quad \frac15\cdot 3\cdot\frac1{15}=\frac1{25},
\]
\[
G':\quad \frac15\cdot 2\cdot\frac1{10}=\frac1{25}.
\]

Thus, in both graphs,
\[
\phi_r^2(\theta)
=
5f_\theta(3/25)+10f_\theta(1/25)
\]
for **every** real frequency, and
\[
s_r^2=(3/25,12/25,10/25).
\]

All RoleWalk coordinates through \(L=2\) agree, while 2-round 1-WL separates the roots.

### A counterexample at the default \(L=3\)

Attach a new leaf \(z\) to \(r\) in each graph and root at \(z\).

The two new roots have identical RoleWalk features through three steps:

- step one: \(z,r\) each have probability \(1/2\);
- step two: \(z,r\) each have probability \(1/3\), and the other four core vertices each have probability \(1/12\);
- step three:
  \[
  p_z^3(z)=\frac29,\qquad
  p_z^3(r)=\frac{23}{90},
  \]
  each of the other four core vertices has probability \(4/45\), and each outside vertex has probability \(1/60\).

Corresponding vertices are in the same layers, so the layer masses also agree.

But 3-round 1-WL separates \(z\): the difference at \(r\) after two rounds propagates to \(z\).

These counterexamples concern the curvature-free method. They do not establish the same failure for the optional curvature augmentation.

### Consequences for the paper

The empirical result on graphs up to seven nodes may still be correct. It does not support the conjecture beyond that domain.

The appropriate theoretical relationship is:

- RoleWalk separates some 1-WL-indistinguishable graphs;
- 1-WL separates some nodes that finite-depth RoleWalk does not.

Thus the two are **incomparable**, not a general refinement hierarchy.

> “At equal depth RoleWalk is strictly finer … on all graphs up to 7 nodes…”

The experiment described concerns **\(L=3\)** and the **996 connected graphs** up to seven nodes. State exactly that domain. It does not establish all depths or all disconnected graphs.

The counts themselves are correct:
\[
1+1+2+6+21+112+853=996,
\]
\[
1+2+6+24+105+672+5971=6781.
\]

Numerical grouping also needs specification. “Within a tolerance” is not automatically a transitive equivalence relation, and near-equality of sampled features is not exact equality.

---

## 11. Proposition: relation to GraphWave

### Algebraic identity

**Status: correct.**
\[
n(\chi_u(t)-1)
=
\sum_j(e^{it\Psi_{ju}}-1).
\]

This is a useful and important connection. It also makes clear that the characteristic-function mechanism itself is closely related to GraphWave, rather than a new principle.

### Locality statement

**Status: correct with \(s>0\) and a connected graph, for the usual heat kernel.**

The heat kernel has positive support throughout the connected component, so exact fixed-radius locality generally fails.

The graph-adaptive scale and \(1/n\) normalization can make coordinates depend on global graph properties.

### Overstatements

> “The same rooted neighbourhood gets different coordinates…”

Replace “gets” with “can get.” Nonlocal dependence does not imply a difference in every particular case or at every finite sampled grid.

> “This is why GraphWave transfers at chance level…”

**Status: unsupported causal attribution.**

The theorem identifies possible sources of mismatch. It does not establish which caused the observed failure.

An essential baseline is:
- centered, unnormalized GraphWave;
- fixed scales shared across graphs;
- matched feature budget and preprocessing.

Also, “nonlocal” does not mean “non-inductive.” A fixed heat-kernel feature definition can be evaluated on unseen graphs. The table’s categorical “Transfer: no” is too strong.

---

## 12. Proposition: cost

### Support bound

**Status: correct if pruned rows remain substochastic.**

If retained entries are at least \(\tau>0\) and the row sum is at most one, then there are at most \(\lfloor1/\tau\rfloor\) retained entries.

### Sparse multiplication cost

**Status: a valid operation-count upper bound, with implementation qualifications.**
\[
O\!\left(
\sum_u\sum_{j\in\operatorname{supp}p_u^{w-1}}(d_j+1)
\right)
\]
counts generated contributions.

It does not bound:
- transient accumulator size;
- sorting/deduplication overhead;
- peak memory before thresholding;
- worst-case degree dependence.

The support cap after pruning is not a cap on intermediate work.

### CF and layer costs

The \(O(m\,\mathrm{nnz})\) CF cost is correct.

“Layers \(L\) element-wise products” is underspecified: per step \(w\) there are \(w+1\) layers. With stored distance labels, mass accumulation can instead be done in one pass over nonzeros.

Again, support-based distance recovery is valid only without pruning unless extra reachability information is retained.

### Curvature cost

**Status: reasonable for the specified symmetrized unweighted curvature and suitable adjacency data structures.**

Computing closed-neighborhood intersections along edges in
\[
O(d_x+d_y)
\]
per edge gives
\[
O\!\left(\sum_u d_u^2\right),
\]
apart from input/preprocessing costs.

For directed graphs these are degrees in the symmetrized graph.

### Local computation

Correct without curvature; with curvature, use the repaired weak-neighborhood statement.

“Online” needs a definition. The ability to embed a queried node locally is not the same as an efficient dynamic algorithm maintaining all embeddings after arbitrary updates.

---

## 13. Additional mathematical claims in the analysis section

### 13.1 Degree marks and graph size

> “Anything that depends on graph size (degree marks, counting bounds) hurts transfer.”

**Status: mathematically and empirically overgeneralized.**

Degree is a local invariant, not explicit dependence on \(|V|\). RoleWalk already exactly encodes degree through \(s_u^1(0)\).

The experiments may show that particular unnormalized degree marks worsen transfer. They do not establish that size-dependent features universally hurt transfer.

### 13.2 Path parity

> “\(\#\mathrm{intra}\equiv\#\mathrm{moves}-d(u,j)\pmod2\).”

**Status: correct for non-self-loop moves in an undirected graph’s BFS layers; not generally for directed layers.**

For an undirected edge, layer changes are \(-1,0,+1\). Therefore
\[
\#\mathrm{out}-\#\mathrm{in}=d(u,j),
\]
and the parity identity follows.

Directed edges can decrease distance by more than one, so the argument does not carry over unchanged.

> “Obtained for free from … \(\widetilde D^{-1}(A-I)\).”

For loopless \(A\), this signed walk weights a path by
\[
(-1)^{\#\mathrm{stays}}=(-1)^{w-\#\mathrm{moves}}.
\]
To obtain intra-layer parity, an additional endpoint factor
\[
(-1)^{w-d(u,j)}
\]
is needed.

Computing an additional signed power stream is not literally free; quantify the extra arithmetic and memory.

### 13.3 Commuting path labels

> “With commuting labels the product along a path only depends on its end layer…”

**Status: false in general.**

Commutativity makes the product depend on label counts, not only the endpoint. Endpoint dependence requires special relations, such as inward and outward labels being inverses and all other labels being trivial.

Paths ending at the same layer can have different numbers of backtracks, stays, and intra-layer moves.

The experimental negative result may be valid, but the proposed general explanation is not.

### 13.4 Frequency scaling heuristic

> “\(p\approx1/d^w\): a change of degree rescales \(p\)….”

**Status: a limited heuristic, not a general property.**

Even regular graphs have backtracking and return mass; irregular walks involve products of different reciprocal degrees. The star is again a counterexample to uniform thinning.

Log spacing can be motivated as sampling multiple probability scales, but there is no general degree-rescaling equivariance here.

### 13.5 Standardization and layer scaling

The experiments say classification uses standardized embeddings.

For a positive constant \(c\), featurewise standardization gives
\[
\frac{cx-c\mu}{c\sigma}=\frac{x-\mu}{\sigma}.
\]
Thus multiplying all layer coordinates by \(\|\Theta\|\) has no effect on a pipeline that independently standardizes every coordinate, apart from numerical details.

The scaling result can matter for raw-distance \(k\)-means, Ward clustering, or kernels. Make clear which claims concern those settings and which concern standardized classification.

---

## 14. Numerical and experimental consistency

### 14.1 Correct visible arithmetic

- Feature counts \(69/75\): correct for one direction.
- Connected-graph counts \(996/6781\): correct.
- BREC totals:
  - \(60+50+97=207\);
  - \(60+50+100+3=213\);
  - \(58+48+97=203\);
  - \(60+50+100+4=214\).
- Centrality “Mean” row: all entries agree with the arithmetic mean of the seven displayed centrality scores, to rounding.
- Base RoleWalk airport mean:
  \[
  (0.743+0.583+0.657)/3=0.661.
  \]

### 14.2 Small but real discrepancies

The displayed curvature airport scores average to
\[
(0.741+0.596+0.666)/3=0.66767,
\]
i.e. \(0.668\), not the \(0.669\) in the geometry table.

The directed accuracy for the layer-equipped base method is \(0.680\) in the main table and \(0.682\) in the geometry table.

These could reflect different runs or configurations. State that explicitly rather than presenting them as interchangeable.

The CF summary table’s \(0.656/0.445\) appears to correspond to **no layers**, whereas “RoleWalk” elsewhere includes layers. Label the ablation configuration.

### 14.3 “On par with 16-step RWSE”

This is reasonable only as a qualified empirical summary:

- 3-step RoleWalk: 207 BREC pairs;
- 5-step RoleWalk: 213;
- 16-step RWSE: 214.

On CSL, 3-step RoleWalk yields six classes versus ten for RWSE.

The methods also have different dimensions and computational implementations. Report the depth–dimension–time tradeoff rather than one headline comparison.

### 14.4 “CFI pairs” limitation contradicts the table

> “Walk statistics cannot separate … CFI pairs…”

The BREC table reports **three CFI pairs separated by 5-step RoleWalk**. The limitation should say “most CFI pairs in this benchmark,” unless a narrower theoretical class is intended and defined.

### 14.5 “Only method that uses directions”

This contradicts the paper’s own table and related-work discussion:

- Role2Vec is marked directed;
- RWSE is explicitly evaluated on an out-walk;
- directed struc2vec extensions are acknowledged;
- directed/weighted ReFeX-style features are natural and should be checked against the original method.

A reference implementation that discards direction does not establish that the competing method cannot use direction.

### 14.6 Speed claims

At the displayed USA time of \(0.2\) seconds:

- struc2vec: \(564/0.2=2820\times\);
- GraphWave: \(90\times\);
- Role2Vec: \(50\times\);
- RESD: \(40\times\);
- RWSE: \(8.5\times\);
- ReFeX is **twice as fast as RoleWalk**.

Thus “10–1,000× lower cost” is at best an approximate statement about selected expensive baselines, not the whole comparison set.

The source does not include the actual scalability CSV values, so I cannot audit that table’s timings.

### 14.7 Wikipedia talk comparison

The struc2vec average is over the three smallest networks; other methods appear averaged over more networks. Those means should not share a best/second-best ranking without a matched-subset table.

### 14.8 Degree baselines and headline rankings

Separating degree visually is fine. Excluding it from rankings does not justify claiming the best predictive performance overall.

Examples:
- degree beats RoleWalk on Brazil;
- degree beats every reported embedding on airport transfer;
- in/out degree is already at \(0.922\) on the three-class RfA task.

These results matter to the substantive claim about structural roles.

### 14.9 Top probabilities and frequency range

The top-10-probabilities ablation achieves airport transfer \(0.512\), exceeding the default full method’s \(0.488\), and is faster in the displayed timing.

The \([1,30]\) frequency range also reaches \(0.515\) transfer.

Therefore the experiments do not show that the default characteristic-function choice is uniformly the strongest or most transferable summary.

### 14.10 ZINC

The paper appropriately acknowledges limited seeds. The reported \(0.077\to0.073\) GPS change is not yet convincing evidence of a robust improvement with two seeds, particularly given the displayed baseline variation.

Also:
- the published RWSE result is better than all reproduced GPS results;
- “RWSE only” versus “RoleWalk + curvature + RWSE” changes feature dimension and potentially encoder capacity;
- a matched-parameter comparison and more seeds are needed.

### 14.11 Graph-level expressiveness protocol

The paper never precisely defines how node embeddings are converted into a graph-level distinction. A multiset comparison, sorted coordinates, pooled moments, and a learned readout have different guarantees.

The expressiveness table also mixes **numbers of pairs separated** with **numbers of CSL classes distinguished**. Its caption should reflect that.

> “Degree (baseline, = 1-WL)”

Degree is not equivalent to 1-WL in general. At most, the displayed degree baseline has the same failure outcome as 1-WL on these selected graph families.

---

## 15. Film analysis

> “Classes 0–2 are not predictable by any method.”

**Status: stronger than the evidence.**

The reported best score is \(0.637\) versus majority \(0.623\). Without uncertainty, this is neither evidence of meaningful predictability nor proof of impossibility.

> “Edge homophily is 0.28, below the 0.33 of random labels.”

The random-label reference is generally
\[
\sum_c\pi_c^2
\]
(up to finite-sample correction), not \(1/3\), unless the three classes are balanced. The displayed majority fraction \(0.623\) shows that they are not balanced in the within-classes task.

Clarify which nodes and class proportions were used for the homophily calculation.

> “Role2Vec’s lead comes from … position, which an automorphism-invariant embedding cannot capture by [locality].”

**Status: not established.**

An automorphism-invariant embedding cannot distinguish vertices in the same automorphism orbit. It can distinguish many positional properties, and an independent-set node type is not automatically invisible to structural features.

To support this explanation, show:
- actual automorphism-orbit or rooted-neighborhood collisions across labels;
- a controlled positional-versus-structural ablation;
- invariance under random node-ID permutations.

A separate block of node IDs is a reason to check ordering or implementation artifacts, not a proof that the task is inappropriate.

> “Film should not rank role methods.”

That is a benchmark-definition argument, not a consequence of the locality proposition.

---

## 16. Citations and scope of known results

Without `refs.bib`, exact bibliographic accuracy cannot be checked. Conceptually:

- **Gretton et al.:** supports the standard MMD framework, but the paper should explicitly state its extension to finite unnormalized measures.
- **Ollivier/Jost:** the proposed overlap is an upper bound, not an alternative expression for Ollivier curvature.
- **Brouwer et al.:** the distance-regular recurrence and intersection-array argument are appropriate.
- **RWSE citation:** identify that standard simple-walk RWSE and the paper’s self-loop-augmented return features differ on irregular graphs.
- **WL citations:** support standard color-refinement facts, but do not establish RoleWalk’s conjectured refinement.
- **GraphWave:** check whether spectrum-adaptive scale selection is being described as the original default, a required part of the method, or the particular implementation used.
- **Shrikhande/rook:** the “beyond 3-WL” claim needs correction.
- The unfinished RESD/RONE references and unverified property-table entries need resolution before submission.

---

# Part 2 — Reviewer report

## Summary

The paper proposes a deterministic structural node representation built from three components:

1. centered characteristic-function samples of self-loop-augmented random-walk probability multisets;
2. walk mass aggregated by shortest-path distance;
3. optional local overlap-curvature moments.

The intended advantages are local computation, fixed coordinates across graphs, support for directed/weighted graphs, and inexpensive feature extraction. Experiments cover structural-role classification, synthetic directed roles, centrality transfer, graph distinguishability, and ZINC structural encodings.

The curvature-free construction has a valid finite-measure kernel interpretation, a correct local-invariance result, and a useful distance-regular indistinguishability theorem. However, the theoretical positioning is substantially overstated, several claims are false as written, and the experimental comparison is not yet strong enough for a top-tier acceptance.

## Strengths

1. **The construction is explicit and implementable.** Fixed-dimensional, training-free local features are useful when deployment involves many graphs or queried neighborhoods.

2. **The distance-layer block is a meaningful addition to probability multisets.** It retains endpoint distance information discarded by the characteristic-function block, and it includes return-probability features directly.

3. **Several mathematical results are clean.** The injectivity argument, curvature-free kernel identity, Taylor bound, regular-graph triangle identity, and distance-regular failure theorem are sound after scope corrections.

4. **The empirical scope is reasonably broad.** The paper includes transfer, directed tasks, failure cases, and simple degree baselines rather than only favorable classification benchmarks.

These strengths support continued development, but do not resolve the issues below.

## Major weaknesses

### 1. The expressiveness narrative is incorrect

The conjectured refinement of \(L\)-round 1-WL is false, including at the default \(L=3\). The counterexamples above show that the correct relationship is incomparability.

The paper also incorrectly labels Shrikhande/rook as beyond 3-WL and describes a sufficient failure class as an “exact limit.”

These are not cosmetic errors: expressiveness is one of the main advertised contributions.

### 2. The stability claim does not describe the method being evaluated

The proof controls only the CF block. The layer block can move by a constant after an edge insertion between arbitrarily high-degree endpoints, as shown by \(K_{2,d}\).

Thus the abstract’s stability positioning and the statement that high-degree insertions “barely move embeddings” are unsupported for the actual representation.

### 3. The unified-kernel story excludes the reported curvature representation

Raw first and second moments have the claimed linear feature-map form. Mean and standard deviation do not.

The paper currently uses “fixed reparametrisation” to move between information equivalence and metric equivalence. Those are different claims.

### 4. Novelty relative to GraphWave is insufficiently isolated

The paper itself shows the exact centered-CF relationship. The main differences are:
- local finite-step walks instead of heat diffusion;
- removing graph-size normalization;
- distance-layer mass features;
- implementation choices and pruning.

That may constitute a useful method, but the decisive baselines are missing: centered, unnormalized, fixed-scale GraphWave and matched implementations of other walk summaries.

### 5. Directed comparisons are not yet fair enough

The strongest directed results largely compare a direction-aware method against implementations that discard direction.

“Only method that uses directions” is false. A compelling evaluation needs direction-aware ReFeX, directed RWSE in both directions, directed structural-feature baselines, and available directed role-embedding extensions.

### 6. Efficiency may be confounded by implementation and approximation

Three-step pruned RoleWalk is compared against longer or apparently unpruned baselines. GraphWave failing at 11,000 nodes and RWSE taking 51 seconds versus RoleWalk’s four seconds warrant explanation.

The pruning rule itself has severe worst cases and lacks an error study.

### 7. Statistical evidence is incomplete

Most table differences lack uncertainty. Fifty node splits do not replace independent graph seeds or embedding seeds.

The ZINC result has only two GPS seeds, and the reproduced baseline is materially weaker than the cited published result.

### 8. Some of the strongest simple baselines undermine the headline

Degree wins airport transfer. Top-10 probabilities outperform the default on that transfer task and are faster. ReFeX is faster on the USA timing and much stronger for k-core transfer.

These are useful findings, but the paper should analyze them rather than set them outside the main ranking.

## Questions to the authors

1. Which experiments use \(69/75\) features and which use \(138/150\)? Are all directed methods given both incoming and outgoing information?
2. Is curvature computed from the symmetrized unweighted graph or from the original transition distributions?
3. What value is assigned to curvature at isolated vertices?
4. What happens when pruning removes an entire row? How are discarded mass, layer membership, and curvature statistics treated at later steps?
5. Were all expressiveness experiments run with \(\mathrm{tol}=0\)?
6. How exactly are graph embeddings compared, and how is numerical equality defined?
7. What are the baseline walk lengths, feature dimensions, pruning rules, and sparse backends?
8. Were frequency range, layer scale, and other defaults selected using labels from the reported evaluation datasets?
9. Are standardizers and feature-selection rules fitted only on training data and then frozen for transfer?
10. How are signed and repeated wiki-RfA votes converted into the nonnegative adjacency required by the method?
11. Why is the Wikipedia-talk struc2vec mean ranked against means over a different network subset?
12. Does the Film result survive random permutations of node IDs?

## Most important missing baselines and experiments

### Baselines

- Centered, unnormalized GraphWave with scales fixed across graphs.
- RWSE with matched walk definition, depth, pruning, and implementation.
- Incoming and outgoing RWSE concatenated.
- Directed ReFeX with a training-frozen feature schema.
- A simple structural feature set: in/out degree, neighbor-degree moments, egonet density, triangle/clustering statistics, and short return probabilities.
- Top-\(k\) probabilities with the same layer and curvature blocks.
- Available directed/weighted structural embedding extensions.
- For ZINC, a matched-size encoder and basic local-count features alongside RWSE.

### Experiments

- Depth–accuracy–runtime curves, rather than only 3-step versus 16-step comparisons.
- A pruning sweep including exact computation on tractable graphs, with star/hub stress tests.
- Frequency-range and frequency-count sensitivity under a clean validation protocol.
- Separate standardized-classification and raw-distance-clustering analyses.
- Matched-network Wikipedia-talk results.
- Transfer across graph sizes and families, not only new seeds within the same family and size.
- Directed weighted experiments that actually test weight usage.
- At least several additional GPS seeds and matched parameter budgets.
- Confidence intervals for paired method differences.
- Timing breakdowns, peak working memory, and explicit baseline implementation details.

## Overall score and confidence

**Score: 4/10 — weak reject.**

**Confidence: 4/5.** High confidence in the mathematical issues identified above; lower confidence in empirical reproducibility because the referenced code, CSVs, and bibliography were not provided.

The paper has a potentially useful lightweight representation, but the present draft combines an incremental methodological contribution with overstated theory and incomplete comparative evidence.

## Changes most likely to raise the score

1. **Correct the theoretical positioning immediately.** Remove the false conjecture, state incomparability with 1-WL, fix the 3-WL footnote, and rename the distance-regular result as a sufficient failure class.
2. **Separate guarantees by feature block.** State exactly which locality, stability, and kernel results apply to CF, layers, raw curvature moments, and reported standard deviations.
3. **Specify and validate pruning.** The current description leaves important algorithmic behavior undefined.
4. **Run the decisive matched baselines.** In particular, centered/fixed-scale GraphWave and matched directed walk/feature methods.
5. **Strengthen the empirical protocol.** Add uncertainty, matched subsets, clean model selection, and more ZINC seeds.
6. **Narrow the claims to demonstrated advantages.** A defensible paper about fast, transferable local walk-distribution features would be stronger than the current claim of general expressiveness superiority and unified stability.
7. **Explain when the CF is worth using.** The top-probability and degree results make this a central scientific question, not a peripheral ablation.