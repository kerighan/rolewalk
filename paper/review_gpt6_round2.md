## Bottom line

**This is a substantial revision, and I would raise my score from 4 to 5/10. I would not yet recommend acceptance.** The certified-pruning argument is sound for a precisely scoped version of the algorithm, and the mixed-direction witness and incomparability examples are valid. The additional baselines materially improve the paper.

However, there are still important problems:

1. **The implemented triangle-enumeration kernel does not have the claimed \(O(|E|^{3/2})\) complexity.** It takes quadratic time on a star.
2. **The support cap in the cost proposition is false for the implemented algorithm**, both because step one is unpruned and because a mass budget can retain arbitrarily many sub-threshold entries.
3. **Mixed-direction probes invalidate the current pure-out-/pure-in-ball locality statement.**
4. **The pruning certificate needs explicit direction/probe scope and an exact-arithmetic qualification.** “Layers are exact” must distinguish distances from layer masses.
5. Several earlier problems remain in the text: the RWSE inference, Film’s invariance argument, the GraphWave property table, unspecified graph-level comparisons, and some numerical/ranking discrepancies.

I checked the supplied mathematics and code by inspection. I have not executed the library or inspected the external CSVs, benchmark scripts, bibliography, or rendered scalability table.

# 1. Mathematical and implementation audit

## 1.1 Certified pruning: the main argument is correct, with necessary scope corrections

> “Without curvature and for \(L\le3\) …  
> \[
> \|X_u-\widehat X_u\|_2
> \le \sqrt5\|\Theta\|_2
> \left(\sum_{w\le L}(r_u^w)^2\right)^{1/2}.
> \]”

### What is correct

For **one walk direction, sum aggregation, no mixed probes, and exact arithmetic**, the proof is valid.

Write the retained row as
\[
q^1=e_u^\top H,\qquad
q^w=\mathcal P_w(q^{w-1}H),
\]
where \(\mathcal P_w\) deletes entries without renormalizing. Since \(H\) is nonnegative and stochastic,
\[
0\le q^w\le p^w,\qquad
r^w=1-\|q^w\|_1=\|p^w-q^w\|_1.
\]

The implemented corrected CF is
\[
\widehat\phi^w(\theta)
=\sum_j f_\theta(q_j^w)+i\theta r^w.
\]
Therefore
\[
|\phi^w(\theta)-\widehat\phi^w(\theta)|
\le
\theta\|p^w-q^w\|_1+\theta r^w
=2\theta r^w.
\]
Consequently,
\[
\|\phi^w-\widehat\phi^w\|_2
\le2\|\Theta\|_2r^w.
\]

If retained entries have correct distance labels, the layer error is a nonnegative vector whose coordinates sum to \(r^w\). Thus
\[
\|s^w-\widehat s^w\|_2\le r^w.
\]
Combining the orthogonal CF and scaled-layer blocks gives
\[
\|\Delta X^w\|_2^2
\le (4+1)\|\Theta\|_2^2(r^w)^2,
\]
which establishes the stated constant.

The accumulated missing mass is the right quantity. This proof does account for the possibility that previously discarded mass would later concentrate.

### Why the \(L\le3\) distance argument works

> “Supports read before pruning are exact up to step 2, and every node reached at step 3 outside them is at distance 3.”

Correct, because the first step is not pruned:

- Step-one support is exact.
- The pre-pruning step-two row is exact.
- Any positive entry propagated at step three corresponds to an actual walk of length three.
- If its endpoint is outside the exact radius-two support, its distance is exactly three.

This establishes correct distances **for the entries being summarized**. It does not require recovering every endpoint that the exact step-three walk reaches.

### Required corrections

**A. “Layers are exact” is misleading.**

> “Distances are read from supports before pruning, so layers are exact for \(L\le3\).”

The **distance labels** are exact; the **layer masses are approximate**. Indeed, the pruning experiment reports substantial layer-mass errors.

Replace this with:

> “For \(L\le3\), every retained entry is assigned its exact graph distance. Layer masses can still lose mass through pruning.”

The same correction is needed in the library docstrings.

**B. The proposition needs an orientation index for directed embeddings.**

The paper defines directed \(X_u\) as the concatenation of two embeddings. For that representation, the direct bound is
\[
\|X_u-\widehat X_u\|_2
\le
\sqrt5\|\Theta\|_2
\left(
\sum_{\alpha\in\{+,-\}}\sum_{w\le L}
(r_{u,\alpha}^w)^2
\right)^{1/2}.
\]
A uniform per-direction budget gives
\[
\|X_u-\widehat X_u\|_2
\le\sqrt{10L}\|\Theta\|_2\delta.
\]

State the current proposition for a single branch, then give this corollary.

**C. Mixed probes are not covered by the displayed bound.**

“Without curvature” does not exclude them. This matters especially at \(L=1\): pure walks are unpruned, so their \(r^1\) values are zero, while the optional two-step mixed probes can be pruned and change the embedding.

If \(r_{u,\beta}^{\rm mix}\) denotes missing mass in mixed probe \(\beta\), a valid extension is
\[
\|\Delta X_u\|_2^2
\le
\|\Theta\|_2^2
\left[
5\sum_{\alpha,w}(r_{u,\alpha}^w)^2
+
4\sum_{\beta}(r_{u,\beta}^{\rm mix})^2
\right].
\]

Alternatively, explicitly exclude mixed probes from the proposition.

**D. The implementation uses an all-or-nothing threshold decision.**

Both `_prune` and the fused kernel delete **all** sub-threshold entries in a row if doing so respects the budget; otherwise they retain **all** of them. They do not select the largest permissible subset of small entries.

That policy satisfies the certificate. It should nevertheless be described accurately, because it explains abrupt runtime changes and why some hub rows become exactly unpruned.

**E. This is an exact-arithmetic certificate, not yet a rigorous floating-point certificate.**

The code uses float32 output, trigonometric approximations, and Numba `fastmath=True`. In floating point, exact stochasticity and the entrywise domination argument are not automatically certified.

A reasonable statement is:

> “The pruning error bound holds in exact arithmetic; numerical evaluation introduces an additional floating-point error.”

Calling the actual returned float32 vector rigorously certified would require bounding that additional error or using a suitable conservative implementation.

**F. It is an absolute feature-space bound, not a predictive guarantee.**

For the default grid, \(\|\Theta\|_2\approx125\). With \(L=3,\delta=0.05\), the displayed single-direction bound is about \(24.2\); using \(r^1=0\) tightens it to about \(19.8\).

That is a legitimate guarantee, but it does not imply small relative error, unchanged clusters, or small classification error. Featurewise standardization also changes the relevant metric.

Finally:

> “Without a mass budget, pruning error is not bounded on hubs.”

Literally, this is false: the a posteriori bound still applies, and \(r^w\le1\). What is absent is a **small prescribed a priori bound**.

---

## 1.2 Curvature: closed form correct; implemented complexity wrong

> “\(\kappa(x,y)=(c_{xy}+2)/(\max(d_x,d_y)+1)\).”

Correct for the **simple, loopless, unweighted symmetrization**, with \(m_x\) defined from that graph.

Indeed,
\[
m_x=\frac{\mathbf1_{N[x]}}{d_x+1},
\]
and, for an edge \(xy\),
\[
N[x]\cap N[y]=\{x,y\}\cup(N(x)\cap N(y)).
\]
Each shared vertex contributes
\[
\min\left\{\frac1{d_x+1},\frac1{d_y+1}\right\}
=\frac1{\max(d_x,d_y)+1}.
\]

The Ollivier upper-bound argument is also correct for unweighted shortest-path distance. The library correctly assigns curvature zero to isolated vertices; the paper should state that convention.

### The triangle counts appear correct

The rank orientation makes each triangle appear once, with its vertices ordered by rank. The kernel increments its three forward edges, and the subsequent \(C+C^\top\) symmetrizes their counts.

### But the claimed running time does not describe this implementation

> “Orienting edges by degree enumerates each triangle once in \(O(|E|^{3/2})\).”

The crucial loop is:

```python
for f in range(indptr[w], indptr[w + 1]):
    z = indices[f]
    if rank[z] > rank[w] and mark[z] >= 0:
```

It scans **all neighbors of \(w\)**, then tests whether each is forward.

Consider a star with \(d\) leaves:

- Every leaf ranks below the center.
- Every leaf-to-center edge is processed as a forward edge.
- For each such edge, the kernel scans all \(d\) center neighbors.
- There are no triangles.

The work is therefore
\[
\Theta(d^2)=\Theta(|E|^2),
\]
not \(O(|E|^{3/2})\).

**Fix:** construct forward adjacency lists and iterate only over forward neighbors of \(w\). The usual degree-oriented forward-degree bound then supplies the claimed complexity. Account separately for input processing and the current rank sorting.

The fallback
```python
(rows @ U).multiply(rows)
```
also does not have the claimed triangle-enumeration bound. On a star, \(U^2\) can create a dense leaf-by-leaf intermediate before multiplication by \(U\) removes it. This affects the no-Numba path and potentially subset queries.

**This is the most important implementation/theorem mismatch in the revision.**

---

## 1.3 The support-cap cost proposition is now false

> “With pruning, each row of each power keeps at most \(1/\mathrm{tol}\) entries.”

This fails in two separate ways.

1. **The first step is never pruned.** A high-degree node can retain more than \(1/\mathrm{tol}\) entries immediately.
2. **With `max_missing`, sub-threshold entries can be retained.** A budget rejection can preserve an arbitrarily large row.

The counting argument establishes only:

> If every retained positive entry is at least \(\tau\), and the row is substochastic, its support has size at most \(\lfloor1/\tau\rfloor\).

That applies to unconditional thresholding after step one, not to the certified-budget mode generally.

The sparse propagation work expression remains useful, but it should use the actual retained supports. State separately:

- first-step cost;
- unconditional-threshold mode;
- budgeted mode, without a \(1/\tau\) support cap;
- transient accumulator/storage costs.

The certificate trades approximation control for potentially much larger supports. The complexity proposition must reflect that tradeoff.

---

## 1.4 Mixed-direction proposition: correct witness, but locality must change

> “\(e_u^\top H_+H_-\) has values \((2/3,1/6,1/6)\) in \(G\) and \((5/9,2/9,1/9,1/9)\) in \(G'\).”

Correct.

For the pure out-walk, in both graphs,
\[
p_u^w(u)=3^{-w},\qquad
p_u^w(a)=p_u^w(b)=\frac{1-3^{-w}}2.
\]
The in-walk stays at \(u\). Their distance layers agree at every depth.

The curvature calculation is also correct: the symmetrizations are a three-node path and a four-cycle, and every edge has overlap \(2/3\). Thus all curvature means agree and all standard deviations vanish.

The mixed rows follow directly from the incoming transition probabilities of the sinks.

### Finite-frequency qualification

Different multisets do not automatically imply separation on every conceivable finite frequency grid. Here, however, the **default grid contains \(\theta=1\)**, which separates them:
\[
\operatorname{Re}\phi_G(1)\approx-0.241826,\qquad
\operatorname{Re}\phi_{G'}(1)\approx-0.187315.
\]

A brief calculation, or a Taylor bound using their different second moments, would finish that part without appealing only to generic injectivity.

### The current locality proposition does not cover mixed probes

> “\(B_r(u)\) … out- and in-distance for directed graphs.”

The mixed witness itself exposes the problem. The extra source in directed \(K_{2,2}\) lies in neither the pure out-ball nor the pure in-ball of \(u\), at any radius. Nevertheless, it changes \(H_+H_-\).

Therefore:

- The current pure-direction locality statement is valid for the original two branches.
- It is false for the mixed-augmented representation under the stated ball definition.
- Mixed probes are determined by a **weak radius-two neighborhood**, including the directions and weights needed for normalization.

A simple unified statement is to use weak neighborhoods of radius \(\max(L,2)\) when mixed probes are enabled, and the existing curvature radius where applicable.

Also scope the footprint/MMD statement to the pure branches, or add a probe-type mark and orthogonal mixed-probe blocks.

---

## 1.5 Incomparability: correct examples; finish the \(L=3\) proof algebraically

> “For \(L\in\{2,3\}\) there are nodes that \(L\)-round 1-WL … separates and RoleWalk … does not.”

Correct for the curvature-free representation.

The \(L=2\) derivation is complete and correct. For \(L=3\), the current proof ends with:

> “The walk distributions and layers from \(z\) agree up to step 3 (`theory_checks.py`).”

A numerical check should not be the final step of this proposition. The exact values are short enough to include.

After adding the leaf \(z\):

- Step one: \(z,r\) each have mass \(1/2\).
- Step two: \(z,r\) each have mass \(1/3\); each other core vertex has mass \(1/12\).
- Step three:
  \[
  p_z^3(z)=\frac29,\quad
  p_z^3(r)=\frac{23}{90},\quad
  p_z^3(v)=\frac4{45}\quad(v\in\{a,b,c,d\}),
  \]
  and every outside vertex has mass \(1/60\).

The step-three layers are
\[
\left(\frac29,\frac{23}{90},\frac{16}{45},\frac16\right).
\]
All of these agree in the two graphs, while the differing neighbor-degree multisets distinguish \(r\) after two WL rounds and \(z\) after three.

### Scope of “incomparable”

This is a valid **node-level** counterexample. These two particular graphs are not graph-level collisions: their degree multisets already differ.

The separate pendant-cycle example supplies the graph-level converse at fixed depth. Keep these two uses explicit, especially in the abstract.

### The pendant-cycle proof is still incomplete as written

> “Colour refinement … eventually sees the distance between pendants (checked with 40 rounds).”

Replace the check with a proof. One short argument uses the indicator \(h\) of degree-three vertices. WL determines successive neighbor sums \(A^rh\). At a marked cycle vertex:

- for attachments at \(0,15\), \((A^{15}h)=1\);
- for attachments at \(0,20\), \((A^{15}h)=0\), by bipartition parity.

Thus WL distinguishes them. The matching rooted-ball multisets also need the previously requested distinction between cycle roots, pendant roots, and roots whose balls contain no pendant.

---

## 1.6 WL footnote: specify the update convention

> “Shrikhande / \(4\times4\) rook … not separated by 2-FWL (classical 3-WL).”

The failure of **joint pair refinement**, commonly called 2-FWL in ML terminology, is correct for this pair.

The parenthetical “classical 3-WL” is not safe without defining the convention. Under the **joint triple update** used in my earlier report, the graphs are distinguished: a triangle in the rook graph has a common adjacent fourth vertex, while a triangle in Shrikhande does not.

Some literature uses separate-coordinate/oblivious updates with a dimension shift. My earlier report should also have made that convention explicit rather than relying on “usual WL.”

**Recommended repair:** state the actual update convention and cite the relevant equivalence, or simply write:

> “A standard indistinguishable pair for joint pair refinement; also indistinguishable by Proposition …”

Do not leave an unqualified claim about “3-WL.”

---

## 1.7 Other theoretical issues that remain

### A. Several propositions still lack the required graph assumptions

> “Let \(G'\) be \(G\) plus an unweighted edge \(ab\)…”

The stability proof requires the original graph to be **simple, undirected, unweighted**, not merely that the inserted edge is unweighted. Its uniform-row degree calculation is invalid for a general weighted original graph, and the directed concatenation needs separate treatment.

Likewise, the ordinary triangle identity requires a simple undirected unweighted graph. State these assumptions inside the propositions.

For the regular-graph RWSE equivalence, require \(d>0\).

### B. The irregular-RWSE inference remains unsupported

> “On irregular graphs the two walks differ, so the return probabilities do not translate into each other.”

Different transition matrices do not prove that one collection of statistics cannot determine another.

Replace this with:

> “The constant-degree binomial transformation above does not apply on irregular graphs.”

A stronger information-theoretic claim requires counterexamples with specified feature budgets.

### C. The abstract still overextends the kernel result

> “We prove … that its distances are maximum mean discrepancies…”

The body now correctly excludes curvature standard deviations. The abstract should say **“curvature-free embedding distances”**.

The directed footprint also needs an orientation mark. A kernel statement including raw curvature moments requires the corresponding extra kernel term, not just the displayed curvature-free \(K\).

The class docstring is still incorrect:

> “Squared distances … are then MMDs … under \(K(d)=\operatorname{mean}_\theta\cos(\theta d)\).”

That does not describe the default centered, unnormalized, marked representation across graphs. Replace it with the actual centered finite-measure kernel and its scope.

### D. Definitions still need tightening

- Call \(H\) a **self-loop-augmented walk**, or define the intended nonstandard use of “lazy”; it need not be half-lazy.
- Use distances in the **positive-weight support graph**.
- Explicitly define \(m_x\) using the symmetrized unweighted graph.
- The footprint integral is unnormalized; “mean” should not suggest division by its mass.
- Local fixed coordinates enable applying a predictor to another graph; they do not by themselves guarantee successful transfer.

### E. The old degree-mark overclaim remains verbatim

> “Anything that depends on graph size (degree marks, counting bounds) hurts transfer.”

This is false as a general statement, and degree is not explicit graph-size dependence. RoleWalk itself encodes degree.

Report the observed ablation result instead.

---

## 1.8 Numerical and library-specific issues

### Taylor accuracy claim

> “Power sums … \(k\le8\) (error below \(10^{-12}\)).”

This needs a norm, aggregation level, and arithmetic model.

At \(|\theta p|\le0.1\), the sine remainder after degree seven is at most
\[
|\theta p|^9/9!.
\]
For a row of total mass at most one, summing gives
\[
\sum_j\frac{\theta^9p_j^9}{9!}
\le
\frac{\theta^9p_{\max}^8}{9!}.
\]
At \(\theta=100,\ p_{\max}=10^{-3}\), this is approximately
\[
2.76\times10^{-12},
\]
not \(10^{-12}\). The per-atom remainder is much smaller, but the paper is discussing row features.

More importantly, float32 output and the other numerical operations introduce much larger rounding errors. Separate **Taylor truncation error** from **total implementation error**.

### Beyond three steps, the two backends need not compute the same layer features

The fused kernel uses first observed arrival times. The fallback uses differences between stored supports. After pruning, later supports need not be exact or nested.

Thus:

> “Same features as `compute_embedding`…”

needs qualification when `shells=True`, `tol>0`, and \(L>3\). The fused version maintains a consistent first-observation partition, but that partition need not be the true distance partition. The fallback can use nonnested support differences.

Either compute exact distances, disable pruning for distance recovery, or document and test the approximation explicitly.

### The public subset API is not fully local in computational cost

> “With `nodes`, … this only explores their … neighbourhoods, so it is cheap…”

`transform` still constructs the full adjacency, normalizes the full transition matrix, and allocates global-size workspaces. The subset curvature route also performs global setup.

The **feature depends only on a neighborhood**. The supplied API is not an implementation whose total work and memory depend only on that neighborhood. Distinguish those statements.

### Input validation is important

The theory requires finite nonnegative weights, but the code does not enforce this. `normalize(..., norm="l1")` does not turn signed rows into stochastic rows.

This is particularly important for wiki-RfA. Explain how vote signs and repeated votes become the required nonnegative adjacency. Reject invalid inputs rather than silently violating the mathematics.

# 2. Empirical and presentation audit

## What is materially better

The new baselines genuinely address important confounds:

- Directed ReFeX is now stronger than RoleWalk on noisy supervised synthetic tasks.
- Incoming-plus-outgoing RWSE removes a directional-information imbalance.
- Unnormalized GraphWave substantially narrows the airport-transfer gap.
- The fixed-scale heat-kernel baseline is competitive on airport classification.
- The matched Wikipedia subset changes the substantive conclusion for admins.
- The pruning study exposes large hub errors instead of hiding them.
- Three GPS seeds provide more evidence than two, though still preliminary.

These are meaningful improvements even where they weaken the headline.

## Remaining issues

### 2.1 The “fixed-scale GraphWave” baseline changes more than scale and normalization

> “\(\exp(s(P-I))\) … GraphWave with fixed scales and no \(1/n\).”

This is a random-walk heat kernel. It is not automatically the same operator as the spectral heat kernel used by the original GraphWave baseline. On irregular graphs, combinatorial, symmetric-normalized, and random-walk heat kernels differ.

The current baseline is valuable, but label it accurately. To isolate causes, include a controlled variant that holds the diffusion operator fixed while changing only:

1. \(1/n\) normalization;
2. graph-adaptive versus fixed scales.

The no-\(1/n\) result now provides actual evidence for the normalization explanation on airport transfer. Credit that result, but do not turn it into a universal explanation of GraphWave transfer.

### 2.2 The properties table is still wrong or unfinished

> “GraphWave: Transfer — no.”

Nonlocality does not imply that a feature map cannot be evaluated on a new graph. The new fixed-scale baseline itself demonstrates the distinction.

Use separate notions such as:

- evaluable on an unseen graph;
- fixed coordinate definition;
- strictly local;
- demonstrated transfer performance.

The unresolved property-table and reference TODOs also prevent this from being submission-ready.

### 2.3 Centrality transfer still has a baseline-alignment problem

> “ReFeX … Only where its number of features matches across graphs.”

Matching feature count does not establish matching feature semantics. If recursive feature selection is graph-dependent, two equal-length vectors may still contain different features.

Freeze the ReFeX feature schema on training graphs and apply that schema to test graphs. Also report methods on the same graph-family/seed subset. Otherwise the headline mean comparison remains questionable.

### 2.4 The CF’s incremental value remains unresolved

The top-probability baseline still obtains:

> “Top-10 probabilities … Transfer 0.512”

versus \(0.488\) for the full default RoleWalk.

The summary table appears to omit layers, whereas the main RoleWalk configuration includes them. The decisive comparison is:

- CF + layers;
- top-\(k\) probabilities + the same layers;
- optionally the same curvature and mixed blocks.

Match feature dimension, tuning protocol, and implementation as closely as practical. This is central to the paper’s scientific contribution.

### 2.5 Film’s explanation is still mathematically wrong

> “Position, which an automorphism-invariant embedding cannot use…”

Automorphism invariance prohibits distinguishing nodes in the same automorphism orbit. It does not prohibit representing positional information generally.

The explanation remains a hypothesis. Perform the node-ID permutation check and either support the structural-collision claim or remove it.

### 2.6 Small visible discrepancies remain

- The curvature airport mean is
  \[
  (0.741+0.596+0.666)/3=0.66767,
  \]
  not \(0.669\).
- Directed base accuracy remains \(0.680\) in the main table versus \(0.682\) in the geometry table.
- In the matched three-network admin table, ReFeX at \(0.375\) is second, not RoleWalk at \(0.356\).
- The RfA underlining also marks \(0.814\) as second despite \(0.823\) being present.

Different runs or configurations can explain numerical differences, but those differences need labels.

### 2.7 Expressiveness protocol is still underspecified

> “Embeddings grouped at a sup-distance below \(10^{-4}\)…”

A threshold relation is not necessarily transitive. State the actual grouping algorithm.

Also specify how node embeddings define graph-level equality. The expressiveness caption still mixes pair counts and CSL class counts.

For small unweighted graphs, exact rational walk calculations are feasible and would strengthen the mathematical checks substantially.

### 2.8 Statistical and timing claims remain too strong

The paper still lacks paired uncertainty for most differences. In particular, gains of a few thousandths should not drive “best” claims without paired results.

The scalability evidence is useful, but:

- the supplied source does not expose the main CSV values;
- the curvature complexity needs correction;
- JIT warm-up, compilation, conversion, and timing repetitions should be explicit;
- three-step optimized RoleWalk versus sixteen-step sparse RWSE is not an implementation-matched comparison.

The abstract’s ZINC statement should say **“shows a preliminary improvement in our GPS reproduction”** unless the stronger result survives a more complete comparison.

# 3. Updated reviewer report

## Summary

The paper introduces a deterministic structural node representation combining centered Fourier summaries of short random-walk probability multisets, distance-layer masses, optional overlap-curvature moments, and optional mixed-direction probes. It targets inexpensive, fixed-coordinate representations for role classification and transfer.

The revision substantially improves both mathematical honesty and experimental fairness. In particular, the false WL refinement conjecture is replaced by valid counterexamples, stability and kernel claims are better scoped, pruning receives a useful missing-mass analysis, and stronger baselines reveal a more credible accuracy–cost tradeoff.

Nevertheless, the contribution remains insufficiently established for acceptance at a top venue. Several theory–implementation mismatches remain, and the empirical evidence does not yet isolate why the CF construction is preferable to simpler summaries or establish robust transfer beyond closely matched graphs.

## Strengths

1. **Useful deployment-oriented construction.** The representation is explicit, training-free, and has fixed-dimensional coordinates.
2. **Substantial response to criticism.** The new directed and heat-kernel baselines are important, not cosmetic.
3. **Valid and useful approximation analysis.** The missing-mass certificate correctly addresses iterative pruning, once scoped appropriately.
4. **Meaningful directional extension.** The mixed-probe witness cleanly demonstrates information missed by pure forward and reverse walks.
5. **More credible limitations.** The revision acknowledges stronger simple baselines, hub errors, and the limitations of the GPS evidence.

## Main weaknesses

1. **Correctness and implementation alignment are not finished.** The triangle-kernel complexity, support cap, mixed-probe locality, and certificate scope require correction.
2. **The methodological novelty remains modest.** The strongest contribution is a carefully engineered local walk representation, not a new CF principle or a general expressiveness hierarchy.
3. **The decisive comparative question remains open.** It is not clear when the CF is worth its cost relative to top probabilities, local statistics, or matched walk encodings.
4. **Transfer evidence remains narrow and partly compromised by baseline-schema issues.**
5. **Statistical support is insufficient for many headline rankings.**
6. **The GNN result remains preliminary**, with a weaker-than-published RWSE reproduction and no clean isolation of curvature, added information, and encoder capacity.

## Score and confidence

**Score: 5/10 — marginal reject, up from 4/10.**

**Confidence: 4/5.** High confidence in the algebraic and static-code findings; lower confidence in empirical reproducibility because the benchmark artifacts were not supplied.

I can see a realistic route to **6** without inventing a new method. Reaching **7+** would require a convincing empirical result beyond correcting the existing manuscript.

# 4. Ranked changes most likely to raise the score

## Priority 0 — Correctness gate: necessary before acceptance

These are relatively inexpensive and should happen first:

- Fix the triangle kernel to scan forward adjacency lists.
- Correct the cost proposition for unpruned first steps and budgeted pruning.
- State the certificate per branch and extend or exclude mixed probes.
- Repair mixed-probe locality.
- Finish the exact incomparability and pendant-cycle proofs.
- Fix the remaining scope, documentation, and table errors.
- Add tests for star graphs, mixed witnesses, both WL counterexamples, directed branch concatenation, and backend agreement.

**Score effect:** necessary to make a 6 defensible, but not sufficient by itself. Correcting false claims is not an additional novelty contribution.

## 1. Run the decisive matched comparison and add paired uncertainty

This is the highest-value empirical work.

Compare:

- CF + layers;
- top-\(k\) probabilities + identical layers;
- short return probabilities with matched walk definition;
- a compact local-statistics baseline: in/out degree or strength, neighbor-degree moments, clustering/triangles, egonet statistics, and short returns;
- directed ReFeX;
- controlled heat-kernel variants.

Use identical splits and training-only preprocessing. Report paired performance differences with intervals, not just separate means. For synthetic tasks, retain independent graph seeds; uncertainty across node splits is conditional on the observed graph, not evidence across graph populations.

**What would move my score:** consistent gains over these alternatives, or a clear accuracy–runtime Pareto advantage.

**What would not:** statistically significant \(0.002\) improvements on one dataset, or additional comparisons to weak direction-blind baselines.

## 2. Complete genuinely out-of-distribution transfer

The pending **4× size transfer** is substantially more valuable than a fourth GPS seed.

A realistic protocol:

- Train at size \(n\), test at \(n,2n,4n\).
- Use several independent train/test graph seeds.
- Specify whether expected degree or density is held fixed.
- Freeze standardizers, feature schemas, and model selection.
- Include properly aligned ReFeX and the strongest simple-summary baseline.
- Report per-family results, not only an overall mean.

One family-shift experiment would add further value, but is secondary to doing the size experiment cleanly.

**What would move my score:** robust gains at 4× size that are not explained solely by degree.

**What would not:** another same-size, same-generator split with slightly better averages.

## 3. Make certified pruning a practical result, not just a theorem

Measure exact, unbudgeted, and budgeted modes together:

- tolerances such as \(0,10^{-5},10^{-4},10^{-3}\);
- budgets such as \(0.01,0.05,0.1\);
- runtime, peak memory, retained support, missing mass;
- observed error versus certificate;
- downstream accuracy or clustering changes.

Use tractable real graphs and targeted star/hub queries. Expose per-row missing masses or certificates through the API.

Also give depth–accuracy–runtime curves for matched short-walk baselines.

**What would move my score:** evidence that the certified mode remains competitive and materially improves reliability at reasonable cost.

**What would not:** another average-error number that hides the hub tail, or faster timings obtained only with uncontrolled approximation.

## 4. Add one more real directed task, preferably with meaningful weights

One real directed benchmark is too little for the title-level emphasis.

A realistic option is an original directed Wikipedia-talk network with usable bot/admin labels, if node identities can be aligned reliably. Compare against directed ReFeX and the compact directed-statistics baseline.

For weights, use a task where weights change the structural information, with an explicit unweighted ablation. Resolve the RfA sign/repetition preprocessing regardless.

**What would move my score:** a reproducible gain on a second real directed setting, especially beyond activity/degree.

**What would not:** another synthetic motif family designed around the mixed-probe witness.

## 5. Strengthen the GNN experiment only if it remains a headline contribution

For ZINC:

- complete all methods on the same seed set;
- report paired differences;
- match encoder parameter counts;
- compare RWSE + simple local counts/curvature against RWSE + full RoleWalk;
- diagnose the weaker RWSE reproduction.

Five reasonably matched seeds would be more useful than stopping at an arbitrary fourth seed.

**Peptides is optional**, not an automatic acceptance requirement. One clean additional GNN benchmark could help establish generality, but an under-tuned Peptides table would not.

**What would move my score:** a robust, capacity-controlled improvement that survives stronger count-feature controls.

**What would not:** the fourth GPS seed alone, or a second benchmark with overlapping noisy means and an underperforming baseline.

---

## Practical route to 6 versus 7+

**For a 6:** finish the correctness repairs, provide paired evidence for the main comparisons, fix transfer-baseline alignment, and demonstrate either convincing 4× transfer or a strong certified accuracy–cost tradeoff. The paper can then stand as a useful, well-characterized lightweight representation.

**For a 7+:** add one strong result that changes the scientific conclusion—robust out-of-distribution transfer, convincing real directed-role gains against strong structural features, or a reproducible GNN improvement beyond simple count augmentation.

More propositions, more small-graph enumeration, an extra GPS seed, and additional cosmetic softening would **not**, by themselves, move me to acceptance.