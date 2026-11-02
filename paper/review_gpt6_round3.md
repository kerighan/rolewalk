## Bottom line

**I would raise the score from 5 to 6/10: weak accept, not 7.** The revision now provides the main evidence I requested: a dimension-matched alternative, substantially better transfer experiments with frozen ReFeX features, practical pruning measurements, Digraphwave, and a less preliminary ZINC comparison.

This is not a clean mathematical or implementation bill of health. **The change log overstates completion of Priority 0.** In the supplied source:

- The locality proposition still omits the mixed-probe exception at \(L=1\).
- The cost proposition’s displayed support bound still needs a step-index correction.
- The fast triangle kernel is fixed, but the fallback/subset implementation still lacks the claimed complexity.
- The footprint theorem still does not explicitly represent the now-default directed and mixed blocks.
- The old ambiguous WL footnote remains.
- Pruned layers beyond three steps still have backend-dependent semantics.
- Several empirical explanations and cross-table claims remain wrong.

These are mostly repairable scope, implementation, and reporting issues—not evidence that the central construction fails.

**Scope of this audit:** I inspected the supplied manuscript and Python code. I did not execute the library, inspect benchmark scripts/CSVs, render the external figures, or check the bibliography. In particular, I cannot independently verify the million-node timings, paired-test calculations, frozen-schema implementation, or per-seed results.

---

# 1. Mathematical audit: new and changed statements first

## 1.1 Distance laws, footprint kernel, and Taylor bound

### The distance-law definition is correct

> “\(s^w_u(k)=P(d(u,X_w)=k)\)”

With distances in the positive-weight support graph, this is exactly correct for the unpruned walk. Positive self-loops ensure that supports of successive exact powers reveal the distance balls.

The feature counts also check out:

- Undirected: \(60+9=69\), or \(75\) with curvature.
- Directed with mixed probes: \(2(69)+40=178\), or \(190\) with curvature.

### The displayed footprint kernel is correct for a single pure-walk branch

> “Without curvature, \(\|X_u-X_v\|^2=\mathrm{MMD}_K^2(\mu_u,\mu_v)\)”

For the explicitly defined footprint and feature map, this is a valid finite-measure kernel-mean identity. In particular,
\[
\operatorname{Re}\left[(e^{i\theta a}-1)
\overline{(e^{i\theta b}-1)}\right]
=
\cos(\theta(a-b))-\cos(\theta a)-\cos(\theta b)+1.
\]

The centering handles zero-probability atoms correctly. The kernel is positive semidefinite because it is an inner product of the stated realified features.

This establishes an MMD **pseudometric**; the finite-grid kernel need not distinguish every pair of footprints. The injectivity proposition does not remove that qualification for the particular deterministic grid.

### The new Taylor bound is valid

> \[
> |k_\Theta(a,b)-\|\Theta\|^2ab|
> \le \|\Theta\|^2ab\,t^2
> \left(\frac7{12}+\frac{t^2}{36}\right).
> \]

For nonnegative \(a,b\) and the positive frequency grid, the proof is correct:

- The cosine-product contribution is bounded by
  \[
  \frac{\theta^4a^2b^2}{4}
  \le \theta^2ab\,\frac{t^2}{4}.
  \]
- The sine-product error is bounded by
  \[
  \theta^2ab\left(\frac{t^2}{3}+\frac{t^4}{36}\right).
  \]

The constants sum to \(7/12\) and \(1/36\), as claimed.

The interpretation of \(\|\Theta\|\) is also defensible: **with the chosen unweighted frequency inner product and orthonormal layer coordinates**, it is the unique positive scaling that matches the leading \(ab\) coefficient. That is a normalization justification, not a theorem that this weighting is universally optimal.

### But “joint characteristic function” and “samples” need precision

> “Equivalently, the whole embedding samples the joint characteristic function  
> \(\sum_j e^{i\omega d(u,j)}(e^{i\theta p_{uj}}-1)\)…”

This is not literally a probability characteristic function: its value at \((0,0)\) is zero, not one. It is an **unnormalized, centered joint Fourier transform** of the counting footprint. It is also not the characteristic function of \((p_{uX_w},d(u,X_w))\), which would weight atom \(j\) by \(p_{uj}\).

The precise and useful identity is
\[
F_u^w(\theta,\omega)
=
\sum_j e^{i\omega d(u,j)}
(e^{i\theta p^w_{uj}}-1),
\]
with
\[
F_u^w(\theta,0)=\phi_u^w(\theta),
\qquad
\frac{1}{i}\partial_\theta F_u^w(0,\omega)
=
\sum_{k=0}^w s_u^w(k)e^{i\omega k}.
\]

Thus RoleWalk combines:

1. values of \(F\) along \(\omega=0\); and
2. its derivative at \(\theta=0\), equivalently the exact distance law.

That is stronger and clearer than saying the layers are merely approximate samples at finite \(\theta\).

There is also a normalization detail if this becomes a joint-CF proposition. With \(N=w+1\) and unnormalized DFT values
\[
\widehat s_j=\sum_{k=0}^{N-1}s_k e^{2\pi ijk/N},
\]
Parseval gives
\[
\sum_j|\widehat s_j|^2=N\sum_k|s_k|^2.
\]
To identify the Euclidean layer kernel with the Fourier-coordinate kernel, use the **unitary DFT**, or include the factor \(1/N\) in the latter inner product.

### The theorem still omits the default directed representation

The footprint only has the mark \((w,a,\ell)\), whereas the default directed embedding concatenates two directions and two mixed probes.

Without an orientation/probe-type mark, the displayed measure and kernel do not describe that embedding. This is now a default-method issue, not an optional-feature footnote.

A clean repair is to introduce block types
\[
b\in\{(+,w),(-,w):1\le w\le L\}\cup\{+-, -+\}
\]
and use
\[
K((b,a,\ell),(b',a',\ell'))
=
\mathbf1_{b=b'}
\left[
k_\Theta(a,a')
+
\mathbf1_{\{b\text{ pure}\}}\|\Theta\|^2aa'
\mathbf1_{\ell=\ell'}
\right].
\]
Mixed-probe atoms have no layer contribution.

**Verdict:** the kernel algebra and Taylor bound are sound. The “joint CF” interpretation needs the derivative/normalization clarification, and the theorem needs to cover the actual directed default.

---

## 1.2 Certified pruning: the revised constants are correct

> “For one walk direction, without curvature or mixed probes, \(L\le3\) and in exact arithmetic…”

This scope is now appropriate.

The key facts remain
\[
0\le q^w\le p^w,\qquad
\|p^w-q^w\|_1=r^w.
\]
The corrected CF error is at most \(2\|\Theta\|r^w\), and the scaled layer error is at most \(\|\Theta\|r^w\). Therefore
\[
\|\Delta X_u\|^2
\le
5\|\Theta\|^2\sum_w(r_u^w)^2.
\]

The stated consequences are correct:

- One direction: \(\sqrt{5L}\|\Theta\|\delta\).
- Two directions: \(\sqrt{10L}\|\Theta\|\delta\).
- Mixed probes: add
  \[
  4\|\Theta\|^2\sum_\beta r_\beta^2
  \]
  to the squared bound.

The code’s all-or-nothing pruning rule also matches the new description:

> “A row whose small entries would exceed the budget keeps all of them.”

The \(L\le3\) distance-label argument is valid because step one is unpruned and the pre-pruning step-two support is exact.

### Remaining qualifications

**1. The certificate is for the ideal feature evaluation, not literally the returned array.**

> “Floating-point evaluation adds its own error.”

Correct, but the fused kernel also uses a Taylor approximation to the trigonometric features. That is a separate evaluation error even in exact arithmetic.

The clean statement is
\[
\|X-\widehat X_{\rm returned}\|
\le B_{\rm pruning}+\varepsilon_{\rm Taylor}+\varepsilon_{\rm floating}.
\]
Only the first term is currently certified by the proposition.

**2. The pruning experiment must identify the error norm.**

> “A budget \(\delta\) caps the worst layer error at \(\delta\)…”

This is correct for the **unscaled, per-step absolute** layer \(\ell_1\) error, and hence its \(\ell_2\) error. It is not a bound on relative error or the whole concatenated/scaled layer block.

Because the preceding sentences report percentages, the present wording is ambiguous. Label the reported \(0.009,0.047,0.096\) explicitly.

**3. Curvature remains outside the certificate.**

The experiment’s statement

> “the observed error is at most 0.51 of the certificate … everywhere”

must identify the curvature-free configurations and probe scope used. The theorem does not certify the default curvature-augmented vectors.

**4. The first-step zero error permits a tighter corollary.**

Since \(r^1=0\), one can replace \(L\) by \(L-1\) for the pure branches. This is optional, not a correctness requirement.

**Verdict:** the revised mathematical certificate is sound. The remaining work is accurate interpretation and implementation-error accounting.

---

## 1.3 Locality: substantially repaired, but the proposition still has an exception missing

The preamble says:

> “On directed graphs the mixed probes read the undirected 2-ball, so the locality radius is \(\max(L,2)\).”

Correct.

But the proposition still says:

> “\(X_u\) is determined by … \((B_L(u),u)\)…”

That is false for the default directed embedding when \(L=1\). Your mixed-probe witness already supplies a counterexample: the additional source is outside the weak radius-one ball but changes \(H_+H_-\).

State the proposition directly as:

- Pure branches: radius \(L\).
- With mixed probes: weak radius \(\max(L,2)\).
- With curvature: weak radius \(L+2\), for \(L\ge1\).

The proof should explicitly cover the normalization of the second mixed step. The weak two-ball contains all relevant incoming/outgoing edges of the intermediate vertices, so the repaired claim is valid.

Also:

> “invariant and local (hence transferable across graphs)”

still conflates **compatible coordinates** with **successful transfer**. Locality and fixed coordinates make cross-graph application well-defined; they do not imply predictive generalization.

---

## 1.4 Cost: the central repair is right, but the displayed proposition is still not fully correct

> “each row … after the first keeps at most \(1/\mathrm{tol}\) entries”

Correct for threshold-only pruning and positive tolerance.

But the same proposition then writes
\[
|\operatorname{supp}p_u^{w-1}|
\le
\min(|B_{w-1}(u)|,1/\mathrm{tol}).
\]

Two corrections remain:

1. This bound applies to the **retained row \(q^{w-1}\)**, not the exact \(p^{w-1}\).
2. It applies only when \(w-1\ge2\). In particular, it cannot be used for the input support of step two.

A correct formulation is
\[
W_w=
O\!\left(
\sum_u\sum_{j\in\operatorname{supp}q_u^{w-1}}(d_j+1)
\right),
\]
with
\[
|\operatorname{supp}q_u^t|
\le
\min(|B_t(u)|,\lfloor1/\tau\rfloor),
\qquad t\ge2,
\]
in threshold-only mode. The first-step support and budgeted mode need their separate exceptions.

This cap is on the **retained** row; it does not bound the pre-pruning accumulator or sparse product.

### Triangle enumeration: the new fast kernel is correct

The revised code actually constructs forward adjacency and scans only it:

```python
for b in range(fwd_ptr[w], fwd_ptr[w + 1]):
```

That fixes the previous star counterexample. The triangle counts and the three-edge updates are correct.

For degree ordering, forward degree is at most \(\sqrt{2|E|}\), giving the standard \(O(|E|^{3/2})\) enumeration bound. Include \(O(n+|E|)\) setup/output costs and the current rank-sorting cost when describing the entire routine.

### But the fallback/subset complexity is still not fixed

The supplied code retains:

```python
common = (rows @ U).multiply(rows)
```

Without Numba, and whenever `nodes` is supplied, this can create a dense intermediate on a star. Consequently, the unqualified statement

> “curvature costs a triangle enumeration, \(O(|E|^{3/2})\)”

does not describe every library path.

Either use forward enumeration for those paths too, or explicitly scope the bound to the full-graph Numba implementation.

### “Embedding a set touches only its ball” is not an API cost claim

> “Embedding a set \(S\) touches only \(B_L(S)\)…”

As a feature-dependence statement, with the mixed-probe radius correction, this is fine. As a statement about `transform`, it is false: the method converts the full graph, constructs full transitions, and allocates global workspaces.

The subset API is useful, but not a neighborhood-only preprocessing implementation.

---

## 1.5 A new default-directed implementation caveat: mixed probes are not fused and can have large transient memory

This deserves explicit attention now that mixed probes are the default.

> “one pass … without storing the powers: \(O(n)\) memory per thread”

That describes the pure-walk fused kernel, but `_mixed_blocks` does this:

```python
T = (first[block] @ second).tocsr()
```

It materializes a block of the two-step product before pruning.

Moreover, `_row_blocks(first, rows)` estimates work using degrees in `first`, not the degrees in `second` that the mixed product actually scans.

A concrete stress test is a directed in-star: \(d\) sources all point to one sink.

- For \(H_+H_-\), every source reaches all sources through the sink.
- Each mixed row has \(\Theta(d)\) pre-pruning entries.
- The current block-work estimate is only constant per source.
- A block containing all sources can therefore materialize \(\Theta(d^2)\) entries before thresholding.

This does not invalidate the representation or the reported timings on the supplied graph families. It does invalidate a blanket \(O(n)\)-workspace description of the default directed pipeline.

**Fix:** either fuse mixed probes row-by-row, or estimate block work using the second transition matrix and document the remaining transient-memory bound.

---

## 1.6 Mixed-probe proposition: correct

The pure branches, curvature equality, and mixed rows all check out:
\[
(2/3,1/6,1/6)
\quad\text{versus}\quad
(5/9,2/9,1/9,1/9).
\]

The default-grid qualification is now present:

> “at \(\theta=1\) … \(-0.2418\) and \(-0.1873\)”

These values are consistent with the rows, and they separate the graphs. If an entirely algebraic certification is desired, a second-order cosine expansion with its fourth-order remainder suffices. No additional generic-grid injectivity argument is needed.

---

## 1.7 Incomparability and pendant-cycle proof: now substantively complete

The two node-level constructions are correct.

For the leaf-rooted example, the supplied step-three probabilities
\[
\frac29,\quad \frac{23}{90},\quad
\underbrace{\frac4{45},\ldots,\frac4{45}}_{4},\quad
\underbrace{\frac1{60},\ldots,\frac1{60}}_{10}
\]
sum to one and agree across the graphs, with the stated layers.

The assertion that curvature separates these examples is also correct. For example, the unscaled step-one curvature means in the original root construction are \(3/7\) and \(41/90\).

The pendant-cycle WL argument now works:

> “round \(t+1\) colours determine \((A^t h)_v\)”

Starting from the degree-three indicator identified at round one, this follows inductively from neighbor multisets. Both marked vertices in the first graph have the relevant 15-step count one; both in the second have count zero. Since degree information is retained, this yields graph-level separation.

The rooted-ball argument should still be written slightly more carefully. At radius five, distinguish:

- pendant roots;
- cycle roots close enough to see a pendant;
- cycle roots whose induced ball is just a path.

The counts match in both graphs. The conclusion is correct; “fixed by its distance to [a pendant]” is an abbreviated proof rather than a false conclusion.

### The WL footnote was not actually repaired

The supplied paper still says:

> “not separated by 2-FWL (classical 3-WL)”

There is no definition of the higher-dimensional update convention in the supplied source. The uniform initialization of **1-WL** does not resolve this.

Remove the parenthetical or define the update precisely. The safe claim is the one about joint pair refinement. I would not accept the change log’s statement that this item has been completed.

---

## 1.8 Remaining theoretical statements

| Statement | Audit |
|---|---|
| Injectivity over a frequency set with an accumulation point | Correct. |
| Almost-everywhere finite-grid separation of a countable family | Correct, for at least one sampled frequency. Qualitative only. |
| Edge-insertion stability | Correct under the newly stated simple, undirected, unweighted assumptions. |
| \(K_{2,d}\) layer-instability example | Correct. |
| Degree and triangle identities | Correct. |
| Distance-regular failure class | Correct, including curvature. |
| GraphWave centering/rescaling identity | Correct. |
| Overlap-curvature closed form and Ollivier upper bound | Correct with \(m_x=\mathbf1_{N[x]}/(d_x+1)\). State that formula explicitly rather than relying on “lazy.” |

One small assumption is still missing:

> “On a \(d\)-regular graph … \(P=A/d\)…”

Require \(d>0\) in the **RWSE proposition itself**. The condition in the preceding proposition does not automatically carry over. The binomial transformation is valid for fixed/known \(d>0\).

### A significant false explanation has reappeared elsewhere

> “RWSE’s [mean drops] by 0.19 … their coordinates depend on the graph, RoleWalk’s on the neighbourhood only.”

This is wrong for RWSE. Fixed-length return probabilities are also local, invariant, fixed-coordinate statistics. The properties table correctly marks RWSE that way.

A larger transfer drop does not establish a failure of coordinate semantics. It could reflect neighborhood-distribution shift, feature ranges, walk definition, model extrapolation, or loss of informative resolution. Remove this causal explanation unless supported by a controlled diagnosis.

---

# 2. Library–paper alignment

## Correctly aligned now

- Default directed mixed probes and feature dimensions.
- Unpruned first step.
- All-or-nothing budgeted pruning.
- Missing-mass CF correction.
- Exact retained-entry distance labels through three steps.
- Forward-adjacency triangle enumeration in the full-graph Numba path.
- Rejection of negative/non-finite adjacency values in `transform`.
- Wiki-RfA preprocessing into a nonnegative, unweighted adjacency.

## Still misaligned or underspecified

### 2.1 Pruned layers beyond three steps remain backend-dependent

The fused kernel stores first observed arrival times. The fallback subtracts successive stored supports:

```python
layers = np.diff(np.column_stack(within), prepend=0, axis=1)
```

Beyond step three, those supports need not be nested or exact after pruning. Consequently:

- The fused labels can overestimate true distances.
- The fallback differences can cease to represent a partition and can even be negative.
- The two backends need not compute the same features.

The docstring

> “Same features as compute_embedding”

therefore remains false in general.

Choose one of:

1. exact distance recovery;
2. disabling pruning when exact deeper layers are requested;
3. one documented approximate-distance convention shared by both backends.

Also state whether the five-step expressiveness results use `tol=0`. Otherwise the experiment does not necessarily evaluate the theoretical distance-law embedding.

### 2.2 The Taylor claim is fixed in the paper but not in the code comment

The manuscript’s row-wise truncation bound below \(3\times10^{-12}\) is valid at the default frequency range.

The kernel still comments:

```python
# power sums instead of 2 m sines (error below 1e-12)
```

Update it. More importantly, neither number is a total float32 error bound.

### 2.3 The class-level kernel claim remains too broad

The docstring claims an MMD interpretation without specifying:

- `aggregate="sum"`;
- the step/direction/probe marks;
- the layer option;
- no projection, or the corresponding projected kernel;
- exact rather than pruned/evaluated features.

The paper has improved substantially here; the public API documentation should match it.

### 2.4 Other documentation corrections

> “1e-4 … with no measurable loss”

The manuscript reports classification changes up to \(0.005\) and large hub feature errors. Replace this with an empirical, scoped statement.

> “only explores their walk_len-step neighbourhoods, so it is cheap”

This omits full-graph setup and the mixed-probe radius-two exception.

> “Layers … stays nested beyond three steps”

The fused first-observation partition is internally consistent, but that is not the same as recovering true distances. Say so.

Also validate the mathematical parameter domains, especially `max_missing` and nonnegative `tol`. For example, a negative budget is interpreted differently by the fused and fallback implementations. If multigraphs are supported, clarify whether validation concerns original edge weights or the aggregated adjacency.

---

# 3. Table and empirical-claim consistency

## 3.1 Several earlier numerical/ranking problems are genuinely fixed

I checked the visible entries:

- Curvature airport mean is now consistent with \(0.668\).
- Wikipedia admin rankings are corrected.
- RfA second-place markings are corrected.
- BREC totals \(207,213,214\) are correct.
- Centrality means are broadly consistent with the displayed rounded entries.
- The reported ZINC paired mean difference is compatible with the rounded marginal means.

The main tables now tell a substantially more credible story than in round two.

## 3.2 “Best on unsupervised synthetic roles” is contradicted by the table

> “best … on unsupervised synthetic ones”

Local statistics have ARI \(0.409\), exceeding RoleWalk + curvature’s \(0.396\). RoleWalk + curvature is best on mAP, not every unsupervised metric.

Use:

> “best retrieval mAP and competitive clustering on the synthetic directed tasks.”

Likewise, “best on real directed roles” currently means **highest reported means on two tasks from one dataset**, not broad evidence across real directed settings.

## 3.3 The centrality caption contains a denominator inconsistency

> “6 families × 2 seeds”

and seven reported targets imply **84 graph–target pairs**.

But the Digraphwave note says:

> “a constant on 34 of the 96 graph–target pairs”

There may be an eighth target in that CSV, but then the failure count and mean are not transparently aligned with the displayed comparison.

Reconcile the target set, denominator, and mean computation. This matters because the new transfer experiment is the strongest acceptance argument.

## 3.4 “Same ranking” is false

> “Test graphs of the training size give the same ranking…”

At the same size, the quoted ordering is
\[
\text{ReFeX}>\text{RW heat CF}>\text{local stats}.
\]
At \(4\times\) size it is
\[
\text{local stats}>\text{ReFeX}>\text{RW heat CF}.
\]

The top two methods remain the same; the full ranking does not.

## 3.5 The limitations retain a stale ReFeX claim

> “ReFeX … wins … on \(k\)-core transfer”

Not in the revised frozen-schema table:

- RoleWalk + curvature: \(0.658\).
- Top-\(k\) + layers: \(0.655\).
- ReFeX: \(0.635\).

Delete the stale claim. The text already identifies the old \(0.935\) result as an artifact.

## 3.6 The geometry table needs configuration labels, not a rounding disclaimer

> “ablation runs: may differ from the main tables in the third decimal”

The geometry table’s directed \(0.682\) is the **no-mixed** configuration, whereas the new main default is \(0.728\). This is not a third-decimal difference.

Label mixed probes as disabled in that table, and identify any remaining split/seed differences. The same applies to trajectory and other analysis tables compared against old directed defaults.

## 3.7 The old summaries table is not evidently dimension-matched

> “Same budget of features…”

It compares “Top-10 probabilities” against the default CF, which has 20 real coordinates per step. The new main baseline correctly says **Top-20 + layers**.

Explain the actual dimensions in the old analysis table or change its caption. Do not let readers infer that the \(0.512\) versus \(0.445\) transfer comparison uses the same configurations as the new matched comparison.

For the directed matched comparison, also state explicitly whether top-\(k\) receives summaries of **both mixed probes**. “Same layers, same dimension” in the undirected caption does not establish that. The benchmark implementation is not supplied, so I cannot verify it.

## 3.8 The paired-significance summary is overbroad

> “on Brazil, Europe, Reality-call and Actor no difference is significant either way”

Between which methods?

Read literally over the whole table, this is implausible and inconsistent with the quoted interval scales—for example, Brazil has RoleWalk near \(0.74\) and Role2Vec at \(0.476\).

If this means “no difference between RoleWalk and the strongest competing methods,” name those comparisons. Report:

- the paired mean difference;
- corrected interval;
- comparator;
- whether multiplicity is addressed.

Nadeau–Bengio correction is a useful improvement, but 50 overlapping node splits do not provide 50 independent graph realizations.

## 3.9 ZINC: better evidence, still appropriately preliminary

> “below RWSE on each of the 5 seeds”

This is useful and should be reported.

But clarify what
\[
0.0045\pm0.0037
\]
means. If \(0.0037\) is the sample standard deviation of paired differences, the ordinary paired-\(t\) 95% interval is approximately
\[
0.0045\pm 2.776\frac{0.0037}{\sqrt5},
\]
which just includes zero. Five improvements out of five also give a two-sided sign-test \(p=0.0625\).

Thus “preliminary improvement” remains the right wording. The missing capacity/count-feature controls and weaker RWSE reproduction still prevent this from being a strong headline result.

## 3.10 Other interpretation fixes

- **GraphWave scaling:** within one graph, removing \(1/n\) and centering are affine feature transformations that standardization should cancel. The near-identical classification rows are expected; the \(0.001\) Europe discrepancy needs only a numerical/run explanation, not a substantive interpretation.
- **Full per-layer kernel:** “is not better” is too broad. It improves some displayed motif/directed scores while hurting airport transfer. Say it is not consistently better despite its larger dimension.
- **Film:** an ID-permutation check tests identifier/order artifacts. It does not by itself establish a positional/co-occurrence explanation; legitimate positional features can be permutation-equivariant.
- **Centrality terminology:** PageRank is a diffusion/stationary centrality, not a shortest-path centrality. “Path-based centralities, top four rows” should be adjusted.
- **Expressiveness grouping:** “grouped at a sup-distance below…” still does not specify a transitive grouping procedure. The graph-level sorted-row description is an improvement, but tolerance-based grouping and sorting conventions should be reproducible.
- **Scalability:** the main values are imported from an unavailable CSV, so I cannot verify the headline range or the 5× comparison. State warm-up/compilation, repetitions, hardware, and whether budgeted pruning is used.
- **Submission readiness:** reference/property-table TODOs remain. The properties table also still combines locality and fixed-coordinate semantics into one column.

---

# 4. Updated reviewer report

## Summary

The paper proposes a deterministic, training-free structural node representation using centered Fourier summaries of short random-walk probability multisets, distance-layer masses, optional overlap-curvature moments, and mixed-direction probes. It targets inexpensive role analysis and cross-graph application.

The revision now establishes a more convincing scientific contribution than round two. In particular, it compares the CF against a dimension-matched sorted-probability alternative, repairs the strongest implementation complexity error, evaluates transfer across a fourfold size change with a frozen ReFeX schema, and demonstrates the practical cost of missing-mass-controlled pruning.

The contribution is best understood as a well-engineered and well-characterized lightweight representation—not a fundamentally new characteristic-function principle or a general expressiveness hierarchy.

## Strengths

1. **The decisive empirical confounds have been addressed substantially.**  
   Top-\(k\) + layers, local statistics, controlled GraphWave normalization, heat-kernel CFs, and Digraphwave materially strengthen the evaluation.

2. **The transfer result is now meaningful.**  
   The \(2{,}000\to8{,}000\) experiment with frozen ReFeX semantics is much stronger than same-size transfer with graph-dependent feature selection.

3. **The theory has useful, correctly scoped content.**  
   The pruning analysis, mixed-direction witness, stability result, and explicit failure/collision examples are valuable.

4. **The system contribution is credible.**  
   The fused pure-walk implementation and repaired triangle kernel support a useful accuracy–cost tradeoff, subject to the documented fallback and mixed-probe caveats.

5. **The paper increasingly exposes its own limitations.**  
   Degree-dominated airport transfer, stronger supervised directed ReFeX, the Wikipedia admin loss, hub pruning errors, and the weaker RWSE reproduction are not hidden.

## Weaknesses

1. **Correctness and documentation closure is still incomplete.**  
   Several claims marked “fixed” in the change log remain incomplete in the actual source.

2. **The kernel unification is primarily an interpretation.**  
   The direct-sum kernel identity and tangent normalization are valid, but should not be oversold as a new joint-distribution embedding theorem.

3. **The main transfer advantage still lacks graph-level uncertainty and per-family visibility.**  
   The mean advantage over top-\(k\) is \(0.010\). Closeness and harmonic gains are more substantial, but their consistency across families and seeds is not shown here.

4. **The real directed evidence remains narrow.**  
   One unweighted dataset with two related tasks does not establish broad directed/weighted superiority.

5. **The strongest scalability statement does not describe every default implementation path.**  
   Mixed products and fallback curvature deserve explicit qualification.

6. **The GNN evidence remains secondary.**  
   Five seeds improve credibility, but do not resolve baseline reproduction or added-information/capacity confounds.

## Score and confidence

**Score: 6/10 — weak accept, up from 5/10.**

**Confidence: 4/5 for the mathematical/static-code assessment; approximately 3/5 for empirical reproducibility.**

I would require the concrete correctness and reporting repairs above before publication. Nevertheless, I would not keep the paper at 5 merely by moving the empirical goalposts after the authors completed the main requested experiments.

The supplied source is still a working draft rather than submission-ready prose. My score reflects the scientific package now present, not an assertion that the current text can be published unchanged.

---

# 5. Ranked changes that would raise this toward 7

## Priority 0: finish the remaining correctness/artifact repairs

Necessary to sustain the 6:

- Correct the locality proposition itself.
- Correct the retained-support notation and step range.
- Scope or repair fallback curvature complexity.
- Add direction/probe marks to the kernel theorem.
- Replace “joint CF samples” with the precise centered-transform/derivative identity.
- Resolve pruned layers beyond three steps.
- Scope the fused-memory claim and fix mixed-product blocking.
- Remove the unresolved WL parenthetical.
- Reconcile 84 versus 96 transfer cases and stale table explanations.
- Remove the false RWSE-locality explanation.

**These repairs would not, by themselves, raise 6 to 7.** They make the existing contribution trustworthy; they are not additional novelty.

## 1. Establish that the transfer advantage is robust across graphs

This is now the highest-value scientific work.

Provide:

- Per-family results for the current \(4\times\) experiment.
- More independent train/test graph seeds.
- Paired graph-level uncertainty for RoleWalk versus top-\(k\), local statistics, frozen ReFeX, and heat-kernel CF.
- Exact matching of target sets and failure handling.
- A controlled diagnosis of RWSE and Digraphwave transfer degradation.

A modest family/parameter shift would strengthen the result further, but is not a prerequisite for retaining 6.

**What would move the score:** consistent closeness/harmonic or broader transfer gains that are not driven by one family, mismatched feature semantics, or model-range failures.

**What would not:** more overlapping node splits on the same graphs, or another aggregate mean without per-family results.

## 2. Add a second real directed task with meaningful weights

Use a setting where directions and weights plausibly encode roles, and compare against:

- directed local statistics;
- directed ReFeX;
- Digraphwave;
- top-\(k\) with identical probe information.

Include weighted versus unweighted and mixed versus pure ablations.

**What would move the score:** a reproducible gain beyond activity/degree on an independent real setting.

**What would not:** another synthetic motif tailored to the mixed-probe witness, or merely retaining vote magnitudes without demonstrating that they matter.

## 3. Turn certified pruning into a stronger deployment result

The current practical study is useful. The next step is a clear accuracy–runtime–memory frontier:

- exact, threshold-only, and budgeted modes;
- tail errors, not only averages;
- retained supports and peak memory;
- pure versus mixed directed probes;
- per-node missing masses/certificates exposed through the API.

**What would move the score:** competitive certified operation on hub-heavy and directed graphs, with transparent worst-case behavior.

**What would not:** more speed numbers obtained only under uncontrolled approximation.

## 4. Strengthen GNN evidence only if it remains a headline

Compare RWSE against:

- RWSE + curvature/local counts;
- RWSE + dimension/parameter-matched RoleWalk;
- the existing full combination.

Diagnose the RWSE reproduction first.

**What would move the score:** a robust gain beyond simple structural augmentation under a matched encoder.

**What would not:** a sixth GPS seed alone, or an under-tuned Peptides table. **Peptides is not an acceptance requirement.**

---

## Explicitly: what will not move my score

- More propositions restating the direct-sum kernel construction.
- More small-graph enumeration without changing the scientific conclusion.
- Additional weak or direction-blind baselines.
- Tiny significant improvements on one node-split benchmark.
- Cosmetic softening of claims without resolving their underlying inconsistencies.
- Extra GNN seeds without stronger controls.

**The paper has now earned the move to 6. The route to 7 is not “do everything”: it is to make one central claim—robust transfer, real directed-role utility, or certified efficiency—decisively convincing.**