## Bottom line

**I would now give this 8/10 — accept, with bounded technical repairs. Confidence: 3/5 overall, approximately 4/5 on the mathematical and static-code audit.**

The increase is earned by the **parameter-shift experiment**, not by the number of revisions. It addresses the route I specified: the advantage over the dimension-matched probability summary persists when graph size is no longer the only changing variable. The reported gap to top-\(k\) grows to \(+0.080\,[0.048,0.111]\), and the closeness/harmonic advantages persist against local statistics.

The local-statistics win on the overall shifted benchmark does **not** negate this result. It limits the claim, appropriately. I did not require RoleWalk to win every target, and would not introduce that requirement now.

Most earlier correctness issues are closed. **I would not, however, give the library an unrestricted numerical-robustness sign-off:** the new normalization still has a subnormal-weight overflow case, and integer-weight symmetrization can corrupt curvature. Both are localized repairs, not reasons to reopen the research program.

### Scope

This is a review of the supplied manuscript and a static audit of the supplied source. I have not executed the library, tests, or benchmarks, or inspected the referenced CSVs, scripts, labels, SVG, or bibliography. Reported empirical results are therefore evaluated as reported, not independently reproduced.

---

# 1. Mathematical re-check

## The requested repairs are substantially complete

| Item | Assessment |
|---|---|
| Weighted transition matrix | The displayed definition now matches the intended mean-edge-loop walk. |
| Locality proof | Correctly accommodates strengths and degrees under the revised walk. |
| Distance-regular and RWSE results | The required unweighted scope is now explicit. |
| Fourier normalization | The explicit \(1/\sqrt{w+1}\) factor repairs the Euclidean-isometry statement. |
| Kernel interpretation | “Only positive factor” and MMD **pseudometric** are appropriate qualifications. |
| Retained-support bound | Correctly restricted to \(\mathrm{tol}>0\), threshold-only pruning, and steps after the first. |
| Curvature complexity | The compiled implementation and quadratic sparse fallback are now distinguished in the proposition itself. |
| Mixed-product blocking | Budget plus at most one row’s work is the right qualification; it is not presented as a strict budget-sized cap. |
| Expressiveness/BREC settings | `tol=0` is now explicit in both captions. |
| Pruning certificate | The exact-arithmetic scope and separate numerical-evaluation errors are clearly stated. |

I find no new substantive mathematical problem in these repaired arguments.

### One useful clarification in the weighted definition

Write
\[
d_u=\#\{v\ne u:A_{uv}>0\}.
\]

The implementation removes zero-weight entries before counting degree. This should be explicit in the definition too: with nonnegative weights, “unweighted out-degree” could otherwise count zero-weight graph edges. A row containing only zero-weight edges must enter the isolated-node branch, not produce \(s_u/d_u=0\) and a singular normalization.

Also label the displayed expression for \(H_{uv}\) as the **off-diagonal** formula. These are small clarifications, not a recurrence of the previous wrong-walk definition.

## The certificate API matches the theorem’s block accounting

The implementation uses
\[
B_u=\|\Theta\|_2
\sqrt{c\sum_{\text{pure blocks}}r_{u,b}^2+
4\sum_{\text{mixed blocks}}r_{u,b}^2},
\]
with \(c=5\) when layers are included and \(c=4\) otherwise. This agrees with the proof.

It also correctly withholds a certificate for curvature, mean aggregation, and walk lengths beyond the currently proved layer regime. The missing-mass ordering agrees with the documented out/in/mixed ordering.

Importantly, the paper no longer treats this as a certified floating-point error bound. Testing that observed errors fall below it is useful validation, but does not turn it into such a numerical certificate. The revised wording respects that distinction.

---

# 2. Library re-check

## Repairs that check out statically

### The accumulator memory-safety issue is fixed

The separate `listed` marker makes membership in `touched` independent of the accumulated numerical value. An index cannot be appended repeatedly merely because its contributions underflow to zero.

This repairs the actual workspace-overrun mechanism identified last round. Eliminating zero transitions and skipping zero accumulated values are useful additional safeguards.

### Large-weight row-sum overflow is addressed in the intended regime

Scaling before computing strengths prevents several large finite weights from overflowing their unscaled row sum. That is the correct approach, subject to the reciprocal issue below.

### Self-loop semantics are explicit

The default walk ignores graph self-loops; the legacy walk retains its previous behavior. This is much better than silently accepting loops under an incorrect stay-probability description.

### Saved-model semantics are preserved

Saving `walk` and loading older files with `"unit_loops"` resolves the silent weighted-feature migration problem. Previously fitted centers are no longer automatically paired with a different transition rule.

### The requested documentation repairs are present

The shell assumption, centered GraphWave identity, and undirected interpretation of `role_summary.links_to` are now stated.

## Two remaining numerical bugs

These are real issues in the advertised finite-weight domain, although neither appears to undermine the reported unweighted transfer experiments.

### 2.1 Reciprocal row scaling still overflows for sufficiently small positive weights

The new code contains:
```python
A = diags(1 / np.where(top > 0, top, 1)) @ A
```

For a row maximum such as \(10^{-320}\), the reciprocal overflows to infinity even though the original weights are finite and positive.

A two-node graph with a single edge of weight \(10^{-320}\) should produce the same walk as an edge of weight one:
\[
H=
\begin{pmatrix}
1/2&1/2\\
1/2&1/2
\end{pmatrix}.
\]
The displayed implementation instead creates nonfinite intermediate values.

**Repair:** divide each stored weight directly by its row maximum, rather than forming that maximum’s reciprocal. Since every numerator is at most the row maximum, those divisions cannot overflow.

Add a regression test comparing ordinary, very large, and subnormal positive rescalings. The large-weight repair is otherwise sound.

### 2.2 Symmetrization can overflow on integer-weight inputs

Both curvature and subset reachability use:
```python
(A + A.T) > 0
```

`transform` does not convert its original adjacency to floating point before these operations. NetworkX can supply an `int64` adjacency.

For an undirected edge of integer weight \(2^{62}\), adding its two symmetric entries produces \(2^{63}\), which overflows signed `int64`. The subsequent positivity test can therefore remove a real edge.

For the two-node example, the correct node overlap curvature is one; the corrupted support can instead yield zero.

**Repair:** form the positive support first, then symmetrize it using Boolean union or elementwise maximum. Apply this in both `node_curvature` and the subset-reachability construction.

This also avoids unnecessary floating-point overflow warnings when only the support is wanted.

## Smaller remaining API issues

The additional validation is welcome, but not comprehensive:

- Frequency bounds are unchecked, including whether the resulting **float32** frequencies are finite.
- Integer-valued floats such as `n_samples=10.0` pass the integer test but are not converted to integers before use.
- `block_work=np.inf` passes the positivity test.
- `n_jobs` and projection dimensions remain largely unchecked.

These are ordinary API-hardening tasks, not acceptance-level scientific objections.

A few docstrings also deserve tightening:

- The Taylor guarantee is an **absolute per-coordinate bound**, not a uniform relative-error guarantee for possibly cancelling CF coordinates.
- `missing_` need not be literally zero at `tol=0`; floating-point row-sum residuals can remain.
- The `certificate_` documentation should list mean aggregation among the cases returning `None`.

**Library verdict:** the important earlier safety and compatibility repairs are genuine. Finish the two numerical fixes above; do not describe the current code as robust over every finite nonnegative input yet.

---

# 3. Experimental and reporting consistency

## 3.1 The parameter-shift result materially strengthens the paper

The reported design is appropriate for the question posed:

- five specified shifts;
- ten independent graph pairs per shift;
- unchanged representation/regression protocol;
- the same matched baselines;
- graph-pair-level uncertainty;
- explicit failures rather than selective reporting.

The strongest supported conclusions are:

1. **The CF summary retains an advantage over matched top-\(k\) under these shifts.**
2. **Closeness and harmonic centrality remain particular strengths**, including against local statistics.
3. **RoleWalk is not the best general-purpose centrality predictor under parameter shift.**
4. **Superiority to frozen ReFeX is not established by the reported interval**, despite a higher point estimate.

That is a useful and substantially more informative result than another size-transfer repetition.

### Three wording adjustments remain

**“Ahead of every other embedding.”**  
This is defensible as a statement about the displayed means, but not as established superiority over each comparator. For frozen ReFeX, \(+0.023\,[-0.007,0.054]\) includes zero. Say “highest mean among the embedding baselines,” then identify which paired advantages are established.

**“Across generators.”**  
The experiment changes parameters **within generator families**. Replace this with “across the tested within-family parameter shifts.” This is precisely the experiment I requested; there is no need to inflate its scope.

**“For targets that are themselves local statistics…”**  
Clustering is directly included among the local features. Betweenness and \(k\)-core are not local statistics in the relevant bounded-neighborhood sense. The paragraph should distinguish the direct clustering-feature advantage from successful prediction of global quantities using local features.

Likewise, name **closeness and harmonic centrality** rather than using “distance-based centralities” so broadly that readers might infer betweenness is also a shifted-transfer strength.

## 3.2 The old centrality exclusions and aggregation ambiguity are repaired

The revised exclusions sum correctly:
\[
10+9+1=20.
\]

The main table now explicitly distinguishes valid-cell averaging from pair-level averaging. Its RoleWalk mean remains consistent with the displayed values:
\[
\frac{60(0.864+0.797+0.861+0.989+0.805+0.673)
+40(0.651)}{400}
=0.81345.
\]

The numerical per-family results are now present.

### Give the new shift table the same aggregation precision

Its displayed grand mean is consistent with averaging the 330 valid graph–target cells:
\[
\frac{50(0.614+0.720+0.750+0.893+0.639+0.612)
+30(-0.017)}{330}
\approx0.639.
\]

The caption’s “mean over the 50 pairs” should distinguish:

- per-target means;
- the overall valid-cell mean;
- the pair-level estimand used for paired intervals.

Also identify the families contributing the new twenty constant-target exclusions. This is a short reporting repair, not a request for another experiment.

## 3.3 The significance contradiction is resolved

The new table no longer makes mutually inconsistent statements about Reality-call/ReFeX or introduces an unidentified directed-ReFeX comparison on Brazil.

The visible differences are compatible with the rounded main-table means. I cannot independently check the significance bolding without the underlying paired results. Publishing the corrected intervals alongside the CSV would make those decisions more auditable than significance flags alone.

## 3.4 The pruning frontier is useful—but its strongest result is diagnostic

This is a meaningful improvement:

- exact baselines;
- threshold-only and budgeted settings;
- hub-heavy undirected and directed graphs;
- wiki-RfA;
- process peak memory;
- median and worst-node errors;
- certificate magnitudes.

It demonstrates why reporting only median error would be misleading. It also honestly shows that a small missing-mass budget can cost approximately as much as exact computation, or more.

**What it does not establish is a broadly compelling certified speedup regime.** The budgeted settings are mostly near-exact at near-exact cost. That limits the deployment claim, but does not diminish the value of exposing the trade-off.

### One causal explanation contradicts the implementation

The caption says the budget can be slower:

> “as kept hub rows feed the mixed products.”

They do not. The mixed products are always formed from `H_out` and `H_in`:
```python
T = (first[block] @ second).tocsr()
```
and pruning occurs afterward. Retained rows from the multi-step pure walks do not feed these products.

Budget overhead and retention can increase pure-walk work and feature summarization, but the proposed dependency is absent from the code. Correct this explanation; use profiling if assigning the measured overhead to a specific component.

The frontier caption should also explicitly say **curvature-free**.

## 3.5 Food webs and directed comparisons

The food-web presentation is now appropriately cautious:

- proxy labels are identified;
- model confidence is not mistaken for annotation accuracy;
- model and prompt/label locations are identified;
- “comparable” replaces an unsupported equivalence claim;
- mixed-probe information is disclosed;
- weighted failures remain visible.

This improves transparency, not label validity. The prompt and labels are referenced rather than supplied here, and no independent annotation audit has been added.

The directed CF-isolation limitation also remains correctly important: on food webs, top-\(k\) exceeds RoleWalk **without mixed probes**. The full model’s advantage over that baseline cannot be attributed solely to the CF summary.

Add the same explicit top-\(k\) probe specification to the **wiki-RfA caption**, where it is still missing.

---

# 4. Updated top-venue reviewer report

## Summary

The paper presents a deterministic, training-free structural node representation based on short random-walk probability summaries, distance-layer masses, optional overlap-curvature features, and mixed-direction probes. Its contribution is a useful representation and implementation, supported by a substantial characterization of locality, stability, expressiveness, failure classes, and pruning error.

The latest revision strengthens the central empirical claim through parameter-shift transfer experiments. These show that the probability-summary advantage is not restricted to changing graph size, while also identifying a clear boundary: hand-crafted local statistics outperform RoleWalk overall under the tested parameter shifts.

## Strengths

1. **Transfer evidence now extends meaningfully beyond size.**  
   The matched top-\(k\) comparison and closeness/harmonic results support a substantive representation advantage.

2. **Theoretical characterization is unusually useful for an applied embedding method.**  
   It includes limitations and collisions, not only favorable witnesses.

3. **The implementation is efficient and increasingly usable.**  
   The safety repair, saved-walk semantics, missing-mass API, and pruning frontier are substantive improvements.

4. **The comparisons include strong simple alternatives.**  
   Local statistics and dimension-matched summaries prevent the evaluation from relying only on expensive or weak legacy baselines.

5. **Negative results sharpen the contribution.**  
   Parameter-shift failures, food-web weighting failures, and near-exact budget costs are reported rather than hidden.

## Weaknesses

1. **Conceptual novelty remains moderate.**  
   The CF construction is closely related to GraphWave; much of the contribution lies in the representation choices, characterization, engineering, and evaluation.

2. **The superiority claim is task-specific.**  
   Local statistics win shifted-transfer averages; frozen ReFeX remains statistically competitive there.

3. **Real directed-role evidence is still limited.**  
   Wiki-RfA is positive, while food webs use proxy labels and do not show broad superiority.

4. **Certified pruning is currently more compelling as an error-control interface than as a demonstrated performance breakthrough.**

5. **A few artifact and reporting repairs remain.**  
   In particular, the two finite-weight numerical cases should be fixed before publication.

6. **The GNN result remains preliminary.**  
   It should not carry the paper’s acceptance case.

## Recommendation

**8/10 — accept. Confidence: 3/5.**

The new transfer study satisfies the route toward 8 identified in the previous report. I would not keep the paper at 7 by retroactively demanding cross-family transfer, food-web superiority, or a win over local statistics on every target.

The remaining numerical and reporting corrections are bounded. They should be completed, but they do not erase the strengthened scientific evidence.

---

# 5. What would still move the score?

## To maintain this recommendation

Finish the localized repairs:

- direct division for row-max scaling;
- support-first symmetrization;
- precise shift-table aggregation and exclusions;
- correction of the mixed-product runtime explanation;
- tighter scope around ReFeX, within-family transfer, and closeness/harmonic;
- remaining small API/documentation qualifications.

**These close the present contribution; they do not independently earn another score increase.**

## Toward a higher score

A further increase would require a materially stronger result, not another repair round. Plausible routes include:

- **Reliable real-role utility:** independently validated labels and clear gains beyond strong directed/local-statistics baselines.
- **A genuinely useful certified operating regime:** materially lower runtime or memory while retaining small worst-node error, rather than essentially exact cost.
- **Stronger GNN evidence**, only if that remains a headline, with the previously discussed simple-feature and capacity controls.

None is required for my current acceptance recommendation. Independent reproduction of the present experiments would primarily raise **confidence**, not automatically the score.

**My direct assessment: the paper has now earned the increase to 8. Finish the technical closure and sharpen the narrative; do not keep expanding the benchmark suite merely to make every comparison a win.**