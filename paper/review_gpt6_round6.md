## Bottom line

**8/10 — accept, more firmly than in Round 5. Confidence: 3/5 overall; about 4/5 on the mathematical and static-code assessment.**

The two specific weight-handling bugs are fixed. The step-2 triangle result is correct. The new attribution and clustering-excluded analyses make the transfer evidence more interpretable, and the million-node experiment materially strengthens the practical case for certified pruning.

I would **not automatically move to 9**. The food-web additions establish robustness to several proxy labellings, not ecological ground truth. The scale experiment establishes an affordable certified setting, but does not yet establish a substantial acceleration over exact computation at that scale. These are genuine improvements—not reasons to demand another research programme before acceptance.

There are still a few bounded repairs: incomplete parameter validation, an important walk-length ambiguity in the new attribution experiment, and headline language that has not inherited the qualifications in the body.

### Scope

I reviewed the supplied manuscript and source statically. I have not executed the library, compiled the paper, or inspected the referenced tests, CSVs, benchmark scripts, labels, SVG, or bibliography. In particular, the newly reported test results and intervals are not independently reproduced here.

---

# 1. Mathematical re-check

## The new triangle proposition is correct

For an unweighted \(d\)-regular graph,
\[
H^2=\frac{I+2A+A^2}{(d+1)^2}.
\]
Summing over neighbours of \(u\),
\[
s_u^2(1)
=\frac{2d+\sum_{j\sim u}(A^2)_{uj}}{(d+1)^2}
=\frac{2d+2t_u}{(d+1)^2}.
\]
Each triangle at \(u\) contributes two length-2 walks ending at a neighbour.

Together with
\[
d_u=\frac1{s_u^1(0)}-1,
\]
this recovers
\[
t_u=\frac{(d+1)^2s_u^2(1)-2d}{2}.
\]

Thus **two steps suffice with layers**, including for \(2C_3\) versus \(C_6\). The three-step return formula remains correct. The introduction can now state the stronger two-step result; its current three-step statement is merely weaker, not false.

## Positive-weight support is now explicit

The definition of degree and the theoretical convention that distances use positive-weight edges close the zero-weight ambiguity. The locality proof continues to work: normalising a transition from distance at most \(w-1\) only requires edges contained in the stated ball.

One small notation repair remains: explicitly put **\(u\ne v\)** on the displayed off-diagonal formula for \(H_{uv}\). Otherwise it appears alongside the diagonal formula without distinguishing their domains.

## The certificate remains consistent with the implementation

The coefficients remain correct:

- pure CF block: squared contribution at most \(4\|\Theta\|_2^2r^2\);
- layers: at most \(\|\Theta\|_2^2r^2\);
- mixed CF block: at most \(4\|\Theta\|_2^2r^2\).

The new million-node certificate magnitudes are plausible. For an undirected, three-step walk, the first step is unpruned, so in exact arithmetic a 1% budget gives the sharper bound
\[
B_u\le \sqrt{10}\,\|\Theta\|_2(0.01)\approx 3.95,
\]
rather than the generic bound that allows missing mass at all three steps. Reporting every certificate below 3.9 is consistent with this.

The exact-arithmetic qualification remains essential. These are pruning certificates, **not verified floating-point enclosures or downstream-accuracy guarantees**. The manuscript now maintains that distinction.

I find no new substantive problem in the revised mathematical arguments.

---

# 2. Library re-check

## The two requested weight-handling fixes are genuine

### Direct row-max division

The revised code:
```python
top = np.maximum.reduceat(A.data, A.indptr[:-1][count > 0])
A.data = A.data / np.repeat(top, count[count > 0])
```
does not form a reciprocal. On the nonnegative input domain, each quotient is at most one, including when the maximum is subnormal.

The nonempty-row indexing is also appropriate: removing empty rows from the reduction starts does not mix their neighbours’ data segments.

This closes the specific \(10^{-320}\) reciprocal-overflow bug. The reported unit-walk tests at weights \(1\), \(10^{300}\), \(10^{-320}\), and \(2^{62}\) are the right regressions, although the tests themselves were not supplied.

### Support-first symmetrisation

Both curvature and subset reachability now binarise before adding the transpose. Their intermediate values are only zero, one, or two, so the original integer-weight overflow mechanism is gone.

Using `!= 0` rather than `> 0` is equivalent on the validated nonnegative domain.

**These bugs should be marked closed, not carried forward as unresolved concerns.**

## Parameter validation is improved, but not yet complete

Two concrete issues remain.

### 2.1 Accepted integer-valued floats are not consistently converted

`walk_len` and `n_samples` are converted to integers. `n_jobs` and `embedding_dim` are not.

For example:

- `n_jobs=2.0` passes validation but can reach `numba.set_num_threads` as a float;
- `embedding_dim=10.0` passes validation but is subsequently used as a NumPy array dimension.

Convert these after validation, just as for the other integer parameters. Alternatively, reject non-integer types consistently.

### 2.2 Finite float32 frequencies do not imply a finite computed scale

This check:
```python
np.isfinite(theta.astype(np.float32)).all()
```
does not protect:
```python
np.linalg.norm(self.theta)
```
from overflow in float32 accumulation.

For example, `bounds=(1e20, 1e21)` passes the frequency check, but squaring those float32 frequencies overflows:

- the sparse backend can obtain an infinite layer scale and return nonfinite features;
- `certificate_` can become infinite or NaN, including `inf * 0` when the missing mass is zero;
- the fused backend computes its internal scale in float64, so the backends also disagree in this regime.

Conversely, sufficiently small positive bounds can cast to an all-zero float32 grid while passing the finiteness check.

**Repair:** validate the actual stored grid for positivity and finiteness, compute its norm in float64 everywhere, and check that the scale/output range is supported by the advertised float32 representation.

These are unusual parameter choices, not evidence that the default experiments are invalid. But they prevent an unrestricted numerical-robustness sign-off.

## Small documentation/semantic leftovers

- The fused wrapper now correctly describes an **absolute** Taylor error, but the kernel’s internal comment still says “below 3e-12 of the term.” Remove that stale relative-error wording.
- `role_summary` still constructs its diagnostics with `weight=None`, so a stored zero-weight edge contributes to its degree. That is defensible as a separately documented unweighted diagnostic, but it is not the positive-weight degree used by the embedding. Either align it or explicitly distinguish the conventions.

The saved-walk semantics, accumulator membership marker, certificate block accounting, and mixed-product construction remain consistent with the previous audit.

---

# 3. Experimental and reporting consistency

## 3.1 The component attribution is valuable—but its walk length needs clarification

The new results substantially improve the scientific story:

- layers alone: 0.807;
- CF alone: 0.785;
- full RoleWalk: 0.813;
- full minus layers: \(+0.007\,[0.004,0.009]\);
- full minus CF: \(+0.029\,[0.024,0.035]\).

The appropriate interpretation is now:

> **The distance law carries most of the size-transfer performance; the CF contributes a smaller, consistent complementary gain.**

This is more informative than attributing the full transfer result to the CF construction. Layers alone also exceed top-\(k\) plus layers, so the CF–top-\(k\) gap combines a beneficial CF addition with an apparently unhelpful top-\(k\) addition.

There is, however, a concrete settings ambiguity:

- the method defaults to \(L=3\), giving **9 layer features**;
- the attribution paragraph reports **14 layer features**, which implies \(L=4\);
- its matched lazy RWSE also explicitly uses four steps.

Please state the walk length for **every transfer representation**.

- If the main transfer experiments and attribution all use four steps, this is a documentation repair.
- If the attribution uses four steps while its full-model comparator uses three, it is not a controlled component attribution and should be matched.

The 0.753 lazy-RWSE control is also much stronger than the 0.532 headline RWSE row. That is a useful finding: walk choice matters, not just the representation of a fixed walk. Make both walk and horizon differences explicit rather than leaving readers to interpret that gap.

## 3.2 Keeping clustering ranked is fair with the new disclosure

I agree with the authors’ treatment. Local statistics and ReFeX need not be removed from the clustering ranking simply because they contain that coefficient. The dagger explains what the comparison measures, and the target-excluded analysis supplies the more informative aggregate.

The excluded-target means are compatible with the displayed rows, allowing for rounding. Their interpretation is important:

- **Size transfer:** RoleWalk retains a clear advantage over local statistics without clustering.
- **Parameter shift:** RoleWalk and local statistics are not distinguished by the reported interval once clustering is removed:
  \[
  -0.008\,[-0.033,0.017].
  \]
- The clustering-excluded shifted comparison with frozen ReFeX is positive:
  \[
  +0.049\,[0.020,0.078].
  \]

Thus the earlier “local statistics win shifted transfer overall” remains true for the full target set, but is now properly explained rather than treated as a universal representation ranking.

The valid-cell versus pair-level aggregation distinction and the exclusion counts are substantially repaired.

## 3.3 The claim-scoping repair did not reach all of the paper

The parameter-shift section is appropriately qualified. The introduction and limitations still contain the old language:

> “it stays ahead of every embedding”

and

> “RoleWalk still leads every embedding and the distance-based centralities.”

Replace these with the body’s actual conclusions:

- highest aggregate mean among the embedding baselines;
- no established full-target aggregate advantage over frozen ReFeX;
- strongest shifted-transfer results specifically on **closeness and harmonic centrality**.

Likewise, “clearly ahead on one of six networks” is stronger than the role-classification significance table supports: on USA, the \(+0.013\) comparison with RESD is not marked significant. “Highest mean on one of six networks” is safer.

These are headline consistency repairs, not requests to weaken the supported findings.

## 3.4 Million-node certification is a meaningful advance

This changes my assessment of the deployment evidence.

On the reported million-node BA graph:

- threshold-only: 173 s;
- 1% budget: 227 s;
- ratio: approximately \(1.31\times\);
- threshold-only permits substantial missing mass on many rows;
- the budget gives uniformly small absolute pruning certificates.

That is a **useful certified operating setting at realistic scale**. It is no longer fair to describe the certificate primarily as an interface with little demonstrated practical operating value.

The remaining limitation is narrower: without a million-node exact runtime/memory reference, this does not establish a substantial **pruning-driven speedup over exact computation**. It establishes affordable error control relative to the default approximate setting. Both are worthwhile, but they are different claims.

Two reporting details should be aligned:

- The abstract’s “20 s to 2.5 min” does not cover the new 173-second threshold-only BA run. Use approximately three minutes, or identify the differing settings/runs.
- Specify whether the reported 1.7 GB is process peak including the graph, as in the frontier table.

The corrected mixed-product explanation is now accurate and appropriately avoids assigning unprofiled overhead to a specific component.

## 3.5 Food-web labels: improved robustness evidence, not independent validation

The three labellings are useful. High agreement, majority-label results, and unanimous-subset results show that the broad conclusions do not depend entirely on one generated label set.

They do **not** establish label accuracy:

- two labellings use the same underlying model at different effort settings;
- different models can share systematic ecological misconceptions;
- unanimity selects a potentially easier, less ambiguous population.

I would therefore describe these as **separately generated proxy labellings**, not statistically independent validation.

The reported competitive conclusion is fair: RoleWalk, local statistics, and Digraphwave are not statistically distinguished. That does not establish equivalence, but the paper does not need an equivalence claim.

One wording adjustment: the mixed-probe gain has positive point estimates under all three label choices, but the majority-label interval includes zero. Say:

> “The estimated mixed-probe gain is positive under all three labellings, with intervals excluding zero for the original and unanimous-subset evaluations.”

That preserves the useful result without implying three independently established gains.

I would not require an expert audit to retain acceptance. I would also not treat LLM agreement as having closed the real-label-validity limitation.

## 3.6 Wiki-RfA is better supported, with two remaining qualifications

The reported corrected paired results strengthen the main real directed-role evidence:

- task A: reported advantages over all comparators;
- task B accuracy: highest mean, but not distinguished from local statistics.

I cannot verify the intervals against every comparator because `paired_rfa.csv` was referenced rather than supplied.

Also:

1. The wiki-RfA caption still does **not explicitly state whether top-\(k\) includes mixed probes**. Add the same disclosure used for directed motifs and food webs.
2. State that Nadeau–Bengio corrects for overlapping splits, and separately state whether the comparisons are multiplicity-adjusted. Do not let “corrected” imply both.

---

# 4. Updated top-venue reviewer report

## Summary

RoleWalk is a deterministic, training-free structural representation combining summaries of short-walk transition probabilities, distance-layer masses, optional overlap curvature, and mixed-direction probes. Its contribution is a coherent representation and efficient implementation, accompanied by useful theoretical characterisation and a substantial empirical evaluation.

The latest revision strengthens both attribution and deployment evidence. It shows that distance-layer masses explain most of the centrality-transfer performance, with a smaller complementary CF gain, and demonstrates million-node computation with uniformly controlled pruning error at modest overhead over threshold-only pruning.

## Strengths

1. **Useful, unusually complete characterisation.**  
   Locality, stability scope, expressiveness witnesses, failure classes, and pruning guarantees give a clear account of what the method can and cannot do.

2. **Substantive transfer evidence with strong simple controls.**  
   The matched summary comparison survives within-family parameter shifts. The new target-excluded and component analyses sharpen rather than obscure the conclusions.

3. **A practical certified setting at scale.**  
   Million-node processing with a 1% missing-mass budget is meaningful engineering evidence, not merely an asymptotic claim.

4. **Native directed modelling with identifiable contributions.**  
   Mixed probes have a clear mathematical rationale and useful empirical effects.

5. **Honest negative and competitive results.**  
   Local-statistics strengths, food-web ties, harmful flow weighting, and limitations of pruning speedups are visible.

## Weaknesses

1. **Conceptual novelty is moderate.**  
   The CF relationship to GraphWave is close; much of the contribution lies in the combined representation, analysis, implementation, and evaluation.

2. **Benefits are task- and component-specific.**  
   Layers dominate the strongest transfer result, and strong handcrafted features remain competitive.

3. **Real directed-role validation remains concentrated.**  
   Wiki-RfA is positive; food webs remain proxy-labelled and do not establish superiority.

4. **Certified scalability is demonstrated more strongly than certified acceleration.**

5. **Some bounded artifact and reporting issues remain.**  
   Most importantly, the attribution horizon must be explicit and validation must cover the actual numerical operations.

6. **The GNN result remains preliminary.**  
   It should remain a secondary observation rather than a pillar of acceptance.

## Recommendation

**8/10 — accept. Confidence: 3/5 overall.**

This is a stronger acceptance case than Round 5. The unchanged integer score should not be read as “the revision accomplished nothing.” It closed real defects, clarified what produces the gains, and demonstrated a practically useful certified setting.

I would not withhold acceptance pending expert food-web annotation, cross-family transfer, universal baseline wins, or a larger GNN study.

---

# 5. What would still move the score?

## To close the current paper

The remaining work is bounded:

- finish integer coercion and frequency/scale validation;
- clarify and, if necessary, match the attribution walk lengths;
- propagate the scoped claims into the introduction and limitations;
- disclose the wiki-RfA top-\(k\) probe configuration;
- align runtime and memory definitions.

These repairs protect the current recommendation. They are not another demand for a score-earning benchmark suite.

## Toward 9

A materially stronger result could still move me:

- **Validated real-role utility:** independent domain validation, coupled with a convincing accuracy–cost or transfer advantage over strong directed/simple baselines.
- **A stronger certified performance frontier:** an exact reference, or a clearly documented exact resource limit, showing that small worst-node error enables a substantial runtime or memory advantage on large graphs.

The new million-node result meaningfully advances the second route. It does not yet answer the exact-versus-budgeted comparison.

An expert audit alone or independent reproduction would primarily raise **confidence**. More LLM agreement alone would not raise the score. A larger GNN study is relevant only if that remains a headline claim.

**My direct advice: close the remaining implementation and presentation details and submit. The paper has an acceptance-level contribution now; it does not need to make every dataset or comparison a win.**