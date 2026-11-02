## Bottom line

**I would now give this 7/10: accept, not strong accept.** The reason for the increase is the strengthened transfer experiment—not the volume of revisions. The reported graph-level gains, particularly on closeness and harmonic centrality, substantially satisfy the main route to 7 identified in round 3.

The food-web experiment is also useful, despite not showing superiority: it broadens the evidence, tests meaningful weights, and exposes an important limitation. It should remain in the paper.

**However, this is still not a clean mathematical or implementation bill of health.** Several repairs are genuinely complete; a few are incomplete in the supplied manuscript, and the weighted-walk change exposes additional implementation issues. The most important outstanding items are:

1. The displayed weighted-walk definition still describes the old walk.
2. The Fourier paragraph still needs an explicit unitary normalization.
3. The manuscript’s curvature-complexity proposition remains broader than the implementation.
4. The new centrality exclusion count cannot be explained by the stated family alone.
5. The paired-significance paragraph still contradicts itself.
6. The fused accumulator has an unsafe zero/underflow corner case.
7. Loading older saved weighted models silently changes their feature semantics.

These are bounded repairs, not reasons to demand another wholesale experimental round.

### Scope

This is a **mathematical and static-source audit** of the supplied text and library. I have not executed the code or inspected the benchmark scripts, CSVs, tests, SVG, or `refs.bib`. Thus I can check the visible arithmetic, experimental descriptions, and implementation logic, but not independently reproduce the reported intervals, timings, labels, or backend-equality test.

---

# 1. Mathematical audit

## 1.1 The weighted walk is correctly changed in the library, but not in the displayed definition

The method begins:

> “\(\widetilde D=\operatorname{diag}((I+A)\mathbf1)\) and … \(H=\widetilde D^{-1}(I+A)\).”

It then says:

> “On weighted graphs the self-loop of \(u\) weighs the mean \(s_u/d_u\) of its edge weights…”

Those are different definitions. The first formula still assigns weight one to every added self-loop.

For loopless graphs, the revised definition should be written explicitly as
\[
b_u=
\begin{cases}
s_u/d_u,&d_u>0,\\
1,&d_u=0,
\end{cases}
\qquad
B=\operatorname{diag}(b_u),
\]
\[
H=\operatorname{diag}((A+B)\mathbf1)^{-1}(A+B).
\]

Then, for a nonisolated node,
\[
H_{uu}=\frac1{d_u+1},\qquad
H_{uv}=\frac{d_u}{d_u+1}\frac{A_{uv}}{s_u}.
\]

This is what `lazy_walk` intends to compute. It is invariant under positive global rescaling of the weights. Each individual out-walk is also invariant under rescaling an entire outgoing row; that latter invariance does **not** generally extend to the combined out/in/mixed representation.

A minimal discrepancy: with one outgoing edge of weight two, the displayed formula gives self-loop probability \(1/3\); the revised implementation gives \(1/2\).

The locality proof also retains the old expression:

> “\(H_{xy}=(I+A)_{xy}/\widetilde D_{xx}\)…”

Replace it with the revised expression. **The locality conclusion survives:** both \(s_x\) and \(d_x\) are determined by the incident edges already included in the stated ball.

### Explicitly scope the regular-graph results to unweighted graphs

The distance-regular proposition says:

> “Then all nodes of \(G\) and \(G'\) get the same embedding…”

Its proof requires an unweighted adjacency matrix. Under the section’s general weighted-graph conventions, this needs an explicit qualification.

For example, a triangle with edge weights \(1,1,2\) has distance-regular support, but the revised walk has first-step probability multisets
\[
(1/3,1/3,1/3)
\quad\text{and}\quad
(1/3,2/9,4/9)
\]
at different vertices. The claimed uniformity therefore fails if arbitrary weights are admitted.

Likewise, the RWSE proposition’s \(P=A/d\) requires the stated unweighted interpretation. Add “simple, undirected, unweighted” to these propositions rather than relying on readers to infer it from their proofs.

---

## 1.2 Locality: the changed radii are correct

> “\(r=L\) for pure walks, \(r=\max(L,2)\) … with mixed probes, and \(r=L+2\) with curvature.”

This is now correct, with weak/undirected distance used where stated.

The additional proof sentence is important and correct:

> “including the normalisation of the second step at \(x\).”

The weak two-ball contains all edges needed to normalize the second mixed step. Curvature at a landing vertex within distance \(L\) needs the degrees and common-neighbor information of its neighbors, contained within radius \(L+2\).

The revised transfer interpretation is also appropriate:

> “whether it succeeds is an empirical question.”

That resolves the previous conflation of coordinate compatibility with predictive generalization.

**Verdict:** repaired, apart from updating the weighted transition formula used in the proof.

---

## 1.3 Footprint kernel: repaired; Fourier normalization still needs one explicit factor

The directed kernel now includes block marks and excludes the layer term for mixed probes:

> “\(\mathbf1_{\beta\text{ pure}}\|\Theta\|^2ab\,\delta_{\ell\ell'}\)”

This correctly represents the actual curvature-free default embedding. The direct-sum construction is positive semidefinite, and the finite-measure MMD identity is valid.

The Taylor bound remains correct:
\[
\left|k_\Theta(a,b)-\|\Theta\|^2ab\right|
\le
\|\Theta\|^2ab\,t^2
\left(\frac7{12}+\frac{t^2}{36}\right).
\]

Two qualifications remain appropriate:

- The induced distance is an MMD **pseudometric** on arbitrary footprints; the particular finite frequency grid is not proved characteristic.
- “The only factor” should mean the **only positive factor**, under the specified unweighted frequency inner product and orthonormal layer coordinates.

### The centered-transform interpretation is now right

The identity
\[
-i\,\partial_\theta F_u^w(0,\omega)
=
\sum_k s_u^w(k)e^{i\omega k}
\]
is correct. Calling \(F\) a centered, unnormalized Fourier transform rather than a probability characteristic function resolves the substantive issue.

### But sampled derivative values are not themselves a unitary DFT

The text says:

> “whose values at \(\omega=2\pi k/(w+1)\) (unitary DFT) determine \(s_u^w\) with the same Euclidean geometry.”

Let \(N=w+1\). The displayed derivative gives the **unnormalized** DFT values
\[
g_j=\sum_{k=0}^{N-1}s_k e^{2\pi ijk/N},
\qquad
\sum_j|g_j|^2=N\sum_k|s_k|^2.
\]

Write explicitly
\[
\widetilde g_j=
\frac{-i}{\sqrt N}
\partial_\theta F_u^w(0,2\pi j/N).
\]
Then \(\|\widetilde g\|_2=\|s\|_2\). Merely inserting “unitary DFT” parenthetically does not normalize the displayed values.

This is a small mathematical repair, but it is precisely the normalization issue raised last round.

---

## 1.4 Pruned layers: the revised implementation logic is sound

The documented convention is now clear:

> “the step at which each node is first reached … read before pruning.”

The sparse backend now unions the newly observed support with the previous cumulative support:
```python
S = S + supports[-1]
S.data[:] = 1
```

Consequently, its nested supports encode the same first-arrival partition used by the fused kernel. Taking successive differences of masses on those nested sets gives a genuine partition, rather than the potentially negative quantities produced by the previous implementation.

The exact-distance claim through three steps is also correct:

- Step one is unpruned.
- The pre-pruning step-two support is exact.
- Any newly observed vertex at step three outside those supports has distance three.

Beyond three steps, the labels can overestimate true distance, as now stated.

**Verdict:** the semantic repair checks out statically. I cannot independently verify the reported five-step test, but the code now implements the claimed convention.

One reporting detail remains: explicitly put **`tol=0` in the expressiveness and BREC captions**. The sentence restricting results in the theory section does not clearly specify the experimental tables outside that section.

---

## 1.5 Cost: retained-support bound repaired; curvature scope still missing from the paper

The corrected support statement is right:

> “\(q_u^t\) … for \(t\ge2\) … first step … budget can keep any number.”

State explicitly that the bound assumes **\(\mathrm{tol}>0\)**. With that condition,
\[
|\operatorname{supp}q_u^t|
\le
\min\{|B_t(u)|,\lfloor1/\mathrm{tol}\rfloor\}
\]
is valid in threshold-only mode.

It is also now correctly distinguished from the pre-pruning accumulator.

### Full and subset Numba curvature paths

Both changed implementations match their intended algorithms:

- The full-graph kernel scans forward adjacency under degree ordering.
- The subset kernel scans the shorter neighbor list and binary-searches the longer sorted list.

Thus the stated enumeration and per-edge subset complexities are defensible, including the separate setup costs.

### The manuscript still states an unqualified curvature cost

The proposition says:

> “curvature costs a triangle enumeration, \(O(|E|^{3/2})\)…”

The library correctly documents:

> “Without numba they come from one sparse product … quadratic on a star.”

But that qualification is absent from the manuscript’s proposition. The fallback still executes:
```python
common = (rows @ U).multiply(rows)
```

So the earlier issue is **fixed in the implementation documentation, not fully fixed in the paper**. Add the Numba/full-graph/subset/fallback qualification directly to the proposition or its immediately attached implementation note.

---

## 1.6 Mixed-product blocking: the substantive estimate is fixed, but it is not a hard memory cap

The estimate now uses degrees of the second matrix:
```python
work = (B @ np.diff(second.indptr).astype(np.float64))[rows]
```

That is the right work quantity for `first[block] @ second`. It fixes the directed in-star counterexample from round 3.

However:

> “blocks of rows of bounded work”

and

> “which bounds memory on large graphs”

still need qualification. A single row can exceed `block_work`; the implementation cannot divide that row into smaller products. The block bound is therefore of the order of the target budget **plus the largest individual row’s work**, not a strict `block_work` cap.

Also, in the sparse pure-walk fallback, estimating the first two steps does not bound the size of all later powers.

These are remaining documentation limits, not a recurrence of the original wrong-matrix estimate.

---

## 1.7 Pruning certificate: mathematically correct, with appropriate but incomplete numerical certification

The pruning proof and constants remain sound:
\[
\|X_u-\widehat X_u\|_2
\le
\sqrt5\,\|\Theta\|_2
\left(\sum_w(r_u^w)^2\right)^{1/2},
\]
with the stated additions for directions and mixed probes.

The revised experiment now correctly identifies the norm:

> “worst absolute \(\ell_1\) error of the unscaled layers at each step”

and excludes curvature from the certificate comparison.

The returned-array statement
\[
\|X-\widehat X^{\rm returned}\|
\le B_{\rm prune}+\varepsilon_{\rm Taylor}+\varepsilon_{\rm float}
\]
is also correct error accounting.

**But it is not yet an executable numerical certificate for the returned array.** The library neither returns these terms nor supplies a computed floating-point bound. In particular, `fastmath=True`, probability propagation, feature accumulation, and trigonometric evaluation all contribute to numerical error.

That does not invalidate the pruning theorem. Describe the current result as an **exact-arithmetic pruning certificate with separately acknowledged evaluation errors**, rather than implying that `transform` returns numerically certified vectors.

---

## 1.8 Other theory

I find no new algebraic error in the following, under their intended assumptions:

- Injectivity over an accumulating frequency set.
- Almost-everywhere separation of a countable family.
- Edge-insertion stability.
- The \(K_{2,d}\) layer-instability example.
- Degree and triangle identities.
- The mixed-probe witness and its displayed probabilities.
- The node-level 1-WL collisions.
- The pendant-cycle graph-level collision.
- GraphWave’s centering/rescaling identity.
- The overlap-curvature formula and Ollivier upper bound.

The removal of the ambiguous higher-WL parenthetical is appropriate.

---

# 2. Remaining library issues

## 2.1 Important: the fused accumulator can overrun its workspace on zero/underflow contributions

The kernel uses a zero accumulator as its “not yet touched” marker:
```python
if acc[k] == 0.0:
    touched[n_touch] = k
    n_touch += 1
acc[k] += p * data[e]
```

This assumes that every encountered contribution makes `acc[k]` strictly positive. That is not guaranteed by finite nonnegative input weights:

- Normalization can leave explicitly stored zero entries after underflow.
- A positive `p * data[e]` can itself underflow to zero.

Then the same index can be appended repeatedly while its accumulator remains zero. `touched` has length \(n\), so this can exceed its allocation. With ordinary Numba bounds checking disabled, that is a memory-safety problem, not merely a small numerical discrepancy.

### Static counterexample

Consider an undirected weighted triangle with
\[
A_{01}=10^{300},\qquad A_{02}=A_{12}=10^{-300}.
\]

All supplied weights are finite and positive. In normalized rows 0 and 1, the transition to vertex 2 underflows to zero. `lazy_walk` does not eliminate zeros after normalization.

During the second propagation from root 0, vertex 2 can be appended to `touched` once through row 0 and again through row 1 while `acc[2]` remains zero. The other two vertices have already been appended, so the fourth append exceeds the length-three buffer.

I have not executed this example; it follows directly from the accumulator logic and should be added as a regression test with bounds checking enabled.

**Repair:**

- Eliminate explicit zeros after normalization.
- Skip zero propagation contributions before touching the accumulator, or use a separate touched marker independent of numerical values.
- Test later-step underflow as well as explicitly stored zeros.

This does not implicate the reported 19-order-of-magnitude food-web range, but it does matter for the library’s advertised finite-weight domain.

---

## 2.2 Weighted normalization can overflow despite finite inputs

```python
strength = np.asarray(A.sum(axis=1)).ravel()
```

can overflow when a row contains several large finite weights. The subsequent mean self-loop can become infinite.

For a robust unit-invariant implementation, normalize rows using a row maximum or another stable scaling before computing the transition proportions. This avoids relying on the unscaled strength being representable.

Again, distinguish the exact-arithmetic invariance theorem from unrestricted floating-point robustness.

---

## 2.3 Existing self-loops are accepted, but violate the advertised stay probability

`transform` does not remove or reject original self-loops. `lazy_walk` counts them among the stored entries and then adds another diagonal weight.

Therefore:

> “it stays put with probability `1 / (d + 1)`”

is not generally true for accepted graphs containing self-loops.

Choose and document a policy:

- reject original self-loops;
- remove them before applying the specified walk; or
- retain them, but explicitly define the different transition rule.

Also report the self-loop preprocessing for food webs; within-compartment feeding can plausibly produce them. I am not asserting that the supplied dataset contains such edges—its preprocessing script was not provided.

---

## 2.4 Saved weighted models silently change semantics

`load` handles historical defaults for shells, curvature, and mixed probes, but there is no saved walk-definition/version field.

An older weighted model’s cluster centers were fitted using unit self-loop weights. Loading it now computes new features using mean-edge self-loop weights while retaining the old centers.

The fact that earlier **paper benchmarks** were unweighted does not protect previously saved **library models**.

Store a walk-definition version and either:

- preserve the legacy rule when loading old models; or
- explicitly reject/warn about incompatible weighted-model reuse.

---

## 2.5 Smaller documentation and helper issues

These are not acceptance-level issues, but they remain incorrect or underspecified.

### Kernel docstring omits the `shells` condition

The class docstring includes the layer kernel whenever aggregation is `"sum"` and curvature/pruning/projection are absent. But users can also set `shells=False`.

Include an indicator for `shells`, or state that the displayed kernel assumes `shells=True`.

### `compute_iter` omits centering in its verbal identity

> “the characteristic function over all \(n\) nodes … rescaled by \(n\)”

The actual identity is \(n(\chi-1)\), not \(n\chi\). Say “centered and rescaled.”

### Parameter validation is improved, not comprehensive

The new checks reject negative values and NaNs for `tol` and `max_missing`. They do not validate the other parameter domains, including positive finite `block_work`, positive integer step/sample counts, and sensible finite frequency bounds.

Values of `max_missing` above one are mathematically harmless but vacuous; either document this or restrict it to \([0,1]\).

### Directed `role_summary` uses symmetrized links

`_node_stats` symmetrizes a directed adjacency and returns it; `role_summary` then uses that matrix for `links_to`.

Thus:

> “where the role’s edges go”

actually describes undirected incidence, not outgoing links. Either label it accordingly or retain a directed adjacency for the mixing calculation.

---

# 3. Text, tables, and experimental consistency

## 3.1 The new transfer result is materially stronger

The key reported comparison is now:

> “top-\(k\) + layers \(+0.013\) \([0.009,0.018]\), 77%”

with 60 independent graph pairs, and approximately \(+0.04\) on closeness/harmonic centrality.

This is much better evidence than the previous aggregate advantage over two seeds:

- uncertainty is calculated over graphs rather than overlapping node splits;
- a dimension-matched comparator is included;
- frozen ReFeX semantics are retained;
- the gain is not presented as universal across targets;
- the per-family summary suggests it is not driven entirely by one family.

**This substantially meets route 1 from the previous report.**

The supplied manuscript gives a **prose per-family summary**, not the numerical per-family results. Put the small numerical table or a forest plot in the appendix. That is a presentation/reproducibility repair, not a demand for another experiment.

---

## 3.2 The centrality exclusion count still needs correction

The caption says:

> “20 of the 420 graph–target cells have a constant target (\(k\)-core on Barabási-Albert) and are left out.”

There are only ten Barabási–Albert test graphs in the stated design. One \(k\)-core target per graph accounts for **ten**, not twenty, test cells.

The displayed means are consistent with twenty missing \(k\)-core cells. For RoleWalk:
\[
\frac{
60(0.864+0.797+0.861+0.989+0.805+0.673)
+40(0.651)
}{400}
=0.81345,
\]
which rounds to the reported \(0.813\).

So this is not evidence that the mean is arithmetically wrong. It indicates that the explanation of the excluded cells is incomplete—possibly another family also has constant \(k\)-core, but the supplied text does not establish which.

Specify the exact excluded families/cells. Do not count both training and test targets as two exclusions from a denominator defined over test graph–target cells.

### Also state the aggregation weights

The table appears to average valid graph–target cells, whereas the paired intervals average targets within each graph and then average graphs. Those are different weightings when graphs have different numbers of valid targets.

That can legitimately explain why the paired mean difference is not exactly the difference between displayed table means. State it explicitly, or use one common estimand.

The “25 of 80 scored pairs” Digraphwave denominator is internally plausible under four exclusions from the old 84 cases, but those exclusions should be identified consistently too.

---

## 3.3 The paired-significance paragraph still contradicts itself

It says:

> “on Reality-call, ReFeX and RWSE”

are significantly worse, then says:

> “against GraphWave, the heat-kernel CF, RESD, ReFeX or degree the intervals include 0”

over the same listed datasets.

Both cannot hold for the Reality-call/ReFeX comparison.

It also refers to:

> “on Brazil … directed ReFeX”

but the Brazil table contains “ReFeX,” not a separately identified directed ReFeX row.

Replace this long paragraph with a small table of the actual relevant paired differences and corrected intervals. This will be clearer and less error-prone. The multiplicity disclaimer and acknowledgment that the splits come from one graph are good additions.

---

## 3.4 Food webs: useful evidence, appropriately negative, but not validated ground truth

The protocol is substantially better than a random split over related webs:

> “seasons and sub-webs of one ecosystem stay together”

is the right grouping decision.

The central conclusions agree with the visible numbers:

- RoleWalk is close to local statistics on the unweighted webs.
- Digraphwave has the highest unweighted mean.
- Mixed probes help RoleWalk.
- Raw flow weights hurt the methods evaluated in both forms.

The main limitations are:

### LLM labels are proxy labels

> “given by an LLM from the compartment names and the ecosystem alone”

avoids direct graph-to-label leakage, but does not establish biological correctness. In particular, “63% high confidence” is model confidence, not measured annotation accuracy.

Provide the model/version, prompt, frozen labels, and preferably a modest expert or source-based audit stratified across guilds and confidence levels. This would strengthen the task’s validity without requiring a new dataset.

### “Ties” is stronger than the interval supports

> “RoleWalk ties the local statistics (\(-0.003\) \([-0.08,0.08]\))”

The point estimates are close, but this is not an equivalence result. The interval permits fairly substantial differences in either direction.

Use “similar mean, with no established difference,” or simply “comparable.”

### Qualify “every walk-based method”

The table does not show unweighted RWSE. Write “every walk-based method evaluated both ways,” unless the missing comparison is supplied.

### Clarify top-\(k\)’s probe information

The synthetic directed caption now explicitly says top-\(k\) excludes mixed probes. That is an honest and useful correction.

The corresponding configurations remain unclear for food webs and wiki-RfA. Without identical probes, a full RoleWalk-versus-top-\(k\) comparison does not isolate the CF summary. This does **not** undermine the undirected matched centrality result, but it limits the directed interpretation.

---

## 3.5 Other visible consistency checks

### Correct or substantially repaired

- Default dimensions: 69/75 undirected and 178/190 directed.
- BREC totals: 207, 213, 214.
- Main-table ranking marks are generally consistent.
- The summaries table now admits its dimension mismatch.
- The geometry table now identifies the no-mixed configuration.
- The RWSE locality explanation has been removed.
- ZINC’s paired interval agrees with the stated mean, standard deviation, and five seeds.
- The two-sided sign-test value is \(0.0625\), appropriately reported as \(0.06\).
- Film is now presented as a hypothesis rather than a demonstrated causal explanation.
- Scalability hardware, compilation exclusion, and single-run status are disclosed.

### Remaining presentation qualifications

- The trajectory paragraph still compares against the old no-mixed directed ARI \(0.366\); label that configuration explicitly.
- “A fraction of the cost of every method but ReFeX” is too broad if it includes the newly evaluated top-\(k\) and local-statistics baselines. On wiki-RfA they take 5 s and 3 s versus RoleWalk’s 7 s. Say “most learned and diffusion baselines.”
- The same-size versus \(4\times\)-size drop now compares a two-seed estimate with a ten-seed estimate. It is a reasonable descriptive comparison, not a matched estimate of the size-change effect.
- The header still says “results and proofs; prose to come,” and the abstract contains “is has.” There are no substantive TODO passages visible, but this is not yet final submission prose.
- I cannot verify the exact references or imported scalability values from the supplied material.

---

# 4. Updated reviewer report

## Summary

The paper proposes a deterministic, training-free structural node representation based on short random-walk probability summaries, distance-layer masses, optional overlap-curvature moments, and mixed-direction probes.

Its contribution is principally a useful, efficient representation with substantial mathematical characterization and increasingly careful empirical evaluation. The central Fourier construction is closely related to GraphWave; the direct-sum kernel interpretation is explanatory rather than a major new kernel principle.

The latest revision materially strengthens the scientific case. Most importantly, the centrality-transfer comparison now uses 60 independent graph pairs, graph-level paired uncertainty, frozen baseline semantics, and a dimension-matched alternative. The new food-web evaluation broadens the directed evidence while openly reporting negative results for weights and lack of superiority over local statistics or Digraphwave.

## Strengths

1. **The transfer advantage is now supported by meaningful replication.**  
   The reported gains over top-\(k\), local statistics, frozen ReFeX, and heat CF are not merely one aggregate result over a handful of graph realizations.

2. **The dimension-matched comparison supports the probability summary itself.**  
   In particular, the closeness/harmonic gains are large enough to be scientifically more interesting than the small overall mean difference alone.

3. **The theory is useful and mostly correct.**  
   Locality, pruning, mixed-direction separation, explicit collisions, stability, and failure classes provide a substantive characterization.

4. **The implementation repairs address real earlier problems.**  
   Forward triangle enumeration, subset intersections, shared pruned-layer semantics, and correct mixed-product work estimates are meaningful improvements.

5. **The negative results improve credibility.**  
   Food-web weights, local-statistics competitiveness, directed ReFeX’s strengths, and the tentative ZINC result are not hidden.

## Weaknesses

1. **Definition and artifact closure are still incomplete.**  
   The weighted transition formula, numerical corner cases, serialization semantics, and several reporting statements need correction.

2. **Novelty is moderate rather than foundational.**  
   Much of the value lies in the combination, engineering, and evaluation.

3. **Transfer is still primarily within synthetic families.**  
   Generalization across sizes is convincing; generalization across substantially different graph-generating mechanisms remains less established.

4. **Real directed-role superiority is not broad.**  
   Wiki-RfA is positive; food webs are informative but do not show superiority, and their labels are weakly validated proxies.

5. **Certified efficiency remains partly a theoretical promise.**  
   There is no returned per-node certificate API or comprehensive accuracy–runtime–peak-memory frontier for difficult directed graphs.

6. **GNN evidence remains secondary.**  
   The ZINC improvement is promising but not statistically decisive and still lacks the relevant simple-feature/capacity controls.

## Score and confidence

**Score: 7/10 — accept.**

**Confidence: 3/5 overall; approximately 4/5 on the mathematical/static-code findings.**

The move to 7 is earned by the strengthened transfer evidence. I would not keep the paper at 6 by demanding that the food-web experiment also produce a win, or by introducing an entirely new mandatory benchmark.

I would require the concrete correctness and reporting repairs above before publication. They are feasible corrections to the present contribution, not a new research program.

---

# 5. What would still move the score—and what would not

## First: close the remaining repairs

These would make the current 7 trustworthy, rather than raise it by themselves:

- Write the correct weighted transition matrix everywhere.
- Make the unweighted scope of the relevant regular-graph results explicit.
- Insert the explicit \(1/\sqrt{w+1}\) Fourier normalization.
- Scope the manuscript’s curvature complexity to the actual backend.
- Fix the fused zero/underflow accumulator issue.
- Define self-loop handling and weighted-model migration.
- Reconcile the excluded centrality cells and averaging weights.
- Correct the significance paragraph.
- Publish the per-family numbers and identify exact configurations.

## The most realistic route toward 8: test transfer beyond size alone

You do **not** need another sixty-pair repetition of the same experiment.

A focused study could freeze the current pipeline and test a small, prespecified set of parameter shifts—for example, mean degree, rewiring probability, community structure, or spatial connectivity—while retaining the same matched baselines.

**What would move the score:** evidence that the CF advantage persists under meaningful shifts where graph size is not the only changing variable, with clear failure cases and graph-level uncertainty.

**What would not:** another pooled mean from more seeds of the identical generators.

## Alternative: strengthen real-role validity and utility

For food webs, the first step is annotation validity, not another classifier:

- independently audit a stratified label sample;
- report sensitivity to uncertain labels;
- make the top-\(k\) probe information identical;
- provide ecosystem-level results.

A clear benefit on reliable labels beyond directed local statistics would be valuable. A reliable negative result would strengthen confidence and scope, but would not by itself make this an 8.

**What would not:** adding another LLM-labelled dataset and treating its labels as unquestioned ground truth.

## Alternative: make certified pruning a deployment contribution

Expose missing masses and pruning bounds through the API, then report an accuracy–runtime–peak-memory frontier on a small set of hub-heavy undirected and directed graphs:

- exact;
- threshold-only;
- budgeted;
- pure versus mixed;
- tail errors and worst rows, not only averages.

**What would move the score:** a demonstrated, useful certified operating regime with transparent memory behavior.

**What would not:** faster numbers obtained by dropping more mass without accounting for the resulting errors.

## GNN work is optional, unless it remains a headline

The useful next comparison is not simply more seeds. It is RWSE plus simple local counts/curvature versus a dimension- and parameter-matched RoleWalk augmentation, after diagnosing the RWSE reproduction.

There is still **no requirement to add Peptides or another large GNN benchmark**.

---

## Explicitly, what will not move my score

- More propositions restating the direct-sum kernel identity.
- More small-graph enumeration with the same conclusion.
- Another synthetic witness tailored to mixed probes.
- More overlapping node splits on existing graphs.
- A sixth ZINC seed without stronger controls.
- Cosmetic claim softening without fixing the underlying inconsistency.
- Turning the food-web result into a “win” through selective reporting.

**The revision has now made one central claim—useful transfer to larger graphs—substantially convincing. That earns the increase to 7. The remaining task is to finish the technical closure and present that contribution accurately, not to keep expanding the paper indefinitely.**