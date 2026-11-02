## Bottom line

**8/10 — accept. Confidence: 3/5 overall; 4/5 on the textual, mathematical and static-code assessment.**

The split is a good decision. The condensed submission has a much clearer scientific story, and the new alignment experiment materially strengthens it: this is a real cross-graph identification task against the closest directed baseline, not merely another synthetic ablation.

**The condensed paper is largely faithful to the report, but not yet completely faithful.** Most displayed results agree. The important remaining problems are a false blanket statement about experimental settings, a few qualifications lost during compression, and some mathematical definitions that no longer describe the implementation precisely. These are bounded submission repairs—not reasons to undertake another research programme.

**Scope:** I reviewed the supplied LaTeX, Typst and Python statically. I did not execute experiments, compile either document, inspect the referenced CSVs/figures/tests, or independently check Digraphwave’s original protocol and published results.

# 1. Reviewer assessment

### What works well

- **The main contribution is now understandable.** The concentration/distance distinction is effective, and the mixed probes have a concrete motivation.
- **The centrality attribution is substantially more honest.** The paper explicitly says that layers carry most of the performance and that matched top-\(k\)+layers essentially ties the CF version under size transfer. That correction belongs in the main paper, and it is there.
- **The alignment addition is valuable.** The reported Enron margins over Digraphwave are substantial and consistent across noise levels. The weighted high-noise ReFeX win is disclosed in the body.
- **Several important negative results survive condensation.** Food-web non-separation, strong simple baselines, expressiveness failures, pruning problems on hubs, and the preliminary status of ZINC remain visible.
- **The supplementary report is genuinely useful.** It contains the component analyses, alternative constructions and negative results needed to interpret the main submission.

### What still limits the contribution

Conceptual novelty remains moderate: the CF connection to GraphWave is close, while the contribution is principally the combination of distance layers, directed probes, analysis and implementation. Benefits remain task-specific, and simple features are competitive. Certified computation at scale is demonstrated more convincingly than a substantial speedup from certified pruning.

Those limitations support an **8 rather than an automatic 9**. The new alignment result makes the acceptance case stronger, but does not make the remaining scope distinctions irrelevant.

# 2. Important edits before submission

## A. Correct the experimental settings: the current defaults paragraph is false

The condensed paper says:

> “All experiments use the defaults … except centrality transfer, which uses \(L=4\)…”

But the report explicitly states that centrality transfer uses:

- RoleWalk: **four steps, exact computation, \(\mathrm{tol}=0\)**;
- top-\(k\)+layers: four steps, **pruned at \(10^{-5}\)**.

The condensed paragraph instead implies that both use the default \(10^{-4}\). This matters particularly because the pruning theorem covers layers only through three steps.

Other exceptions include exact expressiveness evaluations, five-step variants and the explicitly varied pruning settings.

**Replace the blanket statement with a short settings paragraph or table.** For example:

> Unless otherwise stated, predictive experiments use the default three-step configuration. Centrality transfer uses four-step RoleWalk with exact walks (\(\mathrm{tol}=0\)); the matched top-\(k\)+layers control uses four steps and threshold \(10^{-5}\). Expressiveness tests use exact walks. Pruning and walk-length experiments vary the stated settings explicitly.

Also:

- Rename the transfer-table column **“GraphWave, no \(1/n\)”**, as in the report. Otherwise readers may wrongly attribute its poor transfer entirely to a normalization that this experiment already removes.
- Restore the **matched lazy-RWSE result, 0.753**, beside the layers-only and CF-only attribution. The headline RWSE result of 0.532 uses a different walk and horizon; that distinction is important to understanding where the gains come from.
- Restore the aggregation convention: the overall table mean averages **400 valid cells**, whereas paired intervals average targets within each graph pair and then average pairs.

The horizon mismatch is fixed. However, “dimension-matched” should not silently become “identical numerical computation”: the remaining pruning difference should be disclosed, or removed in the control if convenient.

## B. Handle the role-benchmark selection more carefully

The abstract/contributions advertise six role-classification benchmarks, but the main table contains five and explicitly excludes Film.

Excluding Film from the primary role analysis is defensible. The stronger statement is not:

> “which we show to be a node-type rather than a role task”

The report offers evidence and an interpretation, not a definitive demonstration of that taxonomy. This is also the dataset where RoleWalk has a substantial loss: **0.581 versus Role2Vec’s 0.743**.

I would accept either:

1. retaining Film as a clearly marked secondary column; or
2. changing the count to five and adding one transparent sentence:

> We treat Film separately because its labels appear to reflect node types rather than the structural roles targeted here; Role2Vec substantially outperforms RoleWalk on that dataset (0.743 versus 0.581; supplementary analysis).

That is sufficient. The full Film analysis need not return to the main paper.

**Restore the fixed-scale random-walk heat-CF row in the role table.** It is the closest controlled diffusion comparator and has the highest Europe mean and second-highest Brazil mean in the report. Omitting it also leaves stale ranking marks: RoleWalk+curvature is underlined on Europe although no displayed method is bold.

Finally, narrow:

> “not significantly worse than any method on any network”

to the retained datasets and the comparisons actually tested. The supplied significance table does not cover every displayed baseline. Failure to detect a disadvantage is also not an equivalence result.

## C. Keep the alignment result strong, but describe its provenance and scope precisely

The main text correctly identifies the baselines as published results. It should also retain the report’s explicit disclosure that **noise draws are not paired**.

Suggested caption addition:

> Baseline results are quoted from Digraphwave rather than rerun; noise draws are therefore unpaired. Noisy RoleWalk results average five draws; noiseless Enron uses one deterministic run.

Variability for RoleWalk would be helpful if available, but I would not require rerunning all baselines. Without matched draws or suitable uncertainty estimates, these are descriptive accuracy comparisons, not paired significance claims.

The abstract and conclusion should replace:

> “best scores … on the Enron network-alignment benchmark”

with the more precise—and still strong—claim:

> RoleWalk exceeds the published Digraphwave top-1 accuracy at every tested Enron noise level.

The body can retain the ReFeX exception at weighted \(p=0.1\). Similarly, replace broad claims about leading on “real directed roles” with **“highest mean scores on wiki-RfA”**: the food-web result does not support a general real-directed-role superiority claim.

## D. Finish the statistical and selection disclosures

### Multiplicity

The condensed setup now correctly says that Nadeau–Bengio addresses overlapping splits. It still does not say whether comparisons are multiplicity-adjusted.

The report explicitly says the role comparisons are not. Add:

> Classification intervals account for overlapping splits but are not adjusted for multiple comparisons.

Confirm the corresponding status for wiki-RfA. I am not demanding a new multiple-testing procedure; I am asking that “significant against every comparator” be understood as the reported nominal pairwise evidence.

### Hyperparameter selection

Disclosing the 48-configuration search is an improvement. But “synthetic and real networks” is too vague. Identify:

- which datasets and metrics selected the defaults;
- which reported evaluations also participated in selection;
- how the four-step transfer setting was selected;
- whether the alignment datasets were outside that selection process.

Leaving each dataset out and retaining the same optimum is useful robustness evidence, but is not a substitute for documenting the selection/evaluation overlap.

For alignment, **“no alignment-specific tuning”** is clearer than “nothing tuned.”

### Parameter-shift claims

“RoleWalk remains the best embedding” still exceeds the report’s inferential conclusion. Use:

> RoleWalk has the highest aggregate mean among embedding baselines, but its full-target advantage over frozen ReFeX is not established.

The matched top-\(k\) advantage under parameter shifts, \(+0.024\,[0.003,0.044]\), is worth one sentence in the main text. It gives the CF a supported, specific benefit after the size-transfer tie. Retain the report’s run/version provenance in the supplementary description.

## E. Remove the unsupported causal claim about pruning and scalability

This sentence should go:

> “Threshold pruning is what makes these times possible”

The report does not establish that at a million nodes. At 50k nodes, the budgeted runs are close to exact cost and sometimes slower: **43.1 s versus 38.2 s** on the directed graph.

A faithful replacement is:

> Threshold pruning reduces runtime in the measured comparisons but can produce large worst-node errors. A 1% mass budget provides affordable error control at a million nodes; an exact-runtime comparison at that scale is not available.

Also distinguish:

- default, curvature-free million-node timing: **18–193 s** in the scale table;
- budgeted BA timing: **227 s**;
- the separate **173 s** threshold-only run.

Explain whether 173 versus 193 reflects different runs or timing boundaries. Keep conversion, warm-up and memory definitions consistent.

The introduction’s claim that learned methods “do not reach a hundred thousand” is contradicted by the report’s RESD result at 100k. Replace it with an empirical statement about the tested implementations and resource limits:

> In our benchmarks, RoleWalk is substantially faster than most learned and diffusion baselines; ReFeX remains faster.

# 3. Mathematical compression needs a careful repair

The core arguments remain useful, but several definitions lost necessary qualifications.

## Directed layer distances are currently misstated

The appendix says distances on directed graphs are taken in the underlying undirected graph. **That is not how the implementation computes layers.** Supports of \(H_+^w\) give directed out-distances; supports of \(H_-^w\) give directed in-distances.

For example, in \(u\to a\to v\to u\), the implementation places \(v\) at out-distance two from \(u\), although its underlying-undirected distance is one.

Distinguish explicitly:

- **layer distances:** shortest paths in the relevant walk direction;
- **locality balls:** underlying-undirected balls sufficient to contain everything read by both directions and mixed probes;
- **curvature:** computed on the simple unweighted symmetrisation.

Enabling curvature does not change the distance-layer convention.

## Restore the omitted walk conventions

In the main method section:

- say self-loops in the input are ignored;
- define \(d_u\) using positive-weight neighbours;
- put \(u\ne v\) on the off-diagonal formula;
- define zero-outdegree nodes as absorbing, avoiding \(A_{uv}/s_u=0/0\).

The curvature overlap formula also needs its **simple, unweighted, symmetrised-graph** qualification in the main definition. Otherwise it appears to apply to the weighted walk just introduced.

## State the actual pruning algorithm

The report and code include two details missing from the condensed method:

1. the first pure-walk step is never pruned;
2. missing mass contributes the first-order correction \(i\theta r\) to the CF block.

The proof currently invokes the second detail without the method having defined it. The first detail matters to the three-step distance argument.

Also say that beyond three steps, pruning can delay first discovery and hence change assigned layer labels. This is another reason to state clearly that the centrality result uses exact walks.

## Scope the kernel proposition to the directed embedding

The displayed footprint/kernel uses only the step mark \(w\), but the default directed embedding concatenates two directions and mixed probes. Restore the report’s short extension:

> For directed embeddings, replace the step mark by a block mark identifying direction and step or mixed probe; mixed blocks have no layer term.

Also restore the qualification that the finite-grid construction gives an **MMD pseudometric**, not a proved characteristic embedding of arbitrary footprints.

The first-order calculation motivates the layer scale under the chosen construction. It does not establish that this is the uniquely optimal geometry. A short qualification would improve the discussion:

> This fixes the scale under first-order matching; the approximation applies when \(\theta p\) is small.

The report’s observation that feature-wise standardisation cancels constant block scaling is also worth retaining, so the raw-distance motivation is not mistaken for an explanation of supervised-classification gains.

## Tighten the certificate language

Replace:

> “floating-point rounding is separate and small”

with:

> The certificate bounds pruning error in exact arithmetic; it does not certify floating-point/Taylor errors or downstream prediction accuracy.

The abstract’s locality statement should say **bounded-radius neighbourhood**, rather than unconditionally \(L\)-hop: curvature requires \(L+2\). Its kernel statement should likewise be scoped to the curvature-free representation.

These are wording/definition repairs, not demands for new theory.

# 4. What can remain supplementary?

I would **not** restore every negative result to the main paper. The present food-web qualification and ZINC caveat are appropriate; change “proxy trophic labels” to **“LLM-generated proxy trophic labels.”** Full food-web tables, alternative kernels, trajectory features and detailed pruning frontiers can stay supplementary.

A brief note that simple statistics win airport-to-airport activity-label transfer would further balance the broad transfer framing, but is less urgent than the settings and Film corrections above.

The supplementary report itself needs a final synchronization pass: it still contains “online,” “clearly ahead on one of six,” and “leads every embedding” language. The wiki-RfA top-\(k\) mixed-probe disclosure is no longer needed in the main table because that row was removed, but it remains needed in the report.

The supplied library also still contains the previously identified nondefault-parameter issues: accepted integer-valued floats are not consistently coerced, and frequency/scale validation does not protect the float32 norm calculation. **The two weight-handling bugs remain fixed.** These remaining artifact issues do not invalidate the default experiments, but should be closed before release.

# 5. Recommendation and priority

**Keep the 8/10.** The condensed submission is more publishable than the full report as a main paper, and the alignment result strengthens the contribution. I would not withhold acceptance for lack of expert food-web labels, a larger GNN study, universal baseline wins or an exact million-node comparison.

The highest-value final edits are:

1. **Correct experimental settings and comparator labels.**
2. **Disclose the Film loss and restore the closest heat-CF control.**
3. **Scope alignment, parameter-shift and significance claims precisely.**
4. **Remove the unsupported pruning-driven scalability assertion.**
5. **Repair directed distances and the compressed method/theorem definitions.**

My direct advice remains: **make these consistency repairs and submit.** The paper does not need more results to justify acceptance; it needs the shorter presentation to preserve the distinctions that made the full report credible.