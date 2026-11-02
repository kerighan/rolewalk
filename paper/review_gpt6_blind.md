## 1. Summary

RoleWalk is a training-free structural node embedding based on short, self-loop-augmented random walks. At each walk length, it concatenates:

- A centred, unnormalised characteristic-function summary of the multiset of transition probabilities.
- The probability mass in each shortest-path distance layer around the starting node.
- Optionally, the mean and standard deviation of a local neighbourhood-overlap quantity interpreted as an upper bound on Ollivier–Ricci curvature.

Directed graphs additionally use separate incoming and outgoing walks and two mixed-direction products. The implementation combines sparse propagation, a fused compiled kernel, and optional pruning with a missing-mass budget.

The paper establishes locality, invariance, an MMD interpretation, stability of the characteristic-function block, expressiveness results and limitations, and an exact-arithmetic pruning bound. Experiments cover structural-role prediction, directed roles, graph discrimination, cross-graph centrality prediction, scalability, and structural encodings for ZINC.

**Overall assessment:** This is a technically careful and practically promising contribution. The strongest evidence concerns efficient extraction of transferable structural features, especially centrality transfer across graph sizes. The mathematical results are largely correct, although mostly elementary consequences of the construction. The claims of state-of-the-art role-classification performance are broader than the results justify.

**Review scope:** I inspected the supplied manuscript and `rolewalk.py`. I did not execute the code. The referenced benchmark scripts, CSV files, figure, and bibliography file were not included in the supplied material, so I cannot independently verify the reported measurements or benchmark implementations.

## 2. Soundness

### Mathematics

I did not find a fatal error in the central propositions under their stated exact-arithmetic assumptions, with one clarification needed regarding zero-weight edges.

**Locality and invariance.** The radii are justified. Computing a length-\(L\) walk requires transition normalisations only at vertices reached before the final step, whose incident edges lie inside the stated ball. Mixed probes require an underlying-undirected radius of two. Curvature at endpoints reached within \(L\) steps requires up to two further hops. These arguments establish consistent coordinates, not a guarantee of successful distribution-shift generalisation; the paper appropriately distinguishes these.

**Injectivity.** The analytic-function argument is correct. The finite-grid statement is also correct for any *fixed countable family* of unequal multisets. Its limitations matter:

- It does not establish injectivity over arbitrary weighted graphs, an uncountable family.
- It does not directly certify the particular deterministic frequency grid.
- It provides no separation margin or numerical robustness.

The manuscript mostly acknowledges these restrictions.

**MMD interpretation and scaling.** Expanding the real inner product of the complex Fourier features gives the stated kernel, and the layer term is a positive-semidefinite inner-product kernel. The Taylor remainder bound is valid. The result is an MMD pseudometric on finite measures, not a characteristic-kernel guarantee.

The interpretation is useful, but mathematically any explicit feature map induces such a kernel. Likewise, the “only positive factor” statement is conditional on the chosen first-order matching criterion and orthonormal layer coordinates; it is not an optimality result for the scaling.

**Stability.** The endpoint-row change \(2/(d_a+2)\), the telescoping argument, and the locality of the perturbation are correct. The restriction to the characteristic-function block is essential. The supplied \(K_{2,d}\) example correctly demonstrates why the full embedding does not inherit this degree-dependent stability.

**Expressiveness.** The expansion
\[
(H^3)_{uu}=\frac{1+3d+2t_u}{(d+1)^3}
\]
is correct, as are the regular-graph relationship to RWSE and the distance-regular failure-class argument. I also checked the probabilities in the explicit node-level 1-WL counterexamples and the mixed-direction example; they are consistent.

The cycle-with-pendants construction correctly separates bounded-radius information from sufficiently many rounds of 1-WL. Consequently, the claimed incomparability is justified, with the stated distinctions between node-level and graph-level examples.

One observation could sharpen the presentation: on a \(d\)-regular graph,
\[
s_u^2(1)=\frac{2d+2t_u}{(d+1)^2}.
\]
Thus the layer block already recovers triangle counts in **two** steps. The three-step claim is true but conservative. This would also demonstrate more directly what layers add beyond return probabilities.

**Curvature.** The overlap is \(1-\operatorname{TV}(m_x,m_y)\). Since transport between distinct vertices costs at least one, the claimed upper bound on Ollivier–Ricci curvature is correct. However, this nonnegative overlap statistic should not be interpreted as recovering curvature’s sign or its full transport geometry.

**Pruning certificate.** The proof is sound: nonnegative propagation and deletion give \(q^w\leq p^w\), missing mass bounds the characteristic-function error, and the layer error is controlled by the same missing mass. The \(L\leq3\) restriction is justified because the first step is unpruned and the exact second-step support is recorded before pruning.

The certificate’s limitations are consequential but explicitly disclosed: it excludes curvature moments and numerical error, and it bounds absolute feature error rather than downstream predictions. It should consistently be described as an **exact-arithmetic pruning bound**, not a certified bound on the returned floating-point array.

**Zero-weight definition.** As written, “non-negative weights” together with an ordinary unweighted out-degree allows \(d_u>0\) but \(s_u=0\), making the displayed transition formula undefined. The implementation removes zero entries. The mathematical definition should therefore specify that \(d_u\) counts **positive-weight neighbours**.

### Implementation

For ordinary inputs, the code appears consistent with the method, including the feature counts, mixed probes, pre-pruning support tracking, cumulative missing-mass correction, and all-or-nothing row budget. The fused and sparse backends implement the same intended computation up to numerical approximation.

I found two concrete numerical edge cases that contradict unrestricted claims about finite nonnegative weights:

1. **Reciprocal overflow in `lazy_walk`.** Scaling through
   ```python
   diags(1 / top) @ A
   ```
   overflows when a positive row maximum is sufficiently small, for example \(10^{-310}\). A two-node graph with that edge weight should have transition probabilities \(1/2\), just as with weight one, but the implementation creates nonfinite intermediates. Directly dividing each stored entry by its row maximum avoids forming the overflowing reciprocal.

2. **Integer overflow in curvature symmetrisation.** `node_curvature` computes `(A + A.T) > 0` before converting to floating point. For an undirected edge with integer weight \(2^{62}\), an `int64` adjacency can overflow under this addition, incorrectly removing the edge from the support. Binarising before symmetrising avoids this issue.

These are limited edge cases, not evidence that the reported unweighted experiments are invalid, but they should be fixed and tested.

The “online” property also needs qualification. Local recomputation is possible mathematically, but `transform` still converts the full graph and constructs full transition matrices. There is no demonstrated dynamic-update implementation or update-time benchmark.

### Experimental support

**The centrality-transfer evidence is the strongest part.** Sixty independently generated train/test graph pairs, frozen ReFeX preprocessing, dimension-matched top-\(k\) features, local-statistics baselines, and paired intervals constitute a meaningful evaluation. The reported improvements over top-\(k\) features and local statistics are reasonably supported within the selected generator families.

However, this is principally **within-family transfer across size**, plus within-family parameter shifts. It does not establish general transfer between unrelated graph-generating processes. The closing language about transfer “across generators” should be narrowed unless cross-family experiments are added.

**Role-classification superiority is not established broadly.** Among the six main classification datasets, RoleWalk variants clearly lead the reported means on USA, while other methods lead Brazil, Europe, Actor, and Film; several remaining results are effectively ties. The paired comparisons do not establish superiority over the closest competitors on most datasets. “Competitive accuracy at low cost” is supported better than “at or above the state of the art.” The speed advantage remains valuable even without an accuracy advantage.

**Directed evidence is useful but limited.** The mixed-probe ablation and exact counterexample convincingly show additional information. The synthetic results favour RoleWalk for retrieval, but directed ReFeX wins logistic-regression accuracy under noise. On wiki-RfA, improvements over local statistics are modest and lack the paired uncertainty analysis provided for other tasks. Task B is also largely solved by degree features.

**Food-web results are exploratory.** Ecosystem-held-out evaluation is appropriate, and the manuscript is commendably explicit about LLM-generated labels. Nevertheless, without expert validation, these are results against proxy annotations rather than established trophic-role ground truth. They also do not demonstrate a predictive benefit from using weights: weighted RoleWalk trails unweighted RoleWalk substantially.

**Scalability and approximation quality must remain separate.** The million-node headline uses threshold pruning without a mass budget. Elsewhere, that setting produces large worst-node errors. The certified setting can approach exact-computation cost and loses much of the directed speedup. These facts are disclosed, but the headline could still suggest a simultaneous speed-and-accuracy guarantee that has not been demonstrated at million-node scale.

**Baseline attribution could be stronger.** In particular, the paper needs more systematic comparisons with:

- Layer-only features.
- Return probabilities for the same transition matrix and walk lengths.
- Characteristic-function-only features across the main tasks.
- A dimension-matched directed top-\(k\) baseline including mixed probes.

Comparing three-step RoleWalk with a sixteen-step RWSE implementation is informative for one accuracy–cost trade-off, but it does not isolate the source of the computational advantage. Baseline implementation quality and propagation strategy also matter.

**ZINC supports only the stated preliminary conclusion.** The paired interval includes zero, and the reproduced RWSE baseline is weaker than the published result. The combined encoding is promising, but neither a reliable general improvement nor a new best result is established. The manuscript appropriately reports these caveats.

Finally, numerical graph-discrimination tests are useful empirical evidence, not mathematical separation guarantees. The missing benchmark code prevents checking whether the sorted-row comparison robustly handles nearly tied coordinates and numerical row-order changes.

## 3. Novelty and significance

The closest predecessor is clearly **GraphWave**. Applying a characteristic function to diffusion coefficients is not new, and the manuscript correctly exposes the algebraic relationship. Replacing heat diffusion by finite walk powers and removing graph-size-dependent normalisation are straightforward but useful modifications.

The more distinctive contribution is the **combination** of:

- Fixed-coordinate, bounded-radius walk summaries.
- Distance-layer masses.
- Mixed incoming/outgoing probes.
- An efficient row-wise implementation.
- A missing-mass-controlled approximation with an explicit bound.

This combination appears practically valuable. The centrality-transfer results suggest that it is more than merely a faster reimplementation of GraphWave.

The theoretical novelty is modest. The MMD identity follows directly from the explicit feature map; stability uses standard stochastic-matrix perturbation arguments; exceeding 1-WL through return probabilities is already familiar from RWSE; and the distance-regular limitation follows from the adjacency algebra. The explicit collision examples and pruning analysis are nevertheless useful and well aligned with the algorithm.

The related-work discussion should also position the method against transferable structural-signature approaches such as **xNetMF/REGAL**. Inductive or cross-network structural representations are not unique to the methods currently emphasised.

Overall, the contribution is strongest as a coherent, efficient graph-representation method with careful analysis, rather than as a major new expressiveness or kernel-learning result.

## 4. Strengths

- Clear, compact representation with graph-independent coordinate semantics.
- Native treatment of direction, including a well-motivated mixed-probe construction.
- Largely correct theory that includes explicit limitations and counterexamples.
- Efficient implementation and a useful approximation-control mechanism.
- Substantial experimental breadth, especially the independently generated centrality-transfer pairs.
- Strong transparency about negative results, approximation errors, weak benchmarks, and inconclusive statistical evidence.

## 5. Weaknesses

1. **Incremental conceptual novelty:** most components have close precedents, and several theoretical results are straightforward consequences of the construction.
2. **Overbroad accuracy claims:** the evidence supports competitiveness more consistently than superiority.
3. **Incomplete component attribution and efficiency controls:** matched layer-only, short-RWSE, and directed mixed-probe baselines are particularly important.
4. **Limited externally validated directed and weighted evidence:** one substantive real directed network and proxy-labelled food webs leave substantial uncertainty.
5. **Fastest scalability results do not establish uniformly accurate or budget-certified embeddings at that scale.**
6. **Reproducibility cannot be fully assessed from the supplied material:** benchmark scripts, configurations, raw results, and the CSV-dependent scalability table are unavailable here.
7. **Minor implementation/definition defects** affect extreme weights and zero-weight-edge semantics.

## 6. Questions to the authors

1. **Component attribution:** Can you report layer-only, matched-length lazy-RWSE, and characteristic-function-only results on centrality transfer and wiki-RfA? How much performance comes from each block?
2. **Directed matching:** Can top-\(k\)+layers be extended with equally dimensioned mixed-direction summaries, so that the main directed comparison isolates the characteristic function rather than access to additional walks?
3. **Evaluation protocol:** Were scalers, classifier hyperparameters, and other learned preprocessing fitted using training data only? Which datasets and labels were used to select the default frequency range, walk length, and layer scaling?
4. **Certified scalability:** What are runtime, peak memory, and missing-mass distributions at one million nodes with a fixed budget? How does pruning affect predictions for the highest-centrality nodes?
5. **Transfer baselines:** Can fixed-scale, consistently normalised diffusion baselines be evaluated under the same transfer protocol? How does RoleWalk compare with a transferable structural-signature method such as xNetMF?
6. **Real directed evidence:** What are paired intervals and class-balanced metrics for wiki-RfA? Can a domain expert audit a representative sample of the food-web labels?
7. **Artifacts and robustness:** Can you provide the exact benchmark scripts and configurations, robust graph-multiset comparison code, and tests covering zero-weight edges, subnormal positive weights, and large integer weights?

## 7. Overall recommendation

**Score: 6 — Weak accept**  
**Confidence: 4/5**

The core method is sound, useful, and supported by unusually broad and candid evaluation. The strongest contribution is an efficient, interpretable, transferable structural representation with explicit approximation control. I would lean toward acceptance on that basis, especially at a graph-learning venue.

I would not assign a stronger score because the conceptual novelty is incremental, superiority on real role tasks is limited, and some efficiency and component-level comparisons remain insufficiently controlled. The numerical implementation issues appear readily fixable and do not undermine the main unweighted results.