## 1. Summary

RoleWalk is a training-free structural node embedding based on short lazy random walks. At each walk length, it concatenates:

- A centred, unnormalised characteristic-function summary of the nonzero transition probabilities.
- The probability mass assigned to each shortest-path distance from the starting node.
- Optionally, moments of a local neighbourhood-overlap statistic that upper-bounds an Ollivier–Ricci curvature.

Directed graphs additionally use walks on the reversed graph and two forward/backward mixed probes. The resulting coordinates are defined without a learned reference frame or graph-level scale selection. The implementation propagates sparse rows individually and optionally prunes small probabilities.

The paper establishes locality, invariance, a kernel-embedding interpretation, expressiveness results relative to 1-WL, a distance-regular failure class, and an exact-arithmetic pruning-error bound. Experiments cover structural-role classification, directed roles, network alignment, centrality transfer, expressiveness, molecular structural encodings, and scalability.

The most convincing empirical findings are the improvements over Digraphwave on Enron alignment, strong wiki-RfA results, and effective within-family centrality transfer. The ablations also reveal an important qualification: distance layers carry most of the centrality-transfer performance, whereas the characteristic-function block is more useful for alignment, where adding layers actually hurts.

**Assessment scope:** I checked the supplied proofs and implementation and cross-checked the reported results across the paper and report. I did not execute the code. The benchmark scripts and CSV files referenced in the report are not included in the supplied material, so my experimental assessment concerns the reported evidence rather than independently reproduced results.

## 2. Soundness

### Mathematics

**The central propositions appear correct under their stated assumptions.** In particular:

1. **Locality and invariance.**  
   The radius argument correctly accounts for transition normalisation. A length-\(L\) walk only uses transitions originating at vertices within distance \(L-1\), whose incident edges are contained in the radius-\(L\) ball. Mixed probes need radius two in the underlying undirected graph. The additional two hops for curvature are sufficient. On weighted graphs, the relevant isomorphisms must preserve weights.

2. **Characteristic-function injectivity.**  
   The analytic-continuation argument is valid. The generic finite-grid statement is also valid for a prescribed countable family of distinct multisets. It does not establish injectivity or a useful separation margin for the particular default grid on arbitrary weighted graphs, but the paper does not claim such a theorem.

3. **Kernel interpretation.**  
   The displayed kernel follows correctly from the inner product of the explicit feature map, and the Taylor bound supports the stated leading-order coefficient \(\|\Theta\|^2ab\). The finite-measure formulation is important: this is not the usual comparison of probability measures obtained by normalising the footprint’s atom count.

   The significance should nevertheless be kept modest. The MMD identity is principally an explicit-feature-map interpretation, not an additional expressiveness guarantee. Moreover, “not proved characteristic” is weaker than the actual general limitation: a finite-dimensional feature map is not characteristic on unrestricted finite measures. Even with all frequencies, this particular decomposition retains the marginal probability multiset and layer masses, rather than the full probability–layer association.

4. **Expressiveness.**  
   The triangle calculation is correct:
   \[
   s_u^2(1)=\frac{2d+2t_u}{(d+1)^2}.
   \]
   I also checked the constructed node-level collisions for two and three steps; the stated probabilities agree. The distinction between these node-level collisions and graph-level indistinguishability is handled appropriately in the report. The distance-regular argument, including constant curvature, is sound.

5. **Mixed probes.**  
   The source-to-two-sinks and directed-\(K_{2,2}\) example is correct, including the two mixed-walk probability multisets. This is a useful, concrete justification for mixed probes beyond simply concatenating in- and out-walks.

6. **Certified pruning.**  
   The proof is valid for the stated setting. Nonnegative propagation and deletion give \(q^w\le p^w\), with missing mass equal to their \(\ell_1\) difference. The corrected characteristic-function error is bounded by \(2\|\Theta\|r^w\), and the layer error by \(\|\Theta\|r^w\), yielding the factor \(\sqrt{5}\).

   The restriction to at most three steps is substantive and correctly explained: recording supports before pruning gives exact distances through step two, which suffices to label retained step-three entries correctly. It need not suffice thereafter. The supplementary characteristic-function stability proposition also appears correct; importantly, it does not extend to the distance-layer block.

### Mathematical and specification issues requiring correction

- **The “Beyond square adjacency” comparison is incorrect as written.** Matrix exponentiation requires a square operator, not a symmetric one. Directed diffusion is already a counterexample within the paper’s own related work. Cross-rank walks can also be represented by a square block operator on the disjoint union of cell types, so this is not an intrinsic capability unavailable to diffusion methods.

  Furthermore, if \(B'\) means \(B^\top\), a row-normalised incidence matrix \(B\) does **not** generally make \(BB^\top\) row-stochastic. The reverse incidence transition needs its own normalisation. If \(B'\) denotes that separately normalised operator, it should be defined explicitly.

- The main-text RWSE equivalence should explicitly assume \(d>0\), as the supplementary report does. Otherwise the simple walk \(A/d\) is undefined and the claimed nonzero triangular diagonal is absent.

These are repairable issues and do not invalidate the core method.

### Implementation

The supplied implementation is generally consistent with the method:

- The default walk removes input self-loops, uses positive-weight neighbour counts, and implements mean-edge-weight self-loops.
- The fused kernel avoids materialising full powers, records first encounters before pruning, and retains the first step without pruning.
- The budget implementation keeps all small entries in a row when deleting them would exceed the budget. This is conservative but satisfies the claimed guarantee.
- Mixed probes are formed from the original transition matrices and pruned separately.
- The API correctly withholds certificates for curvature, more than three steps, and the alternative mean aggregation.

I did not identify a major algorithmic bug in the supplied core implementation. The distinction between an **exact-arithmetic pruning certificate** and a certificate for the actual floating-point output is appropriately documented. This qualification should remain prominent in any headline claim about certification.

### Experimental support

**The reported experiments support useful practical performance, but not universal superiority.**

#### Particularly convincing evidence

- **Enron alignment:** The direct Digraphwave reproduction, including shared noisy instances, materially strengthens the comparison to published numbers. The advantage is substantial rather than merely a small statistically significant difference.
- **Component attribution:** The matched no-mixed-probe comparison isolates the characteristic-function contribution reasonably well. Reporting that layers worsen alignment is valuable negative evidence.
- **Centrality transfer:** Sixty independently generated graph pairs provide a substantially stronger basis than repeated splits of one graph. Layers-only, characteristic-function-only, lazy-return, and dimension-matched top-\(k\) ablations make the attribution informative.
- **Pruning:** The report does not hide severe worst-node errors under default threshold pruning. Exact comparisons and budgeted runs substantiate the practical value of the certificate.

#### Important limitations and inconsistencies

1. **The main paper and report disagree about the matched top-\(k\) transfer experiment.**  
   The main paper states that both RoleWalk and the dimension-matched ablation use exact walks, whereas the report specifies tolerance \(10^{-5}\) for top-\(k\). This matters because these experiments use four steps, beyond the guaranteed correctness of support-derived distances under pruning. It probably does not overturn the reported near-tie, but the experiment is not exactly matched as currently described. Both methods should be rerun exactly with the same library version.

2. **Some classification benchmarks participate in hyperparameter selection.**  
   Agreement between \(k\)-means clusters and labels is label-based model selection, even though the clustering algorithm is unsupervised. Using a different downstream metric does not automatically make the classification evaluation independent. The reported leave-one-dataset-out stability is encouraging and could substantially address this concern, but it should be presented as an explicit selection protocol rather than only a robustness statement. Wiki-RfA, alignment, and centrality transfer remain more persuasive because they did not participate in default selection.

3. **The GIN comparison is limited.**  
   One untuned, multitask GIN configuration with a different prediction architecture does not isolate representation quality or establish superiority over trained inductive models generally. Target transformations, relative loss scaling, optimisation settings, and tuning budget are especially important when jointly predicting seven heterogeneous centralities. The conclusion should remain “better than this GIN configuration.”

4. **Relevant transferable baselines are missing.**  
   FEATHER and a shared-reference xNetMF/REGAL configuration would be particularly useful for testing the claimed advantages over established transferable structural representations. The paper appropriately acknowledges that learned or landmark-based coordinates can be shared by freezing an encoder or reference basis; the experiments should engage more directly with that possibility.

5. **The RWSE gap needs careful attribution.**  
   The report’s four-step lazy-return baseline reaches 0.753, substantially above the headline 16-step RWSE result of 0.532. Thus a considerable part of that comparison concerns the transition operator and representation configuration, not just the new distance statistic. A matched lazy-RWSE length comparison would be informative.

6. **The strongest real-role claims are narrow.**  
   Wiki-RfA is one network, task B is already largely explained by degree, and the accuracy improvement over local statistics is not established. On task A, the nominal interval against local statistics only narrowly excludes zero; the multiple comparisons are explicitly uncorrected. These results are promising, but do not establish broad superiority on real directed roles.

7. **Film should remain visible in the main results table.**  
   The substantial negative result is disclosed, which is commendable. However, an independent-set class and separated identifier ranges do not by themselves establish that this is an invalid structural-role task. The proposed explanation of Role2Vec’s advantage remains untested. Retaining Film with an explicit caveat would be more convincing than excluding it from the main table.

8. **Scalability is demonstrated for particular implementations and sparse graph families.**  
   The results support impressive practical throughput, not a general linear-time guarantee. Comparisons against 16-step RWSE and potentially dense diffusion implementations also combine algorithmic and implementation differences. Importantly, the headline threshold-pruned timings should always be accompanied by the budgeted timings, because default approximation errors can be large.

The food-web labels are appropriately described as proxies; agreement among LLM annotators is not independent biological validation. Likewise, the restrained interpretation of the ZINC experiment is justified: it is preliminary, not evidence of a new state of the art.

## 3. Novelty and significance

The novelty is **incremental but coherent and potentially useful**.

The characteristic-function block is closely related to GraphWave; short-walk probabilities, return statistics, distance encodings, and shared-source/shared-target operators all have substantial precedent. Locality and fixed coordinates are also properties of many handcrafted and fixed-operator structural features. The MMD formulation does not, by itself, constitute a major new learning-theoretic result.

The more distinctive contribution is the combination of:

- A compact distance-law statistic alongside probability-multiset summaries.
- Mixed directed probes in a fixed-coordinate embedding.
- Efficient row-wise computation with a practical missing-mass certificate.
- Extensive empirical attribution of which components help which transfer tasks.

The result is more than a cosmetic recombination because the implementation and evaluation demonstrate a useful accuracy–cost trade-off. The strongest scientific observation is arguably the effectiveness of the short-walk distance law for within-family centrality transfer. The alignment results provide complementary evidence for retaining the characteristic-function block.

The significance would be stronger with comparisons to additional transferable structural methods and more convincing real cross-graph role-transfer evidence. Airport transfer favours simple statistics, while food-web transfer is inconclusive.

## 4. Clarity and presentation

The main paper is generally clear and unusually candid about limitations. The division into concentration and distance is intuitive, the mixed-probe example is effective, and the report makes negative results and historical ablation corrections visible.

Points to improve:

- Distinguish consistently between the centred transform of a **counting measure over probabilities** and a conventional probability-distribution characteristic function.
- Explain that finite walk length is essential to strict locality; removing GraphWave’s normalisation and graph-dependent scale alone does not make heat diffusion finitely local.
- Resolve the exact/pruned top-\(k\) discrepancy and identify the precise software version used for each central result.
- Report uncertainty more directly in the principal tables, rather than relying heavily on prose and supplementary comparisons.
- Keep within-graph repeated-split evidence separate from evidence across independently sampled graphs.
- Shorten or correct the speculative higher-order extension, which currently makes an unjustified contrast with diffusion.
- Preserve the clear qualifications that locality enables consistent coordinates but does not guarantee successful transfer of global roles.

The supplement is thorough but reads partly as an experiment log. A concise configuration and reproducibility table would help readers navigate it.

## 5. Strengths

- Simple, interpretable, fixed-coordinate representation with native directed and weighted support.
- Correct and useful core mathematical analysis, including explicit failure cases.
- Efficient implementation rather than only a conceptual proposal.
- Strong Enron comparison supported by a direct baseline reproduction.
- Extensive component ablations, including matched-dimensional alternatives.
- Useful distinction between average pruning accuracy and potentially severe worst-node errors.
- Unusually transparent reporting of negative and inconclusive findings.

## 6. Weaknesses

- Most ingredients are established; novelty lies primarily in their integration and empirical validation.
- Missing comparisons to several relevant transferable structural representations.
- Limited independent real directed-role and real cross-graph evidence.
- Label-assisted default selection overlaps some classification evaluations.
- The GIN comparison is insufficiently controlled for broader conclusions.
- Main/report inconsistency concerning pruning in the key top-\(k\) ablation.
- Default pruning can materially distort embeddings despite attractive headline runtimes.
- Incorrect claims about symmetry and diffusion in the higher-order extension.

## 7. Questions to the authors

1. **Matched transfer ablation:** Was four-step top-\(k\) computed exactly or with tolerance \(10^{-5}\)? Can you provide a fully exact comparison using the same library version, graphs, and regressors?

2. **Default selection:** Can you present the explicit leave-one-dataset-out selection results, identifying which labels were available when choosing the configuration for each classification benchmark?

3. **Transferable baselines:** How does RoleWalk compare with FEATHER and with xNetMF/REGAL using a fixed or shared reference basis? What about lazy RWSE with matched transition operators and several walk lengths?

4. **GIN protocol:** What target transformations, loss weights, optimiser settings, and model-selection procedure were used? Does the conclusion persist for separately trained targets or a modest tuning budget?

5. **Practical pruning:** Given the large default worst-node errors, why not recommend a small missing-mass budget by default? How do exact and budgeted embeddings affect Enron alignment accuracy and runtime?

6. **Film:** Can you retain this result in the principal table and test the proposed explanation of Role2Vec’s advantage, rather than treating the task categorisation as established?

7. **Reproducibility:** Can you provide a single pinned-version configuration manifest covering feature preprocessing, baseline settings, centrality definitions, alignment noise generation, and nearest-neighbour tie handling?

## 8. Overall assessment

**Overall score: 6 — Weak accept**  
**Confidence: 4/5**

This is a technically sound and practically useful contribution with a strong implementation, meaningful directed-alignment results, and unusually thorough, candid ablations. The core mathematical claims survive inspection. The paper is not a major conceptual breakthrough, and missing transferable baselines, limited real cross-graph evidence, and some evaluation inconsistencies prevent a stronger recommendation.

On balance, the coherent method, demonstrated efficiency, and informative analysis of distance-law versus characteristic-function features constitute a worthwhile contribution at these venues. The identified mathematical wording errors and experimental-description discrepancy should be corrected.