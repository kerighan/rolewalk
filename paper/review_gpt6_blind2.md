## 1. Summary

RoleWalk is a training-free structural node embedding based on short, self-loop-augmented random walks. At each walk length, it combines:

- A centered, unnormalized characteristic-function summary of the multiset of transition probabilities.
- The probability mass assigned to each shortest-path distance from the starting node.
- Optionally, moments of a local neighborhood-overlap statistic that upper-bounds Ollivier–Ricci curvature.

Directed graphs additionally use incoming and outgoing walks and two mixed-direction probes. The implementation uses fused sparse propagation and optionally prunes small probabilities, with an error bound controlled by missing probability mass.

The paper claims graph-independent coordinates, locality and invariance, connections to MMD and RWSE, expressiveness incomparable with 1-WL, identifiable failure classes, and favorable accuracy–runtime trade-offs. Experiments cover structural-role classification and retrieval, directed roles, centrality transfer, expressiveness benchmarks, scalability, and structural encodings for GNNs.

**Overall assessment:** This is a technically careful and practically promising contribution. The main mathematics appears sound, and the empirical evidence supports competitive role embeddings and useful transfer within the tested graph families. The contribution is principally a well-designed extension and implementation of diffusion-based structural descriptors, rather than a fundamentally new representation principle. Several experimental claims require tighter qualification and more directly matched baselines.

*Review scope:* I inspected the supplied source and `rolewalk.py` statically. The referenced benchmark scripts, CSV files, bibliography, and generated figures were not supplied here, so I cannot independently reproduce the reported results.

## 2. Soundness

### Mathematics

I did not identify a major error in the central propositions. In particular:

**Walk definition, locality, and invariance.**  
The average-edge-weight self-loop gives the stated holding probability \(1/(d_u+1)\) and invariance to changing weight units. The locality argument correctly accounts for transition normalization: a length-\(L\) walk needs outgoing edges only from vertices reached before the final step. The additional radii for mixed probes and curvature are sufficient. Isomorphisms must, naturally, preserve edge weights and directions.

**Characteristic-function injectivity.**  
The entire-function argument is correct for finite multisets of strictly positive probabilities. The almost-everywhere separation statement for a countable collection of pairs is also correct. Importantly, this does **not** establish injectivity of the particular ten-frequency grid, or simultaneous injectivity over arbitrary weighted graphs. The manuscript largely respects this distinction.

**MMD interpretation.**  
Expanding the real inner product of the complex Fourier features produces the stated kernel. The layer contribution is another positive-semidefinite feature inner product, so the finite-measure MMD identity is valid. The small-probability expansion and its bound also check out.

This result should nevertheless be interpreted modestly: an explicit feature map always induces such a kernel. It provides a useful description of the geometry, not a general characteristic-kernel or optimal-weighting result. Matching the leading bilinear term justifies the proposed scale under the stated convention, not its universal statistical optimality.

**Stability.**  
The row-wise change \(2/(d_a+2)\), stochastic-matrix telescoping argument, and support-locality statement are correct. The limitation to the characteristic-function block is essential. The supplied \(K_{2,d}\) example correctly demonstrates why the full embedding does not inherit this degree-decaying stability guarantee.

**Triangles, RWSE, and expressiveness.**  
The triangle identities are correct. In particular,
\[
s_u^2(1)=\frac{2d+2t_u}{(d+1)^2}
\]
already establishes separation of suitable regular graphs at two steps. The explicit node-level collisions with 1-WL also check out, as does the separated-pendants construction establishing a graph-level collision for fixed-radius RoleWalk.

The distance-regular failure theorem follows correctly from the distance-matrix algebra, and the curvature statistic is constant for graphs sharing the intersection array.

A minor qualification is needed in the RWSE equivalence proposition: the triangular transformation is conditional on knowing \(d\). If the statement is intended to concern the return-probability vectors alone for arbitrary \(L\), it fails at \(L=1\): simple-walk returns are zero for every loopless regular graph, whereas lazy returns identify \(d\). For \(L\geq2\), \(d\) is recoverable from the simple-walk second return probability.

**Pruning certificate.**  
The proof is sound under its stated restrictions. Since deletion does not renormalize,
\[
0\leq q_u^w\leq p_u^w,\qquad
\|p_u^w-q_u^w\|_1=r_u^w.
\]
The Fourier block, including its first-order correction, contributes at most \(2\|\Theta\|r_u^w\), and the scaled layer block at most \(\|\Theta\|r_u^w\), yielding the factor \(\sqrt{5}\).

The exact-distance argument through three steps is also valid: the first step is unpruned, the second-step support is recorded before pruning, and any newly encountered retained vertex at step three must have distance three.

The restrictions matter:

- The certificate is not for curvature moments.
- With layers, the proof does not cover arbitrary walk lengths.
- It is an exact-arithmetic pruning bound, not a certified floating-point error bound.
- It applies to raw feature-space distance, not directly to standardized features, predictions, or clustering assignments.

The manuscript commendably states most of these limitations explicitly.

### Implementation

The implementation broadly matches the definitions and proofs:

- The first pure-walk step is not pruned.
- Missing mass is propagated without renormalization.
- Supports are recorded before pruning.
- The budget rejects an entire proposed row-wise deletion when it would exceed the allowed mass.
- Mixed probes use the stated matrix-product order.
- The returned certificate uses the correct coefficients for pure and mixed blocks.
- The curvature implementation computes the stated normalized common-neighbor overlap.

I did not find a clear implementation error invalidating the default method in the supplied code. However, “exact” computation at `tol=0` still uses floating-point arithmetic and, in the fused backend, Taylor evaluation of small probabilities. This is consistent with the manuscript’s caveat, but should remain explicit in expressiveness and certification reporting.

The **online** designation is stronger than what is demonstrated. Locality enables selective recomputation in principle, but `transform` still converts the full graph and constructs full transition matrices. There is no implemented streaming update mechanism or dynamic-update evaluation. “Supports embedding selected nodes” or “locally recomputable” would be more precise.

### Experimental support

**Role classification supports competitiveness, not broad superiority.**  
The reported results show RoleWalk near the strongest methods on several datasets, with attractive runtime. They do not establish uniform accuracy leadership. Even on USA, where RoleWalk has the highest reported accuracy, its difference from RESD is not significant in the provided paired table. “Clearly ahead on one of six networks” therefore needs qualification. The paired comparison table also omits Digraphwave, despite its competitiveness.

**The directed evidence is useful but mixed.**  
Wiki-RfA is the strongest real-data result, particularly task A. On task B, the gain over local statistics is not established in accuracy, and degree alone is already strong. Synthetic results convincingly demonstrate the utility of direction and mixed probes, but directed ReFeX wins noisy supervised classification under the main classifier.

The food-web experiment is appropriately presented as inconclusive between RoleWalk, Digraphwave, and local statistics. Agreement among LLM-generated labels is not validation against ecological ground truth, especially when two labeling configurations use the same underlying model. This should remain secondary evidence. The experiments establish that weights are supported computationally, but not that using them improves the tested downstream tasks.

**Centrality transfer is substantial evidence, with an important attribution issue.**  
The 60 independent training/test graph pairs and pair-level uncertainty estimates are stronger than repeated splits of a single graph. The closeness and harmonic-centrality improvements are particularly convincing within the tested families.

However, most of the transfer performance comes from the distance layers: 0.807 versus 0.813 for the full method. The Fourier contribution is small but reportedly consistent. Also, matched lazy-return features achieve 0.753, substantially above the 0.532 reported for the main RWSE baseline. These comparisons should be elevated into the main table to avoid attributing improvements from walk choice and distance information primarily to the characteristic function.

There is also a **reporting inconsistency requiring clarification**: the default method uses \(L=3\), giving nine layer coordinates, but the transfer attribution reports fourteen layer coordinates and four-step lazy returns. Fourteen coordinates correspond to \(L=4\). The actual walk length, pruning settings, and dimension matching for this experiment must be stated. If pruned four-step layers are used, they are outside the exact-distance/certificate guarantee.

**ZINC is preliminary, as the authors acknowledge.**  
The combined encoding improves the authors’ reproduced RWSE baseline, but the paired confidence interval includes zero and the result does not improve on published GPS–RWSE performance. It is evidence of possible complementarity, not an established architectural improvement.

**Scalability is promising but implementation- and accuracy-dependent.**  
The fused implementation and million-node results are valuable. Nevertheless, the fastest default configuration can substantially distort hub embeddings, and the budgeted version can cost approximately as much as exact computation. The paper reports this honestly.

Comparisons against exact 16-step RWSE or relatively expensive diffusion implementations do not establish an intrinsic efficiency advantage over all approximate or optimized alternatives. A same-length lazy-RWSE/layer-only timing baseline and approximation-matched diffusion comparisons would make the efficiency claim stronger.

**Statistical reporting is better than average, but incomplete.**  
The use of overlap-corrected tests, graph-pair-level intervals, and explicit nonsignificant results is welcome. Remaining concerns include multiple comparisons, dependence within individual networks, and embedding-seed variability for stochastic baselines beyond struc2vec. The relationship between benchmark-driven hyperparameter selection and final evaluation also needs clearer documentation.

## 3. Novelty and significance

The closest precursor is GraphWave, and the manuscript correctly exposes the algebraic relationship: the characteristic-function block is GraphWave-style summarization with a different propagation operator and centering/normalization.

The more substantive contributions are:

1. Fixed, strictly local walk-based coordinates without graph-size normalization.
2. Distance-layer masses combined with probability-multiset summaries.
3. Mixed-direction probes for directed structural similarity.
4. An efficient fused implementation with explicit missing-mass accounting.
5. A coherent analysis of expressiveness, collisions, and approximation limits.

These constitute a meaningful practical contribution, although several theoretical results are elementary consequences of the representation. Triangle sensitivity and going beyond 1-WL are not themselves new capabilities among structural encodings. The overlap statistic is also a simple common-neighbor/degree quantity, rather than a new curvature construction.

Related work should better distinguish RoleWalk from characteristic-function random-walk descriptors such as FEATHER, diffusion-scattering representations, and local distance-based structural encodings. These are not identical methods, but they help identify which elements are genuinely new.

The likely significance is as a useful, efficient structural-feature baseline and transferable descriptor—not as a major advance in graph-isomorphism expressiveness or kernel theory.

## 4. Strengths

- **Strong technical transparency.** The paper gives explicit failure classes, constructive collisions, instability examples, and clear restrictions on its guarantees.
- **A coherent directed construction.** Mixed probes have a simple interpretation, a separating example, and empirical ablations.
- **Practical implementation effort.** Fused propagation, sparse support handling, numerical care for small angles, and per-node missing-mass reporting are valuable.
- **Broad evaluation with nontrivial simple baselines.** Degree, local statistics, matched top-\(k\) summaries, frozen ReFeX, and modified heat-kernel features make the comparisons more informative.
- **Good treatment of negative results.** The manuscript acknowledges stronger baselines, weak labels, parameter-shift failures, pruning errors, and inconclusive ZINC evidence.
- **Useful transfer results.** The independent graph-pair experiments provide credible evidence beyond within-graph classification.

## 5. Weaknesses

- **Incremental representation novelty.** Much of the framework extends established diffusion/characteristic-function summaries; several propositions formalize expected properties.
- **The strongest simple ablations are not sufficiently prominent.** Layer-only and matched lazy-RWSE features explain much of the transfer improvement.
- **Some comparisons are not fully matched.** In directed experiments, top-\(k\) baselines lack mixed probes; runtime comparisons differ substantially in propagation depth and approximation.
- **Experimental specification needs tightening.** The three-versus-four-step discrepancy is material, and the supplied implementation alone does not permit auditing the benchmark protocols.
- **Real-world evidence is narrower than the breadth of tables suggests.** Many labels correlate strongly with degree, the food-web labels are proxies, and most centrality transfer is within synthetic generator families.
- **Some headline terminology overreaches.** “Online,” “clearly ahead,” and general efficiency claims should be scoped more precisely.
- **Presentation is overly dense.** The numerous propositions and long table notes obscure the main contribution and its strongest evidence.

## 6. Questions to the authors

1. **What are the exact parameters for centrality transfer?** Does it use three or four steps? Please reconcile the fourteen layer features with the stated default and specify pruning and dimension matching for every relevant baseline.

2. **How do the simplest fully matched alternatives perform?** Please prominently report layer-only and lazy-RWSE baselines using the same walks, and top-\(k\) summaries with the same mixed probes on directed tasks.

3. **What was the model-selection protocol?** Which datasets informed the frequency range, walk length, scaling, curvature choice, and pruning settings? How were baseline hyperparameters selected, and how many independent embedding seeds were used?

4. **Can the complete evaluation artifacts be provided?** In particular, scripts, graph-generation parameters, splits, package versions, CSVs, and relabeling tests are needed to audit the empirical claims.

5. **What operational meaning is intended by “online”?** Can selected-node embedding or dynamic updates avoid full-graph conversion and preprocessing? If not, would you revise this designation?

6. **How robust are the practical conclusions under matched accuracy budgets?** Please compare budgeted RoleWalk with same-depth lazy-walk descriptors and appropriately optimized or approximate diffusion/return-probability methods.

7. **Can the food-web labels receive an expert audit, and can the ZINC result be tested against a stronger reproduction?** These would materially strengthen the two less-established application claims.

## 7. Overall recommendation

**Score: 6 — Weak accept**  
**Confidence: 4/5**

The paper offers a sound, useful, and unusually transparent structural-embedding method, supported by substantial implementation work and credible transfer experiments. I find no major mathematical flaw in the main results. The combination of locality, directed probes, efficient computation, and explicit approximation accounting is a worthwhile contribution.

I would not assign a higher score because the conceptual novelty is moderate, several accuracy claims exceed the strength of the evidence, and the most informative matched ablations and experimental settings need clearer presentation. The walk-length inconsistency and reproducibility questions should be resolved before publication.