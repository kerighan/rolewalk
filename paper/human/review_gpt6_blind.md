# Review

**Scope.** I reviewed the manuscript, technical report, and supplied `rolewalk.py` by static inspection. I did not execute the experiments. The benchmark scripts, result CSVs, and environment specifications referenced in the report are not included in the supplied code, so I can assess internal consistency and implementation logic, but not independently reproduce the reported measurements.

## 1. Summary of the paper and its claims

RoleWalk is a training-free structural node embedding based on short, degree-dependent lazy random walks. At each step, it combines:

1. A centered, unnormalized characteristic-function summary of the multiset of transition probabilities.
2. The probability mass assigned to each shortest-path-distance layer around the starting node.

Directed graphs additionally use separate forward and reverse walks and two mixed-direction products, \(H_+H_-\) and \(H_-H_+\). An optional block summarizes a local neighborhood-overlap quantity interpreted as an upper bound on Ollivier–Ricci curvature.

The principal claims are that RoleWalk provides graph-independent coordinates, incorporates direction and weight information, achieves competitive structural-role classification and strong alignment/transfer performance, and scales efficiently using sparse propagation. The theoretical contribution includes locality and invariance, a kernel-mean-embedding interpretation, comparisons with 1-WL, a distance-regular failure class, and error bounds for probability pruning.

The most convincing empirical contributions are the Enron alignment results, the controlled centrality-transfer experiments, and the accuracy–cost trade-off. The supplementary report usefully shows that distance-layer masses account for most of the centrality-transfer benefit; the characteristic function is not uniquely responsible for that result.

## 2. Soundness

### 2.1 Mathematics

Overall, the mathematical arguments are mostly correct under their stated restrictions. I found one important inconsistency between the mathematical specification and implementation, discussed below.

**Walk and locality.** The transition rule is row-stochastic once sinks are explicitly assigned an absorbing self-loop, as the supplement and code do. The locality argument is correct: an \(L\)-step walk needs transition normalizers only at vertices reached before the final step, so the induced rooted \(L\)-ball suffices. The additional radii for mixed probes and curvature are sufficient. For directed graphs, this neighborhood-containment statement should use weak/underlying-undirected balls, regardless of the distance convention used for the layer features.

**Injectivity lemma.** The analytic-function argument correctly establishes recovery of a finite multiset of positive probabilities from an interval of frequencies. The almost-everywhere statement for a countable family of pairs is also correct. It does **not** establish injectivity of the particular deterministic ten-frequency grid, or universal injectivity over weighted graphs. The supplementary report appropriately acknowledges this limitation.

**Kernel interpretation and scaling.** The proposed kernel is positive semidefinite, and the feature-map calculation correctly gives an MMD extended to finite measures. The Taylor remainder bound is also consistent with the stated inequalities.

However, the interpretive conclusion is stronger than the mathematics warrants. Any relative weighting of these additive feature blocks defines a positive-semidefinite direct-sum kernel. The factor \(\|\Theta\|\) is uniquely selected only after choosing the particular first-order matching criterion and orthonormal layer coordinates. This is a reasonable justification, but not a general proof that the concatenation or its metric is uniquely natural. Moreover, the expansion is useful only when \(\theta a,\theta b\) are small, which need not hold for low-degree vertices or concentrated return probabilities. The supplement explains this more carefully than the main paper.

**Expressiveness.** I checked the main constructions:

- The triangle identity
  \[
  s_u^2(1)=\frac{2d+2t_u}{(d+1)^2}
  \]
  is correct on simple, unweighted, \(d\)-regular graphs.
- The \(K_5\)-based two-round collision is correct. The stated probabilities after attaching the extra leaf also check out, establishing the three-round node-level collision.
- The distance-regular failure result follows correctly from the distance-matrix algebra, including the constant-curvature argument.
- The mixed-probe example yields the stated probability multisets and correctly demonstrates information absent from the separate pure walks.
- The supplementary long-cycle construction provides a valid graph-level locality limitation.

These are useful results, although separating some 1-WL-equivalent graphs is already familiar for random-walk and rooted structural statistics; it should not be read as general dominance over 1-WL.

**Stability and curvature.** The supplementary edge-insertion bound for the characteristic-function block follows from the stochastic-matrix telescoping argument. The report correctly does not extend this bound to the distance layers. The neighborhood-overlap formula and its upper-bound relationship to Ollivier–Ricci curvature are also correct for the simple, unweighted symmetrization. This quantity is an overlap proxy, not the actual Ollivier–Ricci curvature.

**Pruning.** The missing-mass argument is correct for the implemented convention: nonnegative propagation without renormalization, the first step retained exactly, support information recorded before deletion, and the stated restriction \(L\leq3\). The first-order imaginary correction contributes the extra factor in the bound. The implementation also correctly withholds `certificate_` for curvature, longer walks, and unsupported aggregation modes.

The distinction between an exact-arithmetic pruning bound and a certificate on the actual floating-point output is important. The report makes this distinction explicitly; the code does not provide a bound on all numerical errors. This is acceptable if consistently described as a **pruning-error certificate**, rather than a fully verified numerical certificate.

### 2.2 A definite specification–implementation inconsistency

**The directed distance-layer definition is inconsistent across the main paper, supplement, and code.**

The LaTeX appendix says that distances on directed graphs are taken in the underlying undirected graph. In contrast, both implementations derive distances from the supports of successive powers of the particular directed transition matrix. These are forward distances for \(H_+\) and reverse distances for \(H_-\), not underlying-undirected distances.

For example, consider the directed cycle
\[
u\to v\to z\to u.
\]
For the forward walk,
\[
p_u^2=(1/4,1/2,1/4).
\]
The implementation produces layer masses
\[
(1/4,1/2,1/4),
\]
because the forward distances are \(0,1,2\). Under the appendix’s undirected-distance convention, the layers would instead be
\[
(1/4,3/4,0).
\]

This discrepancy exists without pruning. It therefore changes the defined embedding, not merely its approximation.

The likely correction is to define separate directed distances \(d_+\) and \(d_-\) for layer features, while retaining weak distance for locality balls. That would preserve the implementation and the intended proofs. If undirected-distance layers were actually intended, the implementation and directed results require revision.

Two additional method/protocol discrepancies should be corrected:

- The main method description omits the first-order imaginary correction and the rule that the first walk step is never pruned. Both are important for understanding the implemented approximation and the layer-distance argument.
- The main paper says that centrality transfer differs from the defaults only in walk length, whereas the supplement states that RoleWalk uses exact walks, `tol=0`. The top-\(k\) comparator uses a different pruning threshold. These settings should be explicit and, preferably, matched.

### 2.3 Do the experiments support the claims?

**Broadly yes, but the strongest defensible claims are narrower than some headline wording.**

**Role classification.** The results support competitiveness and substantial speed advantages over several evaluated implementations. They do not show general accuracy dominance: degree and local statistics are strong, and several differences are small. “Not significantly worse” is not evidence of equivalence, particularly on the small airport datasets.

**Directed roles.** Wikipedia RfA provides useful real-world evidence. The improvement over local statistics is modest, while the improvement over Digraphwave is clearer. Task B is partly determined by candidate/voter activity structure, as indicated by the 0.922 degree baseline. The paper should distinguish retrospective role identification from prospective prediction of election outcomes; an aggregated graph may contain interactions occurring after the relevant election.

**Alignment.** This is arguably the strongest application result. The gains over published Digraphwave results are substantial, including the deterministic zero-noise case. However, the noisy comparisons use unpaired draws and provide no uncertainty estimates. Re-running the principal baselines on identical perturbed graphs would make the comparison considerably stronger. The alignment section also lacks attribution experiments identifying whether layers, mixed probes, or the characteristic function drive the gains.

**Centrality transfer.** The 60 independent graph pairs and explicit component ablations are strong aspects of the evaluation. The conclusion that distance-layer masses carry much of the benefit is well supported. The dimension-matched top-\(k\) result appropriately limits any claim that the characteristic function itself is essential.

Nevertheless, this is primarily within-generator transfer. The parameter-shift experiment and airport transfer demonstrate a more limited picture: local statistics can win, and airport transfer is substantially better with local statistics or degree than with RoleWalk. The paper reports these findings, but they should temper broad statements about transfer superiority.

**Statistical testing.** The Nadeau–Bengio correction is preferable to treating 50 overlapping splits as independent. It does not, however, create independent graph-level evidence or fully address dependence among connected nodes. The supplementary report also states that multiple comparisons are uncorrected. Claims such as “significantly better than every other method” should identify this qualification. Variability across embedding-training seeds should also be distinguished from variability across classifier splits.

**Model selection.** The defaults were selected using a grid evaluated on synthetic and real benchmark networks. Leave-one-dataset-out stability is reassuring, but does not by itself establish a fully held-out evaluation of the overall method-development process. The authors should identify which datasets informed configuration selection and which were untouched until final evaluation.

**Scaling.** The implementation plausibly supports the reported efficiency: fused root-wise propagation, bounded retained support under threshold pruning, and small-probability Taylor evaluation are sensible engineering choices. The report deserves credit for exposing substantial worst-node errors rather than reporting only averages. Still, the million-node result concerns particular sparse graph families, not a general near-linear complexity guarantee. Comparisons with 16-step RWSE also combine differences in walk length, information extracted, and implementation strategy; they should not be interpreted as showing that random-walk structural encodings inherently cannot scale.

**Additional evidence.** The ZINC conclusion is appropriately preliminary. The food-web experiment is useful exploratory evidence, but agreement among LLM labelers does not establish ecological label correctness.

## 3. Novelty and significance relative to prior work

The novelty is **moderate but practically meaningful**.

The characteristic-function component is a close adaptation of GraphWave, which the authors acknowledge and formalize. Separate forward/reverse diffusion is already central to Digraphwave. Mixed products are normalized, lazy variants of familiar shared-target/shared-source constructions associated with co-citation and bibliographic coupling. The MMD representation is a valid feature-kernel construction rather than a fundamentally new kernel-embedding principle.

The most interesting methodological ingredient is the compact distance-layer mass summary, particularly its demonstrated transfer value. Combining this with fixed local coordinates, mixed-direction probes, and efficient mass-budgeted computation yields a coherent and useful method.

The related-work discussion should be broader around:

- Random-walk and anonymous-walk structural signatures.
- Distance encodings and rooted structural features.
- Other characteristic-function aggregation methods, such as FEATHER.
- Transferable structural approaches such as RiWalk and fixed-reference structural representations.

These are not necessarily equivalent methods, but they are relevant to positioning the contribution. More generally, independently refitting an embedding on every graph is not the only way to use a learned method: frozen encoders or shared reference bases may provide transferable coordinates. The paper should distinguish **training-free, independently computable coordinates** from inductivity in general.

The practical significance is stronger than the theoretical novelty: a small, fast, interpretable descriptor with convincing alignment performance could be useful even without a new expressiveness hierarchy or kernel family.

## 4. Clarity and presentation

The main paper is generally clear and accessible. The “how concentrated/how far” explanation is effective, and the supplement is unusually informative about ablations, failures, and earlier confounds.

Several presentation issues need correction:

- Resolve the directed-distance inconsistency described above.
- State the implemented pruning convention in the method section.
- Label the centrality-transfer GraphWave baseline as **GraphWave without \(1/n\)**, matching the supplement.
- Reconcile the six-network classification wording with the five-network main table.
- The Film exclusion is not convincingly established by the supplied analysis. An independent set, identifier block, and weak predictability of other classes do not by themselves prove that a dataset is unsuitable for structural-role evaluation. Dataset semantics or provenance would be a stronger justification. The substantial Role2Vec advantage should remain visible.
- The strongest fixed-scale heat-kernel comparator is omitted from the main classification table despite being particularly relevant to the claimed novelty. The remaining Europe-column highlighting is consequently inconsistent: 0.596 is underlined, but no entry is bold.
- Claims that the kernel derivation makes the concatenation non-ad-hoc should be softened. It gives a useful interpretation, not a unique design prescription.

## 5. Strengths

- **Useful accuracy–cost trade-off**, especially against the evaluated learned and diffusion implementations.
- **Strong Enron alignment results**, addressing an application where cross-graph comparability genuinely matters.
- **Extensive, informative ablations**, including a properly dimension- and walk-length-matched centrality comparison.
- **Honest negative results:** parameter-shift failures, strong simple baselines, distance-regular blind spots, pruning failures on hubs, and inconclusive ZINC gains are disclosed.
- **Mostly sound theoretical analysis**, including explicit counterexamples rather than only favorable expressiveness examples.
- **Careful implementation design**, including mass budgets and per-node missing-mass reporting.
- **Clear identification of the useful statistic:** distance-layer masses, rather than attributing all gains to the characteristic function.

## 6. Weaknesses

- The directed embedding is not defined consistently between the manuscript and implementation.
- The conceptual contribution is incremental relative to diffusion signatures, walk statistics, and existing directional similarity constructions.
- Real-world cross-graph role-transfer evidence is less decisive than the synthetic centrality-transfer evidence.
- Some headline significance claims rely on one graph, overlapping node splits, and uncorrected comparisons.
- Benchmark-informed configuration selection and uneven baseline tuning need clearer documentation.
- Alignment lacks the component ablations needed to explain its strongest empirical result.
- Several scalability conclusions are implementation- and graph-family-specific.
- The supplied implementation alone is insufficient to reproduce the experimental tables.

## 7. Questions to the authors

1. **Which directed distance is intended?** Please resolve the directed-cycle example above and state separately the distance used for layer features and for locality balls. Were all reported directed experiments produced by the supplied implementation?

2. **What exactly was held out during method selection?** Which datasets and evaluation metrics entered the 48-configuration search or subsequent design decisions? Were classifier preprocessing and hyperparameter selection fitted using training data only?

3. **Can the alignment comparison be rerun on shared noise draws?** Please provide variability and include layers-only, no-mixed-probe, local-statistics, and appropriately matched top-\(k\) variants.

4. **Can experimental settings be made fully consistent?** In particular, clarify exact versus pruned centrality features, equalize approximation settings for matched ablations, label the modified GraphWave baseline, and provide a pinned implementation version for each result.

5. **How were RfA outcomes and timestamps handled?** How are repeated candidacies aggregated, and does the graph include post-election interactions? Is the intended task retrospective role recognition or prospective outcome prediction?

6. **What independently establishes Film as an inappropriate role benchmark?** Can this be supported by dataset documentation rather than inferred from graph structure and model performance?

7. **Can the significance claims be qualified or strengthened?** Please report multiplicity-adjusted comparisons where appropriate and separate classifier-split variability from stochastic embedding-training variability.

## 8. Overall assessment

**Overall score: 6 — Weak accept**  
**Confidence: 4/5**

This is a useful and unusually thoroughly analyzed practical method. The combination of compact local descriptors, strong alignment results, controlled transfer experiments, and transparent approximation analysis is sufficient for a weak acceptance recommendation. The theoretical contribution is mostly elementary but sound, and the extensive negative results improve credibility.

My recommendation is not stronger because novelty is moderate, several empirical claims need tighter qualification, and the directed-distance specification must be corrected. I view that discrepancy as likely repairable by defining the intended directional layer distances consistently, rather than as evidence that the core method or reported advantages are invalid.