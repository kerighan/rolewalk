## Bottom line

**8/10 — accept. Confidence: 3/5 overall; 4/5 on the textual, mathematical and static-code assessment.**

This is a **stronger acceptance case than round 7**, principally because the Digraphwave rerun and matched Enron ablations substantially improve the evidence. The revised paper also preserves most of the qualifications that matter: the Film loss, competitive simple baselines, the centrality top-\(k\) tie, food-web non-separation, pruning failures, and preliminary ZINC evidence.

**The condensed submission is largely faithful to the report, but synchronization is still incomplete.** Most importantly, the two documents directly disagree about whether the centrality top-\(k\) control uses exact walks. Two previously requested centrality disclosures—the matched lazy-RWSE result and the aggregation convention—are still absent. The new “Beyond square adjacency” section also introduces an incorrect contrast with diffusion methods.

These are bounded repairs. I would not require another experimental programme before submission.

**Scope:** I reviewed the supplied LaTeX, Typst and Python statically. I did not execute experiments, compile the documents, inspect the referenced CSVs or figures, run the added tests, or independently verify the original Digraphwave protocol.

# 1. Reviewer assessment

### What is now convincing

**The empirical contribution is clearer and better controlled.** The paper is no longer relying mainly on comparisons against globally normalized diffusion signatures or per-graph learned coordinates.

- The **Digraphwave rerun** addresses a real concern about implementation and protocol fidelity. Reproducing the noiseless published accuracy, obtaining nearby noisy results, and winning on every shared noisy draw make the alignment comparison materially more credible.
- The **Enron ablations provide useful attribution**. The CF beats a dimension-matched top-\(k\) replacement under the no-mixed-probe control; mixed probes improve matching; layers hurt this particular raw-distance task.
- Crucially, the paper **reports the inconvenient outcomes**: CF-only beats the full embedding, and local statistics have no detected aggregate disadvantage to RoleWalk and win at higher noise. Those results belong in the main paper, and they are there.
- The **GIN comparison** adds a meaningful trained inductive baseline. It is explicitly identified as one untuned configuration, rather than presented as a comprehensive GNN benchmark.
- The related-work additions and positioning paragraph make the **moderate, compositional novelty** much easier to assess fairly.

### The scientific conclusion should now be more task-specific

The strongest supported story is:

> Distance layers carry most of the tested centrality size-transfer performance; characteristic functions and mixed probes carry Enron alignment performance.

That is more informative than a universal claim that the distance law is *the* statistic carrying information across graphs. Indeed, Enron is itself cross-graph transfer between graphs of different sizes, and removing the layers improves it.

This does not weaken the paper into an uninteresting collection of features. It makes the contribution a credible, efficient structural-signature framework with **different useful components for different tasks**, rather than a uniformly superior fixed geometry.

### Why 8, not 9

The empirical package is substantial, but:

- conceptual novelty remains principally the combination and implementation of established ingredients;
- the kernel interpretation is coherent but not, by itself, evidence that the chosen geometry is optimal;
- simple features remain highly competitive;
- centrality transfer is demonstrated chiefly within specified synthetic families;
- the trained-GNN evidence is one configuration;
- certified pruning demonstrates affordable error control more clearly than a substantial certified speedup over exact computation.

The new results strengthen acceptance and attribution. They do not remove these scope limitations.

# 2. Highest-priority edits

## A. Resolve the experimental-settings contradiction

This remains the most important factual repair.

The condensed paper says:

> “centrality transfer … uses \(L=4\) with exact walks … for RoleWalk and for its dimension-matched ablation alike.”

The technical report says:

> “RoleWalk exact, \(\mathrm{tol}=0\); top-\(k\) pruned at \(10^{-5}\).”

Both documents retain the same reported top-\(k\) mean, 0.812. From the supplied material, I cannot determine whether the control was rerun exactly and the report is stale, or whether the condensed statement is incorrect.

**State the settings actually used to produce the reported numbers.** Do not call the top-\(k\) computation exact unless the reported results come from that computation. A pruning difference is not fatal to this control, but it must be disclosed.

The opening “All experiments use the defaults … except centrality transfer” is also still too broad. Expressiveness evaluations use exact walks, five-step experiments change the horizon, and pruning experiments explicitly change computation.

A suitable replacement, after resolving the top-\(k\) setting, is:

> Unless otherwise stated, experiments use the default three-step configuration. Centrality transfer uses four-step RoleWalk with exact walks; the matched top-\(k\)+layers control uses [actual setting]. Expressiveness evaluations use exact walks, and walk-length and pruning experiments vary the indicated settings explicitly.

This is especially important because the layer certificate is restricted to three steps.

## B. Restore two centrality disclosures, and finish the selection account

### Matched lazy RWSE

The report gives **0.753 for four-step return probabilities from the same lazy walk**. The main paper still shows only 0.532 for the different, sixteen-step simple-walk RWSE baseline.

Restore one sentence in “What carries the transfer?”:

> Return probabilities alone from the same four-step lazy walk reach 0.753, compared with 0.807 for all distance layers; the 0.532 RWSE baseline in the table uses a different walk and horizon.

This distinguishes gains from the walk convention from gains from retaining the other distance layers. It is an important attribution result, not supplementary clutter.

### Aggregation

The centrality caption should explicitly say:

> Overall table means average the 400 valid graph–target cells. Paired differences first average valid targets within each graph pair, then average the 60 pairs.

“Mean over 60 independent pairs” does not adequately describe the displayed overall mean. The different weighting also explains why a displayed mean difference need not equal a reported paired difference.

### Four-step selection

The default-selection disclosure is much improved: the datasets and selection/evaluation overlap are now visible. What remains unclear is **how \(L=4\) was selected for centrality transfer**.

State whether it was a prior design choice, selected on separate pilots, or chosen using any reported transfer evaluations. The statement that centrality transfer did not participate in choosing the three-step defaults does not answer this separate question.

No new search is necessary; document the choice that was made.

## C. Separate the three kinds of alignment evidence precisely

The alignment section now contains:

1. published baseline results;
2. a Digraphwave rerun on two shared draws at each of three noisy levels;
3. RoleWalk and its controls on 35 noisy draws.

These are all useful, but they should not blur together.

The sentence:

> “compares RoleWalk with its own ablations and with the baselines, all variants on the same noise draws”

is too easy to read as saying that the published baseline rows are paired with the other table rows. The caption partially repairs this, but the body should be explicit.

Suggested wording:

> RoleWalk, its ablations and local statistics are evaluated on identical noise draws. Starred baseline rows are published results and are unpaired with those draws. Separately, our GPU rerun of Digraphwave uses two shared draws at each of \(p=0.01,0.05,0.1\), and RoleWalk leads on all six.

Also:

- Say the noisy rerun results **“agree closely with”** the published values, rather than simply “match” them.
- “Reproduces 0.398 at the reported precision” is more precise than “reproduces exactly.”
- Identify the complete noise-level set underlying the **35-draw aggregate**. The displayed table contains only 25 noisy draws’ worth of levels.
- Give the test or interval supporting the local-statistics comparison, at least in the report. The supplied report gives \(p=0.27\) without identifying the test in that passage.

Finally, distinguish average effects from uniform effect sizes:

> Mixed probes improve accuracy on every noisy draw, by 0.037 on average. Removing layers also improves every noisy draw, by 0.020 on average.

The current “0.02 above … on every draw” can sound like the magnitude is identical on every draw, which the table does not support.

I would also replace **“tie on average”** with:

> Their aggregate means are close, with no statistically detected difference.

Failure to reject a difference is not an equivalence result. The positive alignment result remains strong without that language.

## D. Repair the abstract and make the conclusion reflect the ablations

### The benchmark count remains wrong

The abstract still says:

> “competitive with the best role embeddings on six classification benchmarks”

The primary analysis contains five, and Film is a substantial loss. Disclosing Film in the body is good, but it does not make this abstract statement accurate.

Use:

> competitive on five primary role-classification benchmarks

and retain the Film sentence in the body.

### Scope the transfer comparison

The abstract’s “better than existing inductive embeddings” should be **“higher aggregate performance than the tested inductive baselines”**, or equivalent. The GIN comparison is appropriately phrased as comparison with *a* trained GIN; retain that specificity.

### Revise the concluding attribution

The last sentence currently generalizes the distance-law finding too far. A more faithful ending would be:

> Ablations show complementary strengths: the distance law carries most of the tested centrality size-transfer performance, while characteristic functions and mixed probes carry Enron alignment, where removing the layers improves matching.

This acknowledges the central new result without turning the conclusion into a limitations list.

I do **not** think every negative result needs to enter the abstract. The main body already gives readers the important adverse evidence. The abstract and conclusion simply should not erase its implications.

## E. Correct or shorten “Beyond square adjacency”

This is the weakest new section, because its comparison with diffusion is mathematically inaccurate.

### The reverse incidence operator needs a definition

If \(B'\) means \(B^\top\), then \(BB^\top\) is generally **not row-stochastic**, even when \(B\) is row-stochastic.

Use separately normalized forward and reverse incidences. For a nonnegative incidence matrix \(C\), for example,

\[
P_{VE}=D_V^{-1}C,\qquad
P_{EV}=D_E^{-1}C^\top,
\]

with suitable zero-row conventions. Then \(P_{VE}P_{EV}\) is the node-to-edge-to-node transition operator.

If \(B'\) was intended to mean this independently normalized reverse operator, say so explicitly.

### Diffusion does not require symmetry

The statement that diffusion signatures need a **“square symmetric operator to exponentiate”** is false. Matrix exponentials require square matrices, not symmetric ones. Your own random-walk heat-CF baseline, and directed diffusion more generally, already contradict that contrast.

Moreover, rectangular transitions can be assembled into a square block operator on the disjoint union of cell types. Diffusion can therefore also operate across types or ranks. Rank-preserving Hodge Laplacians do not establish a general inability to construct coupled operators.

A defensible replacement is:

> RoleWalk can summarize rows of compatible stochastic operators between different entity types. Extending the layer definition to a chosen incidence graph would permit walks between nodes, edges and higher-order cells. Diffusion on suitable block operators is another possible construction; evaluating these extensions is future work.

This preserves the useful observation without claiming a unique capability. I would be equally happy to move the discussion to the supplement and use the space for the missing centrality disclosures.

## F. Keep the GIN conclusion at the scope of the experiment

The new GIN result is valuable and worth retaining. Its description as one untuned configuration is honest.

But this sentence is broader than the evidence:

> “the gradient-boosted regressor on fixed features transfers better than message passing.”

Replace it with:

> Under the tested configurations, gradient boosting on RoleWalk transfers better than the trained GIN.

The supplement should also state the GIN loss, target normalization, optimizer, training budget, early-stopping criterion and training-seed treatment. These matter particularly because the GIN predicts seven targets jointly, whereas the feature regressors are fitted per target.

I am **not** requesting a tuned GNN study. The result supports a comparison with this trained baseline, not a general ranking of learning paradigms.

# 3. Smaller but worthwhile corrections

### Method fidelity

- **Curvature scaling:** the report and code append
  \(\|\Theta\|(\mu_u^w,\sigma_u^w)\), whereas the condensed curvature paragraph merely says it adds the mean and standard deviation. State the scaling.
- Define **\(\kappa_u=0\) for isolated nodes**, as the report does.
- Add **\(d>0\)** to the regular-graph RWSE relation, since \(P=A/d\) is otherwise undefined.
- In the computation paragraph, qualify the returned certificate with **“when the conditions of Proposition … hold.”** The implementation correctly returns no certificate for unsupported configurations.

The major round-7 mathematical repairs otherwise look substantially successful: walk directions and locality balls are distinguished, the pruning conventions now describe the implementation, the directed block extension is present, and the finite-grid and numerical-certificate scopes are stated.

### Two compression errors in role classification

The Film sentence says:

> “no method predicts the other classes above the majority rate”

But the report shows 0.637 versus a majority rate of 0.623. Use **“does substantially better than the majority baseline”**, or retain the report’s qualified “clearly above.” The taxonomy should remain an interpretation.

Similarly:

> “differences [on Brazil and Europe] are within \(\pm0.05\)”

should describe **interval widths**, not differences. Several displayed differences are much larger. For example:

> Paired intervals on these small networks have half-widths of roughly 0.05.

### Timing provenance

The scale caption should retain the report’s statement that timings are **after compilation/warm-up**. The separate 173-second run is now identified appropriately; that is an improvement.

Also synchronize actual resource cutoffs: the report describes Role2Vec exceeding 10 GB at 100k, while the condensed text groups it under a one-hour/16-GB failure statement. Report the limit actually established.

### Supplement synchronization

The report still contains stale formulations that the condensed paper has mostly improved:

- abstract: “best scores … on the Enron network alignment”;
- introduction: parameter-shift superiority over every embedding;
- alignment caption: “nothing tuned”;
- the centrality top-\(k\) computation setting;
- wiki-RfA: the top-\(k\) row still needs its **no-mixed-probes** qualification.

These should be reconciled before release. The supplement should not retain stronger conclusions than the main paper now supports.

### Library

The shown constructor now performs the requested integer coercions and checks the float32 frequency norm. Those repairs are visible in the code. I have not run the added tests, and would not describe them as independently verified.

# 4. Final recommendation

**Keep the 8/10 and submit after a focused consistency pass.**

The most important remaining work is:

1. resolve the exact-versus-pruned top-\(k\) contradiction;
2. restore matched lazy RWSE and the aggregation convention;
3. distinguish published, rerun and paired alignment evidence;
4. correct the abstract’s benchmark count and task-general attribution;
5. remove the false diffusion contrast.

The paper’s honesty is now a strength overall. The new ablations do not need to be explained away: they give the work a better scientific conclusion. Present that task dependence plainly, fix the remaining factual mismatches, and resist adding more experiments merely to make every component win everywhere.