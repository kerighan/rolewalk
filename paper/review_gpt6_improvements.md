## My recommendation

I would **keep the characteristic function**. Your ablations are fairly strong evidence that changing the scalar summary of endpoint probabilities is not the productive direction.

The remaining opportunities are more specific:

1. **Compute the current object better**, especially around small probabilities, curvature, and pruning.
2. **Measure something the current object demonstrably discards**—most promisingly, mixed incoming/outgoing walks.
3. For a purer method, replace the separate probability/layer blocks with **a characteristic function of the whole rooted diffusion trajectory**.
4. If near-linear runtime is non-negotiable, accept an expressiveness trade-off and **transport local signatures rather than construct endpoint distributions**.

I would rank the work as follows. The probabilities are subjective estimates, not inferred from your reported standard errors.

| Rank | Proposal | Main intended benefit | Estimated chance of that benefit |
|---|---|---|---|
| 1 | Residual-CF evaluation and terminal streaming | Same features, less numerical error and lower cost | 75–90% |
| 2 | Triangle-based curvature plus exact mark propagation | Much cheaper curvature; remove pruning artefacts | 80% on hub-heavy graphs |
| 3 | Two mixed-direction walk probes | New directed-role information, with a concrete separation witness | 50–65% on directed tasks |
| 4 | Mass-budget pruning and correct shell bookkeeping | Controlled approximation, fewer hub failures | >90% for reliability; 20% for benchmark accuracy |
| 5 | Unit-invariant weighted-walk policy | Comparable weighted embeddings across weight units | High when units differ; no gain on unweighted benchmarks |
| 6 | Rooted trajectory CF | One object, no hard shells, full-block stability | 25–35% for accuracy; 50% for competitive simplification |
| 7 | Transported local-CF signatures | Genuinely near-linear sparse-graph computation | >90% for speed; 15–25% as an overall replacement |

I separate the first five, which preserve most of RoleWalk, from the two reformulations.

---

# Small improvements

## 1. Evaluate a residual CF, and stop pruning the terminal output

This is the first change I would implement.

### 1. What changes

For a probability row \(p\), write
\[
\phi_\theta(p)
=i\theta+\sum_j r_\theta(p_j),
\qquad
r_\theta(a)=e^{i\theta a}-1-i\theta a.
\]

The \(i\theta\) term is identical for every node. Therefore, use
\[
\bar\phi_\theta(p)=\sum_j r_\theta(p_j)
\]
as the coordinate. This is just a translation of the existing exact embedding.

It also reproduces your current pruning correction exactly, up to rounding. For a subprobability row \(q\),
\[
\sum_j f_\theta(q_j)+i\theta(1-\sum_jq_j)-i\theta
=\sum_jr_\theta(q_j).
\]

Thus the code no longer needs to recover a small residual by adding a potentially large missing-mass correction to the imaginary block.

Use stable evaluation:
```python
real = -2 * sin(x / 2)**2       # rather than cos(x) - 1
imag = sin(x) - x              # series near zero
```

For small probabilities, avoid evaluating every frequency separately. Split each row at
\[
a_* = 0.1/\theta_{\max}=10^{-3}.
\]
For \(a\le a_*\), accumulate six moments \(M_k=\sum a^k\), \(k=2,\ldots,7\), and evaluate
\[
\begin{aligned}
\Re\bar\phi_\theta
&=-\frac{\theta^2M_2}{2}
 +\frac{\theta^4M_4}{24}
 -\frac{\theta^6M_6}{720},\\
\Im\bar\phi_\theta
&=-\frac{\theta^3M_3}{6}
 +\frac{\theta^5M_5}{120}
 -\frac{\theta^7M_7}{5040}.
\end{aligned}
\]
Use trigonometric evaluation only for larger entries. Accumulate in float64; store the final features in float32.

**This is not another power-sum embedding.** It is an error-controlled implementation of the existing CF, and only in its small-angle regime.

Also reorder the loop:
```python
Q = T @ H

# Evaluate Q before discarding anything.
emit_features(Q)
save_required_support(Q)

# Only prune what must be propagated.
if another_step:
    T = prune(Q)
```

In particular, never prune the last power merely to compute its summary. Stream its rows into the feature accumulator and discard them.

Finally, put row blocking inside `transform`, rather than relying on benchmark wrappers. The supplied library does not implement the paper’s 50k-row blocking. Prefer a block budget based on estimated expansion work
\[
c_u=\sum_{j\in\operatorname{supp}T_u}\operatorname{nnz}(H_j),
\]
not a fixed number of roots. A block containing hubs can be radically larger than another block of the same row count.

### 2. Why it should help

- The imaginary first-order term carries no node information in the exact method.
- `cos(float32_angle) - 1` loses small real contributions through cancellation.
- The Taylor implementation has a useful absolute bound. For the small-entry real block,
  \[
  |\text{error}|
  \le \frac{\theta^8M_8}{8!}
  \le \frac{\theta^8a_*^7}{8!}\sum_{\rm small}a.
  \]
  At \(\theta=100\), this is at most approximately \(2.5\times10^{-10}\) per row. The imaginary bound is smaller.
- Terminal pruning currently perturbs shell masses and curvature moments even though it saves no subsequent multiplication.
- Adaptive blocking addresses transient SpGEMM memory, which an output threshold does not bound.

### 3. Cost and dimensions

- **Dimensions:** unchanged.
- **Time:** same sparse products; fewer transcendental evaluations when many entries are small. Six scalar moment accumulations per small entry, rather than ten pairs of trigonometric evaluations.
- **Memory:** bounded by the active root block and its expansion, not the whole final power.

This does **not** make worst-case exact walk expansion linear.

### 4. Cheap experiment

On USA, wiki-RfA, and the 20k BA timing graph:

1. Separate timing into sparse products, CF evaluation, shells, curvature.
2. Compare pairwise feature differences against the current implementation and a float64 reference.
3. Compare peak memory with degree-adaptive versus fixed-row blocks.
4. Require unchanged downstream results, apart from numerical tolerance.

Do not count extra Atlas/BREC separations caused solely by changing the numerical grouping threshold after removing \(i\theta\).

### 5. Chance it helps

About **85% for a worthwhile implementation improvement**, but probably **below 10% for measurable classification improvement**. If sparse multiplication accounts for almost all runtime, expect memory and numerical benefits rather than a dramatic speedup.

---

## 2. Compute curvature by triangle enumeration, and propagate its moments independently

Your curvature is much simpler computationally than the current implementation makes it.

### 1. What changes

For an edge \(xy\) in the symmetrized unweighted graph, let \(c_{xy}\) be its number of common neighbors. Then exactly
\[
\kappa(x,y)
=\frac{c_{xy}+2}{\max(d_x,d_y)+1}.
\]

Consequently:

1. Count triangles incident to each edge.
2. Compute the expression above.
3. Average it over incident edges.

Replace
```python
H[x].minimum(H[E.col])
```
with a degree-/degeneracy-oriented triangle enumerator. The current expression materializes repeated sparse rows, which is particularly unfortunate around hubs.

Then compute the transported moments using sparse matrix–vector products:
\[
a_0=\kappa,\quad b_0=\kappa^2,\qquad
a_w=Ha_{w-1},\quad b_w=Hb_{w-1}.
\]
Emit
\[
(a_w,\sqrt{\max(b_w-a_w^2,0)}).
\]

These are exactly the unpruned curvature coordinates, independently of whether endpoint distributions were pruned.

For a kernel-clean variant, emit \((a_w,b_w)\) instead. I would make that a separate, very small ablation—not assume that raw moments classify as well as mean/std.

### 2. Why it should help

The overlap formula follows because the two uniform closed-neighborhood measures overlap on exactly \(c_{xy}+2\) vertices, with common mass
\[
\min\{1/(d_x+1),1/(d_y+1)\}.
\]

Triangle enumeration avoids the current repeated-neighborhood expansion. With a suitable oriented enumeration, the usual bound is \(O(|E|\alpha)\), where \(\alpha\) is arboricity, rather than the current degree-square behavior.

Independent propagation also fixes a subtle pruning effect. With a subprobability row \(q\), your current variance calculation effectively assigns all missing mass curvature zero:
\[
q+\bigl(1-\|q\|_1\bigr)\delta_0.
\]
That is not the curvature distribution of the walk.

### 3. Cost and dimensions

- **Curvature preprocessing:** approximately \(O(|E|\alpha)\), with \(O(|E|)\) edge storage.
- **Propagation:** two SpMVs per step and orientation, \(O(L|E|)\).
- **Dimensions:** unchanged: two per step and orientation.
- **Online subset queries:** use a local dependency computation; do not run global SpMVs merely to embed a few roots.

### 4. Cheap experiment

Use the existing curvature benchmarks plus BA and a large star.

Check:

- bitwise/tolerance agreement of \(\kappa\) with the original on small graphs;
- curvature preprocessing time and peak memory;
- exact propagated moments versus `T @ kappa` under `tol=0`;
- how much the existing pruned moments differ at default tolerance.

Then rerun airports and directed motifs with exact moment propagation. The preprocessing change itself should not alter accuracy.

### 5. Chance it helps

**80–90% for curvature-enabled scaling on hub-heavy graphs.** Probably **10–20% for accuracy**, because your benchmark tolerance may already make the moment errors small.

I would retain curvature as an option. Its directed and ZINC results are too useful to delete on aesthetic grounds.

---

## 3. Add two alternating-direction probes

This is my highest-value **representational** proposal.

### 1. What changes

Let
\[
H_+=\operatorname{rownorm}(I+A),\qquad
H_-=\operatorname{rownorm}(I+A^\top).
\]

Besides powers of each separately, compute
\[
R_{+-}=H_+H_-,
\qquad
R_{-+}=H_-H_+.
\]

Summarize each with the existing CF, initially at three existing frequencies, for example `theta[:, [0, 5, 9]]`. No new shells or curvature marks:
```python
Z_pm = compute_iter(H_out @ H_in, theta_small)
Z_mp = compute_iter(H_in @ H_out, theta_small)
X = hstack([X, Z_pm, Z_mp])
```

That adds **12 real coordinates**.

These are actual edge-direction changes, not labels recording movement relative to a root’s distance layers. The Q8 and intra-layer experiments do not test this.

### 2. Why it should help

Current directed RoleWalk observes:

- where a purely outgoing walk goes;
- where a purely incoming walk goes.

It does not directly observe shared successors or shared predecessors.

There is a particularly clean witness, **even with your curvature option**:

- Graph A: one source \(u\) points to two sinks.
- Graph B: directed \(K_{2,2}\), with both sources pointing to both sinks; root at one source \(u\).

For the root:

- every purely outgoing walk has the same distribution in both graphs;
- the incoming walk stays at \(u\);
- all reached nodes have curvature \(2/3\);
- all existing shell and curvature coordinates agree, at every horizon.

But
\[
e_u^\top H_+H_-
=
\begin{cases}
(2/3,\;1/6,\;1/6), &\text{Graph A},\\
(5/9,\;2/9,\;1/9,\;1/9), &\text{Graph B}.
\end{cases}
\]

Thus the new CF separates a pair the current full method cannot.

This is relevant to wiki-RfA: “voters sharing candidates” and “candidates sharing voters” are meaningful roles, not artificial graph-isomorphism distinctions.

### 3. Cost and dimensions

- **Time:** two additional sparse products. Their support sizes can be large around popular targets/sources; this is not a free addition.
- **Dimensions:** +12 in the diagnostic experiment.
- **Deployment:** replace some existing CF samples rather than retaining an ever-growing concatenation.

For example, with exact shell contrasts described below, a directed 60-dimensional version can use:

- three frequencies for the original three steps: 36 coordinates;
- shell contrasts: 12;
- the two mixed probes: 12.

Compare it against a **60-dimensional baseline** using four frequencies and no mixed probes.

### 4. Cheap experiment

First verify the witness above.

Then run only:

- directed motifs at 0%, 5%, 10% noise;
- wiki-RfA tasks A and B.

Use both:

1. append-only, to establish whether the information is useful;
2. dimension-matched, to establish whether it is worth its budget.

Measure intermediate nnz and time. Reject the default addition if its accuracy improvement comes with, say, a several-fold slowdown on RfA.

### 5. Chance it helps

**50–65% on at least one directed benchmark; 35–45% for a favorable accuracy/time trade-off.**

I would not add it to undirected RoleWalk: there \(H_+=H_-\), so it is just the existing second power.

---

## 4. Replace threshold-only pruning with a mass budget; fix shell support timing

This is primarily an approximation-quality improvement.

### 1. What changes

Maintain a per-root missing-mass budget \(\delta\).

After forming a row \(Q_u\), only discard entries whose **total additional mass** fits within
\[
\delta-\bigl(1-\|Q_u\|_1\bigr).
\]

A simple first implementation:

```python
candidate = Q_row.data < tol
candidate_mass = Q_row.data[candidate].sum()
already_missing = 1 - Q_row.data.sum()

if already_missing + candidate_mass <= delta:
    discard(candidate)
else:
    lower_threshold_or_keep_row()
```

An adaptive threshold can use bins or selection. Do not break equal-probability ties by node ID; that would break relabeling invariance.

For the default \(L=3\), save the support of \(H^2\) **before pruning it**. Since \(H\) is not pruned, that support is exact, and hence all shells required through step 3 are exact.

Your present code saves the pruned support. It need not contain the previous support, so the differenced cumulative masses can even produce negative “layer masses.”

Merely unioning supports repairs nesting, but not distances. Saving the pre-pruning second-step support is the better fix for \(L=3\).

### 2. Why it should help

Let \(p^w_u\) be exact and \(q^w_u\) the retained subdistribution. Nonnegative propagation and deletion imply
\[
0\le q^w_u\le p^w_u,
\qquad
\|p^w_u-q^w_u\|_1=r^w_u=1-\|q^w_u\|_1.
\]

For your tangent-corrected CF,
\[
\left\|
\phi(p^w_u)
-\left[\phi(q^w_u)+i\Theta r^w_u\right]
\right\|_2
\le 2\|\Theta\|_2r^w_u.
\]

If shells use the **true partition**, their scaled error is at most
\[
\|\Theta\|_2r^w_u.
\]

Thus, excluding curvature or using proposal 2,
\[
\|X_u-\widehat X_u\|_2
\le
\sqrt5\,\|\Theta\|_2
\sqrt{\sum_{w=1}^L(r^w_u)^2}.
\]

This gives an actual error certificate. The bound is conservative, but it correctly identifies the star failure rather than treating every small entry as harmless.

For \(L>3\), pre-pruning supports are not necessarily exact after earlier pruning. Exact shell certificates then require independent Boolean reachability or another exact distance computation.

### 3. Cost and dimensions

- **Time:** row reductions and adaptive thresholding; potentially much more retained work.
- **Memory:** exact second-step support can be larger than the pruned numerical matrix.
- **Dimensions:** unchanged. Missing-mass certificates should be diagnostics, not structural coordinates.

There is no simultaneous guarantee of small error and bounded support: a nearly uniform distribution can require retaining almost everything. I would expose that trade-off, not conceal it with another heuristic.

### 4. Cheap experiment

On the existing BA/WS scalability graphs and centrality-transfer graphs:

- plot speed versus certified error;
- report the distribution of \(r_u^w\), not merely average nnz;
- compare against exact embeddings for a small root sample;
- include stars spanning the \(10^4\)-leaf threshold;
- assert nonnegative shell masses.

Then check whether the worst certificate values correlate with actual feature error or centrality-transfer failures.

### 5. Chance it helps

**Very high for correctness and diagnosability.** Only about **20% for visible benchmark accuracy gains**, and there is a real chance it slows the default enough that you retain threshold pruning as an explicitly uncertified fast mode.

---

## 5. Make the weighted-walk convention explicit and optionally unit-invariant

This is conditional on what “weighted roles” should mean.

### 1. What changes

Your current self-loop has weight exactly one. Therefore, changing all edge weights from kilograms to grams changes the walk substantially.

For applications where only relative edge weights matter, use
\[
s_u=\sum_v A_{uv},\qquad
d_u=|\{v:A_{uv}>0\}|,\qquad
P_{uv}=A_{uv}/s_u,
\]
and
\[
H_{uu}=\frac1{d_u+1},\qquad
H_{uv}=\frac{d_u}{d_u+1}P_{uv}\quad(v\ne u).
\]
Isolates remain absorbing.

Equivalently, the artificial self-loop receives the mean positive incident edge weight \(s_u/d_u\), instead of the universal weight one.

This exactly preserves your unweighted walk.

### 2. Why it should help

It makes the embedding invariant under a global rescaling \(A\mapsto cA\), while retaining relative weight information and the existing degree-dependent holding probability.

However:

- it deliberately removes absolute strength as a signal;
- counting positive edges introduces a discontinuity when a new edge’s weight approaches zero;
- consequently, it does not preserve the current weighted CF’s continuity at edge appearance.

I would offer this as a **semantic policy**, not declare it universally superior. If edge weights are meaningful counts in common units, the current convention may be preferable.

### 3. Cost and dimensions

\(O(|E|)\) preprocessing, no additional dimensions or walk products.

### 4. Cheap experiment

Use weighted versions of the existing airport and centrality-transfer graphs:

1. keep topology and relative weights fixed;
2. rescale source and target graphs independently by \(10^{-2},1,10^2\);
3. test transfer.

Also verify exact agreement with current unweighted embeddings.

Your present tables do not establish whether absolute weighted strength or relative-weight structure is the desired target. A unit-rescaling test establishes the semantic behavior, not general predictive superiority.

### 5. Chance it helps

**Zero expected improvement on the current unweighted benchmarks.** High value if weighted graphs arrive in inconsistent units; otherwise uncertain. I would implement this only if weighted support is a serious intended use case.

---

# Purer reformulations

## 6. One CF of the rooted diffusion trajectory

This is the reformulation I would investigate first.

### 1. What changes

For each endpoint \(j\), retain its entire rooted trajectory
\[
z_u(j)=
\bigl(
\mathbf1_{\{j=u\}},
p^1_{uj},p^2_{uj},p^3_{uj}
\bigr).
\]

Define one finite measure
\[
\nu_u=\sum_{j:z_u(j)\ne0}\delta_{z_u(j)}
\]
and embed it by
\[
F_u(\omega)=\sum_j\left(e^{i\langle\omega,z_u(j)\rangle}-1\right).
\]

Delete the separate shell block.

A very concrete 48-dimensional pilot uses three existing frequencies and these eight projections:
\[
\begin{gathered}
e_1,\ e_2,\ e_3,\\
(e_0+e_1)/2,\ (e_0+e_2)/2,\ (e_0+e_3)/2,\\
(e_1+e_2)/2,\ (e_2+e_3)/2.
\end{gathered}
\]
For each projection \(a\), evaluate the ordinary CF of
\[
R_a=\sum_{t=0}^3a_tH^t.
\]

Each \(R_a\) is a walk stopped at a fixed, graph-independent random time. No new walk powers are needed.

For directed graphs use
\[
z=(p^0,p_+^1,p_+^2,p_+^3,p_-^1,p_-^2,p_-^3).
\]
A 60-dimensional pilot can use ten projections:

- the six single-time coordinates;
- \((p^0+p_+^3)/2\), \((p^0+p_-^3)/2\);
- \((p_+^1+p_+^3)/2\), \((p_-^1+p_-^3)/2\);

at three frequencies each.

These particular finite probe sets are starting designs, not canonical optimal quadratures.

### 2. Why it should help

There are three substantive differences from the failed per-layer CF experiment.

**First, it preserves endpoint alignment across time.**  
The current method retains separate probability multisets. It forgets whether a large probability at step 2 and a large probability at step 3 belong to the same endpoint. Mixed-time probes observe this alignment.

**Second, the full object subsumes the curvature-free footprint.**  
With all frequencies/projections, the joint CF identifies the multiset of nonzero trajectories. From that:

- the root is identified by \(p^0\);
- the first positive time identifies distance, because the walk has positive holding probabilities;
- every marginal probability multiset and every shell mass can be recovered.

This is an information statement about the full object—not a guarantee that 24 complex samples dominate the existing embedding.

**Third, it removes hard-distance discontinuities.**  
For a nonnegative projection \(a\) summing to one,
\[
\|e_u^\top R_a(H')-e_u^\top R_a(H)\|_1
\le
\left(\sum_ta_tt\right)\|H'-H\|_\infty.
\]
The usual CF Lipschitz bound then applies to **every coordinate**, including the root-sensitive probes. There is no separate unstable shortest-distance block.

It is genuinely one object and one anchored Fourier kernel. It still has a finite-probe design choice; purity does not make that choice disappear.

### 3. Cost and dimensions

- **Dimensions:** 48 undirected or 60 directed in the pilots above.
- **Walk products:** unchanged.
- **CF cost:** potentially higher. Mixed probes can all touch the largest support; 24 complex evaluations there may cost more than ten at each of three increasingly large supports.
- **Memory:** need aligned numerical values from earlier powers, not merely Boolean supports.

So this is primarily a purity/stability proposal, **not automatically an efficiency proposal**.

### 4. Cheap experiment

Use:

1. Atlas and your explicit WL counterexamples;
2. airport classification and transfer;
3. noisy directed motifs;
4. a random-edge insertion test comparing the movement of the **whole embedding**.

Compare at matched dimension. Also compare “marginal probes only” against “marginal + mixed-time probes,” so any gain is attributable to trajectory alignment rather than frequency-budget changes.

I would stop before ZINC if it materially loses on airports and motifs. BREC’s distance-regular failures are not expected to disappear.

### 5. Chance it helps

About **25–35% for an accuracy improvement**, and **50% for a competitively accurate, cleaner method**. The main risk is that hard shells are a very effective finite-dimensional extraction of information that sparse joint probes recover inefficiently.

---

## 7. Transport local CF signatures instead of endpoint distributions

This is the only proposal here that really removes sparse-power explosion. It also gives up something important.

### 1. What changes

Compute the one-step local signature
\[
b_u(\theta)=\sum_j f_\theta(H_{uj}).
\]

Then propagate that signature as a node signal:
\[
Z_0=b,\qquad Z_{t+1}=HZ_t,
\]
and emit \(Z_0,Z_1,Z_2\).

Interpretation:
\[
Z_t(u,\theta)
=
\sum_v(H^t)_{uv}\sum_j f_\theta(H_{vj}).
\]

This describes the distribution of **local transition laws encountered by a walk**, rather than the distribution of its endpoint probabilities.

For directed graphs, take
\[
b_u=(b_u^+,b_u^-),\qquad
M=(H_++H_-)/2,
\]
and emit \(b,Mb,M^2b\). With five frequencies, this is 60 real coordinates. For undirected graphs, ten frequencies give 60.

This is a single transported-measure construction:
\[
\nu_u^t=\sum_v(M^t)_{uv}\,\nu_v^{\rm local},
\]
with a fixed CF feature map. There are no hard shells, curvature, endpoint pruning, or support-dependent approximations.

### 2. Why it might help—and why it might not

It retains the variation in local transition laws that endpoint aggregation can erase. For example, the neighbor-degree discrepancy in your WL counterexample can affect the transported one-step signatures even when the endpoint distributions coincide.

But there is a decisive limitation:

> On an unweighted regular graph, the one-step signature is constant, so every propagated signature is constant.

It therefore loses your triangle separation and much of the expressiveness table. No amount of additional propagation fixes that.

This is not my proposed new flagship RoleWalk. It is a principled **fast mode** to test if the practical role tasks rely mostly on distributions of local branching behavior. It resembles the economical side of ReFeX, but with fixed transferable coordinates and a specified kernel object.

The previous joint degree-mark result makes me cautious about predictive gains. The reason to try this is chiefly its computational model, not a claim that another degree encoding will suddenly work better.

### 3. Cost and dimensions

- **Time:** \(O(Lm|E|)\), using sparse–dense products.
- **Memory:** \(O(m|V|+|E|)\).
- **Dimensions:** 60 in the designs above.
- **Locality:** radius 3 for \(b,Mb,M^2b\).
- **Pruning:** unnecessary.

This is a much stronger sparse-graph scaling guarantee than retaining at most \(1/\mathrm{tol}\) endpoints per row.

### 4. Cheap experiment

Run:

- the six role-classification datasets;
- centrality transfer, especially k-core;
- wiki-RfA;
- the million-node timing benchmark.

There is no need to spend much time testing regular-graph expressiveness: the failure is analytic.

My acceptance criterion would be something like “within one percentage point on most role tasks, with a substantial memory/runtime reduction,” not “beats the original everywhere.”

### 5. Chance it helps

**Above 90% for scaling**, **perhaps 35–45% for being a useful fast mode**, and only **15–25% as the preferable overall method**.

---

# What I would remove

## Remove unconditionally from the canonical embedding path

1. **Terminal-step pruning.** There is no future propagation to accelerate.
2. **The numerically reconstructed first-order imaginary constant.** Use residual CF coordinates.
3. **The current sparse-row duplication implementation of curvature.**
4. **Pruned supports as if they were exact distance balls.**
5. **`aggregate="mean"` from the scientific method**, retaining it only for legacy model loading.
6. **The random projection as the recommended dimension-control mechanism.** Allocate a fixed feature budget directly.

## Remove exact redundant shell coordinates

At step \(w\), the exact shell vector has \(w+1\) entries summing to one. Let \(C_w\) have orthonormal rows spanning \(\mathbf1^\perp\), such as Helmert contrasts. Replace \(s^w\) by \(C_ws^w\).

Then
\[
\|C_w(s_u^w-s_v^w)\|_2
=
\|s_u^w-s_v^w\|_2.
\]

For \(L=3\), this reduces nine shell coordinates to six **without changing exact Euclidean distances**.

With pruning, varying total mass is an approximation artefact; either retain it as a diagnostic or explicitly acknowledge that deleting that direction changes the approximate geometry.

## Remove only if the corresponding experiment succeeds

- **Hard shells:** only if the trajectory-CF reformulation is competitive.
- **Curvature standard deviation:** only if raw second moments perform comparably. Raw moments restore the linear mean-embedding interpretation, but that is not sufficient reason to accept an accuracy loss.
- **Curvature itself:** not globally. Your directed results argue against that.

## One dimension-budget issue to face directly

The current default is 69/75 dimensions undirected and 138/150 directed. The directed version is already outside a strict “few dozen” budget.

I would therefore report both the existing default and a fixed **60-dimensional track**. For example:

- directed, curvature-free: four frequencies, three steps, shell contrasts;
- directed with curvature: three frequencies, three steps, shell contrasts, curvature;
- directed mixed-walk variant: three frequencies, shell contrasts, and the 12 mixed coordinates.

Reducing frequency count is not itself a claimed improvement. It is the necessary control for deciding whether new information is worth its dimensional budget.

**If I had only one week:** implement proposals 1 and 2, test the mixed-direction witness and benchmarks, and instrument missing mass. **If I had one research month:** test the rooted trajectory CF. I would not spend that month searching for another marginal probability summary.