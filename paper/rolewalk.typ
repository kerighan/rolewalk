// RoleWalk paper, working draft. Numbers come from benchmarks/ (see the
// CSV named under each table). Compile: typst compile rolewalk.typ
#set document(title: "RoleWalk", author: "Maixent Chenebaux")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading: set block(above: 1.3em, below: 0.7em)
#show table: set text(size: 8.5pt)
#show table: set par(justify: false)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt) }, inset: 4pt)
#show figure.where(kind: table): set figure.caption(position: top)

#let statement(kind, name, body) = figure(kind: "thm", supplement: kind,
  numbering: "1", outlined: false, caption: name, body)
#show figure.where(kind: "thm"): set block(breakable: true)
#show figure.where(kind: "thm"): it => block(width: 100%, above: 0.9em, below: 0.9em,
  align(left)[*#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body)]*.* #emph(it.body)])
#let prop(name: none, body) = statement("Proposition", name, body)
#let lemma(name: none, body) = statement("Lemma", name, body)
#let conj(name: none, body) = statement("Conjecture", name, body)
#let proof(body) = block(above: 0.5em, below: 1em)[_Proof._ #body #h(1fr) $square$]
#let todo(body) = text(fill: rgb("#b00020"))[[TODO: #body]]

#align(center)[
  #text(size: 15pt, weight: "bold")[RoleWalk: inductive structural roles for directed graphs \ from the characteristic function of random walks]
  #v(0.6em)
  Maixent Chenebaux \
  #text(size: 9pt)[Draft of #datetime.today().display("[day] [month repr:long] [year]") — results and proofs; prose to come]
]

#v(0.8em)
#block(inset: (x: 1.5cm))[
  *Abstract.* Structural roles group nodes by their position in a graph
  rather than by proximity. RoleWalk describes each node by the
  characteristic function of its lazy random-walk transition
  probabilities, sampled at a few frequencies, together with the
  distribution of the walk's mass over distance layers and, optionally,
  low-order moments of a probabilistic curvature. The embedding is local,
  deterministic, reads directions and weights, and needs no training. We
  prove that it is invariant and local (hence transferable across graphs),
  that its distances are maximum mean discrepancies, that it is Lipschitz
  in edge insertions, that it counts triangles, which places it beyond
  1-WL, and that distance-regular graphs with equal intersection arrays are
  its exact limit. Empirically it matches or beats struc2vec, GraphWave,
  Role2Vec, RESD and ReFeX on role classification at 10–1,000× lower cost,
  is best on directed roles and on transfer of centralities across graphs,
  and improves GPS on ZINC when combined with RWSE.
]

= Introduction

Claims, each backed by a result below:

+ All of: inductive, directed, weighted, deterministic, online, fast (@tab-props).
+ Accuracy at or above the state of the art on role classification at a fraction of the cost (@tab-roles, @fig-speed, @tab-scale).
+ Only method that uses directions; best on synthetic and real directed roles (@tab-directed, @tab-rfa).
+ Transfer: a model fitted on one graph applies to another (@tab-centrality, @tab-airport-transfer; @prop-local).
+ Expressiveness beyond 1-WL with 3 walk steps, on par with 16-step RWSE; exact characterisation of a failure class (@sec-theory, @tab-expr).
+ Useful as a structural encoding for graph transformers (@tab-zinc).

= Related work

Feature-based roles: ReFeX @henderson2011, RolX @henderson2012. Walks
and word2vec: struc2vec @ribeiro2017, Role2Vec @ahmed2018; directed and
weighted extensions of struc2vec @steenfatt2018 #todo[compare if code is
available]. Diffusion: GraphWave @donnat2018, closest to RoleWalk
(@prop-graphwave). Deep role embeddings: RESD and the RONE benchmark
@rone #todo[exact references]. Structural encodings for GNNs: RWSE
@dwivedi2022, used by GPS @rampasek2022. Expressiveness: 1-WL @weisfeiler1968 @xu2019 @morris2019,
CSL @murphy2019, BREC @wang2023. Curvature: Ollivier-Ricci @ollivier2009 @jost2014.
Kernel mean embeddings and MMD: @gretton2012.

= Method <sec-method>

*Walk.* $G = (V, E)$ with non-negative weights $A$ (directed or not),
$tilde(D) = op("diag")((I + A) bold(1))$ and the lazy walk
$H = tilde(D)^(-1) (I + A)$. For a node $u$ and a step $w$, $p^w_u = e_u^top H^w$
is the distribution of the walk after $w$ steps. Directed graphs use the
out-walk ($A$) and the in-walk ($A^top$) and concatenate both embeddings.

*Characteristic function.* With $f_theta (x) = e^(i theta x) - 1$ and a
grid $Theta = {theta_1, ..., theta_m}$ (log-spaced in $[1, 100]$, $m = 10$),
$ phi^w_u (theta) = sum_(j in V) f_theta (p^w_(u j)), quad theta in Theta. $ <eq-phi>
Since $f_theta (0) = 0$, the sum runs over the support of $p^w_u$ only.

*Onion layers.* With $d(u, j)$ the length of the shortest walk from $u$ to $j$,
$s^w_u (k) = sum_(j : d(u, j) = k) p^w_(u j)$ for $k = 0, ..., w$: the mass
the walk keeps at each distance. Supports of earlier powers give the
layers, so no extra walk is computed.

*Curvature (option).* For an edge $x y$ of the symmetrised unweighted graph,
$kappa(x, y) = sum_z min(m_x (z), m_y (z))$ with $m_x$ the one-step lazy
walk from $x$; $kappa_x$ is its mean over the edges at $x$. Each step adds
the mean $mu^w_u = sum_j p^w_(u j) kappa_j$ and the standard deviation
$sigma^w_u$ of $kappa$ under $p^w_u$. Since a unit of mass that does not
overlap moves at least one step, $W_1 (m_x, m_y) >= 1 - kappa(x, y)$: $kappa$ is
an upper bound of Ollivier-Ricci curvature $1 - W_1$, made of probabilities.

*Embedding.* $X_u$ concatenates, for $w = 1, ..., L$ ($L = 3$),
$op("Re") phi^w_u$, $op("Im") phi^w_u$, $norm(Theta) s^w_u$ and, with curvature,
$norm(Theta) (mu^w_u, sigma^w_u)$: $2 m L + L (L + 3) \/ 2 (+ 2 L)$ = 69 (75)
features. The factor $norm(Theta) = (sum_theta theta^2)^(1\/2)$ puts a unit of
mass in the units of $phi$, since $f_theta (p) approx i theta p$ for small $p$;
without it the layers carry 0.02% of the variance and do not change the
roles found by $k$-means (@sec-scaling).

*Pruning.* After each product, entries below $"tol" = 10^(-4)$ are dropped;
their mass is added back to $op("Im") phi$ to first order ($theta dot "mass"$).
Results in @sec-theory are stated for $"tol" = 0$.

*Roles.* Ward clustering with silhouette for the number of roles, or
$k$-means with a given number; fitted centroids label nodes of any graph.

= Properties <sec-theory>

Throughout, $L$ is the number of steps, $B_r (u)$ the subgraph induced by
nodes within distance $r$ of $u$ (out- and in-distance for directed graphs),
and statements hold for $"tol" = 0$.

#prop(name: "locality and invariance")[
  $X_u$ is determined by the isomorphism class of the rooted graph
  $(B_L (u), u)$, and of $(B_(L+2) (u), u)$ with curvature. In particular it
  does not depend on $|V|$, nodes related by an automorphism get equal
  embeddings, and a node can be embedded from its neighbourhood alone.
] <prop-local>
#proof[
  $p^w_(u j) = sum_(u = x_0, ..., x_w = j) product_t H_(x_t x_(t+1))$ runs over walks of
  length $w <= L$ from $u$. Their nodes lie within distance $w$, and
  $H_(x y) = (I + A)_(x y) \/ tilde(D)_(x x)$ needs the edges at $x$, with
  $x$ within distance $w - 1$: all are in $B_L (u)$. An isomorphism of
  rooted balls maps these walks bijectively and preserves their weights,
  so it maps $p^w_u$ to $p'^w_(u')$ entry by entry; @eq-phi is a sum over
  $j$, hence equal. Distances up to $L$ are read inside $B_L (u)$. With
  curvature, $kappa_j$ for $j$ within distance $L$ needs the closed
  neighbourhoods of $j$ and of its neighbours, inside $B_(L+2) (u)$.
]

Coordinates therefore mean the same thing in every graph, which is what
transfer (@tab-centrality, @tab-airport-transfer) requires.

#prop(name: "injectivity")[
  Let $P, Q$ be finite multisets of reals in $(0, 1]$. If
  $sum_(p in P) f_theta (p) = sum_(q in Q) f_theta (q)$ for all $theta$ in a set
  with an accumulation point (e.g. $[1, 100]$), then $P = Q$. For a finite grid,
  given any countable family of pairs $P != Q$, Lebesgue-almost every
  $Theta in RR^m$ separates all of them.
] <prop-inj>
#proof[
  $F(theta) = sum_P e^(i theta p) - sum_Q e^(i theta q) - (|P| - |Q|)$ is entire
  and vanishes on a set with an accumulation point, hence everywhere.
  Exponentials $e^(i theta lambda)$ with distinct $lambda$ are linearly
  independent, so each $lambda > 0$ has the same multiplicity in $P$ and $Q$
  (and $|P| = |Q|$, coefficient of $lambda = 0$). For the second claim, the
  real zeros of each nonzero entire function $F$ are countable; a countable
  union of countable sets is null, so almost every $theta_1$ already
  separates every pair.
]

So with $Theta$ an interval, $phi^w_u$ is exactly the multiset of nonzero
transition probabilities of $u$ after $w$ steps. The second claim is
qualitative: the separation margin, not its existence, motivates
$m = 10$ frequencies across two decades.

#prop(name: "distances are MMDs")[
  $norm(phi^w_u - phi^w_v)^2 = "MMD"_k^2 (mu_u, mu_v)$ between the counting
  measures $mu_u = sum_j delta_(p^w_(u j))$, for the positive-definite kernel
  $k(a, b) = sum_(theta in Theta) [cos theta (a - b) - cos theta a - cos theta b + 1]$.
]
#proof[
  $phi^w_u = sum_j psi(p^w_(u j))$ with the feature map
  $psi(x) = (f_theta (x))_(theta in Theta) in CC^m$, so $phi^w_u$ is the kernel
  mean embedding of $mu_u$ and $norm(phi_u - phi_v)^2$ is the MMD for
  $k(a, b) = op("Re") ⟨ psi(a), psi(b) ⟩$, positive definite as an
  inner product. Expanding $op("Re") (e^(i theta a) - 1) overline((e^(i theta b) - 1))$ gives the formula.
  Atoms at 0 have $psi(0) = 0$: unreached nodes do not count.
]

#prop(name: "stability")[
  Let $G'$ be $G$ plus an unweighted edge $a b$, with $d = min(d_a, d_b)$
  the smaller endpoint degree in $G$. For every node $u$ and step $w$,
  $ norm(phi^w_u (G') - phi^w_u (G))_2 <= w norm(Theta)_2 dot 2 / (d + 2), $
  and the left side is 0 if $u$ is at distance $>= w$ from both $a$ and $b$.
] <prop-stab>
#proof[
  $|e^(i theta x) - e^(i theta y)| <= theta |x - y|$, so
  $norm(phi^w_u (G') - phi^w_u (G))_2 <= norm(Theta)_2 norm(p'^w_u - p^w_u)_1$.
  Telescoping, $H'^w - H^w = sum_(t=0)^(w-1) H'^t (H' - H) H^(w-1-t)$; stochastic
  matrices have $norm(dot)_infinity = 1$, so each row of the difference has
  $ell_1$ norm at most $w norm(H' - H)_infinity$. Only rows $a$ and $b$ of $H$
  change; row $a$ goes from $d_a + 1$ entries $1 \/ (d_a + 1)$ to $d_a + 2$
  entries $1 \/ (d_a + 2)$, an $ell_1$ change of $2 \/ (d_a + 2)$. Row $u$ of
  $H'^t (H' - H)$ is nonzero only if a walk of $t <= w - 1$ steps from $u$
  reaches $a$ or $b$.
]

Insertions at high-degree nodes barely move embeddings, and the effect is
confined to the $(L - 1)$-ball. On 200 random insertions (powerlaw-cluster
graphs, 60 nodes) the largest observed move is 0.52 of the bound
(`theory_checks.py`).

#prop(name: "degree and triangles")[
  $s^1_u (0) = 1 \/ (d_u + 1)$ in an unweighted graph. If $G$ is
  $d$-regular, $s^3_u (0) = (H^3)_(u u) = (1 + 3 d + 2 t_u) \/ (d + 1)^3$, with $t_u$
  the number of triangles at $u$. Hence RoleWalk with $L >= 3$ separates
  1-WL-equivalent graphs, e.g. two triangles and a hexagon, or any two
  $d$-regular graphs with different triangle counts.
] <prop-tri>
#proof[
  $(I + A)^3 = I + 3 A + 3 A^2 + A^3$ with $(A)_(u u) = 0$, $(A^2)_(u u) = d$,
  $(A^3)_(u u) = 2 t_u$, and $H = (I + A) \/ (d + 1)$. 1-WL gives every node
  of two $d$-regular graphs of the same size the same colour.
]

#prop(name: "relation to RWSE")[
  The distance-0 layer $s^w_u (0) = (H^w)_(u u)$ is the return probability
  of the lazy walk, i.e. RWSE @dwivedi2022 for $H$, up to $L$ steps. On a
  $d$-regular graph, ${(H^w)_(u u)}_(w <= L)$ and the simple-walk RWSE
  ${(P^k)_(u u)}_(k <= L)$, $P = A \/ d$, determine each other.
]
#proof[
  $H = (I + d P) \/ (d + 1)$, so
  $(H^w)_(u u) = (d + 1)^(-w) sum_(k=0)^w binom(w, k) d^k (P^k)_(u u)$: a
  triangular system with nonzero diagonal $d^w \/ (d + 1)^w$.
]

On irregular graphs the two walks differ and neither encoding contains
the other; RoleWalk adds the other layers and the full distribution of
transition probabilities.

#prop(name: "distance-regular graphs are the limit")[
  Let $G, G'$ be distance-regular graphs with the same intersection array.
  Then all nodes of $G$ and $G'$ get the same embedding, for every $L$,
  with or without curvature. This covers strongly regular graphs with equal
  parameters (diameter 2).
] <prop-drg>
#proof[
  Let $A_i$ be the distance-$i$ matrices. Distance-regularity gives
  $A A_i = b_(i-1) A_(i-1) + a_i A_i + c_(i+1) A_(i+1)$ with coefficients from
  the intersection array @brouwer1989, so by induction
  $(I + A)^w = sum_i alpha_(w, i) A_i$ with $alpha_(w, i)$ depending only on the
  array. As $H = (I + A) \/ (k + 1)$, $p^w_(u j) = alpha_(w, d(u, j)) \/ (k + 1)^w$:
  each row has value $alpha_(w, i) \/ (k + 1)^w$ with multiplicity $k_i$, the
  number of nodes at distance $i$, fixed by the array. So $phi^w_u$ and
  $s^w_u (i) = k_i alpha_(w, i) \/ (k + 1)^w$ are the same for all nodes of both
  graphs. For an edge $x y$, $kappa(x, y) = |N[x] inter N[y]| \/ (k + 1) = (a_1 + 2) \/ (k + 1)$,
  a constant, so $mu^w_u$ is constant and $sigma^w_u = 0$.
]

Checked on Shrikhande / $4 times 4$ rook, Chang / $T(8)$ and the diameter-3
pair Doob $D(1, 1)$ / Hamming $H(3, 4)$: largest difference between any two
nodes $<= 3 dot 10^(-7)$ (float32). This explains the 0/20
distance-regular pairs of BREC (@tab-brec); RWSE, which is constant on
these graphs for the same reason, fails them too.

#prop(name: "locality limits graph-level separation")[
  Let $A$ and $B$ be a 40-cycle with pendant nodes attached at cycle
  positions ${0, 15}$ and ${0, 20}$. They are not isomorphic and 1-WL
  separates them, but RoleWalk with $L = 3$ (with curvature) does not.
]
#proof[
  The pendants are more than $2(L + 2)$ apart in both graphs, so every
  rooted ball $B_(L+2) (u)$ contains at most one pendant and is fixed by
  $u$'s distance to it; both graphs have the same multiset of balls.
  Conclude by @prop-local. Colour refinement propagates along the cycle
  and eventually sees the distance between pendants (checked with 40
  rounds).
]

*Empirical comparison with 1-WL at equal depth.* A walk of $L$ steps sees
degrees up to distance $L - 1$, as does 1-WL after $L$ rounds from uniform
colours. Over all 6,781 nodes of the 996 connected graphs with at most 7
nodes, the partition induced by RoleWalk ($L = 3$) refines that of 3-round
1-WL: no two nodes that 1-WL separates share an embedding, and RoleWalk
splits 677 of the 1-WL classes (`theory_checks.py`).

#conj[
  For every $L$, the partition of nodes (across all graphs) induced by
  RoleWalk with $L$ steps and $Theta$ an interval refines the partition of
  $L$-round 1-WL.
]

Together with @prop-tri and the propositions above: at equal depth
RoleWalk is strictly finer than 1-WL on all graphs up to 7 nodes, while
being local, which 1-WL with unbounded rounds is not.

#prop(name: "relation to GraphWave")[
  GraphWave @donnat2018 embeds $u$ by
  $chi_u (t) = 1 / n sum_j e^(i t Psi_(j u))$ with $Psi = U g_s (Lambda) U^top$ the
  heat kernel at a scale $s$ chosen from the spectrum of each graph. Then
  $n (chi_u (t) - 1) = sum_j f_t (Psi_(j u))$: @eq-phi with the heat kernel in
  place of $H^w$. $chi_u$ depends on $n$ and on $s$, so the same rooted
  neighbourhood gets different coordinates in graphs of different sizes or
  spectra.
] <prop-graphwave>
#proof[
  $sum_j (e^(i t Psi_(j u)) - 1) = n chi_u (t) - n$. $Psi$ has full support on a
  connected graph, so @prop-local fails; $1 \/ n$ and $s$ are graph-level.
]

This is why GraphWave transfers at chance level (@tab-airport-transfer),
while RoleWalk keeps its accuracy.

#prop(name: "cost")[
  With pruning, each row of each power keeps at most $1 \/ "tol"$ entries,
  so step $w$ costs $O(sum_u sum_(j in "supp" p^(w-1)_u) (d_j + 1))$ with
  $|"supp" p^(w-1)_u| <= min(|B_(w-1) (u)|, 1 \/ "tol")$; the
  characteristic function adds $O(m dot "nnz")$ and the layers $L$ element-wise
  products. Curvature costs $O(sum_u d_u^2)$. Embedding a set $S$ touches
  only $B_L (S)$ ($B_(L+2) (S)$ with curvature).
]
#proof[
  Entries are non-negative and each row sums to at most 1, so at most
  $1 \/ "tol"$ of them are $>= "tol"$. The rest follows from sparse products
  and @prop-local.
]

= Experiments

All CPU timings on one core. Classification: logistic regression on
standardised embeddings, 50 stratified 80/20 splits. Best in bold, second
underlined. Code and CSV files in `benchmarks/`.

#figure(caption: [Properties. Checked against each paper; "\~" partial.])[
  #table(columns: 7, align: (left,) + (center,) * 6,
    [], [Transfer], [Directed], [Weighted], [Time (USA)], [Online], [Deterministic],
    [*RoleWalk*], [yes], [yes], [yes], [0.2 s], [yes], [yes],
    [struc2vec], [no], [no#super[a]], [no#super[a]], [564 s], [no], [no],
    [GraphWave], [no (P. #ref(<prop-graphwave>, supplement: none))], [\~], [yes], [18 s], [no], [yes],
    [Role2Vec], [\~], [yes#super[b]], [no], [10 s], [\~], [no],
    [RESD], [no], [no], [no], [8 s], [no], [no],
    [ReFeX], [\~], [\~], [\~], [0.1 s], [\~], [yes],
    [RWSE], [yes], [\~], [yes], [1.7 s], [yes], [yes],
  )
  #text(size: 8pt)[#super[a] struc2vec++ @steenfatt2018 extends it. #super[b] In the paper; the reference implementation is undirected. #todo[verify RESD, ReFeX, RWSE rows]]
] <tab-props>

#figure(caption: [Role classification accuracy. struc2vec is unseeded: mean ± std over 5 runs. `roles.csv`, `struc2vec_runs.csv`.])[
  #table(columns: 7, align: (left,) + (center,) * 6,
    [], [Brazil], [Europe], [USA], [Reality-call], [Actor], [Film#super[\*]],
    [*RoleWalk*], [0.743], [#underline[0.583]], [#underline[0.657]], [*0.860*], [*0.487*], [0.569],
    [*RoleWalk + curv.*], [0.741], [*0.596*], [*0.666*], [*0.860*], [#underline[0.485]], [#underline[0.581]],
    [struc2vec], [0.705 ± .022], [0.561 ± .020], [0.596 ± .004], [–], [–], [–],
    [RESD], [0.728], [0.544], [0.653], [0.857], [0.476], [0.563],
    [GraphWave], [*0.769*], [0.561], [0.587], [*0.860*], [0.481], [–],
    [ReFeX], [0.691], [0.576], [0.622], [0.851], [0.478], [0.579],
    [Role2Vec], [0.476], [0.366], [0.562], [0.844], [0.399], [*0.743*],
    [RWSE], [0.711], [0.572], [0.517], [0.855], [0.363], [0.450],
    [Degree], [#underline[0.756]], [0.554], [0.567], [#underline[0.858]], [0.476], [0.478],
  )
  #text(size: 8pt)[#super[\*] A node-type task (@sec-film). "–": too slow or out of memory.]
] <tab-roles>

#figure(
  image("fig_speed.svg", width: 100%),
  caption: [Accuracy against embedding time (one core, log scale), every method that runs on each benchmark. Airports: mean over Brazil, Europe and USA (struc2vec: mean of 5 runs). Directed motifs: 5% rewired edges, mean of 3 seeds. wiki-RfA: task A of @tab-rfa. Hollow marker: degree, a trivial baseline rather than an embedding. RoleWalk is the only method in the top-left corner of every panel. `speed_figure.py`.],
) <fig-speed>

#figure(caption: [Wikipedia talk networks (RONE): precision\@10 for bots and admins, mean over networks. `roles.csv`.])[
  #table(columns: 3, align: (left, center, center),
    [], [Bots], [Admins],
    [RoleWalk], [#underline[0.129]], [#underline[0.324]],
    [struc2vec (3 smallest networks)], [*0.146*], [*0.391*],
    [GraphWave], [0.115], [0.298],
    [RESD], [0.114], [0.254],
    [ReFeX], [0.101], [0.299],
    [RWSE], [0.118], [0.297],
    [Degree], [0.075], [0.296],
  )
  #text(size: 8pt)[Mixed. Networks are distributed undirected. #todo[directed versions from SNAP; per-network table]]
] <tab-wiki>

#figure(caption: [Directed roles on synthetic graphs (directed motifs hung from a directed cycle; several roles coincide once directions are dropped). 3 seeds; noise = share of edges rewired. `directed.csv`.])[
  #table(columns: 6, align: (left,) + (center,) * 5,
    [], [Acc. 0%], [Acc. 5%], [Acc. 10%], [ARI 5%], [mAP 5%],
    [*RoleWalk + curv.*], [*0.989*], [*0.712*], [*0.547*], [*0.381*], [*0.625*],
    [RoleWalk], [#underline[0.908]], [#underline[0.680]], [#underline[0.510]], [#underline[0.366]], [#underline[0.574]],
    [RoleWalk, undirected], [0.767], [0.560], [0.420], [0.351], [0.479],
    [Role2Vec], [0.730], [0.451], [0.368], [0.148], [0.295],
    [ReFeX], [0.532], [0.429], [0.431], [0.342], [0.356],
    [GraphWave], [0.586], [0.388], [0.383], [0.317], [0.416],
    [RESD], [0.547], [0.420], [0.423], [0.253], [0.348],
    [struc2vec], [0.510], [0.366], [0.277], [0.141], [0.260],
    [In/out degree], [0.516], [0.466], [0.405], [0.310], [0.349],
  )
] <tab-directed>

#figure(caption: [Real directed roles: Wikipedia requests for adminship @west2014 (11,377 users, 188k votes voter → candidate). A: elected among the 3,494 candidates. B: voter / failed candidate / admin. `roles_rfa.csv`.])[
  #table(columns: 4, align: (left, center, center, center),
    [], [A: elected], [B: 3 classes], [Time],
    [*RoleWalk + curv.*], [*0.823*], [*0.943*], [5 s],
    [RoleWalk], [0.808], [#underline[0.938]], [4 s],
    [RoleWalk, undirected], [0.791], [0.863], [],
    [RoleWalk + curv., undirected], [#underline[0.814]], [0.900], [],
    [RESD], [0.798], [0.843], [68 s],
    [Role2Vec], [0.798], [0.857], [82 s],
    [ReFeX], [0.794], [0.847], [1 s],
    [In/out degree], [0.755], [0.922], [0 s],
    [RWSE (out-walk)], [0.690], [0.826], [51 s],
  )
  #text(size: 8pt)[struc2vec and GraphWave do not run at this size.]
] <tab-rfa>

#figure(caption: [Expressiveness: pairs told apart. Every set is 1-WL-indistinguishable. No false difference on random relabellings. `expressiveness.csv`.])[
  #table(columns: 7, align: (left,) + (center,) * 6,
    [], [Atlas ≤ 7 (20)], [Regular (3,181)], [CSL (10 cl.)], [Decalin / bicyc.], [$2 C_3$ / $C_6$], [SRG pair#super[\*]],
    [*RoleWalk (3 steps)*], [*20*], [*3,181*], [#underline[6]], [yes], [yes], [no],
    [RoleWalk (5 steps)], [*20*], [*3,181*], [*10*], [yes], [yes], [no],
    [RoleWalk, no layers], [#underline[16]], [#underline[3,180]], [#underline[6]], [yes], [yes], [no],
    [RWSE (16 steps)], [*20*], [#underline[3,180]], [*10*], [yes], [yes], [no],
    [Degree (= 1-WL)], [0], [0], [1], [no], [no], [no],
  )
  #text(size: 8pt)[#super[\*] Shrikhande / $4 times 4$ rook, beyond 3-WL; not separable by @prop-drg.]
] <tab-expr>

#figure(caption: [BREC @wang2023 (400 pairs).])[
  #table(columns: 8, align: (left,) + (center,) * 7,
    [], [Basic (60)], [Regular (100)], [Extension (100)], [CFI (100)], [4-vertex (20)], [Dist.-reg. (20)], [Total],
    [RoleWalk (3 steps)], [60], [50], [97], [0], [0], [0], [207],
    [RoleWalk (5 steps)], [60], [50], [100], [3], [0], [0], [#underline[213]],
    [RoleWalk, no layers], [58], [48], [97], [0], [0], [0], [203],
    [RWSE (16 steps)], [60], [50], [100], [4], [0], [0], [*214*],
  )
  #text(size: 8pt)[Distance-regular: 0 by @prop-drg. The 50 regular pairs missed are exactly the 50 strongly regular pairs, all with equal parameters (checked).]
] <tab-brec>

#figure(caption: [Centrality transfer: a gradient-boosting regressor from embedding to centrality is fitted on one graph and evaluated (Spearman) on another graph of the same family and size (other seed) (Barabási-Albert, powerlaw-cluster, Watts-Strogatz, SBM, random geometric, motifs). `centrality_transfer.csv`.])[
  #table(columns: 7, align: (left,) + (center,) * 6,
    [], [RoleWalk], [\+ curv.], [ReFeX#super[†]], [RWSE], [GraphWave], [Degree],
    [Betweenness], [#underline[0.901]], [*0.905*], [0.844], [0.708], [0.723], [0.634],
    [Closeness], [*0.866*], [#underline[0.864]], [0.684], [0.732], [0.597], [0.428],
    [Clustering], [*0.873*], [#underline[0.870]], [0.852], [0.803], [0.392], [0.212],
    [k-core], [0.747], [#underline[0.748]], [*0.935*], [0.660], [0.566], [0.692],
    [Eigenvector], [*0.656*], [#underline[0.649]], [0.492], [0.531], [0.462], [0.428],
    [Harmonic], [*0.918*], [#underline[0.916]], [0.782], [0.784], [0.699], [0.500],
    [PageRank], [*0.990*], [*0.990*], [#underline[0.968]], [0.864], [0.857], [0.927],
    [Mean], [*0.850*], [#underline[0.849]], [0.794], [0.726], [0.614], [0.546],
  )
  #text(size: 8pt)[#super[†] Only where its number of features matches across graphs.]
] <tab-centrality>

#figure(caption: [Role transfer between airport networks: classifier fitted on one network, tested on another (6 ordered pairs, accuracy). Labels are activity quartiles, close to degree. `role_transfer.csv`.])[
  #table(columns: 6, align: center,
    [Degree], [RoleWalk], [\+ curv.], [RoleWalk, no layers], [RWSE], [GraphWave],
    [0.542], [*0.488*], [#underline[0.470]], [0.445], [0.375], [0.249 (chance)],
  )
] <tab-airport-transfer>

#figure(caption: [Structural encodings for GNNs on ZINC (test MAE at the best validation epoch). Top: GINE, 4 layers, ≈230k parameters, 300 epochs, 3 seeds. Bottom: GPS @rampasek2022 protocol, 10 layers, ≈470k parameters, 500 epochs, 2 seeds. `pe_zinc_v13.csv`, `pe_zinc_gps.csv`.])[
  #table(columns: 3, align: (left, center, center),
    [GINE], [Test MAE], [],
    [none], [0.202 ± 0.010], [],
    [RWSE], [0.107 ± 0.004], [],
    [RoleWalk], [0.108 ± 0.008], [],
    [RoleWalk + RWSE], [#underline[0.104 ± 0.006]], [],
    [*RoleWalk + curv. + RWSE*], [*0.095 ± 0.002*], [],
    table.hline(stroke: 0.6pt),
    [GPS], [Test MAE], [Val. MAE],
    [RWSE (published: 0.070 ± 0.004)], [0.077 ± 0.006], [0.103 ± 0.000],
    [RoleWalk + curv.], [#underline[0.076 ± 0.001]], [#underline[0.092 ± 0.003]],
    [*RoleWalk + curv. + RWSE*], [*0.073 ± 0.001*], [*0.089 ± 0.005*],
  )
  #text(size: 8pt)[#todo[GPS seeds 2-3 running; Peptides (LRGB)]]
] <tab-zinc>

#figure(caption: [Embedding time against size, one thread, defaults (tol $10^(-4)$, 3 steps, layers). `scalability.csv`.])[
  #let rows = csv("../benchmarks/scalability.csv", row-type: dictionary)
  #let sizes = rows.map(r => r.n_nodes).dedup()
  #let cell(graph, n, curv) = {
    let hit = rows.filter(r => r.graph == graph and r.n_nodes == n and r.curvature == curv)
    if hit.len() == 0 { [–] } else { [#hit.first().time s] }
  }
  #table(columns: 1 + sizes.len(), align: (left,) + (center,) * sizes.len(),
    [Nodes], ..sizes.map(n => [#calc.round(int(n) / 1000) k]),
    [Barabási-Albert ($m = 3$)], ..sizes.map(n => cell("barabasi_albert", n, "False")),
    [  \+ curvature], ..sizes.map(n => cell("barabasi_albert", n, "True")),
    [Watts-Strogatz ($k = 6$)], ..sizes.map(n => cell("watts_strogatz", n, "False")),
    [  \+ curvature], ..sizes.map(n => cell("watts_strogatz", n, "True")),
  )
  #text(size: 8pt)[Peak memory 6.6 GB at a million nodes (blocks of 50k rows). For comparison, at 1,190 nodes (USA airports): struc2vec 564 s, GraphWave 18 s, Role2Vec 10 s, RESD 8 s; struc2vec and GraphWave do not run on wiki-RfA (11k nodes).]
] <tab-scale>

= Analysis

== Summaries of the walk distribution

#figure(caption: [Same budget of features, replacing the characteristic function. Airports: mean accuracy; Transfer: airport transfer; Motifs: synthetic accuracy.])[
  #table(columns: 5, align: (left,) + (center,) * 4,
    [], [Airports], [Transfer], [Motifs (varied / houses)], [Time 20k],
    [*Characteristic function*], [#underline[0.656]], [#underline[0.445]], [*0.744* / #underline[0.733]], [6.5 s],
    [Laplace transform], [*0.657*], [0.345], [0.614 / 0.717], [18 s],
    [Dyadic histogram (Haar)], [0.631], [0.353], [0.614 / 0.725], [15 s],
    [Power sums (Rényi)], [0.653], [0.420], [0.647 / 0.692], [9.7 s],
    [Top-10 probabilities], [0.652], [*0.512*], [#underline[0.679] / *0.758*], [3.5 s],
  )
]

== Geometry: moments rather than integrated characteristic functions

#figure(caption: [Ways to add local geometry. Anything that depends on graph size (degree marks, counting bounds) hurts transfer.])[
  #table(columns: 5, align: (left,) + (center,) * 4,
    [], [Airports], [Transfer], [Motifs (varied / houses)], [Directed acc.],
    [No geometry], [0.656], [0.445], [0.744 / 0.733], [0.678],
    [\+ onion layers (moments)], [0.661], [#underline[0.488]], [0.744 / #underline[0.758]], [0.682],
    [\+ layers, one CF per layer], [0.654], [0.424], [#underline[0.751] / 0.742], [0.695],
    [\+ degree marks (joint CF)], [0.65–0.67], [0.34–0.36], [–], [–],
    [\+ curvature, counting bound], [#underline[0.667]], [0.388], [*0.754* / *0.767*], [#underline[0.721]],
    [\+ curvature, probabilistic (moments)], [*0.669*], [0.470], [0.725 / 0.742], [0.712],
    [\+ curvature, exact Ollivier-Ricci], [0.662], [*0.494*], [0.746 / 0.750], [*0.743*],
    [\+ curvature, integrated CF], [0.663], [0.400], [#underline[0.751] / 0.750], [0.697],
  )
]

The probabilistic bound is kept over exact Ollivier-Ricci for cost
(no optimal transport) at a small loss.

== Kernel scaling <sec-scaling>

In raw embeddings $theta = 100$ carries 53% of the variance; unscaled
layers carry 0.02% and leave $k$-means roles unchanged. A grid of 48
configurations (frequency weighting, range, steps, layers), scored on
synthetic and real networks with equal weight (`kernel_grid.csv`): scaling
the layers by $norm(Theta)$ is what matters (score 0.274 → 0.323), the
frequency weighting barely does, and the defaults win whichever dataset is
left out.

== Film is a node-type task <sec-film>

#figure(caption: [Film (RONE). Class 3 is an independent set in a separate block of node ids.])[
  #table(columns: 3, align: (left, center, center),
    [], [Class 3 vs rest], [Within classes 0–2],
    [Role2Vec], [*0.946*], [0.637],
    [RoleWalk + curv.], [0.740], [#underline[0.634]],
    [ReFeX], [#underline[0.745]], [0.627],
    [Degree], [0.536], [0.623],
    [Majority class], [0.593], [0.623],
  )
]

Classes 0–2 are not predictable by any method; edge homophily is 0.28,
below the 0.33 of random labels. Role2Vec's lead comes from walk
co-occurrence, i.e. position, which an automorphism-invariant embedding
cannot capture by @prop-local. Film should not rank role methods.

= Limitations

- Walk statistics cannot separate distance-regular graphs with equal arrays (@prop-drg), CFI pairs, or graphs whose balls coincide (locality).
- Middle-tier roles on real networks often form a continuum; per-node stability should be reported next to roles.
- Degree-like labels (airports) favour trivial baselines.
- One real labelled directed benchmark.
- GPS on ZINC: 2 seeds; the test gain is within published seed noise, clearer on validation.

#bibliography("refs.bib", style: "ieee")
