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
  that its distances are maximum mean discrepancies, that its
  characteristic-function block is Lipschitz in edge insertions, that it
  recovers triangle counts on regular graphs, so that it separates graphs
  1-WL cannot (and conversely: the two are incomparable), and that
  distance-regular graphs with equal intersection arrays are a provable
  failure class. With a budget on the probability mass that pruning may
  drop, the approximation error is certified. Empirically it matches or
  beats struc2vec, GraphWave, Role2Vec, RESD and ReFeX on role
  classification at a fraction of the cost of every method but ReFeX, is
  best on real directed roles and on transfer of centralities across
  graphs, embeds a million nodes in 20 s to 2.5 min on one core, and
  improves GPS on ZINC when combined with RWSE.
]

= Introduction

Claims, each backed by a result below:

+ All of: inductive, directed, weighted, training-free, deterministic, online, fast (@tab-props).
+ Accuracy at or above the state of the art on role classification at a fraction of the cost; a million nodes in minutes on one core (@tab-roles, @fig-speed, @tab-scale).
+ Reads edge directions natively, including walks that change direction; best on real directed roles and on unsupervised synthetic ones (@tab-directed, @tab-rfa, @prop-mixed).
+ Pruning with a certified error (@prop-prune).
+ Transfer: a model fitted on one graph applies to another (@tab-centrality, @tab-airport-transfer; @prop-local).
+ Separates graphs that 1-WL cannot with 3 walk steps (the two are incomparable), close to 16-step RWSE on BREC; a provable failure class (@sec-theory, @tab-expr).
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
out-walk $H_+$ ($A$) and the in-walk $H_-$ ($A^top$) and concatenate both
embeddings (138 features, 150 with curvature). As an option, the
characteristic functions of the rows of $H_+ H_-$ (one step forward, one
back: nodes sharing targets with $u$) and $H_- H_+$ (shared sources) add
$4 m = 40$ features (@prop-mixed).

*Characteristic function.* With $f_theta (x) = e^(i theta x) - 1$ and a
grid $Theta = {theta_1, ..., theta_m}$ (log-spaced in $[1, 100]$, $m = 10$),
$ phi^w_u (theta) = sum_(j in V) f_theta (p^w_(u j)), quad theta in Theta. $ <eq-phi>
Since $f_theta (0) = 0$, the sum runs over the support of $p^w_u$ only.

*Onion layers.* With $d(u, j)$ the length of the shortest walk from $u$ to $j$,
$s^w_u (k) = sum_(j : d(u, j) = k) p^w_(u j)$ for $k = 0, ..., w$: the mass
the walk keeps at each distance, which the multiset of probabilities
ignores. Supports of earlier powers give the layers, so no extra walk is
computed.

*Curvature (option).* For an edge $x y$ of the symmetrised unweighted graph,
$kappa(x, y) = sum_z min(m_x (z), m_y (z))$ with $m_x$ the one-step lazy
walk from $x$; $kappa_x$ is its mean over the edges at $x$. Each step adds
the mean $mu^w_u = sum_j p^w_(u j) kappa_j$ and the standard deviation
$sigma^w_u$ of $kappa$ under $p^w_u$. Since a unit of mass that does not
overlap moves at least one step, $W_1 (m_x, m_y) >= 1 - kappa(x, y)$: $kappa$ is
an upper bound of Ollivier-Ricci curvature $1 - W_1$, made of probabilities.

*One measure, one kernel.* Every node $j$ reached at step $w$ is an atom
marked by its step and its layer, which gives the _footprint_ of the walks
from $u$:
$ mu_u = sum_(w = 1)^L sum_(j : p^w_(u j) > 0) delta_((w, p^w_(u j), d(u, j))). $
The embedding is the mean of a single feature map over the footprint,
$X_u = integral Psi thin d mu_u$, with
$ Psi(w, a, ell) = e_w ⊗ (psi(a) ⊕ norm(Theta) a e_ell), quad psi(a) = (f_theta (a))_(theta in Theta), $ <eq-psi>
which gives, for each step $w = 1, ..., L$ ($L = 3$), $op("Re") phi^w_u$,
$op("Im") phi^w_u$ and $norm(Theta) s^w_u$. With curvature the mark is
$(d(u, j), kappa_j)$ and $Psi$ adds $norm(Theta) a (kappa, kappa^2)$; we report
$norm(Theta) (mu^w_u, sigma^w_u)$, which carries the same information but,
being nonlinear, is not an integral of $Psi$: kernel statements below are
for the curvature-free embedding (or raw moments). In all, $2 m L + L (L + 3) \/ 2 (+ 2 L)$ = 69 (75) features. The
probability block ignores the layer; the layer block is the first-order
part of one characteristic function per layer, which is what fixes the
factor $norm(Theta) = (sum_theta theta^2)^(1\/2)$ (@prop-mmd). Without it
the layers carry 0.02% of the variance and do not change the roles found
by $k$-means (@sec-scaling).

*Pruning.* After each product, entries below $"tol" = 10^(-4)$ are dropped;
their mass is added back to $op("Im") phi$ to first order ($theta dot "mass"$),
the same tangent approximation as the layer block. The first step is never
pruned. Dropped mass can gather again: in a star with more than $10^4$
leaves, the leaf mass of step 2 falls below tol although half of it returns
to the centre at step 3. With a budget $delta$ (`max_missing`), a row drops
its small entries only while its missing mass $r^w_u = 1 - norm(q^w_u)_1$
stays below $delta$, which bounds the error (@prop-prune). Distances are
read from supports before pruning, so layers are exact for $L <= 3$.
Results in @sec-theory are otherwise stated for $"tol" = 0$.

*Implementation.* For each root, one pass propagates a sparse row with a
dense accumulator, prunes it, and summarises it (characteristic function,
layers from the step at which each node is first reached, curvature
moments) without storing the powers: $O(n)$ memory per thread, and rows
are independent. Below $p = 10^(-3)$ every angle $theta p$ is under 0.1,
and the characteristic function comes from the power sums $sum p^k$,
$k <= 8$ (error below $10^(-12)$) instead of $2 m$ sines.

*Roles.* Ward clustering with silhouette for the number of roles, or
$k$-means with a given number; fitted centroids label nodes of any graph.

= Properties <sec-theory>

Throughout, $L$ is the number of steps, $B_r (u)$ the subgraph induced by
nodes within distance $r$ of $u$ (out- and in-distance for directed graphs;
with curvature, distance in the underlying undirected graph, on which
$kappa$ is defined), graphs have no self-loops, and statements hold for
$"tol" = 0$.

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

#prop(name: "one kernel on the footprint")[
  Without curvature, $norm(X_u - X_v)^2 = "MMD"_K^2 (mu_u, mu_v)$ (extended
  to finite measures) for the positive-semidefinite kernel on marked atoms
  $ K((w, a, ell), (w', b, ell')) = delta_(w w') [k_Theta (a, b) + norm(Theta)^2 a b thin delta_(ell ell')], $
  $ k_Theta (a, b) = sum_(theta in Theta) [cos theta (a - b) - cos theta a - cos theta b + 1]. $
  The first term compares probabilities whatever their layer, the second
  compares masses in the same layer. Moreover, with
  $t = max(Theta) max(a, b)$,
  $ abs(k_Theta (a, b) - norm(Theta)^2 a b) <= norm(Theta)^2 a b thin t^2 (7 \/ 12 + t^2 \/ 36), $
  so the layer term is the first-order part of $k_Theta (a, b) thin delta_(ell ell')$,
  the kernel of one characteristic function per layer, and $norm(Theta)$ is
  the only factor for which the two agree to first order.
] <prop-mmd>
#proof[
  $X_u = sum Psi$ over the atoms of $mu_u$, so $X_u$ is the kernel mean
  embedding of $mu_u$ and $norm(X_u - X_v)^2$ is the MMD @gretton2012 for
  $K = op("Re") ⟨ Psi, Psi' ⟩$, positive semidefinite as an inner product. The
  blocks $e_w$ and $e_ell$ are orthonormal, which gives $delta_(w w')$ and
  $delta_(ell ell')$; expanding $op("Re") (e^(i theta a) - 1) overline((e^(i theta b) - 1))$
  gives $k_Theta$. Unreached nodes have $Psi = 0$ and do not count.
  For the bound, fix $theta$ and write the real part as
  $(cos theta a - 1)(cos theta b - 1) + sin theta a sin theta b$. Since
  $abs(cos x - 1) <= x^2 \/ 2$, the first product is at most
  $theta^4 a^2 b^2 \/ 4 <= theta^2 a b thin t^2 \/ 4$. With
  $sin theta a = theta a + r_a$ and $abs(r_a) <= theta^3 a^3 \/ 6$,
  $abs(sin theta a sin theta b - theta^2 a b) <= theta^4 a b (a^2 + b^2) \/ 6 + theta^6 a^3 b^3 \/ 36 <= theta^2 a b thin t^2 (1 \/ 3 + t^2 \/ 36)$.
  Summing over $theta$ gives the bound, and $norm(Theta)^2 = sum_theta theta^2$
  is the coefficient of $a b$.
]

The concatenation is therefore a sum of kernels on one object: a mark-blind
term on probabilities and a mark-aware term on masses. The first-order
regime $t << 1$ holds where walks spread thin, not for low-degree nodes
nor for returns to a hub ($max(Theta) = 100$): the linear layer kernel is a
choice, $L (L + 3) \/ 2 = 9$ coordinates instead of $m L (L + 3) = 180$ for
one characteristic function per layer, and the full
per-layer kernel is not better (@tab-geometry). Further marks enter the
same way, as factors of the second term: curvature, or labels on the
steps of each path.

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

Insertions at high-degree nodes barely move the characteristic-function
block, and the effect is confined to the $(L - 1)$-ball. On 200 random
insertions (powerlaw-cluster graphs, 60 nodes) the largest observed move of
this block is 0.52 of the bound (`theory_checks.py`). The bound does not
extend to the layer block, which reads distances: in $K_(2, d)$, joining
the two hubs $a, b$ moves $s^2_a (2)$ from $d \/ (3 (d + 1))$ to 0, whatever $d$.

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

On irregular graphs the two walks differ, so the return probabilities do
not translate into each other; RoleWalk adds the other layers and the
full distribution of transition probabilities.

#prop(name: "a failure class: distance-regular graphs")[
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

#prop(name: "RoleWalk and 1-WL are incomparable")[
  @prop-tri gives graphs that RoleWalk separates and 1-WL does not.
  Conversely, without curvature, for $L in {2, 3}$ there are nodes that
  $L$-round 1-WL (uniform initial colours) separates and RoleWalk with $L$
  steps does not, for every set of frequencies.
] <prop-wl>
#proof[
  Take a clique $K_5$ on ${r, a, b, c, d}$ and ten further nodes, adjacent
  to $b, c, d$ in $G$, and five to $a, b$ and five to $c, d$ in $G'$. The
  neighbour degrees of $r$ are ${4, 14, 14, 14}$ in $G$ and ${9, 9, 9, 9}$
  in $G'$, so two rounds separate $r$. Walks from $r$ agree: at step 1 each
  clique node has probability $1 \/ 5$; at step 2 each clique node has
  $1 \/ 5 (1 \/ 5 + 2 \/ 5) = 3 \/ 25$ (the inverse augmented degrees of the
  neighbours of $r$ sum to $1 \/ 5 + 3 \/ 15 = 2 \/ 5$ in $G$ and
  $4 \/ 10 = 2 \/ 5$ in $G'$) and each outer node
  $1 \/ 25 = 1 \/ 5 dot 3 \/ 15 = 1 \/ 5 dot 2 \/ 10$, with
  the same layers. For $L = 3$, add a leaf $z$ at $r$ and root at $z$:
  three rounds separate $z$, while the walk distributions and layers from
  $z$ agree up to step 3 (`theory_checks.py`). With curvature, both pairs are
  separated.
]

On the 6,781 nodes of the 996 connected graphs with at most 7 nodes, no
pair separated by 3-round 1-WL is confused by RoleWalk ($L = 3$), which
splits 571 of the 1-WL classes (embeddings grouped at a sup-distance
below $10^(-4)$ of the largest coordinate; `theory_checks.py`); the
counterexamples above have 15 and 16 nodes. RoleWalk is also local, which
1-WL with unbounded rounds is not.

#prop(name: "walks that change direction")[
  Let $u$ be a source pointing to two sinks ($G$), or one of the two
  sources of a directed $K_(2, 2)$ ($G'$). For every $L$, with or without
  curvature, $u$ gets the same out- and in-walk embedding in $G$ and $G'$;
  but $e_u^top H_+ H_-$ has values $(2 \/ 3, 1 \/ 6, 1 \/ 6)$ in $G$ and
  $(5 \/ 9, 2 \/ 9, 1 \/ 9, 1 \/ 9)$ in $G'$, so the mixed probes separate them.
] <prop-mixed>
#proof[
  Out-walk: $u$ keeps $1 \/ 3$ and sends $1 \/ 3$ to each sink, where it
  stays; the layers agree. In-walk: $u$ has no in-edge and stays. The
  symmetrised graphs are a path $a - u - b$ and a 4-cycle; no edge has a
  common neighbour, and every edge has $max(d_x, d_y) = 2$, so every node
  has $kappa = 2 \/ 3$ in both. In $H_-$, a sink with in-neighbours $N$
  spreads uniformly over itself and $N$: $1 \/ 2$ on itself and $u$ in $G$,
  $1 \/ 3$ on itself, $u$ and the other source in $G'$. Hence
  $e_u^top H_+ H_- = 1 \/ 3 (e_u + sum_("sinks" s) e_s^top H_-)$, which gives the
  two rows.
]

The out- and in-walks see where $u$ sends and receives, not who shares its
targets (voters of the same candidates) or its sources; @tab-directed and
@tab-rfa measure what this adds.

#prop(name: "relation to GraphWave")[
  GraphWave @donnat2018 embeds $u$ by
  $chi_u (t) = 1 / n sum_j e^(i t Psi_(j u))$ with $Psi = U g_s (Lambda) U^top$ the
  heat kernel at a scale $s$ chosen from the spectrum of each graph. Then
  $n (chi_u (t) - 1) = sum_j f_t (Psi_(j u))$: @eq-phi with the heat kernel in
  place of $H^w$. $chi_u$ depends on $n$ and on $s$, so the same rooted
  neighbourhood can get different coordinates in graphs of different sizes or
  spectra.
] <prop-graphwave>
#proof[
  $sum_j (e^(i t Psi_(j u)) - 1) = n chi_u (t) - n$. $Psi$ has full support on a
  connected graph, so @prop-local fails; $1 \/ n$ and $s$ are graph-level.
]

This explains most of why GraphWave transfers at chance level: without
its $1 \/ n$, it reaches 0.455 on airport transfer, against 0.488 for
RoleWalk (@tab-airport-transfer).

#prop(name: "cost")[
  With pruning, each row of each power keeps at most $1 \/ "tol"$ entries,
  so step $w$ costs $O(sum_u sum_(j in "supp" p^(w-1)_u) (d_j + 1))$ with
  $|"supp" p^(w-1)_u| <= min(|B_(w-1) (u)|, 1 \/ "tol")$; the
  characteristic function adds $O(m dot "nnz")$, the layers $O("nnz")$. As
  $kappa(x, y) = (c_(x y) + 2) \/ (max(d_x, d_y) + 1)$ with $c_(x y)$ the
  common neighbours of $x$ and $y$, curvature costs a triangle enumeration,
  $O(|E|^(3\/2))$. Embedding a set $S$ touches only $B_L (S)$
  ($B_(L+2) (S)$ with curvature).
]
#proof[
  Entries are non-negative and each row sums to at most 1, so at most
  $1 \/ "tol"$ of them are $>= "tol"$. Both closed neighbourhoods are uniform
  ($1 \/ (d + 1)$ on $d + 1$ nodes) and share $x$, $y$ and their common
  neighbours, so the overlap is $c_(x y) + 2$ times the smaller mass;
  orienting edges by degree enumerates each triangle once in
  $O(|E|^(3\/2))$. The rest follows from sparse products and @prop-local.
]


#prop(name: "certified pruning")[
  Without curvature and for $L <= 3$, let $hat(X)_u$ be computed with
  pruning and $r^w_u$ the mass row $u$ misses at step $w$. Then
  $ norm(X_u - hat(X)_u)_2 <= sqrt(5) norm(Theta)_2 (sum_(w <= L) (r^w_u)^2)^(1\/2), $
  so with a budget $delta$, $norm(X_u - hat(X)_u)_2 <= sqrt(5 L) norm(Theta)_2 delta$.
] <prop-prune>
#proof[
  Propagation and deletion keep entries non-negative, so the pruned row
  $q^w_u$ satisfies $0 <= q^w_u <= p^w_u$ entrywise and
  $norm(p^w_u - q^w_u)_1 = r^w_u$. For each $theta$,
  $abs(sum_j f_theta (p_j) - f_theta (q_j)) <= theta r^w_u$, and the first-order
  correction $i theta r^w_u$ adds at most $theta r^w_u$: the characteristic-function
  block moves by at most $2 norm(Theta)_2 r^w_u$. Layers use the exact
  distances (supports read before pruning are exact up to step 2, and
  every node reached at step 3 outside them is at distance 3), so each
  layer loses $sum_(j in "layer") (p_j - q_j) >= 0$, in total $r^w_u$: the
  scaled layer block moves by at most $norm(Theta)_2 r^w_u$. Summing
  squares over blocks and steps gives the bound; a budget enforces
  $r^w_u <= delta$, since propagation by a stochastic matrix keeps the
  missing mass and a row only drops entries while it stays below $delta$.
]

= Experiments

All CPU timings on one core. Classification: logistic regression on
standardised embeddings, 50 stratified 80/20 splits. Best in bold, second
underlined, among embeddings; the degree baseline (log-degree alone, which
shows how far labels follow activity) is set apart and not ranked. Code and CSV files in `benchmarks/`.

#let mark(c, g) = text(font: "DejaVu Sans Mono", fill: rgb(c), weight: "bold")[#g]
#let Y = mark("#1a7f37", "✓")
#let N = mark("#c62828", "✗")
#let P = mark("#b26a00", "∼")
#figure(caption: [Properties, checked against each paper: #Y yes, #N no, #P partial.])[
  #table(columns: 8, align: (left,) + (center,) * 7,
    [], [Transfer], [Directed], [Weighted], [No training], [Online], [Deterministic], [Time (USA)],
    [*RoleWalk*], Y, Y, Y, Y, Y, Y, [0.2 s],
    [struc2vec], N, [#N#super[a]], [#N#super[a]], N, N, N, [564 s],
    [GraphWave], [#N #text(size: 7pt)[(P. #ref(<prop-graphwave>, supplement: none))]], P, Y, Y, N, Y, [18 s],
    [Role2Vec], P, [#Y#super[b]], N, N, P, N, [10 s],
    [RESD], N, N, N, N, N, N, [8 s],
    [ReFeX], P, P, P, Y, P, Y, [0.1 s],
    [RWSE], Y, P, Y, Y, Y, Y, [1.7 s],
  )
  #text(size: 8pt)[#super[a] struc2vec++ @steenfatt2018 extends it. #super[b] In the paper; the reference implementation is undirected. "No training": struc2vec and Role2Vec fit skip-gram models, RESD a variational autoencoder. #todo[verify RESD, ReFeX, RWSE rows]]
] <tab-props>

#figure(caption: [Role classification accuracy. struc2vec is unseeded: mean ± std over 5 runs. The heat-kernel CF is RoleWalk's characteristic function on the rows of $exp(s (P - I))$, $s in {1, 2, 3}$, with the same frequencies: GraphWave with fixed scales and no $1 \/ n$. `roles.csv`, `roles_gw.csv`, `struc2vec_runs.csv`.])[
  #table(columns: 7, align: (left,) + (center,) * 6,
    [], [Brazil], [Europe], [USA], [Reality-call], [Actor], [Film#super[\*]],
    [*RoleWalk*], [0.743], [0.583], [#underline[0.657]], [*0.860*], [*0.487*], [0.569],
    [*RoleWalk + curv.*], [0.741], [#underline[0.596]], [*0.666*], [*0.860*], [#underline[0.485]], [#underline[0.581]],
    [struc2vec], [0.705 ± .022], [0.561 ± .020], [0.596 ± .004], [–], [–], [–],
    [RESD], [0.728], [0.544], [0.653], [0.857], [0.476], [0.563],
    [GraphWave], [*0.769*], [0.561], [0.587], [*0.860*], [0.481], [–],
    [GraphWave, no $1 \/ n$], [*0.769*], [0.560], [0.587], [*0.860*], [0.481], [–],
    [Heat-kernel CF, fixed scales], [#underline[0.750]], [*0.598*], [0.633], [#underline[0.859]], [0.484], [#todo[…]],
    [ReFeX], [0.691], [0.576], [0.622], [0.851], [0.478], [0.579],
    [Role2Vec], [0.476], [0.366], [0.562], [0.844], [0.399], [*0.743*],
    [RWSE], [0.711], [0.572], [0.517], [0.855], [0.363], [0.450],
    table.hline(stroke: 0.4pt),
    [#emph[Degree (baseline)]], [#emph[0.756]], [#emph[0.554]], [#emph[0.567]], [#emph[0.858]], [#emph[0.476]], [#emph[0.478]],
  )
  #text(size: 8pt)[#super[\*] A node-type task (@sec-film). "–": too slow or out of memory.]
] <tab-roles>

#figure(
  image("fig_speed.svg", width: 100%),
  caption: [Accuracy against embedding time (one core, log scale), every method that runs on each benchmark. Airports: mean over Brazil, Europe and USA (struc2vec: mean of 5 runs). Directed motifs: 5% rewired edges, mean of 3 seeds. wiki-RfA: task A of @tab-rfa. Hollow marker: degree, a trivial baseline rather than an embedding. RoleWalk is the only method in the top-left corner of every panel. `speed_figure.py`.],
) <fig-speed>

#figure(caption: [Wikipedia talk networks (RONE): precision\@10 for bots and admins, mean over the 6 networks, and over the 3 smallest (446–2,101 nodes), the only ones struc2vec runs on. `roles.csv`.])[
  #table(columns: 5, align: (left,) + (center,) * 4,
    [], [Bots (6)], [Admins (6)], [Bots (3)], [Admins (3)],
    [RoleWalk], [*0.129*], [*0.324*], [*0.179*], [#underline[0.356]],
    [RoleWalk + curv.], [#underline[0.121]], [#underline[0.322]], [0.162], [#underline[0.356]],
    [struc2vec], [–], [–], [0.146], [*0.391*],
    [GraphWave], [0.115], [0.298], [0.163], [0.341],
    [RESD], [0.114], [0.254], [0.158], [0.334],
    [ReFeX], [0.101], [0.299], [0.161], [0.375],
    [Role2Vec], [0.119], [0.016], [0.158], [0.041],
    [RWSE], [0.118], [0.297], [#underline[0.178]], [0.309],
    table.hline(stroke: 0.4pt),
    [#emph[Degree (baseline)]], [#emph[0.075]], [#emph[0.296]], [#emph[0.121]], [#emph[0.341]],
  )
  #text(size: 8pt)[Mixed: RoleWalk leads on bots and on admins over all networks, struc2vec on admins where it runs. Networks are distributed undirected. #todo[directed versions from SNAP]]
] <tab-wiki>

#figure(caption: [Directed roles on synthetic graphs (directed motifs hung from a directed cycle; several roles coincide once directions are dropped). 3 seeds; noise = share of edges rewired. Accuracy: logistic regression; ARI: $k$-means; mAP: retrieval. ReFeX, directed: in-, out- and mutual degrees and recursive features over out- and in-neighbours (450 one-hot features). `directed.csv`.])[
  #table(columns: 6, align: (left,) + (center,) * 5,
    [], [Acc. 0%], [Acc. 5%], [Acc. 10%], [ARI 5%], [mAP 5%],
    [*RoleWalk + curv. + mixed*], [*1.000*], [#underline[0.732]], [#underline[0.595]], [*0.396*], [*0.629*],
    [RoleWalk + mixed], [*1.000*], [0.728], [0.590], [0.373], [0.615],
    [RoleWalk + curv.], [#underline[0.989]], [0.712], [0.547], [#underline[0.381]], [#underline[0.625]],
    [RoleWalk], [0.908], [0.680], [0.510], [0.366], [0.574],
    [RoleWalk, undirected], [0.767], [0.560], [0.420], [0.351], [0.479],
    [ReFeX, directed], [*1.000*], [*0.756*], [*0.638*], [0.355], [0.607],
    [RWSE, out + in], [0.357], [0.336], [0.336], [0.048], [0.179],
    [Role2Vec], [0.730], [0.451], [0.368], [0.148], [0.295],
    [ReFeX], [0.532], [0.429], [0.431], [0.342], [0.356],
    [GraphWave], [0.586], [0.388], [0.383], [0.317], [0.416],
    [RESD], [0.547], [0.420], [0.423], [0.253], [0.348],
    [struc2vec], [0.510], [0.366], [0.277], [0.141], [0.260],
    table.hline(stroke: 0.4pt),
    [#emph[In/out degree (baseline)]], [#emph[0.516]], [#emph[0.466]], [#emph[0.405]], [#emph[0.310]], [#emph[0.349]],
  )
] <tab-directed>

#figure(caption: [Real directed roles: Wikipedia requests for adminship @west2014 (11,377 users, 188k votes voter → candidate). A: elected among the 3,494 candidates. B: voter / failed candidate / admin. `roles_rfa.csv`, `roles_rfa_new.csv`, `roles_rfa_mixed.csv`.])[
  #table(columns: 4, align: (left, center, center, center),
    [], [A: elected], [B: 3 classes], [Time],
    [*RoleWalk + curv. + mixed*], [*0.827*], [*0.944*], [7 s],
    [RoleWalk + mixed], [0.814], [0.940], [7 s],
    [RoleWalk + curv.], [#underline[0.823]], [#underline[0.943]], [5 s],
    [RoleWalk], [0.808], [0.938], [4 s],
    [RoleWalk, undirected], [0.791], [0.863], [],
    [RoleWalk + curv., undirected], [#underline[0.814]], [0.900], [],
    [RESD], [0.798], [0.843], [68 s],
    [Role2Vec], [0.798], [0.857], [82 s],
    [ReFeX], [0.794], [0.847], [1 s],
    [ReFeX, directed], [0.778], [0.930], [3 s],
    [RWSE (out-walk)], [0.690], [0.826], [51 s],
    [RWSE, out + in], [0.710], [0.836], [84 s],
    table.hline(stroke: 0.4pt),
    [#emph[In/out degree (baseline)]], [#emph[0.755]], [#emph[0.922]], [#emph[0 s]],
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
    table.hline(stroke: 0.4pt),
    [#emph[Degree (baseline)]], [#emph[0]], [#emph[0]], [#emph[1]], [#emph[no]], [#emph[no]], [#emph[no]],
  )
  #text(size: 8pt)[#super[\*] Shrikhande / $4 times 4$ rook: strongly regular with equal parameters, not separated by 2-FWL (classical 3-WL) nor by @prop-drg.]
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
    [], [RoleWalk], [\+ curv.], [ReFeX#super[†]], [RWSE], [GraphWave], [#emph[Degree (baseline)]],
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

#figure(caption: [Role transfer between airport networks: classifier fitted on one network, tested on another (6 ordered pairs, accuracy). Labels are activity quartiles, close to degree. Removing GraphWave's $1 \/ n$ lifts it from chance to 0.455 (@prop-graphwave); fixed scales and a heat kernel instead of walk powers give 0.425. `role_transfer.csv`, `role_transfer_gw.csv`.])[
  #table(columns: 8, align: center,
    [#emph[Degree (baseline)]], [RoleWalk], [\+ curv.], [RoleWalk, no layers], [RWSE], [GraphWave], [GraphWave, no $1 \/ n$], [Heat-kernel CF],
    [#emph[0.542]], [*0.488*], [#underline[0.470]], [0.445], [0.375], [0.249 (chance)], [0.455], [0.425],
  )
] <tab-airport-transfer>

#figure(caption: [Structural encodings for GNNs on ZINC (test MAE at the best validation epoch). Top: GINE, 4 layers, ≈230k parameters, 300 epochs, 3 seeds. Bottom: GPS @rampasek2022 protocol, 10 layers, ≈470k parameters, 500 epochs, 3 seeds. `pe_zinc_v13.csv`, `pe_zinc_gps*.csv`.])[
  #table(columns: 3, align: (left, center, center),
    [GINE], [Test MAE], [],
    [none], [0.202 ± 0.010], [],
    [RWSE], [0.107 ± 0.004], [],
    [RoleWalk], [0.108 ± 0.008], [],
    [RoleWalk + RWSE], [#underline[0.104 ± 0.006]], [],
    [*RoleWalk + curv. + RWSE*], [*0.095 ± 0.002*], [],
    table.hline(stroke: 0.6pt),
    [GPS], [Test MAE], [Val. MAE],
    [RWSE (published: 0.070 ± 0.004)], [0.078 ± 0.004], [0.099 ± 0.008],
    [RoleWalk + curv.], [#underline[0.075 ± 0.002]], [*0.091 ± 0.003*],
    [*RoleWalk + curv. + RWSE*], [*0.072 ± 0.002*], [#underline[0.092 ± 0.006]],
  )
  #text(size: 8pt)[#todo[4th seed running (RWSE: 0.073); Peptides (LRGB)]]
] <tab-zinc>

#figure(caption: [Embedding time against size with the fused kernel, defaults (tol $10^(-4)$, 3 steps, layers), including the conversion of the networkx graph. `scalability.csv`.])[
  #let rows = csv("../benchmarks/scalability.csv", row-type: dictionary)
  #let sizes = rows.map(r => r.n_nodes).dedup()
  #let cell(graph, n, curv, jobs: "1") = {
    let hit = rows.filter(r => r.graph == graph and r.n_nodes == n and r.curvature == curv and r.n_jobs == jobs)
    if hit.len() == 0 { [–] } else { [#hit.first().time s] }
  }
  #table(columns: 1 + sizes.len(), align: (left,) + (center,) * sizes.len(),
    [Nodes], ..sizes.map(n => [#calc.round(int(n) / 1000) k]),
    [Barabási-Albert ($m = 3$), 1 core], ..sizes.map(n => cell("barabasi_albert", n, "False")),
    [  \+ curvature], ..sizes.map(n => cell("barabasi_albert", n, "True")),
    [  8 threads], ..sizes.map(n => cell("barabasi_albert", n, "False", jobs: "8")),
    [Watts-Strogatz ($k = 6$), 1 core], ..sizes.map(n => cell("watts_strogatz", n, "False")),
    [  \+ curvature], ..sizes.map(n => cell("watts_strogatz", n, "True")),
    [  8 threads], ..sizes.map(n => cell("watts_strogatz", n, "False", jobs: "8")),
  )
  #text(size: 8pt)[Peak memory 2.5 GB at a million nodes, of which 1.6 GB hold the networkx graph. Baselines on the same Barabási-Albert graphs, one core (`scalability_baselines.csv`): ReFeX 0.1 s / 1.8 s / 31 s at 10k / 100k / 1M nodes, faster than RoleWalk; 16-step RWSE 68 s at 10k and over 16 GB at 100k; the heat-kernel CF 196 s at 10k. At 1,190 nodes (USA airports): struc2vec 564 s, GraphWave 18 s, Role2Vec 10 s, RESD 8 s; struc2vec and GraphWave do not run on wiki-RfA (11k nodes).]
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
] <tab-geometry>

The probabilistic bound is kept over exact Ollivier-Ricci for cost
(no optimal transport) at a small loss.

Marks on the steps of a path rather than on its end were also tried: the
parity of steps inside a layer (closing odd cycles), obtained from one
extra signed walk $tilde(D)^(-1) (A - I)$ and the layer of arrival, since
on undirected graphs $\#"intra" equiv \#"moves" - d(u, j) space (mod 2)$;
its exact count; and
the order of moves out and in, as products in the quaternion group $Q_8$.
On the 6,781 nodes of connected graphs with at most 7 nodes, none
separates a pair of nodes that $X$ does not, and none improves airport
accuracy (`intra_layer_study.py`, `quaternion_study.py`). With labels
$g$ (out) and $g^(-1)$ (in) in an abelian group and trivial labels
otherwise, the product along a path only depends on its end layer, which
gives back the layers; short cycles already change the multiset of
probabilities.

A purer object was also tried: one characteristic function of each node's
whole trajectory $(1[j = u], p^1_(u j), p^2_(u j), p^3_(u j))$, sampled along
marginal and mixed-time projections (walks stopped at a random time),
in place of per-step functions and layers. It sees whether large
probabilities at different steps sit on the same node, which per-step
multisets forget, and makes every coordinate Lipschitz. At 48 to 160
features it separates the same small-graph nodes as RoleWalk and matches it
on airports (0.664–0.666 against 0.662), but transfers worse (0.454–0.471
against 0.486) and loses on directed motifs (ARI 0.295–0.314 against 0.366)
(`trajectory.csv`): hard layers are a better use of a few dimensions.

== Pruning <sec-pruning>

On the real networks, pruning at $10^(-4)$ changes accuracy by at most
0.005 and the characteristic-function block by a median relative error
below $10^(-4)$; layers move by up to 16% on wiki-RfA and 18% on Actor
(`pruning.csv`). The error concentrates on hubs: for the centre of a star
with 20,000 leaves and the largest hub of a 100k-node Barabási-Albert
graph, the embedding is off by 49% and 27%. A budget $delta = 0.05$
makes both exact (no row drops anything that comes back), certifies every
other row by @prop-prune, and costs $2.2 times$ the time on the
Barabási-Albert graph, where hub rows are kept whole.

== Kernel scaling <sec-scaling>

In raw embeddings $theta = 100$ carries 53% of the variance; unscaled
layers carry 0.02% and leave $k$-means roles unchanged. A grid of 48
configurations (frequency weighting, range, steps, layers), scored on
synthetic and real networks with equal weight (`kernel_grid.csv`): scaling
the layers by $norm(Theta)$ is what matters (score 0.274 → 0.323), the
frequency weighting barely does, and the defaults win whichever dataset is
left out.
Spacing matters less than range (`theta_scheme.csv`): on $[1, 100]$,
log-spaced frequencies score 0.323 against 0.316 for uniform ones, with the
same airport accuracy (0.661 / 0.662) and transfer (0.488 / 0.485). A
narrower range helps real networks and transfer but loses on synthetic
motifs, whatever the spacing: $[1, 30]$ gives transfer 0.515 (log or
uniform) and score 0.312–0.315; log $[1, 50]$ gives 0.506 and 0.326, within
noise of the default. Log spacing is kept as a heuristic: probabilities
spread over several decades, which a log grid samples evenly, while a
uniform grid puts 9 of its 10 frequencies above 10. Feature-wise
standardisation, as in our classifiers, cancels any constant factor on a
block: the scaling of the layers matters for raw distances ($k$-means,
Ward, nearest neighbours), not for the classification tables.

== Film is a node-type task <sec-film>

#figure(caption: [Film (RONE). Class 3 is an independent set in a separate block of node ids.])[
  #table(columns: 3, align: (left, center, center),
    [], [Class 3 vs rest], [Within classes 0–2],
    [Role2Vec], [*0.946*], [0.637],
    [RoleWalk + curv.], [0.740], [#underline[0.634]],
    [ReFeX], [#underline[0.745]], [0.627],
    [#emph[Degree (baseline)]], [#emph[0.536]], [#emph[0.623]],
    [Majority class], [0.593], [0.623],
  )
]

No method predicts classes 0–2 clearly above the majority class; edge
homophily is 0.28, below the 0.32 expected from random labels with the
same (degree-weighted) class proportions. Role2Vec's lead on class 3 is
consistent with walk co-occurrence, i.e. position, which an
automorphism-invariant embedding cannot use; we read Film as a node-type
task rather than a role benchmark.

= Limitations

- Walk statistics cannot separate distance-regular graphs with equal arrays (@prop-drg), most CFI pairs (3 of 100 with 5 steps), or graphs whose balls coincide (locality); without curvature, some nodes separated by 1-WL are confused (@prop-wl).
- Without a mass budget, pruning error is not bounded on hubs (@sec-pruning); the budget costs time on hub-heavy graphs.
- ReFeX is faster (5× at a million nodes) and wins on noisy synthetic directed motifs in supervised accuracy and on $k$-core transfer; a fixed-scale heat-kernel CF matches RoleWalk on two of three airport networks.
- Middle-tier roles on real networks often form a continuum; per-node stability should be reported next to roles.
- Degree-like labels (airports) favour trivial baselines.
- One real labelled directed benchmark.
- GPS on ZINC: 3 seeds; our RWSE reproduction (0.078) is weaker than the published 0.070, and the gain of RoleWalk + curvature + RWSE (0.072) is of the size of the seed noise.

#bibliography("refs.bib", style: "ieee")
