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
  rather than by proximity. RoleWalk follows a short random walk from
  each node and describes, at every step, two laws: that of the transition
  probabilities, through its characteristic function sampled at a few
  frequencies (how concentrated the walk is), and that of the walker's
  distance to its start (how far it has gone); optionally, low-order
  moments of a probabilistic curvature where it lands. The embedding is local,
  deterministic, reads directions and weights, and needs no training. We
  prove that it is invariant and local (so its coordinates mean the same in every graph),
  that its curvature-free distances are maximum mean discrepancies, that its
  characteristic-function block is Lipschitz in edge insertions, that it
  recovers triangle counts on regular graphs, so that it separates graphs
  1-WL cannot (and conversely: the two are incomparable), and that
  distance-regular graphs with equal intersection arrays are a provable
  failure class. With a budget on the probability mass that pruning may
  drop, the approximation error is certified. Empirically it matches or
  beats struc2vec, GraphWave, Role2Vec, RESD and ReFeX on role
  classification at a fraction of the cost of every method but ReFeX, is
  has the best scores on a real directed-role dataset and on the transfer
  of centralities to larger graphs, embeds a million nodes in 20 s to 2.5 min on one core, and
  shows a preliminary improvement of GPS on ZINC when combined with RWSE.
]

= Introduction

Claims, each backed by a result below:

+ All of: inductive, directed, weighted, training-free, deterministic, online, fast (@tab-props).
+ Accuracy at or above the state of the art on role classification at a fraction of the cost; a million nodes in minutes on one core (@tab-roles, @fig-speed, @tab-scale).
+ Reads edge directions natively, including walks that change direction; highest scores on both wiki-RfA tasks, best retrieval and competitive clustering on synthetic directed motifs; on food webs, comparable to local statistics (@tab-directed, @tab-rfa, @tab-foodweb, @prop-mixed).
+ Pruning with a certified error (@prop-prune).
+ Transfer: a regressor fitted on one graph predicts path-based centralities (betweenness, closeness, harmonic) of another, including one four times larger, better than every baseline (@tab-centrality; @prop-local).
+ Separates graphs that 1-WL cannot with 3 walk steps (the two are incomparable), close to 16-step RWSE on BREC; a provable failure class (@sec-theory, @tab-expr).
+ Useful as a structural encoding for graph transformers (@tab-zinc).

= Related work

Feature-based roles: ReFeX @henderson2011, RolX @henderson2012. Walks
and word2vec: struc2vec @ribeiro2017, Role2Vec @ahmed2018; directed and
weighted extension of struc2vec, struc2vec++ @steenfatt2018 (not evaluated
here). Diffusion: GraphWave @donnat2018, closest to RoleWalk
(@prop-graphwave), and Digraphwave @ceylan2022, its extension to directed
and weighted graphs (heat diffusion on the out- and in-graph, timescales
read from each graph, characteristic functions aggregated over neighbours). Deep role embeddings: RESD @resd, from the RONE survey and benchmark
@rone, whose code and Wikipedia talk datasets we use. Structural encodings for GNNs: RWSE
@dwivedi2022, used by GPS @rampasek2022. Expressiveness: 1-WL @weisfeiler1968 @xu2019 @morris2019,
CSL @murphy2019, BREC @wang2023. Curvature: Ollivier-Ricci @ollivier2009 @jost2014.
Kernel mean embeddings and MMD: @gretton2012.

= Method <sec-method>

*Walk.* $G = (V, E)$ with non-negative weights $A$ (directed or not),
$tilde(D) = op("diag")((I + A) bold(1))$ and the self-loop-augmented walk
$H = tilde(D)^(-1) (I + A)$. On weighted graphs the self-loop of $u$ weighs
the mean $s_u \/ d_u$ of its edge weights ($s_u$ weighted, $d_u$ unweighted
degree), which is 1 on unweighted graphs: the walk stays put with
probability $1 \/ (d_u + 1)$ in both cases, and does not depend on the unit
of the weights. For a node $u$ and a step $w$, $p^w_u = e_u^top H^w$
is the distribution of the walk after $w$ steps. Directed graphs use the
out-walk $H_+$ ($A$) and the in-walk $H_-$ ($A^top$), concatenate both
embeddings, and add the characteristic functions of the rows of
$H_+ H_-$ (one step forward, one back: nodes sharing targets with $u$) and
$H_- H_+$ (shared sources), the _mixed probes_ (@prop-mixed): 178 features,
190 with curvature. On undirected graphs $H_+ = H_-$ and the probes would
repeat $H^2$, so they are left out.

*Characteristic function.* With $f_theta (x) = e^(i theta x) - 1$ and a
grid $Theta = {theta_1, ..., theta_m}$ (log-spaced in $[1, 100]$, $m = 10$),
$ phi^w_u (theta) = sum_(j in V) f_theta (p^w_(u j)), quad theta in Theta. $ <eq-phi>
Since $f_theta (0) = 0$, the sum runs over the support of $p^w_u$ only.

*Distance of the walker.* With $d(u, j)$ the length of the shortest walk
from $u$ to $j$ and $X_w$ the walker's position after $w$ steps,
$ s^w_u (k) = P(d(u, X_w) = k) = sum_(j : d(u, j) = k) p^w_(u j), quad k = 0, ..., w, $
is the law of the walker's distance to its start, which the multiset of
probabilities ignores: whether the walk stays near $u$ (loops, triangles)
or moves out. It takes $w + 1$ values, so these $w + 1$ numbers describe it
exactly. We call the sets of nodes at each distance _onion layers_.
Supports of earlier powers give the distances, so no extra walk is
computed.

At each step, RoleWalk thus describes two laws of the same walk: how
concentrated it is (the characteristic function of its probabilities) and
how far it has gone (the law of its distance). Both are integrals over one
measure, under one kernel (@prop-mmd).

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
The embedding is the integral of a single feature map over the footprint
(a finite measure, not normalised),
$X_u = integral Psi thin d mu_u$, with
$ Psi(w, a, ell) = e_w ⊗ (psi(a) ⊕ norm(Theta) a e_ell), quad psi(a) = (f_theta (a))_(theta in Theta), $ <eq-psi>
which gives, for each step $w = 1, ..., L$ ($L = 3$), $op("Re") phi^w_u$,
$op("Im") phi^w_u$ and $norm(Theta) s^w_u$. With curvature the mark is
$(d(u, j), kappa_j)$ and $Psi$ adds $norm(Theta) a (kappa, kappa^2)$; we report
$norm(Theta) (mu^w_u, sigma^w_u)$, which carries the same information but,
being nonlinear, is not an integral of $Psi$: kernel statements below are
for the curvature-free embedding (or raw moments). In all,
$2 m L + sum_(w <= L) (w + 1) (+ 2 L) = 60 + 9 (+ 6) = 69 (75)$ features: the
characteristic function, then the law of the distance at each step. The
probability block ignores the distance; the distance block is the
first-order part of one characteristic function per layer, which is what
fixes the factor $norm(Theta) = (sum_theta theta^2)^(1\/2)$ (@prop-mmd).
On directed graphs the step mark $w$ becomes a block mark $b$, a direction
and a step $(plus.minus, w)$ or a mixed probe ($+-$, $-+$), and mixed-probe
atoms carry no layer: $Psi(b, a) = e_b ⊗ psi(a)$.
Equivalently, with the centred joint Fourier transform of the footprint
$F^w_u (theta, omega) = sum_j e^(i omega d(u, j)) (e^(i theta p^w_(u j)) - 1)$
(unnormalised: $F(0, 0) = 0$; not the law of $(p_(u X_w), d(u, X_w))$, which
would weight atom $j$ by $p_(u j)$), the characteristic function is
$F^w_u (theta, 0)$ and the law of the distance is its derivative at
$theta = 0$, $-i partial_theta F^w_u (0, omega) = sum_k s^w_u (k) e^(i omega k)$, whose
values at $omega = 2 pi k \/ (w + 1)$ (unitary DFT) determine $s^w_u$ with the
same Euclidean geometry. Without it
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
read from supports before pruning, as the step at which each node is
first reached: its distance when $"tol" = 0$; with pruning, for $L <= 3$
every retained entry still gets its exact distance, while layer masses lose
what pruning drops; beyond three steps, a node first reached through a
pruned entry is labelled later than its distance. Both implementations
use this convention. A
row whose small entries would exceed the budget keeps all of them.
Results in @sec-theory are otherwise stated for $"tol" = 0$.

*Implementation.* For each root, one pass propagates a sparse row with a
dense accumulator, prunes it, and summarises it (characteristic function,
layers from the step at which each node is first reached, curvature
moments) without storing the powers: $O(n)$ memory per thread, and rows
are independent. Mixed probes are sparse products computed by blocks of
rows of bounded work (entries before pruning). Below $p = 10^(-3)$ every angle $theta p$ is under 0.1,
and the characteristic function comes from the power sums $sum p^k$,
$k <= 8$ instead of $2 m$ sines (truncation error below
$theta^9 p_max^8 \/ 9! < 3 dot 10^(-12)$ per coordinate and row in exact
arithmetic; outputs are float32).

*Roles.* Ward clustering with silhouette for the number of roles, or
$k$-means with a given number; fitted centroids label nodes of any graph.

= Properties <sec-theory>

Throughout, $L$ is the number of steps, $B_r (u)$ the subgraph induced by
nodes within distance $r$ of $u$ (out- and in-distance for directed graphs;
with curvature, or on directed graphs, distance in the underlying undirected
graph, which contains everything these read), graphs have finite
non-negative weights and no self-loops, distances are taken in the graph
of positive-weight edges, $m_x$ and $kappa$ are defined on the simple
unweighted symmetrisation ($kappa = 0$ at isolated nodes), and statements
hold for $"tol" = 0$, in exact arithmetic.

#prop(name: "locality and invariance")[
  $X_u$ is determined by the isomorphism class of the rooted graph
  $(B_r (u), u)$, with $r = L$ for pure walks, $r = max(L, 2)$ (undirected
  distance) on directed graphs with mixed probes, and $r = L + 2$ with
  curvature. In particular it
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
  neighbourhoods of $j$ and of its neighbours, inside $B_(L+2) (u)$. A
  mixed probe goes from $u$ to a neighbour $x$ (one direction) then to a
  neighbour of $x$ (the other): it needs the edges at $u$ and at $x$, all
  within undirected distance 2 of $u$, including the normalisation of the
  second step at $x$.
]

Coordinates therefore mean the same thing in every graph, which makes
transfer (@tab-centrality, @tab-airport-transfer) well defined; whether it
succeeds is an empirical question.

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
  and on directed graphs, with block marks, for
  $K((beta, a, ell), (beta', b, ell')) = delta_(beta beta') [k_Theta (a, b) + bb(1)_(beta "pure") norm(Theta)^2 a b thin delta_(ell ell')]$,
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
per-layer kernel is not consistently better despite its larger dimension:
it helps on synthetic and directed motifs and hurts airport transfer
(@tab-geometry). Further marks enter the
same way, as factors of the second term: curvature, or labels on the
steps of each path.

#prop(name: "stability")[
  Let $G$ be simple, undirected and unweighted, and $G'$ be $G$ plus an edge $a b$, with $d = min(d_a, d_b)$
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
  Let $G$ be simple, undirected and unweighted. Then
  $s^1_u (0) = 1 \/ (d_u + 1)$, and if $G$ is $d$-regular ($d > 0$), $s^3_u (0) = (H^3)_(u u) = (1 + 3 d + 2 t_u) \/ (d + 1)^3$, with $t_u$
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
  $d$-regular graph with $d > 0$, ${(H^w)_(u u)}_(w <= L)$ and the simple-walk RWSE
  ${(P^k)_(u u)}_(k <= L)$, $P = A \/ d$, determine each other.
]
#proof[
  $H = (I + d P) \/ (d + 1)$, so
  $(H^w)_(u u) = (d + 1)^(-w) sum_(k=0)^w binom(w, k) d^k (P^k)_(u u)$: a
  triangular system with nonzero diagonal $d^w \/ (d + 1)^w$.
]

On irregular graphs the binomial transformation above does not apply; RoleWalk adds the other layers and the
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
] <prop-locality-graph>
#proof[
  The pendants are more than $2(L + 2)$ apart in both graphs, so every
  rooted ball $B_(L+2) (u)$ contains at most one pendant and is fixed by
  $u$'s distance to it; both graphs have the same multiset of balls.
  Conclude by @prop-local. For 1-WL, let $h$ mark the two degree-3 nodes,
  which round 1 identifies; round $t + 1$ colours determine $(A^t h)_v$,
  the number of $t$-step walks from $v$ to a marked node. Both graphs are
  bipartite (an even cycle with pendant trees). In $A$, the only 15-step
  walk from node 0 to node 15 is the short arc, and no odd walk returns
  to 0, so $(A^(15) h)_0 = 1$; in $B$, nodes 0 and 20 are on the same
  side, so $(A^(15) h)_0 = 0$. After 16 rounds the marked nodes have
  different colours.
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
  three rounds separate $z$. From $z$, step 1 puts $1 \/ 2$ on $z$ and $r$;
  step 2 puts $1 \/ 3$ on $z$ and $r$ and $1 \/ 12$ on each of $a, b, c, d$;
  step 3 puts $2 \/ 9$ on $z$, $23 \/ 90$ on $r$, $4 \/ 45$ on each of
  $a, b, c, d$ and $1 \/ 60$ on each outer node, in both graphs, with layers
  $(2 \/ 9, 23 \/ 90, 16 \/ 45, 1 \/ 6)$ (exact rational computation,
  `theory_checks.py`). With curvature, both pairs are separated. These
  are node-level collisions; @prop-locality-graph gives a graph-level one.
]

On the 6,781 nodes of the 996 connected graphs with at most 7 nodes, no
pair separated by 3-round 1-WL is confused by RoleWalk ($L = 3$), which
splits 571 of the 1-WL classes (embeddings grouped by single linkage at a
sup-distance below $10^(-4)$ of the largest coordinate, i.e. connected
components of that graph; `theory_checks.py`); the
counterexamples above have 15 and 16 nodes. RoleWalk is also local, which
1-WL with unbounded rounds is not.

#prop(name: "walks that change direction")[
  Let $u$ be a source pointing to two sinks ($G$), or one of the two
  sources of a directed $K_(2, 2)$ ($G'$). For every $L$, with or without
  curvature, $u$ gets the same out- and in-walk embedding in $G$ and $G'$;
  but $e_u^top H_+ H_-$ has values $(2 \/ 3, 1 \/ 6, 1 \/ 6)$ in $G$ and
  $(5 \/ 9, 2 \/ 9, 1 \/ 9, 1 \/ 9)$ in $G'$; at $theta = 1$ (in the default
  grid) $op("Re") phi$ is $-0.2418$ and $-0.1873$, so the mixed probes separate them.
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
  Let $q^t_u$ be the retained (pruned) row after step $t$. Step $w$ costs
  $O(sum_u sum_(j in "supp" q^(w-1)_u) (d_j + 1))$; with threshold pruning
  (no budget), $|"supp" q^t_u| <= min(|B_t (u)|, floor(1 \/ "tol"))$ for
  $t >= 2$, while the first step keeps all $d_u + 1$ entries and a budget
  can keep any number. The cap is on retained rows, not on the
  accumulator before pruning. The characteristic function adds
  $O(m dot "nnz")$, the layers $O("nnz")$. As
  $kappa(x, y) = (c_(x y) + 2) \/ (max(d_x, d_y) + 1)$ with $c_(x y)$ the
  common neighbours of $x$ and $y$, curvature costs a triangle enumeration,
  $O(|E|^(3\/2))$ plus $O(n log n + |E|)$ set-up (on a subset of nodes,
  $O(min(d_x, d_y) log d)$ per edge at those nodes). The embedding of a set
  $S$ depends only on its ball (@prop-local), and only its walks are
  computed; the transition matrices are still built for the whole graph.
]
#proof[
  Entries are non-negative and each row sums to at most 1, so at most
  $1 \/ "tol"$ of them are $>= "tol"$. Both closed neighbourhoods are uniform
  ($1 \/ (d + 1)$ on $d + 1$ nodes) and share $x$, $y$ and their common
  neighbours, so the overlap is $c_(x y) + 2$ times the smaller mass;
  orienting edges by degree enumerates each triangle once, scanning at
  most $sqrt(2 |E|)$ forward neighbours per node: $O(|E|^(3\/2))$. The rest
  follows from sparse products and @prop-local.
]


#prop(name: "certified pruning")[
  For one walk direction, without curvature or mixed probes, $L <= 3$ and
  in exact arithmetic, let $hat(X)_u$ be computed with pruning and $r^w_u$
  the mass row $u$ misses at step $w$. Then
  $ norm(X_u - hat(X)_u)_2 <= sqrt(5) norm(Theta)_2 (sum_(w <= L) (r^w_u)^2)^(1\/2), $
  so with a budget $delta$, $norm(X_u - hat(X)_u)_2 <= sqrt(5 L) norm(Theta)_2 delta$
  ($sqrt(10 L) norm(Theta)_2 delta$ for the two directions of a directed
  graph). Mixed probes with missing masses $r_beta$ add
  $4 norm(Theta)_2^2 sum_beta r_beta^2$ to the squared bound.
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
  Directions are orthogonal blocks; a mixed probe is a characteristic
  function without layers, hence $2 norm(Theta)_2 r_beta$.
]

The bound is a posteriori without a budget ($r^w_u <= 1$) and a priori with
one; it is absolute in feature space ($norm(Theta)_2 approx 125$, so about
24 at $delta = 0.05$, $L = 3$), not a guarantee on downstream accuracy.
The certificate is for the exact features: the returned array also
carries the Taylor truncation of the fused kernel and float32 rounding,
$norm(X_u - hat(X)_u^"returned") <= B_"prune" + epsilon_"Taylor" + epsilon_"float"$, and
curvature moments are not covered.


= Experiments

All CPU timings on one core. Classification: logistic regression on
standardised embeddings, 50 stratified 80/20 splits. Best in bold, second
underlined, among embeddings; the degree baseline (log-degree alone, which
shows how far labels follow activity) is set apart and not ranked. Code and CSV files in `benchmarks/`.

#let mark(c, g) = text(font: "DejaVu Sans Mono", fill: rgb(c), weight: "bold")[#g]
#let Y = mark("#1a7f37", "✓")
#let N = mark("#c62828", "✗")
#let P = mark("#b26a00", "∼")
#let note(m, k) = [#m#box(width: 0pt)[#super[#k]]]
#figure(caption: [Properties, checked against each paper: #Y yes, #N no, #P partial.])[
  #table(columns: 8, align: (left,) + (center,) * 7,
    [], [Local, fixed coords], [Directed], [Weighted], [No training], [Online], [Deterministic], [Time (USA)],
    [*RoleWalk*], Y, Y, Y, Y, Y, Y, [0.2 s],
    [struc2vec], N, note(N, "a"), note(N, "a"), N, N, N, [564 s],
    [GraphWave], note(N, "c"), P, Y, Y, N, Y, [18 s],
    [Role2Vec], P, note(Y, "b"), N, N, P, N, [10 s],
    [RESD], N, N, N, N, N, N, [8 s],
    [ReFeX], P, P, P, Y, P, Y, [0.1 s],
    [Digraphwave], P, Y, Y, Y, P, Y, [1.1 s],
    [RWSE], Y, P, Y, Y, Y, Y, [1.7 s],
  )
  #text(size: 8pt)[#super[a] struc2vec++ @steenfatt2018 extends it. #super[b] In the paper; the reference implementation is undirected. #super[c] Scale chosen from the spectrum and $1 \/ n$ factor (@prop-graphwave). "No training": struc2vec and Role2Vec fit skip-gram models, RESD a variational autoencoder.]
] <tab-props>

#figure(caption: [Role classification accuracy. struc2vec is unseeded: mean ± std over 5 runs. The random-walk heat-kernel CF is RoleWalk's characteristic function on the rows of $exp(s (P - I))$, $s in {1, 2, 3}$, with the same frequencies: fixed scales and no $1 \/ n$, but a random-walk rather than GraphWave's spectral heat kernel. Top-$k$ + layers: the 20 largest probabilities at each step instead of the characteristic function, same layers, same dimension. Local statistics: see @tab-centrality. `roles.csv`, `roles_gw.csv`, `roles_matched.csv`, `struc2vec_runs.csv`.])[
  #table(columns: 7, align: (left,) + (center,) * 6,
    [], [Brazil], [Europe], [USA], [Reality-call], [Actor], [Film#super[\*]],
    [*RoleWalk*], [0.743], [0.583], [#underline[0.657]], [#underline[0.860]], [#underline[0.487]], [0.569],
    [*RoleWalk + curv.*], [0.741], [#underline[0.596]], [*0.666*], [#underline[0.860]], [0.485], [0.581],
    [struc2vec], [0.705 ± .022], [0.561 ± .020], [0.596 ± .004], [–], [–], [–],
    [RESD], [0.728], [0.544], [0.653], [0.857], [0.476], [0.563],
    [GraphWave], [*0.769*], [0.561], [0.587], [#underline[0.860]], [0.481], [–],
    [GraphWave, no $1 \/ n$], [*0.769*], [0.560], [0.587], [#underline[0.860]], [0.481], [–],
    [Random-walk heat-kernel CF], [#underline[0.750]], [*0.598*], [0.633], [0.859], [0.484], [0.549],
    [Digraphwave], [0.724], [0.593], [0.641], [0.859], [*0.488*], [#underline[0.606]],
    [Top-$k$ + layers], [0.672], [0.590], [0.625], [0.858], [0.482], [0.564],
    [Local statistics], [0.716], [0.584], [0.616], [*0.862*], [0.486], [0.517],
    [ReFeX], [0.691], [0.576], [0.622], [0.851], [0.478], [0.579],
    [Role2Vec], [0.476], [0.366], [0.562], [0.844], [0.399], [*0.743*],
    [RWSE], [0.711], [0.572], [0.517], [0.855], [0.363], [0.450],
    table.hline(stroke: 0.4pt),
    [#emph[Degree (baseline)]], [#emph[0.756]], [#emph[0.554]], [#emph[0.567]], [#emph[0.858]], [#emph[0.476]], [#emph[0.478]],
  )
  #text(size: 8pt)[#super[\*] A node-type task (@sec-film). "–": too slow or out of memory. Paired tests over the 50 splits (Nadeau–Bengio corrected intervals, `paired_tests.csv`): RoleWalk + curv. against each other method, at 5% without correction for the 13 comparisons per dataset; the splits are of one graph, not independent graphs. On USA it is significantly better than every method except RoleWalk and RESD. On Brazil, Europe, Reality-call and Actor it is significantly better than Role2Vec (and on Brazil, top-$k$ + layers and directed ReFeX; on Reality-call, ReFeX and RWSE; on Actor, RWSE), and no method is significantly better than it: against GraphWave, the heat-kernel CF, RESD, ReFeX or degree the intervals include 0 (±0.05 on Brazil and Europe, 131 and 399 nodes). On Film, Role2Vec wins. Within one graph, GraphWave without $1 \/ n$ is GraphWave up to an affine map that standardisation cancels: the two rows differ by numerical noise. Appending the local statistics to RoleWalk gives 0.752 / 0.595 / 0.659 / 0.861 / 0.486 / 0.598.]
] <tab-roles>

#figure(
  image("fig_speed.svg", width: 100%),
  caption: [Accuracy against embedding time (one core, log scale), every method that runs on each benchmark. Airports: mean over Brazil, Europe and USA (struc2vec: mean of 5 runs). Directed motifs: 5% rewired edges, mean of 3 seeds. wiki-RfA: task A of @tab-rfa. Hollow marker: degree, a trivial baseline rather than an embedding. On directed graphs RoleWalk includes the mixed probes. RoleWalk is alone in the top-left corner on airports and wiki-RfA; on directed motifs, directed ReFeX is more accurate and 40 times slower. `speed_figure.py`.],
) <fig-speed>

#figure(caption: [Wikipedia talk networks (RONE): precision\@10 for bots and admins, mean over the 6 networks, and over the 3 smallest (446–2,101 nodes), the only ones struc2vec runs on; only two of these have admins (8 and 16), so the "Admins (3)" column rests on 24 queries. `roles.csv`.])[
  #table(columns: 5, align: (left,) + (center,) * 4,
    [], [Bots (6)], [Admins (6)], [Bots (3)], [Admins (3)],
    [RoleWalk], [*0.129*], [*0.324*], [*0.179*], [0.356],
    [RoleWalk + curv.], [#underline[0.121]], [#underline[0.322]], [0.162], [0.356],
    [struc2vec], [–], [–], [0.146], [*0.391*],
    [GraphWave], [0.115], [0.298], [0.163], [0.341],
    [RESD], [0.114], [0.254], [0.158], [0.334],
    [ReFeX], [0.101], [0.299], [0.161], [#underline[0.375]],
    [Role2Vec], [0.119], [0.016], [0.158], [0.041],
    [RWSE], [0.118], [0.297], [#underline[0.178]], [0.309],
    table.hline(stroke: 0.4pt),
    [#emph[Degree (baseline)]], [#emph[0.075]], [#emph[0.296]], [#emph[0.121]], [#emph[0.341]],
  )
  #text(size: 8pt)[RoleWalk leads on bots and on admins over all networks. Where struc2vec runs, its lead on admins comes from br-wiki (8 admins; per-query paired Wilcoxon $p = 0.06$ against RoleWalk, $p = 0.04$ for ReFeX, uncorrected); exact walks, a mass budget, other walk lengths, wider frequency ranges, local statistics or whitened distances do not change it. Admins differ from other hubs by low clustering and low neighbour degree: they write to many newcomers who do not write to each other. Networks are distributed undirected. #todo[directed versions from SNAP]]
] <tab-wiki>

#figure(caption: [Directed roles on synthetic graphs (directed motifs hung from a directed cycle; several roles coincide once directions are dropped). 3 seeds; noise = share of edges rewired. Accuracy: logistic regression; ARI: $k$-means; mAP: retrieval. ReFeX, directed: in-, out- and mutual degrees and recursive features over out- and in-neighbours (450 one-hot features). Top-$k$ + layers reads the out- and in-walks, not the mixed probes: compare it with RoleWalk without them (note). `directed.csv`.])[
  #table(columns: 6, align: (left,) + (center,) * 5,
    [], [Acc. 0%], [Acc. 5%], [Acc. 10%], [ARI 5%], [mAP 5%],
    [*RoleWalk + curv.*], [*1.000*], [#underline[0.730]], [#underline[0.593]], [#underline[0.396]], [*0.629*],
    [*RoleWalk*], [*1.000*], [0.728], [0.588], [0.373], [#underline[0.615]],
    [ReFeX, directed], [*1.000*], [*0.756*], [*0.638*], [0.355], [0.607],
    [Digraphwave], [*1.000*], [0.717], [0.575], [0.292], [0.539],
    [Local statistics (in / out)], [0.863], [0.675], [0.584], [*0.409*], [0.609],
    [Top-$k$ + layers], [#underline[0.908]], [0.671], [0.516], [0.348], [0.568],
    [RWSE, out + in], [0.357], [0.336], [0.336], [0.048], [0.179],
    [Role2Vec], [0.732], [0.440], [0.373], [0.153], [0.296],
    [ReFeX], [0.532], [0.429], [0.431], [0.342], [0.356],
    [GraphWave], [0.586], [0.388], [0.383], [0.317], [0.416],
    [RESD], [0.547], [0.420], [0.423], [0.253], [0.348],
    [struc2vec], [0.510], [0.366], [0.277], [0.141], [0.260],
    table.hline(stroke: 0.4pt),
    [#emph[In/out degree (baseline)]], [#emph[0.516]], [#emph[0.466]], [#emph[0.405]], [#emph[0.310]], [#emph[0.349]],
  )
  #text(size: 8pt)[Ablations. Without the mixed probes: RoleWalk + curv. 0.989 / 0.717 / 0.547 / 0.381 / 0.625, RoleWalk 0.908 / 0.682 / 0.510 / 0.366 / 0.580 (never better on any seed and noise level for RoleWalk; within noise for RoleWalk + curv.). On the symmetrised graph: RoleWalk 0.767 / 0.560 / 0.420 / 0.351 / 0.474. Classifier: over 10 splits per graph, with gradient boosting instead of logistic regression, RoleWalk + curv. reaches 1.000 / 0.744 / 0.586 and directed ReFeX 1.000 / 0.739 / 0.578 (logistic: 1.000 / 0.741 / 0.585 against 1.000 / 0.753 / 0.610), so the two are comparable on these motifs; no other method changes rank (`directed_gb*.csv`).]
] <tab-directed>

#figure(caption: [Real directed roles: Wikipedia requests for adminship @west2014 (11,377 users, 188k votes voter → candidate; signs dropped, repeated votes merged, unweighted). A: elected among the 3,494 candidates. B: voter / failed candidate / admin. `roles_rfa*.csv`.])[
  #table(columns: 4, align: (left, center, center, center),
    [], [A: elected], [B: 3 classes], [Time],
    [*RoleWalk + curv.*], [*0.827*], [*0.944*], [7 s],
    [*RoleWalk*], [#underline[0.814]], [0.940], [7 s],
    [RESD], [0.798], [0.843], [68 s],
    [Role2Vec], [0.798], [0.857], [82 s],
    [ReFeX], [0.794], [0.847], [1 s],
    [ReFeX, directed], [0.778], [0.930], [3 s],
    [Digraphwave], [0.807], [0.939], [360 s],
    [Top-$k$ + layers], [0.804], [0.936], [5 s],
    [Local statistics (in / out)], [#underline[0.814]], [#underline[0.941]], [3 s],
    [RWSE, out + in], [0.710], [0.836], [84 s],
    table.hline(stroke: 0.4pt),
    [#emph[In/out degree (baseline)]], [#emph[0.755]], [#emph[0.922]], [#emph[0 s]],
  )
  #text(size: 8pt)[struc2vec and GraphWave do not run at this size. Ablations. Without the mixed probes: RoleWalk + curv. 0.823 / 0.943 (5 s), RoleWalk 0.808 / 0.938 (4 s); on the symmetrised graph: 0.814 / 0.900 and 0.791 / 0.863. RWSE on the out-walk only: 0.690 / 0.826.]
] <tab-rfa>

#figure(caption: [Trophic roles in 20 real food webs @ulanowicz1998 (1,086 compartments, 20–125 per web; edge prey → predator, weighted by carbon flow; boundary nodes removed). Labels: 8 trophic guilds (producer, detritus, microbial, herbivore, detritivore, omnivore, invertivore, top predator), given by an LLM from the compartment names and the ecosystem alone, without the graph (`foodweb_labels.py`; 63% high confidence). Leave one ecosystem out (12 ecosystems; seasons and sub-webs of one ecosystem stay together): fitted on the others, tested on an unseen ecosystem. Balanced accuracy, mean over ecosystems. `foodwebs.csv`.])[
  #table(columns: 3, align: (left, center, center),
    [], [Carbon flows], [Unweighted],
    [*RoleWalk*], [0.428], [0.524],
    [RoleWalk, no mixed probes], [0.420], [0.459],
    [RoleWalk + curv.], [0.413], [0.492],
    [Top-$k$ + layers], [0.377], [0.489],
    [Digraphwave], [0.429], [*0.559*],
    [Local statistics (in / out)], [–], [#underline[0.527]],
    [ReFeX (frozen)], [–], [0.232],
    [RWSE, out + in], [0.214], [–],
    table.hline(stroke: 0.4pt),
    [#emph[In/out degree and flow (baseline)]], [#emph[0.436]], [–],
  )
  #text(size: 8pt)[Carbon flows span 19 orders of magnitude; read as transition weights they send walks along the few largest flows (to detritus), and every walk-based method is better on the unweighted web, where the guild is who eats whom. There, the mixed probes add 0.065 $[0.006, 0.124]$ (paired over ecosystems, better on 10 of 12): sharing prey or predators (competition) is a trophic role. RoleWalk ties the local statistics ($-0.003$ $[-0.08, 0.08]$) and is below Digraphwave on unweighted webs; RoleWalk on the symmetrised flow web: 0.361.]
] <tab-foodweb>

#figure(caption: [Expressiveness: pairs told apart (CSL: number of the 10 isomorphism classes told apart). Every set is 1-WL-indistinguishable. Two graphs are told apart when their multisets of node embeddings differ beyond float32 noise (sorted rows compared at a relative tolerance $10^(-4)$). No false difference on random relabellings. `expressiveness.csv`.])[
  #table(columns: 7, align: (left,) + (center,) * 6,
    [], [Atlas ≤ 7 (20)], [Regular (3,181)], [CSL (10 cl.)], [Decalin / bicyc.], [$2 C_3$ / $C_6$], [SRG pair#super[\*]],
    [*RoleWalk (3 steps)*], [*20*], [*3,181*], [#underline[6]], [yes], [yes], [no],
    [RoleWalk (5 steps)], [*20*], [*3,181*], [*10*], [yes], [yes], [no],

    [RWSE (16 steps)], [*20*], [#underline[3,180]], [*10*], [yes], [yes], [no],
    table.hline(stroke: 0.4pt),
    [#emph[Degree (baseline)]], [#emph[0]], [#emph[0]], [#emph[1]], [#emph[no]], [#emph[no]], [#emph[no]],
  )
  #text(size: 8pt)[#super[\*] Shrikhande / $4 times 4$ rook: strongly regular with equal parameters, hence not separated by @prop-drg. Without layers, RoleWalk (3 steps) separates 16 / 3,180 / 6.]
] <tab-expr>

#figure(caption: [BREC @wang2023 (400 pairs).])[
  #table(columns: 8, align: (left,) + (center,) * 7,
    [], [Basic (60)], [Regular (100)], [Extension (100)], [CFI (100)], [4-vertex (20)], [Dist.-reg. (20)], [Total],
    [RoleWalk (3 steps)], [60], [50], [97], [0], [0], [0], [207],
    [RoleWalk (5 steps)], [60], [50], [100], [3], [0], [0], [#underline[213]],

    [RWSE (16 steps)], [60], [50], [100], [4], [0], [0], [*214*],
  )
  #text(size: 8pt)[Distance-regular: 0 by @prop-drg. The 50 regular pairs missed are exactly the 50 strongly regular pairs, all with equal parameters (checked). Without layers, RoleWalk (3 steps) separates 203 pairs.]
] <tab-brec>

#figure(caption: [Centrality transfer to larger graphs: a gradient-boosting regressor from features to centrality is fitted on a 2,000-node graph and evaluated (Spearman) on a 8,000-node graph of the same family at the same mean degree (other seed); 6 families (Barabási-Albert, powerlaw-cluster, Watts-Strogatz, SBM, random geometric, motifs) × 10 independent pairs of graphs. Path-based centralities and PageRank (top four rows) are where walks matter; 20 of the 420 graph–target cells have a constant target ($k$-core on Barabási-Albert) and are left out. ReFeX's feature selection, scaling and bins are fitted on the training graph and frozen. `centrality_transfer_x4_10seeds.csv`, `transfer_paired.csv`.])[
  #table(columns: 10, align: (left,) + (center,) * 9,
    [], [RoleWalk], [\+ curv.], [Top-$k$ + layers], [Local stats], [ReFeX (frozen)], [RW heat CF], [RWSE], [GraphWave, no $1 \/ n$], [#emph[Degree]],
    [Betweenness], [#underline[0.864]], [*0.866*], [0.862], [0.852], [0.845], [0.849], [0.412], [0.659], [#emph[0.623]],
    [Closeness], [*0.797*], [#underline[0.793]], [0.761], [0.716], [0.725], [0.738], [0.464], [0.529], [#emph[0.407]],
    [Harmonic], [*0.861*], [#underline[0.860]], [0.823], [0.778], [0.788], [0.806], [0.517], [0.597], [#emph[0.460]],
    [PageRank], [*0.989*], [*0.989*], [#underline[0.988]], [0.984], [0.973], [0.986], [0.741], [0.611], [#emph[0.929]],
    table.hline(stroke: 0.4pt),
    [Clustering], [0.805], [0.802], [0.803], [*0.844*], [#underline[0.813]], [0.785], [0.723], [0.347], [#emph[0.201]],
    [k-core], [#underline[0.651]], [*0.652*], [*0.652*], [0.650], [0.631], [0.647], [0.494], [0.594], [#emph[0.614]],
    [Eigenvector], [#underline[0.673]], [0.671], [0.667], [0.661], [*0.700*], [0.586], [0.358], [0.415], [#emph[0.463]],
    table.hline(stroke: 0.4pt),
    [Mean], [*0.813*], [#underline[0.812]], [0.801], [0.790], [0.790], [0.777], [0.532], [0.533], [#emph[0.524]],
  )
  #text(size: 8pt)[Local stats: degree, neighbour-degree moments, triangles, egonet boundary, clustering, lazy-walk returns at steps 2 and 3. RW heat CF: random-walk heat-kernel characteristic function. Paired over the 60 pairs of graphs (RoleWalk minus other, mean over targets, 95% $t$-interval, share of pairs won): top-$k$ + layers $+0.013$ $[0.009, 0.018]$, 77%; local stats $+0.022$ $[0.016, 0.029]$, 80%; frozen ReFeX $+0.024$ $[0.018, 0.029]$, 82%; RW heat CF $+0.038$ $[0.030, 0.046]$, 98%. On closeness and harmonic, RoleWalk wins every pair against local stats and ReFeX ($+0.07$ to $+0.08$) and leads top-$k$ + layers, at the same dimension, by $+0.04$ $[0.03, 0.04]$; on betweenness it ties top-$k$ ($+0.002$ $[-0.003, 0.007]$); local stats win clustering ($-0.04$) and ReFeX eigenvector ($-0.03$). Per family, RoleWalk (with or without curvature) has the best mean on Barabási-Albert, random geometric and Watts-Strogatz, and is within 0.007 of the best on the other three. With 2 seeds, test graphs of the training size give the same top two (means: RoleWalk 0.851, top-$k$ + layers 0.835, ReFeX 0.822, RW heat CF 0.819, local stats 0.817, RWSE 0.726, GraphWave 0.633; `centrality_transfer.csv`): from the same size to $4 times$ the size, RoleWalk's mean drops by 0.04, RWSE's by 0.19 and GraphWave's (no $1 \/ n$) by 0.10. GraphWave's scale is chosen from each graph's spectrum (@prop-graphwave); RWSE's coordinates are local like RoleWalk's, so its drop is not a matter of coordinates, and we have not diagnosed it. With 2 seeds, appending the local statistics to RoleWalk gives 0.875 / 0.780 / 0.863 / 0.990 / 0.838 / 0.657 / 0.654, mean 0.808. Digraphwave does not transfer (2 seeds, mean 0.08; `centrality_transfer_x4_dgw.csv`): its coordinates are near 1 with spreads that shrink with the graph's size, so on the larger graph they fall outside the bins learned on the smaller one and the regressor predicts a constant on 25 of the 80 scored pairs (counted as 0). With an unfrozen ReFeX (only where feature counts happen to match), k-core reached 0.935 at the same size: an artefact of the subset.]
] <tab-centrality>

#figure(caption: [Role transfer between airport networks: classifier fitted on one network, tested on another (6 ordered pairs, accuracy). Labels are activity quartiles, close to degree: this transfer is won by degree-level statistics, and the centrality transfer of @tab-centrality is the relevant test of what walks transfer. Removing GraphWave's $1 \/ n$ lifts it from chance to 0.455 (@prop-graphwave); fixed scales and a heat kernel instead of walk powers give 0.425. `role_transfer.csv`, `role_transfer_gw.csv`.])[
  #table(columns: 5, align: center,
    [RoleWalk], [\+ curv.], [Top-$k$ + layers], [Local statistics], [ReFeX (frozen)],
    [#underline[0.488]], [0.472], [0.475], [*0.565*], [0.438],
    [RWSE], [GraphWave], [GraphWave, no $1 \/ n$], [RW heat CF], [#emph[Degree (baseline)]],
    [0.375], [0.249 (chance)], [0.455], [0.425], [#emph[0.542]],
  )
] <tab-airport-transfer>

#figure(caption: [Structural encodings for GNNs on ZINC (test MAE at the best validation epoch). Top: GINE, 4 layers, ≈230k parameters, 300 epochs, 3 seeds. Bottom: GPS @rampasek2022 protocol, 10 layers, ≈470k parameters, 500 epochs, 5 seeds. `pe_zinc_v13.csv`, `pe_zinc_gps*.csv`.])[
  #table(columns: 3, align: (left, center, center),
    [GINE], [Test MAE], [],
    [none], [0.202 ± 0.010], [],
    [RWSE], [0.107 ± 0.004], [],
    [RoleWalk], [0.108 ± 0.008], [],
    [RoleWalk + RWSE], [#underline[0.104 ± 0.006]], [],
    [*RoleWalk + curv. + RWSE*], [*0.095 ± 0.002*], [],
    table.hline(stroke: 0.6pt),
    [GPS], [Test MAE], [Val. MAE],
    [RWSE (published: 0.070 ± 0.004)], [#underline[0.077 ± 0.004]], [0.100 ± 0.005],
    [RoleWalk + curv.], [0.078 ± 0.006], [#underline[0.093 ± 0.003]],
    [*RoleWalk + curv. + RWSE*], [*0.072 ± 0.001*], [*0.092 ± 0.006*],
  )
  #text(size: 8pt)[RoleWalk + curv. + RWSE is below RWSE on each of the 5 seeds (paired difference 0.0045, standard deviation 0.0037 over seeds, 95% interval $[-0.0001, 0.0091]$; two-sided sign test $p = 0.06$). RoleWalk + curv. alone is unstable across seeds (0.073 to 0.089).]
] <tab-zinc>

#figure(caption: [Embedding time against size with the fused kernel, defaults (tol $10^(-4)$ without budget, 3 steps, layers), including the conversion of the networkx graph; one run after compiling the kernel, Intel Core i9-9900X (3.5 GHz). `scalability.csv`.])[
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
  #text(size: 8pt)[Peak memory 2.5 GB at a million nodes, of which 1.6 GB hold the networkx graph. Baselines on the same Barabási-Albert graphs, one core (`scalability_baselines.csv`): ReFeX 0.1 s / 1.8 s / 31 s at 10k / 100k / 1M nodes, faster than RoleWalk; 16-step RWSE 68 s at 10k and over 16 GB at 100k; RESD 62 s / 631 s at 10k / 100k and over an hour at 1M; Role2Vec 56 s at 10k and over 10 GB at 100k; the heat-kernel CF 196 s and Digraphwave 184 s at 10k, both over an hour at 100k. RoleWalk is the only embedding evaluated, other than ReFeX, to reach a million nodes. At 1,190 nodes (USA airports): struc2vec 564 s, GraphWave 18 s, Role2Vec 10 s, RESD 8 s; struc2vec and GraphWave do not run on wiki-RfA (11k nodes).]
] <tab-scale>

= Analysis

== Summaries of the walk distribution

#figure(caption: [Other summaries in place of the characteristic function, without layers or curvature (an earlier ablation, not dimension-matched: top-10 probabilities give 10 coordinates per step against 20 for the characteristic function; the dimension-matched top-20 + layers is in @tab-roles, @tab-directed and @tab-centrality). Airports: mean accuracy; Transfer: airport transfer; Motifs: synthetic accuracy.])[
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

#figure(caption: [Ways to add local geometry (earlier ablation runs; directed accuracy without the mixed probes, against 0.728 with them in @tab-directed). Here, degree marks and counting bounds hurt transfer.])[
  #table(columns: 5, align: (left,) + (center,) * 4,
    [], [Airports], [Transfer], [Motifs (varied / houses)], [Directed acc.],
    [No geometry], [0.656], [0.445], [0.744 / 0.733], [0.678],
    [\+ onion layers (moments)], [0.661], [#underline[0.488]], [0.744 / #underline[0.758]], [0.682],
    [\+ layers, one CF per layer], [0.654], [0.424], [#underline[0.751] / 0.742], [0.695],
    [\+ degree marks (joint CF)], [0.65–0.67], [0.34–0.36], [–], [–],
    [\+ curvature, counting bound], [#underline[0.667]], [0.388], [*0.754* / *0.767*], [#underline[0.721]],
    [\+ curvature, probabilistic (moments)], [*0.668*], [0.470], [0.725 / 0.742], [0.712],
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

== An observation: betweenness from one small graph <sec-betweenness>

A regressor fitted once on a single 2,000-node graph (exact betweenness,
about a second of training) predicts the betweenness ranking of new graphs
of the same family up to 25 times larger, from RoleWalk alone, without
computing any shortest path on them (@tab-betweenness).

#figure(caption: [Spearman correlation with exact betweenness, and time on one core (embedding and prediction), for a regressor fitted on one 2,000-node graph and applied to new graphs of the same family; mean of 3 seeds (1 at 50k). `betweenness.csv`.])[
  #table(columns: 4, align: (left,) + (center,) * 3,
    [Test graph], [5,000 nodes], [20,000 nodes], [50,000 nodes],
    [Barabási-Albert], [0.953 (0.7 s)], [0.937 (3.7 s)], [0.925 (11.7 s)],
    [Powerlaw-cluster], [0.926 (0.7 s)], [0.907 (3.5 s)], [0.896 (11.4 s)],
    [Watts-Strogatz], [0.960 (0.2 s)], [0.963 (1.1 s)], [0.962 (2.8 s)],
  )
] <tab-betweenness>

The ranking degrades slowly with size on hub-dominated families and not at
all on Watts-Strogatz. This is not a substitute for dedicated algorithms:
exact Brandes (NetworKit) takes 163 s at 20,000 nodes on one core, but
source sampling reaches similar correlations at similar cost, and adding
local statistics to RoleWalk brings it to the same level (0.936–0.964 at
50k). What it shows is that a few walk statistics, read the same way in
every graph, carry most of a global, path-based quantity across sizes.

== Pruning <sec-pruning>

On the six real networks, pruning at the default tol $10^(-4)$ changes
accuracy by at most 0.005 (0.002 with a budget), and by at most 0.012 at
tol $10^(-3)$ (`pruning.csv`). Without a budget, the characteristic-function block
keeps a median relative error below $10^(-4)$ at tol $10^(-4)$, but layers
move by up to 16% (73% at tol $10^(-3)$). The error concentrates on hubs:
for the centre of a star with 20,000 leaves and the largest hub of a
100k-node Barabási-Albert graph, the embedding is off by 49% and 27%. A
budget $delta$ caps the worst absolute $ell_1$ error of the unscaled layers
at each step at $delta$ (observed 0.009, 0.047, 0.096 for
$delta = 0.01, 0.05, 0.10$) and makes both hubs exact; without curvature,
the observed feature error is at most 0.51 of the certificate of
@prop-prune everywhere. On the
real networks the budget costs nothing measurable (3.3 s against 3.1 s in
total); on the hub-heavy Barabási-Albert graph it doubles the time, as
hub rows are kept whole.

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
consistent with walk co-occurrence, i.e. position, a hypothesis we have
not tested (a node-ID permutation check would only rule out identifier
artefacts); we read Film as a node-type task rather than a role
benchmark.

= Limitations

- Walk statistics cannot separate distance-regular graphs with equal arrays (@prop-drg), most CFI pairs (3 of 100 with 5 steps), or graphs whose balls coincide (locality); without curvature, some nodes separated by 1-WL are confused (@prop-wl).
- Without a mass budget, the pruning certificate is only a posteriori and large on hubs (@sec-pruning); the budget costs time on hub-heavy graphs, and has no support cap.
- ReFeX is faster (5× at a million nodes) and wins on noisy synthetic directed motifs in supervised accuracy (with logistic regression; comparable with gradient boosting); a fixed-scale heat-kernel CF matches RoleWalk on two of three airport networks.
- Middle-tier roles on real networks often form a continuum; per-node stability should be reported next to roles.
- Degree-like labels (airports) favour trivial baselines.
- Two real labelled directed settings (wiki-RfA, food webs); on food webs RoleWalk ties local statistics and is below Digraphwave, and raw flow weights hurt every walk-based method.
- GPS on ZINC: 5 seeds; our RWSE reproduction (0.077) is weaker than the published 0.070; RoleWalk + curvature + RWSE (0.072) is better on every seed, but the paired 95% interval just includes 0 (sign test $p = 0.06$).

#bibliography("refs.bib", style: "ieee")
