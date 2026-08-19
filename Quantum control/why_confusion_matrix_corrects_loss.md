# Why a confusion matrix corrects Lindblad loss

A complete derivation, with every step written out. Companion to
[`jc_counting_confusion_matrix.ipynb`](jc_counting_confusion_matrix.ipynb); all numerical
values quoted are produced by that notebook at its current parameters
($N=14$, $T_1^{q} = 20\,\mathrm{ms}$, $T_2^{q} = 80\,\mu s$, $T_1^{c} = 300\,\mu s$,
$T_\phi^{c} = 800\,\mu s$, $t_\pi = 150\,\mathrm{ns}$, $t_m = 1.1\,\mu s$, $g/2\pi = 1$ MHz).

---

## 0. The result, and what it does *not* say

**Theorem.** Let the apparatus be the descending Jaynes–Cummings sweep with any Markovian
noise, run open-loop. Let $p_n = \langle n |\rho_{\rm cav}| n\rangle$ be the photon-number
populations of the input state and let $q_c = P(C = c)$ be the distribution of the click
count. Then there is a fixed matrix $M$, independent of $\rho_{\rm cav}$, with

$$ q \;=\; M p, \qquad M_{cn} \;=\; P\bigl(C = c \,\big|\, \text{cavity prepared in } |n\rangle\bigr) $$

and $M$ is invertible, so $p = M^{-1} q$ recovers the true photon-number distribution
**exactly**, despite the loss.

Three statements must be kept apart, because only the first is true:

| statement | status |
|---|---|
| the photon-number **distribution** $p_n$ is recovered exactly | ✅ true, to machine precision |
| the **quantum state** $\rho_{\rm cav}$ is recovered | ❌ false — coherences are never encoded in the data |
| the photon number **in a single shot** is recovered | ❌ false — one run yields one integer $C$, and $C \neq n$ |

"Corrects loss" therefore means: *the systematic distortion of the estimated distribution is
removed.* The photons are still physically gone; the cavity really does end in vacuum.
Nothing is un-lost. §7 makes this precise.

The logical skeleton is short. Everything else is verification of its two premises:

1. **Linearity** (§3) — the click probabilities are a linear functional of $\rho_{\rm cav}$.
   This gives a response matrix, but one indexed by $(n,m)$: $N^2 = 196$ unknowns feeding
   $14$ observable bins. Hopelessly underdetermined.
2. **Phase covariance** (§4) — the off-diagonal coefficients vanish identically, collapsing
   $196$ unknowns to $14$. *This* is what makes the inverse problem square and solvable.
3. Squareness + a dominant diagonal ⟹ invertible (§6), and inversion is exact (§5, §7).

---

## 1. Setup and notation

Hilbert space $\mathcal{H} = \mathcal{H}_q \otimes \mathcal{H}_c$ with
$\dim = D = 2N$. Tensor products are ordered (qubit, cavity).

$$ \sigma_+ = |e\rangle\langle g|, \qquad \sigma_- = |g\rangle\langle e|, \qquad a|n\rangle = \sqrt n\,|n-1\rangle $$

Resonant Jaynes–Cummings Hamiltonian in the rotating frame,

$$ H \;=\; g\bigl(\sigma_+ a + \sigma_- a^\dagger\bigr) $$

Lindblad generator,

$$ \mathcal{L}_\bullet\rho \;=\; -i[H_\bullet, \rho] \;+\; \sum_k \mathcal{D}[L_k]\rho,
\qquad \mathcal{D}[L]\rho \;\equiv\; L\rho L^\dagger - \tfrac12\bigl\{L^\dagger L, \rho\bigr\} $$

with $H_{\rm drive} = H$ when the pulse is on and $H_{\rm idle} = 0$ when it is off, and four
collapse operators

$$ L_1 = \sqrt{\gamma_1}\,\sigma_-, \quad
   L_2 = \sqrt{2\gamma_\phi}\,|e\rangle\langle e|, \quad
   L_3 = \sqrt{\kappa}\,a, \quad
   L_4 = \sqrt{2\kappa_\phi}\,a^\dagger a $$

$$ \gamma_1 = \frac{1}{T_1^{q}}, \qquad
   \gamma_\phi = \frac{1}{T_2^{q}} - \frac{1}{2T_1^{q}}, \qquad
   \kappa = \frac{1}{T_1^{c}}, \qquad
   \kappa_\phi = \frac{1}{T_\phi^{c}} $$

**One round $r$** consists of four operations, in this order:

| # | operation | map | type |
|---|---|---|---|
| (a) | JC pulse, duration $t^{(r)}$ | $\Phi^{(r)}_{\rm pulse} = e^{\mathcal{L}_{\rm drive} t^{(r)}}$ | CPTP |
| (b) | readout window, duration $t_m$ | $\Phi_{\rm idle}(t_m) = e^{\mathcal{L}_{\rm idle} t_m}$ | CPTP |
| (c) | projective qubit readout | $\rho \mapsto P_g\rho P_g$ or $\rho \mapsto P_e\rho P_e$ | CP, **not** trace-preserving |
| (d) | if outcome $e$: idle $t_\pi$, then flip | $\rho \mapsto X_\pi\,\Phi_{\rm idle}(t_\pi)(\rho)\,X_\pi$ | CPTP |

with $P_g = |g\rangle\langle g|\otimes \mathbb{1}_c$, $P_e = |e\rangle\langle e|\otimes \mathbb{1}_c$,
$X_\pi = \sigma_x \otimes \mathbb{1}_c$. A run is $R$ such rounds.

---

## 2. The exact click distribution

### 2.1 Records and branch operators

A **measurement record** is a string $s = (s_1,\dots,s_R)$, $s_r \in \{g,e\}$, with click count

$$ c(s) \;=\; \#\{\,r : s_r = e\,\} $$

For a fixed record, compose the maps of §1 using those specific outcomes; call the result
$\mathcal{K}_s$. It is a composition of CP maps, hence CP, and it is **not** trace-preserving
(the projections are not). The Born rule reads

$$ P(\text{record} = s) \;=\; \operatorname{Tr}\mathcal{K}_s(\rho_0),
\qquad \rho_0 = |g\rangle\langle g| \otimes \rho_{\rm cav} $$

Group records by their click count and define the **branch superoperator**

$$ \boxed{\;\mathcal{E}_c \;\equiv\; \sum_{s\,:\,c(s) = c} \mathcal{K}_s
\qquad\Longrightarrow\qquad
q_c \;=\; \operatorname{Tr}\mathcal{E}_c(\rho_0)\;} \tag{2.1}$$

This is exactly what `branch[c]` accumulates in the notebook. It is not an approximation of
anything: all decoherence sits inside $\mathcal{L}$, and (2.1) is the exact Born-rule
probability.

### 2.2 Why the branches must stay unnormalised

This is the step that makes everything else possible, and it is easy to get wrong.

To follow a *single* quantum trajectory you must renormalise after each measurement,

$$ \rho \;\longmapsto\; \frac{P_e\,\rho\,P_e}{\operatorname{Tr}\bigl(P_e\,\rho\,P_e\bigr)} $$

and that map is **nonlinear** in $\rho$ — the denominator depends on $\rho$. A nonlinear
response admits no response matrix at all, and the entire construction collapses.

By carrying *unnormalised* branch operators, the denominators never appear. The trace of a
branch **is** its probability, and every map in sight stays linear. The price is bookkeeping
$C_{\max}+1$ operators instead of one; the reward is Lemma 1.

### 2.3 Normalisation

Summing (c) over both outcomes gives $P_g\rho P_g + P_e\rho P_e$, which is trace-preserving;
composing with the CPTP maps (a), (b), (d) keeps that. Hence $\sum_c \mathcal{E}_c$ is CPTP and

$$ \sum_c q_c \;=\; \operatorname{Tr}\Bigl(\textstyle\sum_c \mathcal{E}_c\Bigr)(\rho_0) \;=\; \operatorname{Tr}\rho_0 \;=\; 1 \tag{2.2}$$

*Verified: $\max_n\bigl|\sum_c M_{cn} - 1\bigr| = 6.7\times10^{-16}$.*

---

## 3. Lemma 1 — linearity (the load-bearing property)

**Lemma 1.** $q_c$ is a linear functional of $\rho_{\rm cav}$.

*Proof.* Each of (a), (b), (c), (d) is a linear map on operators. $\mathcal{K}_s$ is a
composition of linear maps, hence linear. $\mathcal{E}_c$ is a finite sum of linear maps,
hence linear. $\operatorname{Tr}$ is linear, and $\rho_{\rm cav}\mapsto |g\rangle\langle g|\otimes\rho_{\rm cav}$
is linear. Compose. $\;\blacksquare$

Therefore there exist fixed coefficients, independent of the input, with

$$ q_c \;=\; \sum_{n,m} A_c[n,m]\,(\rho_{\rm cav})_{mn},
\qquad A_c[n,m] \;=\; \operatorname{Tr}\,\mathcal{E}_c\bigl(|g,n\rangle\langle g,m|\bigr) \tag{3.1}$$

**This is where loss is already fully accommodated, and it is worth pausing on.** A Lindblad
generator produces a CPTP map, and CPTP maps are *linear by definition*. Loss destroys
information and energy; it does **not** destroy linearity. Nothing above required the dynamics
to be unitary, reversible, or lossless. Every claim in this document rests on linearity, and
none on reversibility.

*Verified: with $\rho = \alpha\rho_1 + (1-\alpha)\rho_2$, $\alpha = 0.37$,*
$\bigl|q(\rho) - \alpha q(\rho_1) - (1-\alpha)q(\rho_2)\bigr|_{\max} = 1.7\times10^{-16}$.

At this point we have a response *tensor* $A_c[n,m]$ with $N^2 = 196$ components feeding
$C_{\max}+1 = 14$ measured numbers. Inverting that is impossible. §4 removes the off-diagonal
components entirely.

---

## 4. Lemma 2 — phase covariance

Define the **total excitation number** and its rotation

$$ \hat N \;=\; a^\dagger a + |e\rangle\langle e|, \qquad R_\theta \;=\; e^{i\theta \hat N} $$

**Lemma 2.** Every branch superoperator is covariant:
$\;\mathcal{E}_c\bigl(R_\theta\,\rho\,R_\theta^\dagger\bigr) = R_\theta\,\mathcal{E}_c(\rho)\,R_\theta^\dagger$.

We check each ingredient in turn.

### 4.1 The Hamiltonian conserves $\hat N$

$\sigma_+ a$ destroys a photon and creates a qubit excitation; $\sigma_-a^\dagger$ does the
reverse. Explicitly, on a basis state,

$$ \sigma_+ a\,|g,n\rangle = \sqrt n\,|e,n-1\rangle, \qquad
   \hat N|g,n\rangle = n|g,n\rangle, \quad \hat N|e,n-1\rangle = n|e,n-1\rangle $$

so both sides of the coupling live in the same $\hat N$ eigenspace, giving

$$ [H, \hat N] = 0 \qquad\Longrightarrow\qquad R_\theta^\dagger H R_\theta = H $$

and hence $R_\theta^\dagger\bigl(-i[H, R_\theta\rho R_\theta^\dagger]\bigr)R_\theta = -i[H,\rho]$.

### 4.2 Each collapse operator is an eigenoperator

**Claim.** If $[\hat N, L] = -\lambda L$ then $R_\theta^\dagger L R_\theta = e^{i\lambda\theta} L$.

*Proof.* Let $f(\theta) = e^{-i\theta\hat N} L\, e^{i\theta\hat N}$. Differentiate:

$$ f'(\theta) = e^{-i\theta \hat N}\bigl(-i\hat N L + i L \hat N\bigr)e^{i\theta \hat N}
             = e^{-i\theta \hat N}\bigl(-i[\hat N, L]\bigr)e^{i\theta \hat N}
             = i\lambda\, f(\theta) $$

With $f(0) = L$ this integrates to $f(\theta) = e^{i\lambda\theta}L$. $\;\blacksquare$

Now compute the commutators, using $\sigma_- = |g\rangle\langle e|$:

| $L$ | $[a^\dagger a, L]$ | $\bigl[\vert e\rangle\langle e\vert , L\bigr]$ | $[\hat N, L]$ | $\lambda$ |
|---|---|---|---|---|
| $a$ | $-a$ | $0$ | $-a$ | $1$ |
| $\sigma_-$ | $0$ | $-\sigma_-$ | $-\sigma_-$ | $1$ |
| $a^\dagger a$ | $0$ | $0$ | $0$ | $0$ |
| $\vert e\rangle\langle e\vert$ | $0$ | $0$ | $0$ | $0$ |

(For $\sigma_-$: $|e\rangle\langle e|\cdot|g\rangle\langle e| = 0$ and
$|g\rangle\langle e|\cdot|e\rangle\langle e| = \sigma_-$, so the commutator is $-\sigma_-$.)

### 4.3 The dissipators are covariant

Take any eigenoperator $L$ with parameter $\lambda$, and evaluate term by term:

$$ R_\theta^\dagger\Bigl(L\,R_\theta\rho R_\theta^\dagger\,L^\dagger\Bigr)R_\theta
 = \bigl(R_\theta^\dagger L R_\theta\bigr)\,\rho\,\bigl(R_\theta^\dagger L^\dagger R_\theta\bigr)
 = \bigl(e^{i\lambda\theta}L\bigr)\rho\bigl(e^{-i\lambda\theta}L^\dagger\bigr)
 = L\rho L^\dagger $$

**The phase cancels between $L$ and $L^\dagger$.** For the anticommutator term,

$$ R_\theta^\dagger L^\dagger L R_\theta
 = \bigl(R_\theta^\dagger L^\dagger R_\theta\bigr)\bigl(R_\theta^\dagger L R_\theta\bigr)
 = e^{-i\lambda\theta}L^\dagger e^{i\lambda\theta} L = L^\dagger L $$

so $R_\theta^\dagger\,\mathcal{D}[L]\bigl(R_\theta\rho R_\theta^\dagger\bigr)\,R_\theta = \mathcal{D}[L]\rho$.
Combining with §4.1, both $\mathcal{L}_{\rm drive}$ and $\mathcal{L}_{\rm idle}$ are covariant,
and therefore so are their exponentials $e^{\mathcal{L}t}$.

### 4.4 The projectors are covariant

$P_e = |e\rangle\langle e|\otimes\mathbb{1}_c$ commutes with $a^\dagger a$ (different factor)
and with $|e\rangle\langle e|$ (same operator), so $[P_e, \hat N] = 0$; likewise $P_g$. Hence
$R_\theta^\dagger P_x R_\theta = P_x$ and $\rho\mapsto P_x\rho P_x$ is covariant.

### 4.5 The $\pi$ pulse — the one exception, handled

$X_\pi = \sigma_x\otimes\mathbb{1}_c$ does **not** commute with $\hat N$: it moves population
between $|g\rangle$ and $|e\rangle$, changing $\hat N$ by $\pm 1$. Covariance is rescued by
*where in the circuit it acts*.

**(i) After the readout there is no qubit coherence.** $P_e\rho P_e$ is supported entirely in
the $|e\rangle$ block: it has the form $|e\rangle\langle e|\otimes B$.

**(ii) Idling preserves that.** With $H = 0$, consider the qubit coherence block
$|g\rangle\langle e|\otimes X$. Under $\mathcal{L}_{\rm idle}$: $\sigma_-\,|g\rangle\langle e|\,\sigma_+ = 0$
because $\sigma_-|g\rangle = 0$; the operator $|e\rangle\langle e|$ sandwiched likewise
annihilates it; the anticommutator terms map coherence blocks to coherence blocks; and the
cavity operators $a$, $a^\dagger a$ act on the other tensor factor. So population blocks and
coherence blocks **decouple**, and a state starting with zero qubit coherence keeps zero qubit
coherence for the whole $t_\pi$ window.

Therefore $X_\pi$ only ever acts on states of the form

$$ \rho \;=\; |g\rangle\langle g|\otimes A \;+\; |e\rangle\langle e|\otimes B \tag{4.1}$$

**(iii) On such states $X_\pi$ commutes with $R_\theta$.** Write $U_\theta = e^{i\theta a^\dagger a}$.
Acting on (4.1),

$$ R_\theta\,\rho\,R_\theta^\dagger
 = |g\rangle\langle g|\otimes U_\theta A U_\theta^\dagger
 \;+\; |e\rangle\langle e|\otimes \underbrace{e^{i\theta}}_{\text{from } |e\rangle\langle e|} U_\theta B U_\theta^\dagger \underbrace{e^{-i\theta}}_{\text{from the dagger}} $$

$$ \;=\; |g\rangle\langle g|\otimes U_\theta A U_\theta^\dagger \;+\; |e\rangle\langle e|\otimes U_\theta B U_\theta^\dagger \tag{4.2}$$

**The scalar $e^{i\theta}$ cancels between the two sides of the conjugation**, so on
block-diagonal states $R_\theta$ acts blockwise as $U_\theta$ with *no relative phase between
the blocks*. Since $\sigma_x$ merely swaps the two blocks,

$$ X_\pi\bigl(R_\theta\rho R_\theta^\dagger\bigr)X_\pi
 = |e\rangle\langle e|\otimes U_\theta A U_\theta^\dagger + |g\rangle\langle g|\otimes U_\theta B U_\theta^\dagger
 = R_\theta\bigl(X_\pi\rho X_\pi\bigr)R_\theta^\dagger $$

which is the required covariance. Note the essential role of the *projective measurement*: had
a coherence block $|g\rangle\langle e|\otimes X$ been present, (4.2) would have carried an
uncancelled $e^{i\theta}$ and the argument would fail.

*Verified:* $\bigl\|[X_\pi, R_\theta]\,\text{acting on }\rho\bigr\|$ is $4\times10^{-17}$ for a
block-diagonal $\rho$, and $5\times10^{-2}$ for the same $\rho$ plus qubit coherence.

This completes Lemma 2. $\;\blacksquare$

---

## 5. Theorem — $q = Mp$

**Step 1: the response is phase-blind.** Trace is invariant under unitary conjugation, so
Lemma 2 gives, for every $\theta$,

$$ q_c\bigl(R_\theta\rho_0R_\theta^\dagger\bigr)
 = \operatorname{Tr}\bigl[R_\theta\,\mathcal{E}_c(\rho_0)\,R_\theta^\dagger\bigr]
 = \operatorname{Tr}\mathcal{E}_c(\rho_0) = q_c(\rho_0) \tag{5.1}$$

The input is $\rho_0 = |g\rangle\langle g|\otimes\rho_{\rm cav}$, and $|g\rangle$ carries zero
excitation, so $R_\theta$ acts on it as the cavity rotation alone:

$$ R_\theta\,\rho_0\,R_\theta^\dagger \;=\; |g\rangle\langle g|\otimes U_\theta\,\rho_{\rm cav}\,U_\theta^\dagger,
\qquad U_\theta = e^{i\theta a^\dagger a} $$

so (5.1) says the functional is invariant under **cavity phase rotation**:

$$ q_c(\rho_{\rm cav}) \;=\; q_c\bigl(U_\theta\,\rho_{\rm cav}\,U_\theta^\dagger\bigr)\qquad\forall\,\theta \tag{5.2}$$

**Step 2: average over $\theta$.** The right-hand side of (5.2) is independent of $\theta$, so
it equals its own average. Using **linearity** (Lemma 1) to move the average inside the
functional,

$$ q_c(\rho_{\rm cav}) \;=\; q_c\!\left(\frac{1}{2\pi}\int_0^{2\pi}\!\!d\theta\;
U_\theta\,\rho_{\rm cav}\,U_\theta^\dagger\right) \tag{5.3}$$

**Step 3: the twirl kills the off-diagonals.** Matrix element by matrix element,

$$ \bigl(U_\theta\,\rho\,U_\theta^\dagger\bigr)_{nm}
 = \langle n|e^{i\theta a^\dagger a}\,\rho\,e^{-i\theta a^\dagger a}|m\rangle
 = e^{i\theta n}\,\rho_{nm}\,e^{-i\theta m}
 = e^{i\theta(n-m)}\rho_{nm} $$

$$ \frac{1}{2\pi}\int_0^{2\pi}\!\!d\theta\; e^{i\theta(n-m)} = \delta_{nm}
\qquad\Longrightarrow\qquad
\frac{1}{2\pi}\int_0^{2\pi}\!\!d\theta\; U_\theta\,\rho\,U_\theta^\dagger
 = \sum_n \rho_{nn}\,|n\rangle\langle n| \tag{5.4}$$

**Step 4: assemble.** Substituting (5.4) into (5.3) and expanding by linearity,

$$ q_c \;=\; q_c\Bigl(\sum_n p_n |n\rangle\langle n|\Bigr) \;=\; \sum_n p_n\, q_c\bigl(|n\rangle\langle n|\bigr) $$

$$ \boxed{\;q_c \;=\; \sum_n M_{cn}\,p_n, \qquad
M_{cn} \;=\; q_c\bigl(|n\rangle\langle n|\bigr) \;=\; \operatorname{Tr}\,\mathcal{E}_c\bigl(|g,n\rangle\langle g,n|\bigr)\;} \tag{5.5}$$

Equivalently: $A_c[n,m] = 0$ for $n \neq m$ in (3.1), and $M_{cn} = A_c[n,n]$. $\;\blacksquare$

Two readings of (5.5), both worth holding:

- **Operationally:** column $n$ of $M$ is, by definition, $P(C = c \mid \text{exactly } n \text{ photons in})$
  — the complete statistical description of what the lossy apparatus does to $n$ photons.
  Every loss process is *already inside those numbers*. We never model loss and subtract it;
  we characterise the apparatus on a basis of inputs and let linearity do the rest.
- **Physically:** the apparatus has no phase reference. No displacement, no phase-sensitive
  drive, no homodyne — only an excitation-conserving coupling and number-diagonal
  measurements. It cannot distinguish $|1\rangle + |4\rangle$ from $|1\rangle - |4\rangle$, so
  populations are the only thing it can report on.

*Verified: $\max\bigl|q(\rho) - q(\mathrm{diag}\,\rho)\bigr| = 0$ (bitwise: the coherence
sectors decouple, so off-diagonal entries never feed any trace);
$\max\bigl|q(\rho) - q(U_\theta\rho U_\theta^\dagger)\bigr| \le 5.6\times10^{-17}$; and the
forward law $\bigl|Mp_{\rm true} - q_{\rm sim}\bigr| \le 1.7\times10^{-16}$ for all four
schedules.*

---

## 6. The structure of $M$

### 6.1 $M$ is column-stochastic

Immediate from (2.2): each column is a probability distribution. So $M$ maps probability
vectors to probability vectors, as it must.

### 6.2 $M$ is upper triangular (when $t_\pi = 0$)

**Claim.** With collapse operators $\{\sigma_-, |e\rangle\langle e|, a, a^\dagger a\}$ and
$t_\pi = 0$, we have $C \le n$ always, i.e. $M_{cn} = 0$ for $c > n$.

*Proof.* Track $\hat N$ and ask which operations can **increase** it.

| operation | effect on $\hat N$ |
|---|---|
| $H$ | conserves ($[H,\hat N] = 0$) |
| $\sigma_-$ , $a$ | decrease by 1 |
| $\vert e\rangle\langle e\vert$ , $a^\dagger a$ | conserve |
| $P_g$, $P_e$ | conserve |
| $X_\pi$ | **can increase by 1** |

$X_\pi$ is the only candidate. With $t_\pi = 0$ it acts immediately after the projection onto
$|e\rangle$, so the state is exactly in $|e\rangle$ and the flip sends $|e\rangle\to|g\rangle$,
*decreasing* $\hat N$ by one. So no operation ever increases $\hat N$: it is non-increasing
along the whole run.

Every click decreases $\hat N$ by exactly one (readout finds $|e\rangle$, the reset returns the
qubit to $|g\rangle$). Starting from $\hat N = n$ and never increasing, there can be at most
$n$ clicks. $\;\blacksquare$

There is also no way to click without a photon: the qubit begins each round in $|g\rangle$, and
the only excitation source is $H$, which must take the excitation from the cavity. Loss can
*hide* photons; it cannot invent them.

### 6.3 The one overcounting channel, and its size

With $t_\pi > 0$ the proof above breaks, in exactly one place. During the $t_\pi$ window the
qubit can relax $|e\rangle\to|g\rangle$ (via $\sigma_-$, so $\hat N$ drops by one) — and then
$X_\pi$, which fires **unconditionally** because the readout said $|e\rangle$, flips it *back*
to $|e\rangle$, restoring $\hat N$. The click was already counted, so the excitation budget has
been silently replenished and the run can click more than $n$ times.

Per click the probability is $\sim t_\pi / T_1^{q} = 0.15/20000 = 7.5\times10^{-6}$; over up to
13 clicks this predicts $\sim 10^{-4}$.

*Verified: $\max_{c>n} M_{cn} = 5.2\times10^{-5}$, and it is **identically zero** when
$t_\pi = 0$ — with all coherence times unchanged. That switch is the cleanest possible
confirmation of the mechanism.*

### 6.4 Invertibility

For an exactly triangular $M$ with positive diagonal,

$$ \det M \;=\; \prod_n M_{nn} \;>\; 0 $$

The diagonal is strictly positive because a fully successful run always has nonzero
probability. In practice $M$ is triangular only up to §6.3, so the identity holds only
approximately — but $M$ remains a small perturbation of an invertible matrix.

*Verified: $\det M = 0.143118$ against $\prod_n M_{nn} = 0.143130$ — agreeing to four digits,
the discrepancy being precisely the $5.2\times10^{-5}$ leak of §6.3. $\operatorname{cond} M = 1.761$.*

Diagonal of $M$ (descending schedule, real device):

| $n$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| $M_{nn}$ | 1.0000 | 0.9870 | 0.9726 | 0.9568 | 0.9395 | 0.9207 | 0.9004 |

| $n$ | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|
| $M_{nn}$ | 0.8785 | 0.8549 | 0.8295 | 0.8021 | 0.7724 | 0.7399 | 0.7006 |

---

## 7. What inversion is, and what it is not

The obvious objection: *photon loss is irreversible, so how can inverting a matrix undo it?*

The answer is that **$M$ is not the physical channel.** Two different objects, easily conflated:

| object | acts on | invertible? |
|---|---|---|
| the physical channel $\Lambda$ generated by $\mathcal{L}$ | density matrices | **no** — a genuinely irreversible CPTP map |
| the response matrix $M$ | probability vectors in $\mathbb{R}^N$ | **yes** — $\det M \approx 0.143$ |

$M^{-1}$ exists as a matrix but **is not a physical operation**: it is not stochastic, having
negative entries (*verified: $\min_{nc}(M^{-1})_{nc} = -0.496$*), so it corresponds to no
quantum channel and no device could apply it. It is an *estimator*, applied to numbers on a
computer after the experiment is over. §8 makes this concrete: the inverse of loss is formally
"anti-loss", a thinning with survival probability greater than one.

So the honest statement of what is achieved:

$$ \hat p = M^{-1}q \quad\text{recovers}\quad p_n = \langle n|\rho_{\rm cav}|n\rangle
\qquad\text{— the distribution, in the ensemble limit.} $$

Not the state (coherences were never encoded), not the per-shot photon number (a single run
gives one integer), and not the photons (they are gone).

**Why loss in particular is correctable.** Loss acts *within the same index set as the signal*:
a lost photon moves probability from bin $n$ down to bins $c < n$. That redistribution is

1. **fixed** — the schedule is open-loop, so $M$ does not depend on which state was fed in;
2. **linear** — §3;
3. **dominated by its diagonal** — §6.

A fixed, input-independent, invertible linear redistribution of probability can be undone by
matrix inversion. That is the whole theorem.

**What would genuinely not be correctable:**

- **Input-dependent noise** → the response is nonlinear, no single $M$ exists.
- **A singular $M$** → information truly destroyed. Not hypothetical: with `FIXED_LEVEL = 1`,
  $\sin^2(\tfrac\pi2\sqrt 4) = 0$, so $|4\rangle$ is invisible and the $n=0$ and $n=4$ columns
  become **identical**. $M$ is exactly singular and no amount of data recovers $p_4$.
- **Coherences** → never encoded, hence never recoverable, at any shot count.

---

## 8. An exactly solvable case: binomial thinning

To see every claim in closed form, replace the interleaved dynamics with the caricature *each
of the $n$ photons independently survives to be counted with probability $\eta$, and the
counting itself is perfect*. Then $C \sim \mathrm{Binomial}(n, \eta)$:

$$ M_{cn}(\eta) \;=\; \binom{n}{c}\,\eta^{c}\,(1-\eta)^{n-c} \tag{8.1}$$

This is upper triangular with $M_{nn} = \eta^n > 0$, so

$$ \det M(\eta) \;=\; \prod_{n=0}^{N-1}\eta^{\,n} \;=\; \eta^{\,N(N-1)/2} $$

*Verified for $N = 14$, $\eta = 0.9$: $\det M = 6.856\times10^{-5} = \eta^{91}$.*

**Semigroup structure.** Thinning by $\eta_2$ and then by $\eta_1$ means a photon must survive
both, so

$$ M(\eta_1)\,M(\eta_2) \;=\; M(\eta_1\eta_2), \qquad M(1) = \mathbb{1} \tag{8.2}$$

*Verified: $\bigl|M(0.8)M(\eta) - M(0.8\eta)\bigr|_{\max} \le 1.7\times10^{-16}$.*

**The inverse, in closed form.** Setting $\eta_2 = 1/\eta$ in (8.2) gives
$M(\eta)^{-1} = M(1/\eta)$, i.e. the *same algebraic formula* with $\eta \to 1/\eta$:

$$ \bigl(M^{-1}\bigr)_{nc} \;=\; \binom{c}{n}\,\eta^{-n}\bigl(1 - \eta^{-1}\bigr)^{c-n}
\;=\; \binom{c}{n}\,\eta^{-n}(-1)^{c-n}\left(\frac{1-\eta}{\eta}\right)^{c-n} \tag{8.3}$$

*Verified: $\bigl|M(\eta)^{-1} - M(1/\eta)\bigr|_{\max} \le 8.9\times10^{-15}$.*

This is the sharpest statement of §7. **Correcting loss means applying a thinning with survival
probability $1/\eta > 1$** — manifestly not a physical process, which is exactly why (8.3)
alternates in sign. It is a formula, not a machine.

**And it shows the cost.** The entries of (8.3) grow like $\eta^{-n}$, so the conditioning
degrades exponentially in both $N$ and $-\log\eta$:

| $\eta$ | $\det M$ | $\operatorname{cond} M$ | largest $\bigl\vert M^{-1}\bigr\vert$ entry |
|---|---|---|---|
| 0.9 | $6.9\times10^{-5}$ | $10.1$ | $5.1$ |
| 0.5 | $4.0\times10^{-28}$ | $9.3\times10^{5}$ | $3.7\times10^{5}$ |

Exactness (a statement about means) and usability (a statement about variances) are different
things. Inversion is always exact and eventually useless.

---

## 9. The lossless limit in closed form, and why the schedule matters

Set all rates to zero. Inside the doublet $\{|g,n\rangle, |e,n-1\rangle\}$ we have
$H_n = g\sqrt n\,\sigma_x^{(n)}$ and $H_n^2 = g^2 n\,\mathbb{1}$, so splitting
$e^{-iH_nt}$ into even and odd powers,

$$ U_n = \cos\theta_n\,\mathbb{1} - i\sin\theta_n\,\sigma_x^{(n)},
\qquad \theta_n \equiv gt\sqrt n $$

$$ \Longrightarrow\qquad U|g,n\rangle = \cos\theta_n\,|g,n\rangle - i\sin\theta_n\,|e,n-1\rangle \tag{9.1}$$

Reading off the two measurement branches of (9.1) gives the cavity **Kraus operators** of one
round,

$$ K_g|n\rangle = \cos\theta_n |n\rangle, \qquad K_e|n\rangle = -i\sin\theta_n|n-1\rangle $$

$$ K_g = \cos\bigl(gt\sqrt N\bigr), \qquad
   K_e = -i\,a\,\frac{\sin(gt\sqrt N)}{\sqrt N} = -i\,\frac{\sin\bigl(gt\sqrt{N+1}\bigr)}{\sqrt{N+1}}\,a $$

using $f(N)a = a f(N+1)$; the operator $\sin(gt\sqrt N)/\sqrt N = gt\,\mathrm{sinc}(gt\sqrt N)$
is regular at $N = 0$. Completeness $K_g^\dagger K_g + K_e^\dagger K_e = \mathbb{1}$ is just
$\cos^2 + \sin^2 = 1$ level by level. The $\pi$ pulse $-i\sigma_x$ resets the qubit and
multiplies the $e$-branch by a global phase that drops out of $\rho$, so $K_e$ is defined only
up to a phase.

**The structural consequence.** $K_g$ is *diagonal* and $K_e$ is *strictly subdiagonal*, so the
lossless protocol is exactly a classical Markov chain on photon number that can only step down:

$$ P(n \to n-1) = \sin^2\bigl(gt\sqrt n\bigr)\ \text{(a click)}, \qquad
   P(n \to n) = \cos^2\bigl(gt\sqrt n\bigr) \tag{9.2}$$

Since the chain only descends, the click count is the net displacement, $c = n_{\rm in} - n_{\rm out}$, and

$$ M_{cn}\bigl|_{\rm lossless} \;=\; \bigl[T^{(R)}\cdots T^{(1)}\bigr]_{\,n-c,\;n} $$

a product of $R$ bidiagonal column-stochastic matrices. This recovers §6.2 immediately —
upper triangularity is just "the chain never steps up".

*Verified: the Kraus channel reproduces the full Liouvillian simulation on the ideal machine to
$\le 5.6\times10^{-16}$ for every schedule; completeness holds to $1.1\times10^{-16}$.*

**Why `descending` is exact.** The $\sqrt n$ in (9.2) means no single pulse length steps every
level down with certainty. But $t = t_f \equiv \pi/2g\sqrt f$ gives $\sin^2\theta_f = 1$ at
level $f$ exactly. Sweeping $f = R, R-1, \dots, 1$:

> **Induction.** *Claim: after the round tuned to level $f$, no population remains at level $f$
> or above.* Initially (before $f = R$) this holds for any state with support $\le R$. Suppose
> it holds before the round tuned to $f$, so all population sits at $\le f$. That round moves
> level $f$ down with probability $\sin^2\theta_f = 1$ — with certainty. Levels below $f$ may
> also step down, which is harmless. So afterwards nothing remains at $f$ or above. $\;\square$

After the round tuned to $f = 1$, nothing remains above level $0$: the cavity is empty **with
certainty**, hence $c = n$ with certainty and $M = \mathbb{1}$. Note the induction tolerates
photons falling "early" — they simply sit below the descending front.

**Why one fixed pulse length is not.** With every round tuned to a single level $L$, emptying
$|n\rangle$ requires $n$ successes from a Bernoulli chain with per-round probabilities (9.2).
For $n = R$ you need $R$ successes in $R$ rounds, so *every* round must click:

$$ M_{RR}\bigl|_{\rm lossless} \;=\; \prod_{k=1}^{R}\sin^2\bigl(g t_L\sqrt k\bigr) $$

*Verified for $L = 9$, $R = 13$: the product is $0.03249$, matching the simulated
$M_{13,13} = 0.03249$ to five decimals.* The expected number of rounds needed is
$\sum_k 1/p_k = 18.66$ against $13$ available — which is why tripling the rounds repairs it,
and why the bottleneck is the *bottom* of the ladder ($p_1 = 0.25$ for $L=9$), not the top.

---

## 10. What the argument depends on

### 10.1 It does **not** depend on the noise model

Adding heating — cavity $a^\dagger$ at $\bar n = 0.05$ and qubit $\sigma_+$ — changes nothing
structural, because $a^\dagger$ and $\sigma_+$ are *also* eigenoperators of $R_\theta$
($\lambda = -1$), so §4.2–4.3 go through verbatim. Only §6.2 is lost:

| with heating added | value | meaning |
|---|---|---|
| $M_{1,0}$ | $1.0\times10^{-2}$ | the vacuum now clicks (exactly $0$ without heating) |
| $\max_{c>n}M_{cn}$ | $1.3\times10^{-2}$ | **triangularity lost** |
| $\bigl\vert Mp - q\bigr\vert$ | $1.4\times10^{-16}$ | forward law still exact |
| $\mathrm{TVD}(M^{-1}q, p)$ | $3.2\times10^{-16}$ | unfolding still exact |

Triangularity is a *bonus* from the specific noise model, not a requirement. Only linearity
and phase covariance matter.

### 10.2 It does **not** depend on the protocol being good

A schedule that never cools, a badly conditioned $M$, huge loss: $q = Mp$ still holds exactly.
This is why unfolding is exact for all four schedules in the notebook, including ones that
leave the cavity warm.

### 10.3 It **does** depend on having the *right* $M$

This is the one experimentally load-bearing assumption, and where the method actually fails in
practice. Calibrating on Fock states built by *climbing the ladder* — the only kind a real
device can make — means inverting the wrong matrix: it credits the detector with losses that
*preparation* committed, and therefore **over-corrects**.

$$ M_{\rm climb}^{-1}M_{\rm ideal} \;\neq\; \mathbb{1},
\qquad \bigl\|M_{\rm climb}^{-1}M_{\rm ideal} - \mathbb{1}\bigr\|_{\max} = 0.0714 $$

Calibration B reports $\langle n\rangle = 13.11$ for a true $|13\rangle$. Measured on the real
device with the descending schedule:

| test state | TVD raw | TVD unfolded with **A** | TVD unfolded with **B** |
|---|---|---|---|
| coherent, $\alpha = 1.8$ | $1.11\times10^{-2}$ | $2.2\times10^{-16}$ | $2.38\times10^{-3}$ |
| $\tfrac{1}{\sqrt3}(\vert 0\rangle+\vert 3\rangle+\vert 7\rangle)$ | $5.49\times10^{-2}$ | $3.4\times10^{-16}$ | $9.69\times10^{-3}$ |
| $\tfrac{1}{\sqrt3}(\vert 1\rangle+\vert 4\rangle+\vert 9\rangle)$ | $8.13\times10^{-2}$ | $1.5\times10^{-16}$ | $1.52\times10^{-2}$ |

The exactness of column A is a consistency check on the simulation, **not** an experimental
result — nobody has an ideal Fock source.

### 10.4 It **does** depend on a phase-blind apparatus

Add a cavity displacement, a phase-sensitive drive, or a parity/homodyne measurement and §4
fails: coherences enter, the response tensor $A_c[n,m]$ is genuinely $N^2$-dimensional, and no
matrix acting on $p_n$ suffices.

### 10.5 It **does** depend on having enough shots

Exactness concerns the *mean*, not the variance. With $\hat q$ estimated from $N_{\rm shots}$ runs,

$$ \operatorname{Cov}(\hat q) = \frac{\operatorname{diag}(q) - qq^{\mathsf T}}{N_{\rm shots}},
\qquad \operatorname{Cov}(\hat p) = M^{-1}\operatorname{Cov}(\hat q)\,M^{-\mathsf T} $$

$\operatorname{cond}M$ bounds the amplification — but it is a worst case over all directions in
$q$, and real data rarely points along the worst one. $M^{-1}\hat q$ can also come out negative
in low-population bins, since it is an unbiased estimator and not a probability distribution;
constraining the fit to $\hat p \ge 0$ (non-negative least squares) removes that.

### 10.6 Truncation

Both the Fock space ($N = 14$) and the click axis ($C_{\max} = 13$) are truncated. Clicking past
$C_{\max}$ is only reachable through the channel of §6.3, so the leak inherits that channel's
$t_\pi/T_1^{q}$ scaling exactly:

| schedule | rounds | weight dropped past $C_{\max}$ | worst column |
|---|---|---|---|
| `descending` | 13 | $8.9\times10^{-16}$ | — (round-off) |
| `fixed` | 13 | $1.1\times10^{-15}$ | — (round-off) |
| `descending x3` | 39 | $6.2\times10^{-5}$ | $n = 13$ |
| `fixed x3` | 39 | $6.2\times10^{-5}$ | $n = 13$ |

A single pass cannot overflow at all: with $R = 13$ rounds the round count itself caps $C$ at
$13 = C_{\max}$, so the array bound is never binding and the residual is pure round-off. The
three-pass schedules run 39 rounds, so only the physics limits $C$, and reaching $c = 14$ needs a
state starting at $n = 13$ *and* the $\pi$-pulse channel manufacturing an excitation — hence the
leak sits entirely in the single top column $n = C_{\max}$, where none of the test states has
appreciable weight. (At the earlier $T_1^{q} = 200\,\mu s$ this leak was $\sim 6\times10^{-3}$;
raising $T_1^{q}$ by $100\times$ reduced it by $100\times$, as the scaling predicts.)

---

## 11. Numerical verification, end to end

| § | claim | check | result |
|---|---|---|---|
| 2.3 | branch bookkeeping is exact | $\sum_c q_c = 1$ | $6.7\times10^{-16}$ |
| 3 | $q_c$ linear in $\rho_{\rm cav}$ | $q(\text{mix})$ vs mix of $q$ | $1.7\times10^{-16}$ |
| 4.5 | $X_\pi$ covariant *after* readout | $\Vert[X_\pi,R_\theta]\Vert$ | $4\times10^{-17}$ (vs $5\times10^{-2}$ with coherence) |
| 5 | coherences irrelevant | $q(\rho)$ vs $q(\mathrm{diag}\,\rho)$ | $0$ (bitwise) |
| 5 | phase covariance | $q(\rho)$ vs $q(U_\theta\rho U_\theta^\dagger)$ | $5.6\times10^{-17}$ |
| 5 | forward law $q = Mp$ | $\bigl\vert Mp_{\rm true} - q_{\rm sim}\bigr\vert$ | $\le 1.7\times10^{-16}$, all schedules |
| 6.1 | $M$ column-stochastic | $\max\bigl\vert\text{colsum}-1\bigr\vert$ | $6.7\times10^{-16}$ |
| 6.3 | overcount channel is the $\pi$ pulse | $\max_{c>n}M_{cn}$, then $t_\pi\to0$ | $5.2\times10^{-5} \to 0$ exactly |
| 6.4 | $\det M = \prod M_{nn}$ | both computed | $0.143118$ vs $0.143130$ |
| 7 | $M^{-1}$ is not a channel | $\min(M^{-1})_{nc}$ | $-0.496$ |
| 7 | inversion recovers $p$ | $\mathrm{TVD}(M^{-1}q, p_{\rm true})$ | $\le 3.4\times10^{-16}$ |
| 8 | binomial semigroup | $M(\eta_1)M(\eta_2)$ vs $M(\eta_1\eta_2)$ | $1.7\times10^{-16}$ |
| 8 | closed-form inverse | $M(\eta)^{-1}$ vs $M(1/\eta)$ | $8.9\times10^{-15}$ |
| 9 | Kraus form (lossless) | vs full Liouvillian simulation | $\le 5.6\times10^{-16}$ |
| 9 | completeness | $\Vert K_g^\dagger K_g + K_e^\dagger K_e - \mathbb{1}\Vert$ | $1.1\times10^{-16}$ |
| 9 | $M_{RR} = \prod_k\sin^2$ (fixed, lossless) | product vs simulation | $0.03249$ vs $0.03249$ |
| 10.1 | robust to noise model | same checks, with heating | $3.2\times10^{-16}$ |

---

## Summary in one paragraph

The apparatus is a composition of Lindblad channels and projective measurements. Keeping the
measurement branches **unnormalised** makes the click probabilities a *linear* functional of
the input state — and linearity, not reversibility, is what loss leaves intact. Every element
of the protocol commutes with the total-excitation rotation $e^{i\theta(a^\dagger a + |e\rangle\langle e|)}$
— including the $\pi$ pulse, because the readout has already destroyed the qubit coherence
that would have spoiled it — so the response is blind to the cavity's phase, and averaging over
that phase annihilates every coherence. What survives is a linear map on the $N$ photon-number
populations alone: $q = Mp$, with $M_{cn}$ the click distribution of a Fock state, loss and all.
Because loss can only move probability *down* in click number and leaves a strong diagonal,
$M$ is triangular with positive determinant, hence invertible, and $M^{-1}q$ returns the true
distribution exactly. What $M^{-1}$ is *not* is a physical operation: it is anti-loss, with
negative entries, existing only as an estimator applied after the fact — which is why this
undoes the *statistics* of loss and never the loss itself.
