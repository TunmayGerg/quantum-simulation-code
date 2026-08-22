# The photon-counting record as a hidden Markov model

*A derivation from the Born rule, for a reader who knows quantum mechanics and has never met a
Markov model.*

Companion to `jc_counting_confusion_matrix.ipynb`. That notebook computes the distribution of
the **click count** — how many rounds of a photon-counting sequence reported "excited" — and
inverts a confusion matrix to recover the cavity's photon-number distribution. This document
derives the sharper object: the distribution of the **whole ordered measurement record**, shows
that it is *exactly* the output of a classical hidden Markov model with $2(N{+}1)$ states, and
works out what that buys.

Nothing here is assumed about Markov models. Everything is assumed about quantum mechanics:
Hilbert spaces, the Born rule, unitary evolution, projective measurement, and the Lindblad
equation. Every claim is stated and proved.

The implementation is the section headed `MARKOV ANALYSIS` at the bottom of
`msmt_workbench/rfsoc/runtimes/quantum_simulation/resonator_number_measurement.py`. Function
names in `code font` refer to it.

---

## Contents

- **Plain words** — what the model does, and what every input parameter does to it (no maths)
- **§0** The protocol, and the notation
- **§1** What a record's probability *is*: quantum instruments (Thm 1.1, 1.2)
- **§2** What a hidden Markov model is, from scratch (Def 2.1, Thm 2.4)
- **§3** The record process is a hidden Markov model on $D^2$ dimensions (Thm 3.1)
- **§4** It collapses to $D$ dimensions: the grading theorem (Lem 4.1–4.5, Thm 4.6)
- **§5** Corollary: the record sees the cavity only through its populations (Thm 5.1) — this
  is what licenses the confusion matrix
- **§6** The matrices, derived: swap (Lem 6.1–6.2), decay (Thm 6.4), readout, reset
- **§7** The one approximation, and its error (Prop 7.1)
- **§8** Inference: the forward algorithm (Thm 8.1), scaling (Lem 8.2), MAP and posterior
- **§9** Unfolding: why $q = Mp$ is exact and why $M$ must be rectangular (Thm 9.1–9.3)
- **§10** What the click count throws away, proved (Thm 10.1) — and the sharp condition under
  which it throws away *nothing* (Thm 10.3)
- **§11** Measured results: hardware and simulation
- **§12** Assumptions, in one list

---

# Plain words, before any maths

## What you have, and what you want

One shot of the experiment gives you a string of bits, one per round:

```
round     0    1    2    3    4    5    6    7    8    9   10   11   12
record    0    0    0    1    0    1    0    1    0    0    0    0    0
```

You want a number: how many photons were in the cavity when the counting started.

The obvious rule is **count the 1s** — here, 3. That is the click count, and it is what the
rest of `jc_counting_confusion_matrix.ipynb` does.

The Markov model does something different. For each candidate photon number
$n = 0, 1, 2, \ldots$ it asks:

> *If the cavity really had $n$ photons, how likely was **this exact string, in this exact
> order**?*

Then it picks the $n$ with the highest answer. That is the whole idea. Everything else is
bookkeeping.

## Why the order can possibly matter

Two records from a 13-round run, both with three clicks:

```
A:  1 1 1 0 0 0 0 0 0 0 0 0 0     clicks in the first three rounds
B:  0 0 0 0 0 0 0 0 0 0 1 1 1     clicks in the last three rounds
```

The click count says these are the same measurement: "3 photons". The physics says they are
not, and the reason is that **the rounds are not identical**. The schedule walks the ladder
down: round 0 is tuned to level 8, round 1 to level 7, and so on, with the last several rounds
all tuned to level 1.

- **A** clicked on the rounds tuned to levels 8, 7 and 6 — the rounds that only a highly
  occupied cavity responds strongly to.
- **B** clicked only on the level-1 mop-up rounds at the very end. Population that sits quietly
  through every high-level round and only responds to level-1 pulses was at $n = 1$ the whole
  time. Three separate late clicks are much better explained by *a qubit that heated up twice*
  than by three photons that waited politely for the end of the sequence.

The Markov model knows which round is which, so it can tell A from B. The click count cannot.
That is the entire source of its advantage — and §10 proves the sharp version: if the hardware
were perfect, the ordering would carry *exactly zero* extra information. The ordering only
helps because of errors.

## How it actually computes that likelihood

Think of it as carrying a little ledger of **"where might the system be right now?"** — a list
of possibilities with a probability attached to each. A possibility is a pair:

> *(how many photons are in the cavity, is the qubit up or down)*

Start the ledger with 100% on *(n photons, qubit down)* — that is the hypothesis being tested.
Then walk through the rounds. Each round does three things to the ledger:

1. **The pulse.** Move some probability from *(n photons, qubit down)* to
   *(n−1 photons, qubit up)*. How much depends on the round's length and on $n$ — a round tuned
   to level 4 moves almost all of $|4\rangle$ but only half of $|1\rangle$. While this happens,
   also bleed a little probability towards lower photon numbers and towards a relaxed qubit,
   because time is passing.
2. **Compare with the bit you actually recorded.** If the record says 1 for this round, then
   every entry with the qubit *down* just became unlikely — scale it down hard. If the record
   says 0, scale down the entries with the qubit *up*. The total weight you lose here *is* the
   likelihood contribution of this round; multiply it into a running product.
3. **Apply what happened next.** If the bit was 1, the reset $\pi$ fired: swap up and down in
   the ledger. If it was 0, nothing fired. Either way more time passes, so bleed a bit more.

After the last round, the running product is $P(\text{this record} \mid n)$. Do the whole walk
once per candidate $n$, compare, and take the winner. `counting_loglikelihood` is that walk;
`classify_photon_number` compares and picks.

Two things worth noticing about step 2. It is where the *data* enters — everything else is the
model talking to itself. And it is why the answer depends on the order: the ledger at round 7
is different depending on what happened in rounds 0 through 6, so the same bit means different
things at different points in the string.

## What every input parameter is for

Each parameter's job is to make some *excuse* available to the model. Without any of them the
model believes the hardware is perfect and can only ever answer "$\hat n$ = number of clicks".
Each parameter you turn on lets it say "…or maybe *that* happened instead."

**The schedule** — not device physics, but the model is wrong without it.

| input | what it is | what it does |
|---|---|---|
| `fock_ladder_top`, `count_extra_rounds` | how many rounds, and which level each is tuned to | Defines the rounds. Comes from the runtime, must match what the hardware played. |
| `level_swap_times_from` | where the per-level full-swap times come from: the ideal $t_1/\sqrt n$ law, or the measured `<swap>_N_<n>_swap` pulses | **The most influential input in the whole model.** These times set how strongly each round moves each photon number, which is the entire basis of telling A from B above. The two choices disagree by up to 0.25 in a single transfer probability. Default is `"sqrt"` because that is what `main()` plays. |

**The excuses.** Every row below is a way for the model to explain a bit it would not otherwise
expect.

| input | physical meaning | the excuse it enables | what happens if you raise it |
|---|---|---|---|
| `swap_transfer_scale` | swap contrast left unexplained after everything else | "the pulse fired but the photon didn't move" | The model trusts a *missing* click less. Makes it more willing to believe a high $n$ despite few clicks. |
| `qubit_t1` | qubit relaxation time | "the qubit was up, then fell down before I read it" | Missing clicks get cheaper to explain, so high-$n$ hypotheses survive longer. |
| `qubit_thermal_population` | residual $\|e\rangle$ population / heating | "that click was a hot qubit, not a photon" | **The key knob for late clicks.** Raise it and the model starts dismissing trailing clicks, pushing $\hat n$ down. |
| `cavity_t1` | single-photon lifetime — note decay runs at $n/T_1$, so $\|8\rangle$ leaks 8× as fast as $\|1\rangle$ | "the photon left before its round came round" | Missing clicks at *high* $n$ get cheaper. This is the parameter that fixes the systematic bias against large photon numbers. |
| `cavity_thermal_population` | photons appearing from the environment | "that click was a photon that wasn't there at the start" | Lets the $\|0\rangle$ hypothesis survive a click. Small effect unless the cavity is genuinely hot. |
| `readout_g_infidelity`, `readout_e_infidelity` | probability the recorded bit disagrees with the qubit | "I wrote down the wrong bit" | Every individual bit becomes less trustworthy, so the model leans more on the *pattern* and less on any single round. This is the parameter that softens step 2 above. |
| `reset_pi_error` | probability the reset $\pi$ did nothing | "the qubit was still up going into the next round" | Lets a click be followed by a spurious click. Explains click *pairs*. |
| `extra_round_time` | per-round dead time the pulse config doesn't know about | (not an excuse — it scales the clock) | Every round is longer, so all the decay excuses above get proportionally stronger. |

**Housekeeping** — no physics in these.

| input | what it does |
|---|---|
| `max_photons` | Cavity truncation. Keep it above the highest photon number you score, so heating has somewhere to go instead of piling up at the top. |
| `markov_analysis` | Switch the whole thing off; the click-count analysis is unaffected. |
| `bitstring_map_max_rounds` | Only build the complete "every possible string → photon number" table below this many rounds. The table has $2^{\rm rounds}$ rows. |
| `unfold_prep_index` | Which prep state the distribution plots show. `-1` = the last one, i.e. the test state. Display only; changes no number. |
| `shots_prep_index`, `shots_iterations_shown` | Which prep and how many iterations the per-round shots raster draws. Display only. |

## Which of these actually move the answer

Measured, on real data (§11):

- **The level swap times dominate.** Switching `"sqrt"` ↔ `"calibrated"` changes individual
  transfer probabilities by up to 0.25. If your answer depends on which one you pick, that is a
  calibration problem, not an analysis problem.
- **The thermal populations barely matter.** Moving the qubit's from 0.010 to 0.006 changed the
  mean fidelity by $10^{-4}$. Do not spend time on them.
- **The readout infidelity and reset error matter in proportion to how bad they are** — and
  they are the errors the ordering is *good* at catching, so they are the ones that make the
  whole exercise worthwhile. See the heating sweep in §10: at 10% heating per round the click
  count collapses to 0.32 while the chain still returns 0.46.
- **Cavity $T_1$ matters for the answer but, counter-intuitively, not for the Markov
  *advantage*.** A photon that already left is *absent* evidence rather than *ambiguous*
  evidence, and no cleverness recovers it. §10 has the sweep and the argument.

## One honest caveat before you read on

A better confusion matrix is **not** the same as a better final answer. The confusion matrix
already removes the systematic error, whichever reading you use — that is Theorem 9.1 — so both
readings land near the truth after unfolding. What the Markov reading actually buys is a
matrix closer to the identity, hence better conditioned, hence less shot noise amplified into
the result, and less exposure to your Fock-prep calibration being slightly off. Worth having.
Not the same as "more accurate". §11 shows a case where the Markov reading wins the
classification comparison and loses the unfolded-distribution comparison.

---

*Everything from here on is the derivation. The claims above are all proved below.*

---

## §0 The protocol, and the notation

A transmon qubit is dispersively coupled to a cavity mode through a parametric sideband drive.
The drive is tuned to the transition

$$|g,n\rangle \longleftrightarrow |e,n-1\rangle$$

which moves one photon out of the cavity and onto the qubit. One **round** of the counting
sequence is three operations in a fixed order:

1. a sideband pulse of duration $t_r$ (the "swap");
2. a projective measurement of the qubit in the $\{|g\rangle, |e\rangle\}$ basis, whose outcome
   is recorded as a bit $b_r \in \{0,1\}$ ($0 = $ read ground);
3. **conditional feedback**: if $b_r = 1$, a $\pi$ pulse returns the qubit to $|g\rangle$; if
   $b_r = 0$, nothing is played.

Run $R$ rounds. The datum of one shot is the **record**

$$\mathbf{b} = (b_0, b_1, \ldots, b_{R-1}) \in \{0,1\}^R .$$

**Hilbert space and basis.** Truncate the cavity at $N$ photons:

$$\mathcal{H} = \mathbb{C}^2 \otimes \mathbb{C}^{N+1}, \qquad
\{\,|q,n\rangle : q \in \{g,e\},\ n = 0 \ldots N \,\}, \qquad D \equiv \dim\mathcal{H} = 2(N{+}1).$$

`state_dim` returns $D$. Flatten the index as $i = 2n + q$ with $q = 0$ for $g$ and $1$ for
$e$, which is the convention every matrix in the code uses.

**Total excitation number.** Define the operator

$$\mathcal{N} \;=\; \hat n \otimes \mathbb{1} + \mathbb{1} \otimes |e\rangle\langle e|
\qquad\text{i.e.}\qquad \mathcal{N}|q,n\rangle = \big(n + [q = e]\big)\,|q,n\rangle .$$

$\mathcal{N}$ will do most of the work in §4. Its eigenspaces are

$$B_0 = \mathrm{span}\{|g,0\rangle\}, \qquad
B_n = \mathrm{span}\{|g,n\rangle,\ |e,n-1\rangle\} \ \ (1 \le n \le N), \qquad
B_{N+1} = \mathrm{span}\{|e,N\rangle\},$$

so $\mathcal{H} = \bigoplus_{n=0}^{N+1} B_n$, of dimensions $1, 2, 2, \ldots, 2, 1$, totalling
$2(N{+}1) = D$. ✓

**The schedule.** The Jaynes–Cummings matrix element on level $n$ is $g\sqrt{n}$, so a *full*
swap on that level takes a time $t_n \propto 1/\sqrt n$ (proved as Lemma 6.2). The runtime plays
the ladder **downwards**: rounds tuned to levels $N_{\max}, N_{\max}-1, \ldots, 1$, followed by
$K$ further rounds all at the level-1 time. `counting_ladder_levels` returns that list;
`counting_geometry` returns the durations actually played, reproducing the runtime's own
arithmetic so that the model can never be handed a schedule the hardware did not run.

---

## §1 What a record's probability is

This section fixes the exact object we are computing. There is no approximation in it.

A projective measurement with feedback is not a unitary and not a channel — it is a **quantum
instrument**: a collection $\{\mathcal{E}_b\}$ of completely positive maps, one per outcome,
whose sum $\sum_b \mathcal{E}_b$ is trace preserving. The physical content is the Born rule:
$\mathcal{E}_b(\rho)$ is the *unnormalised* post-measurement state on outcome $b$, and its trace
is the probability of $b$.

For one round $r$ of our protocol, write

$$\boxed{\ \mathcal{E}^{(r)}_b \;=\; \mathcal{F}_b \circ \mathcal{P}_b \circ \mathcal{U}_r \ }
\tag{1.1}$$

where

- $\mathcal{U}_r$ is the pulse: the completely positive trace-preserving (CPTP) map generated by
  the Lindbladian with the drive on, for duration $t_r$;
- $\mathcal{P}_b(\rho) = \Pi_b \rho \Pi_b$ with $\Pi_g = |g\rangle\langle g|\otimes\mathbb{1}$
  and $\Pi_e = |e\rangle\langle e|\otimes\mathbb{1}$ — the projective qubit measurement;
- $\mathcal{F}_b$ is the feedback: $\mathcal{F}_0$ is free evolution for the readout window;
  $\mathcal{F}_1$ is free evolution, then $\rho \mapsto X_\pi \rho X_\pi$ with
  $X_\pi = \sigma_x \otimes \mathbb{1}$, then free evolution for the $\pi$-pulse duration.

*(Readout misassignment is deferred to §6.4 and does not change anything structurally.)*

> **Theorem 1.1 (record probability).** For an initial state $\rho_0$,
> $$P(\mathbf{b}) \;=\; \mathrm{Tr}\Big[\ \mathcal{E}^{(R-1)}_{b_{R-1}} \circ \cdots \circ
> \mathcal{E}^{(0)}_{b_0}(\rho_0)\ \Big].$$

**Proof.** Induction on $R$. For $R=1$ this is the Born rule together with the statement that
$\mathcal{F}_b$ is trace preserving: $\mathrm{Tr}[\mathcal{F}_b(\Pi_b\rho_0\Pi_b)] =
\mathrm{Tr}[\Pi_b \rho_0 \Pi_b] = \mathrm{Tr}[\Pi_b \rho_0] = P(b_0 = b)$, using
$\Pi_b^2 = \Pi_b$ and cyclicity. For the inductive step, the state conditioned on
$b_0 \ldots b_{r-1}$ having been observed is, by the projection postulate,
$$\hat\rho_r = \frac{\mathcal{E}^{(r-1)}_{b_{r-1}}\circ\cdots\circ\mathcal{E}^{(0)}_{b_0}(\rho_0)}
{\mathrm{Tr}[\,\cdot\,]},$$
and the Born rule applied at round $r$ gives $P(b_r \mid b_0\ldots b_{r-1}) =
\mathrm{Tr}[\mathcal{E}^{(r)}_{b_r}(\hat\rho_r)]$. Multiplying the chain of conditionals
telescopes the normalisations. $\square$

> **Theorem 1.2 (normalisation).** $\sum_{\mathbf{b}\in\{0,1\}^R} P(\mathbf{b}) = 1$.

**Proof.** $\sum_b \mathcal{E}^{(r)}_b$ is trace preserving for each $r$, because
$\sum_b \Pi_b \rho \Pi_b$ is (as $\{\Pi_g,\Pi_e\}$ is a complete set of orthogonal projectors)
and $\mathcal{U}_r, \mathcal{F}_b$ are. Summing over $b_{R-1}$ first, then $b_{R-2}$, and so on,
each sum removes one trace-preserving layer, leaving $\mathrm{Tr}\,\rho_0 = 1$. $\square$

Theorem 1.1 is already a recursion — an *unnormalised* state is pushed forward one round at a
time and its trace read off. The rest of this document is about how small that recursion can be
made. Right now the object being propagated is a $D \times D$ operator: $D^2 = 4(N{+}1)^2$ real
numbers. We will get it down to $D$.

---

## §2 What a hidden Markov model is

Nothing quantum in this section.

> **Definition 2.1 (Markov chain).** A sequence of random variables $s_0, s_1, \ldots$ on a
> finite set $S$ is a **Markov chain** if
> $$P(s_{r+1} \mid s_r, s_{r-1}, \ldots, s_0) = P(s_{r+1} \mid s_r) \quad\text{for all } r.$$
> The **transition matrix** is $A[i,j] = P(s_{r+1} = j \mid s_r = i)$. It is *row-stochastic*:
> $A[i,j] \ge 0$ and $\sum_j A[i,j] = 1$.

The content of the definition is the phrase *"the future depends on the past only through the
present"*. A state that carries enough information to make that true is called a **sufficient
state**; finding one is the whole game.

> **Definition 2.2 (hidden Markov model).** A **hidden Markov model** (HMM) is a Markov chain
> $s_0, s_1, \ldots$ that is *not observed*, together with observations $b_0, b_1, \ldots$ on a
> finite alphabet, such that each $b_r$ depends only on the contemporaneous hidden state:
> $$P(b_r \mid s_0 \ldots s_r,\ b_0 \ldots b_{r-1}) = P(b_r \mid s_r) \;\equiv\; E[s_r, b_r].$$
> $E$ is the **emission matrix**, row-stochastic.

> **Definition 2.3 (HMM with output feedback).** Allow the transition to depend on the
> *observation* as well as the state: $P(s_{r+1} \mid s_r, b_r) = B_{b_r}[s_r, s_{r+1}]$, with
> each $B_b$ row-stochastic.

Definition 2.3 is what a feedback-reset experiment needs, and it is why an "averaged reset" model
would be describing a different experiment. Nothing about the theory below is harder for it.

> **Theorem 2.4 (the likelihood of a record).** For an HMM with output feedback, initial
> distribution $\pi$, transition matrices $A_r$, emission $E$, and feedback transitions $B_b$,
> $$P(\mathbf{b}) \;=\; \pi\, \Big[\textstyle\prod_{r=0}^{R-1} A_r\,\mathrm{diag}(E[:,b_r])\,
> B_{b_r}\Big]\, \mathbf{1},$$
> where $\pi$ is a row vector, $\mathbf{1}$ a column of ones, and the product is taken in
> increasing $r$ from the left.

**Proof.** Write the joint probability of states and observations, using Definitions 2.1–2.3 to
factor it, and then sum over the unobserved states:
$$P(\mathbf{b}) = \sum_{s_0,\ldots,s_R} \pi[s_0] \prod_{r=0}^{R-1}
\underbrace{A_r[s_r, s_r']}_{\text{pulse}}\,
\underbrace{E[s_r', b_r]}_{\text{emit}}\,
\underbrace{B_{b_r}[s_r', s_{r+1}]}_{\text{feedback}} .$$
Each sum over an intermediate index is exactly one matrix multiplication, performed left to
right; multiplying by $\mathbf{1}$ performs the last one. $\square$

Theorem 2.4 is the **forward algorithm**. Evaluated left to right it costs $O(R\,|S|^2)$ instead
of the $O(|S|^R)$ of the naive sum. That is the entire computational content of HMMs, and it is
what `counting_loglikelihood` runs.

---

## §3 The record process is an HMM

> **Theorem 3.1.** The record process of §1 is an HMM with output feedback whose hidden state is
> the (normalised) conditional density matrix $\hat\rho_r$.

**Proof.** By Theorem 1.1's inductive step, $P(b_r \mid b_0\ldots b_{r-1}) =
\mathrm{Tr}[\mathcal{E}^{(r)}_{b_r}(\hat\rho_r)]$ — a function of $\hat\rho_r$ and $b_r$ alone.
And $\hat\rho_{r+1} = \mathcal{E}^{(r)}_{b_r}(\hat\rho_r)/\mathrm{Tr}[\cdot]$ is determined by
$\hat\rho_r$ and $b_r$ alone. So $\hat\rho_r$ is a sufficient state in the sense of Definition
2.1, with the observation-dependent transition of Definition 2.3. $\square$

This is true but not yet useful: the state space is the set of density matrices, a continuum of
dimension $D^2 - 1$. Theorem 1.1's recursion propagates all of it. The next section shows that
almost all of it is unreachable.

---

## §4 The collapse to $D$ dimensions

The whole reduction rests on one bookkeeping device. For basis operators define the **grading**

$$\Delta\big(|i\rangle\langle j|\big) \;=\; \mathcal{N}_i - \mathcal{N}_j
\qquad\text{where } \mathcal{N}_{(q,n)} = n + [q = e],$$

and decompose the space of operators as $\mathcal{B}(\mathcal{H}) =
\bigoplus_{d} \mathcal{V}_d$, where $\mathcal{V}_d$ is spanned by the basis operators with
$\Delta = d$. Note $\mathcal{V}_0$ contains all the diagonal operators, and also the
*within-block* coherences $|g,n\rangle\langle e,n-1|$ (both have $\mathcal{N} = n$), and nothing
else.

> **Lemma 4.1 (the pulse preserves the grading).** The Jaynes–Cummings Hamiltonian
> $H = g(\sigma_+ \otimes a + \sigma_- \otimes a^\dagger)$ satisfies $[H, \mathcal{N}] = 0$.
> Consequently the superoperator $\rho \mapsto -i[H,\rho]$ maps $\mathcal{V}_d$ into
> $\mathcal{V}_d$.

**Proof.** $\sigma_+\otimes a$ raises the qubit ($\mathcal{N} \to \mathcal{N}+1$) and lowers the
cavity ($\mathcal{N}\to\mathcal{N}-1$), net zero; likewise $\sigma_-\otimes a^\dagger$. So $H$
maps each $\mathcal{N}$-eigenspace $B_n$ into itself, i.e. $[H,\mathcal{N}]=0$. Then for
$|i\rangle\langle j| \in \mathcal{V}_d$, both $H|i\rangle\langle j|$ and $|i\rangle\langle j|H$
are supported on basis operators $|i'\rangle\langle j'|$ with $\mathcal{N}_{i'} = \mathcal{N}_i$
and $\mathcal{N}_{j'} = \mathcal{N}_j$, hence $\Delta = d$. $\square$

> **Lemma 4.2 (the dissipators preserve the grading).** Let $L$ be any of
> $\sigma_-\otimes\mathbb{1}$, $\sigma_+\otimes\mathbb{1}$, $\mathbb{1}\otimes a$,
> $\mathbb{1}\otimes a^\dagger$, or any operator diagonal in $\{|q,n\rangle\}$ (dephasing). Then
> $\rho \mapsto L\rho L^\dagger - \tfrac12\{L^\dagger L, \rho\}$ maps $\mathcal{V}_d$ into
> $\mathcal{V}_d$.

**Proof.** Each listed $L$ maps a basis vector to a multiple of a single basis vector and shifts
$\mathcal{N}$ by a fixed amount $\delta_L$ ($\delta = -1$ for $\sigma_-$ and for $a$, $+1$ for
$\sigma_+$ and $a^\dagger$, $0$ for diagonal $L$). Hence
$L|i\rangle\langle j|L^\dagger \propto |i'\rangle\langle j'|$ with $\mathcal{N}_{i'} =
\mathcal{N}_i + \delta_L$ and $\mathcal{N}_{j'} = \mathcal{N}_j + \delta_L$, so $\Delta$ is
unchanged. And $L^\dagger L$ is diagonal, so $\{L^\dagger L, \cdot\}$ acts on
$|i\rangle\langle j|$ by a scalar. $\square$

> **Lemma 4.3 (the measurement preserves the grading, and kills every within-block coherence).**
> $\mathcal{P}_g + \mathcal{P}_e$ maps $\mathcal{V}_d \to \mathcal{V}_d$. Moreover
> $(\mathcal{P}_g + \mathcal{P}_e)$ annihilates $|g,n\rangle\langle e,n-1|$ and its conjugate,
> i.e. its restriction to $\mathcal{V}_0$ is exactly the projection onto the **diagonal**.

**Proof.** $\Pi_b$ is diagonal in $\{|q,n\rangle\}$, so $\Pi_b|i\rangle\langle j|\Pi_b$ is either
$|i\rangle\langle j|$ or $0$: the grading is preserved. For the second claim,
$|g,n\rangle$ lies in the range of $\Pi_g$ and $|e,n-1\rangle$ in the range of $\Pi_e$, so
$\Pi_g |g,n\rangle\langle e,n-1| \Pi_g = 0$ (right factor killed) and
$\Pi_e|g,n\rangle\langle e,n-1|\Pi_e = 0$ (left factor killed). The only $\mathcal{V}_0$ basis
operators surviving are the diagonal ones. $\square$

Lemma 4.3 is the physical heart of the reduction: **the qubit readout, every single round,
destroys exactly the coherences the swap creates.** Nothing needs to be assumed about
decoherence for this; it is the projection postulate.

> **Lemma 4.4 (the feedback preserves the grading).** On operators already in the range of
> $\mathcal{P}_g + \mathcal{P}_e$, the map $\rho \mapsto X_\pi \rho X_\pi$ preserves $\Delta$.

**Proof.** Such operators are supported on $|q,n\rangle\langle q,m|$ with the *same* $q$ on both
sides (Lemma 4.3). $X_\pi$ flips $q$ on both sides, shifting $\mathcal{N}$ by the same
$\pm 1$ on each, so $\Delta = n - m$ is unchanged. (On a mixed-sector operator $X_\pi$ would
*not* preserve $\Delta$; it never sees one.) $\square$

> **Lemma 4.5 (free evolution preserves diagonality).** With the drive off, the Lindbladian
> $\mathcal{L}_0$ built from the operators of Lemma 4.2 maps diagonal operators to diagonal
> operators, hence so does $e^{\mathcal{L}_0 t}$.

**Proof.** For diagonal $\rho = \sum_i p_i |i\rangle\langle i|$: each $L\rho L^\dagger =
\sum_i p_i \,|L i\rangle\langle L i|$ is diagonal because $L$ sends basis vectors to multiples of
basis vectors; $L^\dagger L$ is diagonal so the anticommutator term is diagonal; and there is no
Hamiltonian term. The diagonal operators form a closed subspace, so the exponential preserves
it. $\square$

Now the theorem.

> **Theorem 4.6 (exact reduction to $D$ classical states).** Suppose the initial state is
> $\rho_0 = |g\rangle\langle g| \otimes \rho_{\rm cav}$. Then for every $r$ the unnormalised
> conditional operator $\tilde\rho_r = \mathcal{E}^{(r-1)}_{b_{r-1}}\circ\cdots\circ
> \mathcal{E}^{(0)}_{b_0}(\rho_0)$ appearing in Theorem 1.1 is **diagonal** in the basis
> $\{|q,n\rangle\}$ up to terms of grading $\Delta \ne 0$ that never contribute to any record
> probability. Consequently the record distribution is *exactly* that of an HMM with output
> feedback on $|S| = D = 2(N{+}1)$ hidden states.

**Proof.** *Step 1 — the $\Delta \ne 0$ sectors are inert.* By Lemmas 4.1–4.4, every map in the
composition preserves the grading, so a component of $\rho_0$ in $\mathcal{V}_d$ stays in
$\mathcal{V}_d$ for all time. Record probabilities are traces (Theorem 1.1), and
$\mathrm{Tr}\,|i\rangle\langle j| = \delta_{ij}$ vanishes on $\mathcal{V}_d$ for $d\ne 0$ —
indeed $\Delta = 0$ is *necessary* for $i = j$. So no $\Delta\ne0$ component can ever influence
a record probability, and may be discarded.

*Step 2 — inside $\mathcal{V}_0$, only the diagonal survives each round.* $\mathcal{V}_0$ is
spanned by the diagonal operators together with the within-block coherences
$|g,n\rangle\langle e,n-1|$ and their conjugates. By Lemma 4.3 the measurement annihilates the
latter. Since the measurement occurs once per round, at the *start* of every round the operator
is diagonal within $\mathcal{V}_0$; and $\rho_0$ is diagonal at $r=0$ (its off-diagonal parts
$|g,n\rangle\langle g,m|$, $n\ne m$, have $\Delta = n-m \ne 0$ and were discarded in Step 1).

*Step 3 — identify the HMM.* Let $\alpha_r \in \mathbb{R}^D$ be the diagonal of $\tilde\rho_r$.
By Steps 1–2 the recursion of Theorem 1.1 closes on $\alpha_r$. Define
$$A_r[i,j] = \big\langle j\big|\,\mathcal{U}_r\big(|i\rangle\langle i|\big)\,\big|j\big\rangle ,
\qquad
B_b[i,j] = \big\langle j\big|\,\mathcal{F}_b\big(|i\rangle\langle i|\big)\,\big|j\big\rangle ,$$
with $E$ the readout channel of §6.4. $A_r$ is
row-stochastic because $\mathcal{U}_r$ is trace preserving and positive, and the projection onto
the diagonal (Lemma 4.3) does not change traces; $B_b$ likewise. The factorisation is exactly
Theorem 2.4. $\square$

Two remarks, both worth keeping.

- **The inert corners.** $B_0 = \{|g,0\rangle\}$ has no swap partner, and $B_{N+1} =
  \{|e,N\rangle\}$ would need $|g,N+1\rangle$, which the truncation does not carry. Both are
  fixed points of the swap. The first is physical (vacuum cannot emit); the second is a
  truncation artifact, and the remedy is to keep $N$ at least one level above the largest photon
  number being scored — which is what `max_photons` defaults to.
- **What was *not* assumed.** No secular or rotating-wave approximation beyond the one already
  in $H$; no weak-coupling limit; no assumption that decoherence is small. Theorem 4.6 is exact
  given the operator list of Lemma 4.2.

---

## §5 The record sees the cavity only through its populations

This is the theorem that licenses the entire confusion-matrix method, in this notebook and in the
runtime. It is a corollary of Step 1 above and deserves to be stated on its own.

> **Theorem 5.1.** Let $\rho_0 = |g\rangle\langle g|\otimes\rho_{\rm cav}$. Then
> $$P(\mathbf{b}) \;=\; \sum_{n=0}^{N} p_n\, P(\mathbf{b}\mid n), \qquad
> p_n = \langle n|\rho_{\rm cav}|n\rangle,$$
> where $P(\mathbf{b}\mid n)$ is the record distribution for the pure initial state
> $|g,n\rangle$. In particular the record distribution is **completely independent** of the
> off-diagonal elements of $\rho_{\rm cav}$: two cavity states with the same photon-number
> populations are indistinguishable to this measurement, no matter how different their
> coherences.

**Proof.** $\rho_0 = \sum_{n,m} \langle n|\rho_{\rm cav}|m\rangle\, |g,n\rangle\langle g,m|$, and
$\Delta(|g,n\rangle\langle g,m|) = n - m$. By Step 1 of Theorem 4.6, every term with $n \ne m$
lies in an inert grading sector and contributes nothing to $P(\mathbf{b})$. What remains is
$\sum_n p_n |g,n\rangle\langle g,n|$, and $P(\mathbf{b})$ is linear in $\rho_0$ by Theorem 1.1.
$\square$

Theorem 5.1 is also a *warning*: a photon counter of this kind measures $\{p_n\}$ and nothing
else. It cannot distinguish $\tfrac{1}{\sqrt2}(|0\rangle + |3\rangle)$ from
$\tfrac12(|0\rangle\langle0| + |3\rangle\langle3|)$, and no amount of post-processing will
change that.

---

## §6 The matrices, derived

### 6.1 The swap

> **Lemma 6.1 (transfer probability).** Let the drive be resonant with the sideband and let
> decoherence be neglected for the moment. A pulse of duration $t$ implements, on the block
> $B_n$ in the ordered basis $(|g,n\rangle, |e,n-1\rangle)$,
> $$U_n(t) = \exp\!\big(-i\,g\sqrt{n}\,t\,\sigma_x\big),$$
> so the probability of moving one photon out of the cavity is
> $$p_n(t) \;=\; \big|\langle e,n-1|U_n(t)|g,n\rangle\big|^2 \;=\; \sin^2\!\big(g\sqrt n\, t\big).$$

**Proof.** With $H = g(\sigma_+\otimes a + \sigma_-\otimes a^\dagger)$,
$$H|g,n\rangle = g\,\sigma_+\otimes a\,|g,n\rangle = g\sqrt n\,|e,n-1\rangle, \qquad
H|e,n-1\rangle = g\,\sigma_-\otimes a^\dagger|e,n-1\rangle = g\sqrt n\,|g,n\rangle,$$
so $H|_{B_n} = g\sqrt n\,\sigma_x$ in that basis. Exponentiate; the off-diagonal element of
$\exp(-i\theta\sigma_x)$ is $-i\sin\theta$. $\square$

> **Lemma 6.2 (the $1/\sqrt n$ law).** A *full* swap on level $n$ — $p_n = 1$ — requires
> $t_n = \pi/(2g\sqrt n) = t_1/\sqrt n$. Consequently a round of duration $t$ rotates level $n$
> by $\theta_n = \pi t / t_n$ and
> $$p_n(t) \;=\; \sin^2\!\big(\theta_n/2\big), \qquad \theta_n = \pi\,t/t_n .$$

**Proof.** $\sin^2(g\sqrt n\,t) = 1 \iff g\sqrt n\,t = \pi/2$. Substituting
$g\sqrt n = \pi/(2 t_n)$ into Lemma 6.1 gives $p_n = \sin^2(\pi t / (2 t_n)) =
\sin^2(\theta_n/2)$. $\square$

Lemma 6.2 is the form the code uses (`swap_map`), and the reason it is written that way rather
than in terms of $g$ is that $t_n$ is *measurable* while $g$ is inferred. It also makes the model
automatically honest about hardware quantisation: the runtime's register-driven pulse length
lands on a 5 ns grid, and `counting_geometry` reports the length actually played, so $\theta_n$ is
computed from what happened rather than from what was intended.

**The consequence that matters.** $p_n(t)$ depends on *both* the round and the level, and not as
a product of a per-round factor and a per-level factor. A round tuned to level $L$ is a full swap
on $L$ and something else everywhere else, and *that something else is the information*. Real
numbers from the runtime's schedule ($t_1 = 570$ ns, $\sqrt n$ law):

| round tuned to | $\vert 1\rangle$ | $\vert 2\rangle$ | $\vert 4\rangle$ | $\vert 6\rangle$ | $\vert 8\rangle$ |
|---|---|---|---|---|---|
| level 1 | **1.000** | 0.633 | 0.000 | 0.421 | 0.929 |
| level 4 | 0.500 | 0.803 | **1.000** | 0.880 | 0.633 |
| level 8 | 0.278 | 0.500 | 0.803 | 0.956 | **1.000** |

The zero is not a numerical accident and is worth checking by hand:
$\theta_4 = \pi t_1/t_4 = \pi\sqrt4 = 2\pi$, so $\sin^2(\theta_4/2) = \sin^2\pi = 0$. **Under the
$\sqrt n$ law a level-1 pulse is a double swap on $|4\rangle$ and returns it exactly untouched.**

**Residual contrast.** `swap_transfer_scale` multiplies $p_n$ by a factor $\eta_n \le 1$. It is
*not* the level dependence (that is $\theta_n$), *not* off-resonance from per-level Stark shifts
(3–37 kHz against a 0.9–2.5 MHz swap rate on Q1C1 caps the transfer at 0.9998), and it must *not*
absorb decoherence during the pulse, which §6.2 already models exactly. It is only what a swap
Rabi fit fails to explain after both. Note the model limit this introduces: the population that
fails to transfer is left *where it was*, i.e. the photon is still in the cavity for a later round
to catch. If the lost contrast is really leakage out of $\{g,e\}\times\{0\ldots N\}$, the
population should have left the counted subspace, and this model has no sink for it. The two
readings differ most at the top of the ladder.

### 6.2 Decay, exactly

> **Theorem 6.3 (Pauli master equation).** Let $\mathcal{L}_0(\rho) = \sum_k \big(L_k\rho
> L_k^\dagger - \tfrac12\{L_k^\dagger L_k,\rho\}\big)$ with each $L_k$ as in Lemma 4.2. Restricted
> to diagonal $\rho = \sum_i p_i|i\rangle\langle i|$ (legitimate by Lemma 4.5), the dynamics is
> $$\dot p_j \;=\; \sum_i p_i\,G[i,j], \qquad
> G[i,j] = \sum_k \big|\langle j|L_k|i\rangle\big|^2 \ (i \ne j), \qquad
> G[i,i] = -\sum_{j\ne i} G[i,j].$$

**Proof.** Take the $\langle j|\cdot|j\rangle$ matrix element:
$$\dot p_j = \sum_k \Big[\textstyle\sum_i p_i |\langle j|L_k|i\rangle|^2
- p_j \langle j|L_k^\dagger L_k|j\rangle\Big].$$
The first term is $\sum_{i} p_i \sum_k |\langle j|L_k|i\rangle|^2$. For the second, insert a
resolution of the identity: $\langle j|L_k^\dagger L_k|j\rangle = \sum_m |\langle m|L_k|j\rangle|^2$,
so $\sum_k \langle j|L_k^\dagger L_k|j\rangle = \sum_m \sum_k |\langle m|L_k|j\rangle|^2$.
Separating $m = j$ from $m \ne j$ on both sides and cancelling gives the stated $G$. $\square$

> **Corollary 6.4 ($e^{Gt}$ is a stochastic matrix).** $G$ has non-negative off-diagonal entries
> and rows summing to zero; hence $e^{Gt}$ has non-negative entries and rows summing to one, for
> every $t \ge 0$.

**Proof.** Off-diagonal non-negativity and $\sum_j G[i,j] = 0$ are immediate from the
construction. For $c > \max_i |G[i,i]|$, $G + c\,\mathbb{1}$ has all entries $\ge 0$, so
$e^{Gt} = e^{-ct} e^{(G + c\mathbb{1})t} = e^{-ct}\sum_{m\ge0} \frac{t^m}{m!}(G+c\mathbb{1})^m$
is a sum of non-negative matrices. And $G\mathbf{1} = 0 \Rightarrow e^{Gt}\mathbf{1} =
\mathbf{1}$. $\square$

`population_generator` builds $G$; `idle_map` returns $e^{Gt}$ via `scipy.linalg.expm`.
Specialising Theorem 6.3 to the four physical channels, with $\gamma_q = 1/T_1^q$,
$\kappa = 1/T_1^{\rm cav}$:

| $L_k$ | matrix element | rate $|q,n\rangle \to$ |
|---|---|---|
| $\sqrt{\gamma_q}\,\sigma_-\otimes\mathbb{1}$ | $\langle g,n|L|e,n\rangle = \sqrt{\gamma_q}$ | $\gamma_q$, to $\vert g,n\rangle$ |
| $\sqrt{\gamma_q \bar n_q}\,\sigma_+\otimes\mathbb{1}$ | $\sqrt{\gamma_q \bar n_q}$ | $\gamma_q \bar n_q$, to $\vert e,n\rangle$ |
| $\sqrt\kappa\,\mathbb{1}\otimes a$ | $\langle q,n-1|L|q,n\rangle = \sqrt{\kappa n}$ | $\kappa\, \boldsymbol{n}$, to $\vert q,n{-}1\rangle$ |
| $\sqrt{\kappa \bar n_c}\,\mathbb{1}\otimes a^\dagger$ | $\sqrt{\kappa \bar n_c (n{+}1)}$ | $\kappa \bar n_c (n{+}1)$, to $\vert q,n{+}1\rangle$ |

The boldface $n$ is the single most consequential asymmetry in the model: **a Fock state
$|n\rangle$ leaks $n$ times faster than $|1\rangle$**, because $|\langle n-1|a|n\rangle|^2 = n$.
That is a systematic bias against high photon numbers, it is timing-dependent, and it is exactly
what a click count cannot see.

Note what is *absent*: pure dephasing. It appears in $G$ only through diagonal $L_k$, which
contribute nothing off-diagonal, i.e. nothing at all. This is not an omission — by Lemma 4.3 the
readout destroys every coherence once per round, so there is no coherence left for dephasing to
act on. $T_2$ is **not a parameter of this model**, and that is a theorem, not a modelling
choice.

### 6.3 The reset

`pi_map` implements $\rho \mapsto X_\pi\rho X_\pi$ on populations with success probability
$1 - \epsilon_\pi$:

$$B_1 \;=\; e^{G t_{\rm meas}}\;\Pi^{(\epsilon_\pi)}\;e^{G t_\pi}, \qquad
B_0 \;=\; e^{G t_{\rm meas}},$$

where $\Pi^{(\epsilon_\pi)}$ swaps $|g,n\rangle \leftrightarrow |e,n\rangle$ with probability
$1-\epsilon_\pi$ and does nothing with probability $\epsilon_\pi$. Both are stochastic by
Corollary 6.4 and because a mixture of permutation matrices is stochastic.

The **indexing by the observed bit** is the content of Definition 2.3 and it is what makes this
the right chain rather than an approximation of one. It also handles the case that matters most
in practice: the true qubit level was $e$, the readout misread it as $g$, so *no reset fired* and
the qubit stays excited into the next round. A model that averaged the reset would get that
wrong; this one gets it right, because the reset transition is selected by the *recorded* bit
while the emission already accounts for the misread.

$t_{\rm meas}$ is not a free parameter: `readout_round_time` computes it as the capture delay
plus the capture memory's own length, because the CMACC cannot report before its integration
window closes, so the reset decision cannot take effect before then. `extra_round_time` adds
whatever the pulse config does not know about (branch evaluation, cache read, DMA push);
`checks/check_counting_timing.py` measures it. $t_\pi$ comes from `pulse_list_duration` of the
reset pulse list.

### 6.4 The readout

The recorded bit is not the projection outcome. Model the readout as a classical binary channel
applied to the projected qubit level, with

$$E[i, b] = \begin{cases}
1 - \varepsilon_g & i \text{ has } q=g,\ b = 0\\
\varepsilon_g & i \text{ has } q=g,\ b = 1\\
\varepsilon_e & i \text{ has } q=e,\ b = 0\\
1 - \varepsilon_e & i \text{ has } q=e,\ b = 1 .
\end{cases}$$

Row-stochastic by construction. A 99.2% readout gives $\varepsilon \approx 0.008$ each way. Two
points of rigour:

- The projection of Lemma 4.3 is what puts the state into a definite qubit sector; $E$ describes
  only the *classical* corruption of the bit that is written down. Because the projection does not
  change the diagonal (it removes only off-diagonal elements), the populations entering $E$ are
  exactly $\alpha_r A_r$, which is why the code multiplies rather than re-projecting.
- The misassignment is assumed independent between rounds and independent of $n$. Both are
  checkable and neither is guaranteed; a readout whose fidelity depends on cavity occupation
  (a real effect, through the dispersive shift) would violate the second.

### 6.5 One round, assembled

`counting_hmm_model` returns

$$A_r \;=\; e^{G\,t^{\rm wall}_r/2}\;S(t^{\rm eff}_r)\;e^{G\,t^{\rm wall}_r/2},
\qquad E \text{ as above}, \qquad B_b \text{ as in §6.3},$$

with $S$ the swap of Lemma 6.2. Here $t^{\rm eff}$ is the swap *area* delivered (flat plus half
the ramp, the convention the ladder calibration uses) and $t^{\rm wall}$ the pulse's wall-clock
occupancy of the channel. The initial distribution is
$\pi^{(n)} = (1-\bar n_q)\,\delta_{|g,n\rangle} + \bar n_q\,\delta_{|e,n\rangle}$, one row per
candidate $n$.

---

## §7 The one approximation

Everything above is exact except one thing, and it should be named precisely.

> **Proposition 7.1.** The exact round map is $\mathcal{U}_r = \exp\big[(\mathcal{L}_{\rm drive})
> t_r\big]$ with $\mathcal{L}_{\rm drive}(\rho) = -i[H,\rho] + \mathcal{L}_0(\rho)$. The model uses
> the symmetric (Strang) splitting
> $$A_r \;=\; e^{\mathcal{L}_0 t_r/2}\; e^{-i[H,\cdot]t_r}\; e^{\mathcal{L}_0 t_r/2}
> \;=\; \mathcal{U}_r \;+\; O(t_r^3).$$
> The first-order and second-order terms cancel because the splitting is symmetric; the leading
> error is a $t_r^3$ term built from the double commutators
> $[\mathcal{L}_0,[\mathcal{L}_0,-i[H,\cdot]]]$ and $[-i[H,\cdot],[-i[H,\cdot],\mathcal{L}_0]]$,
> with the coefficients given by the Baker–Campbell–Hausdorff expansion for Strang splitting. Its
> magnitude is therefore of order $t_r^3 \,\|\mathcal{L}_0\|\,\|H\|^2$ relative to the leading
> term, i.e. $O\big((t_r/T_1)\,(t_r\|H\|)^2\big)$.

**What this does and does not affect.** It does *not* affect Theorem 4.6: the reduction to $D$
classical states is exact regardless, because the exact map
$K_b^{(r)}[i,j] = \langle j|\Pi_b\,\mathcal{U}_r(|i\rangle\langle i|)\,\Pi_b|j\rangle$ is a
perfectly good (and stochastic) transition kernel. What the splitting approximates is only the
*numerical value* of $A_r$. The scale of the error is set by $t_r \cdot \max(\gamma_q, N\kappa)$
— on the runtime's schedule, $t_r \le 570$ ns against $T_1$'s of 150 µs and 900 µs, so
$t_r/T_1 \lesssim 4\times10^{-3}$ and the cubic term is at the $10^{-8}$ level. It is
irrelevant here and would stop being so only for pulses comparable to $T_1$.

> **Remark 7.2.** If exactness is wanted, replace $A_r$ by the diagonal-to-diagonal block of
> $\exp[\mathcal{L}_{\rm drive} t_r]$, computed once per distinct duration. This costs one
> $D^2 \times D^2$ matrix exponential per distinct round length instead of one $D\times D$ swap
> matrix, and changes nothing else in the pipeline.

---

## §8 Inference

### 8.1 The likelihood

> **Theorem 8.1.** For each candidate initial photon number $n$, define row vectors by
> $$\alpha^{(n)}_0 = \pi^{(n)}, \qquad
> \alpha^{(n)}_{r+1} = \Big(\alpha^{(n)}_r A_r \odot E[:,b_r]^{\!\top}\Big) B_{b_r},$$
> where $\odot$ is elementwise. Then $P(\mathbf{b}\mid n) = \alpha^{(n)}_R \mathbf{1}$.

**Proof.** This is Theorem 2.4 with the product evaluated left to right; the elementwise product
by $E[:,b_r]$ is the multiplication by $\mathrm{diag}(E[:,b_r])$. $\square$

### 8.2 Why the code renormalises every round

A 13-round likelihood is a product of 13 numbers each well below 1 and underflows in double
precision for long records. The fix is exact, not a hack.

> **Lemma 8.2 (scaling identity).** Define $c_r = \hat\alpha_r A_r \odot E[:,b_r]^\top
> \cdot\mathbf{1}$ and $\hat\alpha_{r+1} = \big(\hat\alpha_r A_r \odot
> E[:,b_r]^\top\big)B_{b_r}/c_r$, starting from $\hat\alpha_0 = \pi$. Then
> $$P(\mathbf{b}) = \prod_{r=0}^{R-1} c_r, \qquad\text{equivalently}\qquad
> \log P(\mathbf{b}) = \sum_r \log c_r .$$

**Proof.** Induction. $\hat\alpha_r = \alpha_r / \prod_{s<r} c_s$ by construction, and each
$B_b$ is row-stochastic so it preserves the sum, giving $\hat\alpha_r\mathbf{1} = 1$ for all $r$.
Hence $c_r = (\alpha_r A_r \odot E)\mathbf{1}/\prod_{s<r}c_s$, and telescoping yields
$\alpha_R\mathbf{1} = \prod_r c_r$. $\square$

`counting_loglikelihood` accumulates $\sum_r \log c_r$. It also propagates only the *distinct*
records present in the data (`np.unique`), which is exact because the likelihood is a function of
the record and nothing else: with 13 rounds, thousands of shots collapse onto far fewer patterns.

### 8.3 From likelihood to an answer

`classify_photon_number` returns two things, and they answer different questions.

- The **maximum-likelihood estimate** $\hat n(\mathbf{b}) = \arg\max_n \log P(\mathbf{b}|n)$. A
  hard assignment: one integer per shot. This is what makes a *square* confusion matrix possible.
- The **posterior** $P(n\mid\mathbf{b}) = \pi_n P(\mathbf{b}|n) / \sum_m \pi_m P(\mathbf{b}|m)$
  by Bayes' theorem, computed as a softmax of the log-likelihoods (uniform $\pi$ unless given).
  Averaging it over shots is a photon-number estimate needing **no unfolding at all** — worth
  having as an independent cross-check precisely because it is not exposed to the conditioning of
  any matrix.

---

## §9 Unfolding

### 9.1 Why a linear correction is legitimate

> **Theorem 9.1.** Let $T$ be any statistic of the record — the click count $C(\mathbf{b}) =
> \sum_r b_r$, or the estimate $\hat n(\mathbf{b})$, or anything else. Define
> $M[t, n] = P\big(T = t \mid n\big)$ and $q[t] = P(T = t)$. If the prep leaves the cavity with
> populations $p$, then
> $$\boxed{\ q = M p\ }$$
> exactly.

**Proof.** $P(T = t) = \sum_{\mathbf{b}: T(\mathbf{b}) = t} P(\mathbf{b})$, and by Theorem 5.1
$P(\mathbf{b}) = \sum_n p_n P(\mathbf{b}|n)$. Exchange the finite sums. $\square$

This is the whole justification for confusion-matrix correction: it is *not* an approximation and
does *not* assume small errors. It needs only Theorem 5.1, i.e. that the record depends on the
cavity through its populations alone.

> **Theorem 9.2 (columns sum to one).** $\sum_t M[t,n] = 1$ for every $n$, provided the rows of
> $M$ enumerate **every** value $T$ can take.

**Proof.** $P(\cdot \mid n)$ is a probability distribution on the range of $T$. $\square$

Theorem 9.2 is why the click-count matrix in the runtime is **rectangular**, $(R{+}1)\times
(N_{\rm fock})$: the click count ranges over $0 \ldots R$ while the prepared levels are only
$0\ldots N_{\rm fock}-1$, and with a descent plus mop-up rounds $R > N_{\rm fock}-1$. Truncating
$M$ to a square would discard exactly the rows carrying the overcounting information *and* break
Theorem 9.2. The Markov matrix $M'[\hat n, n]$ has no such problem: both of its axes are photon
number, so it is naturally square.

### 9.2 Inverting it

For rectangular $M$ the inverse does not exist and one solves

$$\hat p_{\rm LS} = \arg\min_p \|Mp - q\|_2 \;=\; M^{+} q, \qquad M^+ = \text{Moore–Penrose
pseudoinverse}.$$

> **Theorem 9.3.** $M^+q$ is the minimum-$\ell_2$-norm minimiser of $\|Mp-q\|_2$.

**Proof.** Standard: the normal equations $M^\top M p = M^\top q$ characterise the minimisers;
writing $M = U\Sigma V^\top$ and $M^+ = V\Sigma^+U^\top$ verifies that $M^+q$ satisfies them and
lies in $\mathrm{row}(M)$, which is orthogonal to the null space along which the solution set
extends. $\square$

`unfold_distribution` returns three things:

- $M^+q$: unbiased, but not constrained to be a probability vector and so may go slightly
  negative;
- $\mathrm{nnls}(M,q)$ renormalised: the same fit subject to $p \ge 0$, hence always a
  distribution, at the cost of a small bias at the boundary;
- $\|M\hat p - q\|$: the residual. For square invertible $M'$ this is $\approx 0$ by
  construction and carries no information. For the **rectangular** $M$ it is a genuine
  goodness-of-fit: it is the distance from $q$ to the column space of $M$, i.e. how well *any*
  distribution over the prepared levels can explain the observed histogram. A residual well above
  $\sqrt{1/4N_{\rm shots}}$ says the state has support outside the prepared levels, or the counter
  is doing something $M$ does not describe.

Error propagation: $q$ is a multinomial estimate with covariance
$\Sigma_q = (\mathrm{diag}(q) - qq^\top)/N_{\rm shots}$, so
$\Sigma_{\hat p} = M^+\Sigma_q (M^+)^\top$, and the code reports its diagonal.

**A caveat about which $M$ you have.** The columns of $M$ are calibrated by *preparing* Fock
states, and a real ladder-climb prep is imperfect. The measured $M$ is therefore
$M^{\rm true} P^{\rm prep}$ for some prep-error matrix $P^{\rm prep}$ — calibration "B" in the
notebook's language, the only one a laboratory has. §8 and §10 of the notebook quantify the
resulting bias. Neither reading of the record fixes it.

---

## §10 What the click count throws away, and when it throws away nothing

The click count is the map $C(\mathbf{b}) = \sum_r b_r$. It is a *deterministic function of the
record*, i.e. a coarse-graining. That single observation settles the qualitative question.

> **Theorem 10.1.** Let $N$ be the (random) initial photon number, $\mathbf{B}$ the record,
> $C = C(\mathbf{B})$. Then
> $$I(N; C) \;\le\; I(N; \mathbf{B}),$$
> with $I$ the Shannon mutual information: the click count can never be more informative about
> the photon number than the record, and every estimator based on $C$ is matched or beaten by
> some estimator based on $\mathbf{B}$.

**Proof.** $N \to \mathbf{B} \to C$ is a Markov chain, because $C$ is a function of $\mathbf{B}$
alone. The data-processing inequality gives the bound. The estimator statement follows because
any $C$-measurable estimator is in particular $\mathbf{B}$-measurable. $\square$

> **Theorem 10.2 (equality condition).** $I(N;C) = I(N;\mathbf{B})$ if and only if $C$ is a
> **sufficient statistic** for $N$, i.e. $P(\mathbf{b}\mid n)$ depends on $\mathbf{b}$ only
> through $C(\mathbf{b})$ — equivalently $P(N \mid \mathbf{B}) = P(N\mid C)$ almost surely.

**Proof.** $I(N;\mathbf{B}) = I(N;C) + I(N;\mathbf{B}\mid C)$ by the chain rule (using
$I(N;C \mid \mathbf{B}) = 0$). The conditional mutual information vanishes iff $N$ and
$\mathbf{B}$ are conditionally independent given $C$, which is the stated sufficiency. $\square$

Now the sharp result, which is more interesting than it first looks.

> **Theorem 10.3 (in the ideal limit, the click count loses nothing).** Suppose the swaps are
> perfect ($\eta \equiv 1$, exact $t_n$), the readout is perfect ($\varepsilon_g = \varepsilon_e
> = 0$), the reset is perfect, and there is no decay or heating. Then for the descending schedule
> with $N_{\max} \ge N$, $C = N$ with probability 1, hence $I(N;C) = I(N;\mathbf{B}) = H(N)$ and
> the ordering of the bits carries **no** information about $N$.

**Proof.** First, the photon number never increases: the swap moves at most one photon out of
the cavity per round, and the perfect reset returns the qubit to $|g\rangle$ without touching the
cavity, so with no heating there is no channel that adds a photon.

Second, the cavity is empty at the end. Let $m_r$ be the photon number just before the round
tuned to level $m$. Claim: after that round, the cavity holds at most $m-1$ photons. Two cases.
If the cavity holds exactly $m$, then by Lemma 6.2 that round is a *full* swap on level $m$
($\theta_m = \pi$, $p_m = 1$), so the photon is moved out with certainty and the cavity holds
$m-1$. If it holds fewer than $m$, it holds at most $m-1$ already, and the previous paragraph says
it cannot grow. The descent visits $m = N_{\max}, N_{\max}-1, \ldots, 1$ in that order, so
applying the claim down the ladder leaves at most $0$ photons after the round tuned to level 1.

Third, each removed photon produced exactly one click: the swap deposits it on the qubit, and a
perfect readout reports $|e\rangle$. Hence $C = n$ with probability 1 — note that *which* rounds
clicked is still random, since a round tuned to level $m > n$ has
$p_n = \sin^2\!\big(\tfrac\pi2\sqrt{n/m}\big) \notin \{0,1\}$ and may move the photon early.

Finally $C = N$ a.s. gives $H(N\mid C) = 0$, so
$I(N;\mathbf{B}\mid C) \le H(N \mid C) = 0$, and Theorem 10.2 applies. $\square$

This is worth dwelling on, because it is easy to get backwards. **In the noiseless limit the
record is still random** — only $|N_{\max}\rangle$ gives a deterministic string, since for
$n < N_{\max}$ the earlier rounds (tuned to levels above $n$) partially swap $|n\rangle$ and the
click can land anywhere. The notebook measures this: preparing $|1\rangle$ losslessly gives 7
distinct records out of 64 shots, while $|8\rangle$ gives 1 out of 64. But that randomness is
*independent of $n$ given $C$*, so it is noise, not signal. **The ordering becomes informative
only in the presence of errors.**

### Which errors, though?

Theorems 10.1–10.3 say the ordering helps only through errors, but not which ones. The
simulation in `jc_counting_confusion_matrix.ipynb` §12 answers that empirically, drawing records
from the *full* Lindblad master equation — including cavity dephasing and qubit $T_2$, neither of
which the model contains, so the test is not circular. Two sweeps, and they point opposite ways.

**Cavity loss (removes clicks):**

```
cavity T1 [us]   rounds/T1   click count   Markov    gain
        1000         0.02       0.9131     0.9631   +0.0500
         300         0.06       0.8822     0.9288   +0.0465
         100         0.17       0.8019     0.8403   +0.0383
          50         0.33       0.7010     0.7375   +0.0365
          30         0.55       0.6128     0.6479   +0.0351
          20         0.83       0.5183     0.5708   +0.0525
```

(The last row turns back up. At 0.83 rounds per $T_1$ both readings are down near 0.5 and the
counter has largely stopped working; read that row as noise on a broken measurement, not as a
trend reversing.)

**Qubit heating (adds clicks):**

```
P(heat)/round    click count   Markov    gain
        0.00        0.8746     0.9256   +0.0510
        0.01        0.7937     0.8594   +0.0657
        0.02        0.7192     0.7971   +0.0779
        0.05        0.5264     0.6386   +0.1122
        0.10        0.3189     0.4649   +0.1460
```

The gain **falls** with cavity loss and grows by a factor of three with heating (+0.051 to +0.146). Note that at $p_{\rm heat} = 0.10$ the click count has collapsed to 0.319 while the Markov reading still returns 0.465 — a 46% relative improvement on a measurement the click count can barely make. The explanation is
one sentence:

> A photon that has already left is not *ambiguous* evidence, it is *absent* evidence.

Cavity loss destroys information rather than scrambling it; $I(N;\mathbf{B})$ itself falls, and
no reading recovers what is not there. A spurious click is different: the photon-number
information is still in the record, with an artifact added *in a round the physics makes
implausible*. That is scrambling, and it is exactly what a chain that knows the round-by-round
transfer probabilities can undo. Consistently, the per-level gain in the simulation is largest at
**low** $n$ ($0.900 \to 0.960$ at $n = 0$), where a spurious click does the most damage relative
to the true signal.

**Rule of thumb, then:** the Markov reading pays for itself against errors that *add* clicks —
readout misassignment, a mistuned reset $\pi$, a hot qubit, a long schedule with many mop-up
rounds. It does not pay against a lossy cavity; for that, shorten the schedule.

---

## §11 Measured results

### On hardware

Run `260820/154655`, Q1/C1: 3385 iterations $\times$ 10 preps $\times$ 13 rounds; Fock
$|0\rangle\ldots|8\rangle$ plus one displaced test state. Model coefficients: qubit $T_1$ = 150
µs, cavity $T_1$ = 900 µs, qubit thermal population 0.006 (measured ~0.6%), cavity thermal
population 0.01 (an upper bound), readout infidelity 0.008 each way, reset $\pi$ error 0.02,
residual contrast 0.98, $\sqrt n$ level times. (The answer is insensitive to the thermal
populations at this level: moving the qubit's from 0.01 to 0.006 changes the mean fidelity by
$10^{-4}$.)

| | click count | Markov |
|---|---|---|
| mean $P$(right answer), 9 Fock columns | 0.9019 | **0.9086** |
| mean $\vert\langle\hat n\rangle - n\vert$ | **0.065** | 0.089 |
| confusion matrix | $14\times9$, cond 1.40 | $9\times9$, cond **1.33** |

```
n                0      1      2      3      4      5      6      7      8
diag(M)  click   0.966  0.976  0.952  0.929  0.909  0.884  0.873  0.828  0.801
diag(M') Markov  0.968  0.976  0.951  0.926  0.910  0.889  0.873  0.858  0.826
```

The gain is $+0.007$ on average and **all of it is at the top of the ladder** ($|7\rangle$:
$0.828 \to 0.858$; $|8\rangle$: $0.801 \to 0.826$), which is where the $\kappa n$ enhancement of
Theorem 6.3 lives. It is small because this counter is already clean: the mean click count tracks
$n$ to a few parts in a thousand ($0.04, 1.008, 2.003, 3.000, 3.977, 4.943, 5.931, 6.876,
7.737$), so the measured records almost never visit the strings where the two readings disagree —
even though 11 of the 14 click counts are ambiguous *in principle*.

### The bit-string map

With $R$ rounds there are $2^R$ possible records, and `bitstring_photon_map` classifies every one
of them with no data at all. That table is the transfer function of the measurement, and it makes
Theorem 10.1 concrete: wherever two records with the same $C$ receive different $\hat n$, the
click count is merging measurements that are not the same. From the hardware run:

```
click count c    0    1    2    3    4    5    6    7    8    9   10   11   12   13
mean n̂         0.0  0.62 1.17 1.81 3.23 4.15 4.80 5.44 6.26 6.90 7.45 7.81 8.0  8.0
```

Not $\hat n = c$. For $c \le 3$ the mean assignment sits *below* $c$: among all records with
three clicks, most are patterns that decay and heating explain better than three photons. The
model is systematically less credulous than the naive rule.

### Classification fidelity is not unfolded accuracy

This is the result most likely to be misread, so it gets its own statement. On the simulated
device the Markov diagonal beats the click diagonal by a wide margin, $0.9285$ against $0.8831$
averaged over nine Fock columns — and yet after unfolding a coherent test state through each
reading's *own* matrix, the click count comes out closer to the truth (total variation $0.0097$
against $0.0145$; the mean posterior is worst at $0.0307$).

There is no contradiction, and Theorem 9.1 explains it: **the confusion matrix already removes
the systematic**, exactly, for either reading. Both inversions are consistent by construction, so
both land near the truth. What a higher diagonal actually buys is

- a better-conditioned matrix (1.25 vs 1.30 simulated, 1.33 vs 1.40 on hardware), hence less
  shot noise amplified through $M^+$ — see the covariance formula in §9.2; and
- less exposure to the calibration itself being wrong, because there is less off-diagonal weight
  for a prep error to hide in.

Both are worth having. Neither is "more accurate". And the mean posterior is the worst of the
three on total variation for the mirror-image reason: it keeps its tails instead of committing to
an integer, which makes it the right thing to inspect when you want to know *how confident* the
classification was, and the wrong thing to quote as a distribution.

### What to do with the two readings

Report both; treat the spread as a systematic. If they agree, the answer does not depend on the
model. If they disagree, the disagreement is the interesting number — and every coefficient
driving it is an argument of `process_current_data`, which `acadia_gui` turns into a live control,
so you can find out which one the answer hangs on without redeploying.

---

## §12 Assumptions, in one list

Exactly these, and nothing else, are what the derivation rests on.

**A1.** The qubit measurement is projective in $\{|g\rangle,|e\rangle\}$ and is performed every
round. *(Lemma 4.3, hence Theorem 4.6 and the absence of $T_2$.)*

**A2.** The drive Hamiltonian commutes with $\mathcal{N}$. *(Lemma 4.1. A drive that is not
sideband-resonant — e.g. a two-photon or counter-rotating term — would break this and the
reduction with it.)*

**A3.** Every Lindblad operator maps basis vectors to multiples of basis vectors: single-photon
loss and gain, qubit relaxation and excitation, and diagonal dephasing. *(Lemma 4.2. Correlated
qubit–cavity jumps would break this.)*

**A4.** The feedback is a $\pi$ pulse conditioned on the recorded bit. *(Lemma 4.4, Definition
2.3.)*

**A5.** The initial state is $|g\rangle\langle g| \otimes \rho_{\rm cav}$. *(Theorem 5.1. Initial
qubit–cavity correlation would put weight in $\Delta \ne 0$ sectors that Step 1 discards, and the
theorem would fail.)*

**A6.** The cavity truncation $N$ is high enough that population reaching $|e,N\rangle$ is
negligible. *(§4, remark on inert corners.)*

**A7.** Readout misassignment is independent between rounds and independent of $n$. *(§6.4. A
dispersive-shift-dependent readout fidelity violates the second half.)*

**A8.** The Strang splitting of Proposition 7.1 is accurate at the level required —
$t_r \ll T_1$. *(The only numerical approximation in the whole construction, and removable per
Remark 7.2.)*

**A9.** Population that fails to swap remains in the cavity rather than leaking out of the
counted subspace. *(§6.1. This is the assumption most likely to be wrong at the top of the
ladder, and the one to interrogate first if the two readings disagree there.)*

---

## Appendix: what the code checks

`checks/verify_on_real_data.py` asserts, on the hardware run above:

- every $A_r$ is row-stochastic and non-negative *(Corollary 6.4, Theorem 4.6)*;
- $E$ and both $B_b$ are row-stochastic *(§6.3, §6.4)*;
- a round tuned to level $L$ transfers $> 0.999$ of $|L\rangle$ *(Lemma 6.2)*;
- with no decay and a perfect readout, the chain recovers every ideal record exactly, $\hat n = n$
  for all $n$ *(Theorem 10.3)*;
- every column of $M$ and $M'$ sums to 1 *(Theorem 9.2)*;
- $M'$ inverts its own columns exactly *(Theorem 9.3)*;
- `enumerate_bitstrings` and `bitstring_index` are mutual inverses;
- the all-zero record is assigned $\hat n = 0$, and the mean assignment increases with $C$.
