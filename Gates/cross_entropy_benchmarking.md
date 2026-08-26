# Cross-Entropy Benchmarking: Complete Derivation

**Source:** Boixo *et al.*, "Characterizing quantum supremacy in near-term devices," *Nature Physics* **14**, 595 (2018), and its Supplementary Information §§I–II.

**Scope.** Only the fidelity-estimation logic. Nothing about classical simulation cost, treewidth, or the supremacy threshold.

**Conventions.**
- Every symbol is defined where it first appears or earlier, never later.
- Boxes marked **▸ Small $N$** carry the exact results for $n=1,2,3$ qubits alongside the large-$N$ statements. These are needed for §13, where XEB is applied to a single non-Clifford gate.
- Numerical claims are verified by Monte Carlo in the Appendix.

---

## Contents

0. [Notation](#0-notation)
1. [The problem and its structure](#1-the-problem-and-its-structure)
2. [The circuit ensemble](#2-the-circuit-ensemble)
3. [The Porter–Thomas distribution](#3-the-porterthomas-distribution)
4. [Two distinct uses of PT, and why (a) is true](#4-two-distinct-uses-of-pt-and-why-a-is-true)
5. [Certifying PT numerically](#5-certifying-pt-numerically)
6. [Constructing the score function; proof that α = F](#6-constructing-the-score-function-proof-that-α--f)
7. [The estimator](#7-the-estimator)
8. [Variance and sample complexity](#8-variance-and-sample-complexity)
9. [The full one-shot distribution](#9-the-full-one-shot-distribution)
10. [On the positivity of σ_U](#10-on-the-positivity-of-σ_u)
11. [Assumption audit](#11-assumption-audit)
12. [Summary: what changes at small N](#12-summary-what-changes-at-small-n)
13. [Interleaved XEB: measuring a non-Clifford gate against a reference](#13-interleaved-xeb-measuring-a-non-clifford-gate-against-a-reference)
14. [Log-XEB versus linear XEB](#14-log-xeb-versus-linear-xeb)
15. [Appendix: numerical verification](#15-appendix-numerical-verification)

---

## 0. Notation

| symbol | meaning |
|---|---|
| $n$ | number of qubits |
| $N \equiv 2^n$ | Hilbert space dimension |
| $\log$ | natural logarithm (never base 2) |
| $x$ | an $n$-bit string; $\lvert x\rangle$ the corresponding computational-basis state |
| $U$ | unitary implemented by the ideal circuit, of depth $d$ |
| $\lvert\psi_0\rangle=\lvert 0\rangle^{\otimes n}$ | input state |
| $\lvert\psi_d\rangle \equiv U\lvert\psi_0\rangle$ | ideal output state |
| $p_U(x)\equiv\lvert\langle x\vert\psi_d\rangle\rvert^2$ | ideal output probability; classically computable at cost exponential in $n$ |
| $\rho$ | state the device actually produces |
| $p_{\exp}(x)\equiv\langle x\rvert\rho\lvert x\rangle$ | probability the device outputs $x$ |
| $F\equiv\langle\psi_d\rvert\rho\lvert\psi_d\rangle$ | fidelity — the target quantity |
| $m$ | number of shots |
| $x_1^{\exp},\dots,x_m^{\exp}$ | measured bit-strings, i.i.d. from $p_{\exp}$ |
| $\gamma\approx0.577216$ | Euler–Mascheroni constant |
| $\mathbb{E}_U[\cdot]$ | average over the circuit ensemble of §2 |
| $\mathbb{E}[\cdot]$ | average over measurement outcomes at fixed circuit |
| $\psi(s)=\Gamma'(s)/\Gamma(s)$ | digamma function |
| $\mathcal H_m \equiv \sum_{k=1}^{m}1/k$ | harmonic number |
| $\mathcal H^{(2)}_m \equiv \sum_{k=1}^{m}1/k^2$ | second-order harmonic number |

---

## 1. The problem and its structure

### 1.1 What must be measured, and why the obvious routes fail

The device produces $\rho$; we want $F=\langle\psi_d\rvert\rho\lvert\psi_d\rangle$.

- **Full tomography** of $\rho$ needs $\sim 4^n$ measurement settings — excluded at $n\sim50$.
- **Randomized benchmarking** characterizes a *group* (Clifford), not the specific chaotic non-Clifford evolution $U$.
- Existing alternatives either append an expensive extra unitary to the circuit or apply only to non-universal circuits.

Wanted: an estimate of $F$ from computational-basis samples of $\rho$ alone, with classical help limited to computing $p_U$ at the observed strings.

### 1.2 The class of estimators available

The experiment yields only bit-strings, so any post-processing is a function of those strings. Restrict to the simplest useful class — *assign a number to each observed string, then average*:

$$\text{choose a }\textbf{score function }o_U:\{0,1\}^n\to\mathbb{R},\qquad
\hat\alpha \equiv \frac1m\sum_{j=1}^{m}o_U\!\left(x_j^{\exp}\right).\tag{1.1}$$

Three objects; each defined now.

**$o_U(x)$ — the score function.** A classical real-valued function of a bit-string: *if the device outputs $x$, write down $o_U(x)$.* This is what we design; constructing XEB **is** choosing $o_U$. Two design constraints: it may depend on $U$ and on anything classically computable from $U$ (hence the subscript) but **not** on $\rho$, which is unknown; and it must be evaluable in practice at the observed strings, since $m$ of its values must be computed.

**$\hat\alpha$ — the estimator.** A number computed from finite data, hence itself a random variable: repeated runs give different $\hat\alpha$. The hat marks "empirical, from $m$ shots." Its expectation over measurement outcomes is a property of $\rho$, not of the run:

$$\alpha \equiv \mathbb{E}[\hat\alpha]=\sum_x p_{\exp}(x)\,o_U(x)=\mathrm{Tr}(\rho\,O_U),\tag{1.2}$$

where the last equality defines the third object.

**$O_U$ — the observable.**

$$O_U \equiv \sum_x o_U(x)\,\lvert x\rangle\langle x\rvert.\tag{1.3}$$

Hermitian ($o_U$ real), diagonal in the computational basis, with eigenvalue $o_U(x)$ on $\lvert x\rangle$. Equation (1.3) adds no content beyond (1.1)–(1.2); it is the operator repackaging of "average a score over basis samples." Its value is that it makes linearity in $\rho$ visible. By the law of large numbers $\hat\alpha\to\alpha$ with fluctuation $O(m^{-1/2})$; the coefficient is computed in §8.

**Diagonal observables are exactly the right class, not a restriction.** A computational-basis measurement returns only $x$, so any per-shot score is a function $o(x)$ and its mean is $\sum_x p_{\exp}(x)o(x)=\mathrm{Tr}(\rho O)$ with $O$ diagonal; conversely every diagonal $O$ arises from some score. Hence

$$\{\text{estimable by averaging a per-shot score}\}=\{\text{expectations of computational-basis-diagonal observables}\}.$$

We now search that entire class for a member whose expectation is $F$.

### 1.3 Why $F$ lies in that class: linearity

$$F=\langle\psi_d\rvert\rho\lvert\psi_d\rangle\ \text{ is \textbf{linear} in }\rho.$$

This is the enabling structural fact. Contrast the total-variation distance $\tfrac12\sum_x\lvert p_{\exp}(x)-p_U(x)\rvert$: not linear in $\rho$, no unbiased per-shot estimator, needs exponentially many samples. Linearity means (i) we can hope to hit $F$ exactly with one expectation value, and (ii) the conditions on $o_U$ decouple into two.

### 1.4 The two conditions on $o_U$

Split $\rho$ into ideal part and remainder:

$$\rho=F\,\lvert\psi_d\rangle\langle\psi_d\rvert+(1-F)\,\sigma_U,\qquad
\langle\psi_d\rvert\sigma_U\lvert\psi_d\rangle=0,\tag{1.4}$$

where $\sigma_U$ is the normalized **error component**. Equation (1.4) exists for every $\rho$ with no assumptions: set $\sigma_U=(\rho-F\lvert\psi_d\rangle\langle\psi_d\rvert)/(1-F)$, Hermitian, unit trace, orthogonal to $\lvert\psi_d\rangle$ by construction. The orthogonality forces the coefficient of the ideal projector to be exactly $F$: sandwiching (1.4) gives $F+(1-F)\cdot0=F$ ✓.

Two facts about $\sigma_U$, both used later:

- $\sigma_U$ need **not** be positive; see §10 for exactly when it is. Nothing in the derivation of $\alpha=F$ uses positivity — only unit trace, in the form $\sum_x p_\sigma(x)=1$, where

  $$p_\sigma(x)\equiv\langle x\rvert\sigma_U\lvert x\rangle\tag{1.5}$$

  is the (possibly signed) diagonal of the error component.
- Sandwiching (1.4) with $\langle x\rvert\cdots\lvert x\rangle$:

  $$p_{\exp}(x)=F\,p_U(x)+(1-F)\,p_\sigma(x).\tag{1.6}$$

Insert (1.4) into (1.2), use linearity of the trace:

$$\alpha=F\,\langle\psi_d\rvert O_U\lvert\psi_d\rangle+(1-F)\,\mathrm{Tr}(\sigma_U O_U).$$

Therefore $\alpha=F$ **identically in the unknown $F$** — not merely at one value — if and only if

$$\textbf{(C1)}\quad \langle\psi_d\rvert O_U\lvert\psi_d\rangle=\sum_x p_U(x)\,o_U(x)=1,
\qquad
\textbf{(C2)}\quad \mathrm{Tr}(\sigma_U O_U)=\sum_x p_\sigma(x)\,o_U(x)=0 .$$

In words: **the score must average to $1$ on ideal output and to $0$ on error output.** A two-point calibration; linearity interpolates and reads off $F$ automatically. That is the entire logical content of cross-entropy benchmarking.

> **▸ Small $N$ — preview.** Nothing above used $n$. The skeleton is dimension-free and holds at $N=2$. What changes at small $N$ is only the *values* of (C1) and (C2): they come out $1-1/N$ and $-1/N$ instead of $1$ and $0$. Derived exactly in §6.4.

### 1.5 Three gaps, and the order of the argument

1. **(C2) refers to $\sigma_U$, which is unknown.** It can be enforced only on ensemble average, and only under a physical assumption about errors in chaotic circuits (Assumption A, §6.2). Hence the theorem is an ensemble statement; single-instance deviations are $O(2^{-n/2})$.
2. **(C1) and (C2) require evaluating $\sum_x f(p_U(x))$** over all $N$ strings, and (C1) requires the *value*. Term by term: impossible. Statistically: easy, because for deep random circuits the histogram of the $N$ numbers $\{p_U(x)\}$ is known in closed form — the **Porter–Thomas (PT) distribution**.
3. **Not every $o_U$ obeying (C1)–(C2) is usable.** If $o_U$ were a fixed function independent of $U$, Levy's lemma would make $\hat\alpha$ concentrate exponentially on its Hilbert-space average, carrying almost no information about $\rho$ and demanding $m\sim2^n$ shots. Usability requires $o_U$ to be *state-specific* — built from $p_U$ itself — with $O(1)$ per-shot variance (§8.3).

Hence the order: PT (§3–5), then the construction and proofs (§6), then estimator and variance (§7–9).

**Paper-notation bridge.** The paper writes $\alpha$ for the cross-entropy difference (my $\alpha$ after the choice of $o_U$ in §6) and $\alpha_f$ for the circuit fidelity (my $F$). Its central claim $\alpha\approx\alpha_f$ is my Eq. (6.9), which is nothing but (C1)+(C2)+linearity.

---

## 2. The circuit ensemble

Ideal preparation:

1. $\lvert0\rangle^{\otimes n}$.
2. Hadamard on every qubit. Necessary because the only two-qubit gate is CZ, which is diagonal and would act trivially on $\lvert0\cdots0\rangle$; the Hadamard layer rotates to the $X$ basis.
3. $d$ cycles. Each cycle applies one of $8$ fixed CZ patterns on the 2D lattice (iterated $1\to8$), plus one-qubit gates from $\{X^{1/2},Y^{1/2},T\}$ on qubits not occupied by a CZ that cycle. $X^{1/2},Y^{1/2}$ are $\pi/2$ Bloch rotations; $T=\mathrm{diag}(1,e^{i\pi/4})$ is the only non-Clifford gate. Eight patterns are needed because two neighbouring superconducting qubits cannot take simultaneous CZs.

Placement rules: the first one-qubit gate on a qubit after the Hadamard layer is always $T$; a one-qubit gate is placed only in the cycle immediately after a CZ on that qubit; if the qubit has already seen a $T$, the gate is drawn uniformly from the two gates *different from the last one applied to it*.

These remove accidental structure and maximize scrambling rate ($T$'s injected immediately so the circuit is non-Clifford from the start; no gate repeats). $\mathbb{E}_U$ averages over this randomness. **The only property of the ensemble used downstream is PT convergence.**

---

## 3. The Porter–Thomas distribution

### 3.1 Exact result for a Haar-random state (Suppl. §I)

**Claim.** Let $\lvert\psi\rangle$ be Haar-uniform in $\mathbb{C}^N$, fix a basis state $\lvert x_0\rangle$, and let $p=\lvert\langle x_0\rvert\psi\rangle\rvert^2$. Then

$$\Pr(p)=(N-1)(1-p)^{N-2},\qquad p\in[0,1].\tag{3.1}$$

**Derivation.** Write $\lvert\psi\rangle=\sum_x(a_x+ib_x)\lvert x\rangle$ with $a_x,b_x\in\mathbb{R}$. Haar measure on pure states is the uniform measure on the unit sphere $S^{2N-1}\subset\mathbb{R}^{2N}$, i.e. the flat measure $\prod_x da_x\,db_x$ subject only to normalization. Hence

$$\Pr(p)=\frac{\displaystyle\int\prod_x da_xdb_x\;\delta\!\Big(\sum_x(a_x^2+b_x^2)-1\Big)\,\delta\!\big(a_{x_0}^2+b_{x_0}^2-p\big)}
{\displaystyle\int\prod_x da_xdb_x\;\delta\!\Big(\sum_x(a_x^2+b_x^2)-1\Big)}\equiv\frac{\mathcal N}{\mathcal D}.$$

Represent each delta as $\delta(u)=\frac1{2\pi}\int dt\,e^{itu}$, with conjugate variables $t$ (normalization) and $w$ (the $x_0$ constraint), each shifted $t\to t+i\epsilon$ so the Gaussians converge. All $2N$ real integrals factorize. Each mode $x\neq x_0$:

$$\int da\,db\;e^{it(a^2+b^2)}=\Big(\sqrt{\tfrac{\pi}{-it}}\Big)^2=\frac{\pi}{-it},$$

while the mode $x_0$ has conjugate variable $t+w$, giving $\pi/(-i(t+w))$. So

$$\mathcal N=\frac1{2\pi}\int dt\,e^{-it}\Big(\frac{\pi}{-it}\Big)^{N-1}\cdot\frac1{2\pi}\int dw\,e^{-iwp}\frac{\pi}{-i(t+w)}.$$

Do $w$ first: shift $w'=w+t$, giving $\pi e^{itp}\cdot\frac1{2\pi}\int dw'\frac{e^{-iw'p}}{-iw'}$. Contour above the pole at $w'=0$; $p>0$ so close in the lower half-plane, residue $i$, bracket $=\frac1{2\pi}(-2\pi i)(i)=1$. The $w$ integral is just $\pi e^{itp}$. Using $(-it)^{-(N-1)}=i^{N-1}t^{-(N-1)}$,

$$\mathcal N=\pi^N i^{N-1}\frac1{2\pi}\int_{-\infty+i\epsilon}^{\infty+i\epsilon}dt\;\frac{e^{-it(1-p)}}{t^{N-1}}.$$

Order-$(N-1)$ pole at $t=0$, contour above it, $1-p>0$, close downward:

$$\frac1{2\pi}\int dt\,\frac{e^{-it(1-p)}}{t^{N-1}}
=-i\,\frac1{(N-2)!}\lim_{t\to0}\frac{d^{N-2}}{dt^{N-2}}e^{-it(1-p)}
=-i\,\frac{(-i)^{N-2}(1-p)^{N-2}}{(N-2)!}.$$

Since $i^{N-1}(-i)^{N-1}=1$, $\ \mathcal N=\pi^N(1-p)^{N-2}/(N-2)!$. The denominator is the same computation without the $x_0$ constraint:

$$\mathcal D=\frac1{2\pi}\int dt\,e^{-it}\Big(\frac{\pi}{-it}\Big)^{N}=\frac{\pi^N}{(N-1)!}.$$

Dividing gives (3.1): $p\sim\mathrm{Beta}(1,N-1)$. Checks: normalized; $\langle p\rangle=1/N$ as forced by $\sum_xp_U(x)=1$ plus permutation symmetry.

**Large-$N$ limit.** With the **rescaled probability** $u\equiv Np$: $(1-u/N)^{N-2}\to e^{-u}$ and $(N-1)/N\to1$, so

$$\boxed{\;\Pr(p)=Ne^{-Np}\;}\qquad\Longleftrightarrow\qquad u=Np\sim\mathrm{Exp}(1).\tag{3.2}$$

This is Porter–Thomas, originally the distribution of nuclear reaction widths, a standard signature of quantum chaos.

### 3.2 One-line rederivation

A Haar state is a normalized i.i.d. complex Gaussian vector: $\langle x\rvert\psi\rangle=g_x/\lVert g\rVert$ with $g_x\sim\mathcal{CN}(0,1)$ independent. Then $p=\lvert g_{x_0}\rvert^2/\sum_x\lvert g_x\rvert^2$ — one exponential over a sum of $N$ of them — which is $\mathrm{Beta}(1,N-1)$ exactly; at large $N$ the denominator concentrates at $N$, giving $p\approx\mathrm{Exp}(1)/N$.

This picture also shows that $p_U(x_j)$ and $p_U(x_k)$, $j\neq k$, are nearly independent: the only coupling is the global normalization, so correlations are $O(1/N)$. Needed in §4 and §8.

### 3.3 Exact constants at any $N$

These are the finite-$N$ forms of everything used below. From $\mathrm{Beta}(1,N-1)$, using $E[\ln X]=\psi(a)-\psi(a+b)$ and $\psi(m+1)=-\gamma+\mathcal H_m$:

$$\langle\log p\rangle=\psi(1)-\psi(N)=-\mathcal H_{N-1}
=-\Big(\log N+\gamma-\tfrac1{2N}+O(N^{-2})\Big).\tag{3.3}$$

Using $E[Xf(X)]=\tfrac1N E[f(Y)]$ with $Y\sim\mathrm{Beta}(2,N-1)$:

$$-N\langle p\log p\rangle=\psi(N+1)-\psi(2)=\mathcal H_N-1
=\log N-1+\gamma+\tfrac1{2N}+O(N^{-2}).\tag{3.4}$$

Second moments, from $\mathrm{Var}(\ln X)=\psi'(a)-\psi'(a+b)$:

$$\langle\log^2p\rangle=\mathcal H_{N-1}^2+\mathcal H^{(2)}_{N-1},
\qquad
N\langle p\log^2p\rangle=(\mathcal H_N-1)^2+\mathcal H^{(2)}_N-1.\tag{3.5}$$

**Every PT result below is exact up to relative $O(1/N)=O(2^{-n})$.** This matters because every XEB quantity is an $O(1)$ difference of $O(\log N)$ numbers; one needs the neglected terms to be exponentially small, not $O(1)$. (The supplement's "$+O(1)$" in its Eq. (3) is loose notation.)

### 3.4 PT integral toolkit (large $N$)

Substituting $u=Np$ and using $\int_0^\infty u^{s-1}e^{-u}du=\Gamma(s)$, with $\psi(1)=-\gamma$, $\psi(2)=1-\gamma$, $\psi'(1)=\pi^2/6$, $\psi'(2)=\pi^2/6-1$:

| $\langle\cdot\rangle_{\rm PT}$ | value | exact finite-$N$ |
|---|---|---|
| $\langle p^k\rangle$ | $k!/N^k$ | $\dfrac{k!\,(N-1)!}{(N+k-1)!}$ |
| $\langle\log p\rangle$ | $-(\log N+\gamma)$ | $-\mathcal H_{N-1}$ |
| $\langle p\log p\rangle$ | $-\frac1N(\log N-1+\gamma)$ | $-\frac1N(\mathcal H_N-1)$ |
| $\langle\log^2p\rangle$ | $(\log N+\gamma)^2+\pi^2/6$ | $\mathcal H_{N-1}^2+\mathcal H^{(2)}_{N-1}$ |
| $\langle p\log^2p\rangle$ | $\frac1N[(\log N-1+\gamma)^2+\pi^2/6-1]$ | $\frac1N[(\mathcal H_N-1)^2+\mathcal H^{(2)}_N-1]$ |

$\gamma$ enters through $\int_0^\infty e^{-u}\log u\,du=\Gamma'(1)=-\gamma$; $\pi^2/6$ through $\psi'(1)=\zeta(2)$.

> **▸ Small $N$ — the exact one-point law.**
>
> | $n$ | $N$ | law of $p$ | $\mathcal H_{N-1}$ | $\log N+\gamma$ | $\mathcal H_N-1$ |
> |---|---|---|---|---|---|
> | 1 | 2 | $\mathrm{Unif}[0,1]$ | 1.00000 | 1.27036 | 0.50000 |
> | 2 | 4 | $3(1-p)^2$ | 1.83333 | 1.96351 | 1.08333 |
> | 3 | 8 | $7(1-p)^6$ | 2.59286 | 2.65666 | 1.71786 |
> | 10 | 1024 | — | 7.50820 | 7.50869 | 6.50918 |
>
> At $n=1$ the amplitude distribution is **uniform**, not exponential — PT in the sense of (3.2) is a large-$N$ statement. But (3.1) is exact at every $N$, so nothing is lost: just use the right-hand column of the table in §3.4 throughout.

---

## 4. Two distinct uses of PT, and why (a) is true

There are two logically different statements, both loosely called "PT." Conflating them is the single easiest way to get lost.

**(b) Over $U$, at fixed $x$.** For a fixed string $x$, the number $p_U(x)$ is a random variable over the circuit ensemble, and it is PT-distributed. This is a direct consequence of Haar-randomness and is exactly §3.1. **Used in §6.2 to evaluate (C2).**

**(a) Over $x$, at fixed $U$.** For a single fixed circuit, the $N$ numbers $\{p_U(x)\}_x$ are spread out in a way that matches the PT density. Stated as a count — which is what it actually means — for any interval $[b,\,b+\delta]$ of probability values,

$$\#\big\{x:\;p_U(x)\in[b,\,b+\delta]\big\}\;\approx\;N\!\!\int_b^{b+\delta}\!\!Ne^{-Np}\,dp .\tag{4.1}$$

The customary shorthand for (4.1) is written with delta functions,

$$\frac1N\sum_x\delta\!\big(p_U(x)-p\big)\approx Ne^{-Np}
\quad\Longrightarrow\quad
\sum_x f\!\big(p_U(x)\big)\approx N\big\langle f(p)\big\rangle_{\rm PT}\tag{4.1$'$}$$

for smooth $f$ — this is Suppl. Eq. (5) — but (4.1$'$) is only notation for (4.1) and is *never* meant pointwise in $p$: the left side is a sum of spikes, so it can only equal a smooth function after smearing over an interval. **Used in §6.3 to evaluate (C1), and four times in §8 for the variance.**

### 4.1 (a) does not follow from (b)

Once $U$ is fixed, the $N$ numbers $p_U(x)$ are **deterministic** — a number has no distribution. So (4.1) is a statement about the *geometry of one state in one basis*: a form of ergodicity in the computational basis. Statement (b) says nothing about how one circuit's $N$ numbers are arranged among themselves.

Counterexamples showing (a) must be generated dynamically:

- **$d=0$** (Hadamards only): $\lvert\psi\rangle=\lvert+\rangle^{\otimes n}$, so $p_U(x)=1/N$ for every $x$. The histogram is a single spike at $p=1/N$ — maximally un-PT, even though the mean of $p_U(x)$ is correct.
- An ensemble in which each circuit is uniform over a *random half* of the strings: the per-string marginal can look reasonable while no instance's histogram is exponential.

### 4.2 Why (a) is nevertheless true — the counting argument

No change of variables is used anywhere in this subsection; everything is written directly in terms of $p_U(x)$.

#### 4.2.0 Two senses of "probability," kept apart

Almost all confusion here comes from conflating two unrelated probabilities.

- **The Born probability $p_U(x)$.** Once the circuit $U$ has been chosen, $p_U(x)$ is a fixed real number, one for each of the $N$ strings. So a fixed circuit carries a list of $N$ definite numbers. **Nothing about that list is random.**
- **The ensemble probability, written $\Pr_U(\cdot)$ and $\mathbb E_U[\cdot]$.** The circuit itself is drawn at random from the ensemble of §2. Any quantity that depends on $U$ — each individual number $p_U(x)$, and also the whole list — thereby becomes a random variable with respect to *this* randomness.

Every $\mathbb E_U$ below averages over **which circuit was drawn**. It never averages over which string was measured.

#### 4.2.1 Naming the objects

Statement (a) is a claim about the histogram of the list $\{p_U(x)\}_x$, so first define the histogram concretely.

**Bin.** Fix an interval of probability values

$$B=[\,b,\;b+\delta\,]\subset[0,1],$$

with $b$ its left edge and $\delta$ its width, taken small enough that the PT density $Ne^{-Np}$ is essentially constant across it.

**Membership indicator.** For each string $x$ define

$$\chi_x(U)=\begin{cases}1&\text{if }p_U(x)\in B,\\[2pt]0&\text{otherwise.}\end{cases}$$

For fixed $U$ this is a definite $0$ or $1$; as $U$ varies it is a random variable taking values in $\{0,1\}$.

**Bin count.** 

$$C_B(U)\;=\;\sum_{x}\chi_x(U)\;=\;\#\big\{x:\;p_U(x)\in B\big\},$$

an integer between $0$ and $N$: *how many of this circuit's $N$ output probabilities fall inside the bin.* This is literally the height of one bar of the histogram. Formally, integrating the left-hand side of (4.1) over the bin gives exactly

$$\int_B dp\;\frac1N\sum_x\delta\big(p_U(x)-p\big)\;=\;\frac{C_B(U)}{N}.$$

**What (a) asserts.** That for the single circuit actually run,

$$\frac{C_B(U)}{N}\;\approx\;\int_B dp\;Ne^{-Np}\;\approx\;Ne^{-Nb}\,\delta,
\qquad\text{i.e.}\qquad
C_B(U)\;\approx\;N\cdot Ne^{-Nb}\delta.\tag{4.2}$$

#### 4.2.2 Step 1 — the expected bin count

Statement **(b)** says: for one fixed string $x$, the number $p_U(x)$ is PT-distributed as $U$ varies. The chance that it lands inside $B$ is therefore the area of the PT density over $B$. Since the expectation of an indicator is the probability of the event it indicates,

$$\mathbb E_U[\chi_x]\;=\;\Pr_U\big(p_U(x)\in B\big)\;=\;\int_B dp\;Ne^{-Np}\;\approx\;Ne^{-Nb}\delta.\tag{4.3}$$

This number is **the same for every string $x$**, because Haar measure is invariant under permutations of the computational basis: no string is special.

Now sum (4.3) over the $N$ strings. Expectation is linear — $\mathbb E[A+B]=\mathbb E[A]+\mathbb E[B]$ holds for *any* random variables, correlated or not — so

$$\mathbb E_U\big[C_B\big]\;=\;\mathbb E_U\Big[\sum_x\chi_x\Big]\;=\;\sum_x\mathbb E_U[\chi_x]\;=\;N\cdot Ne^{-Nb}\delta.\tag{4.4}$$

That is exactly the right-hand side of (4.2). In one sentence: **there are $N$ strings, each has chance $Ne^{-Nb}\delta$ of landing in the bin, so on average $N\times Ne^{-Nb}\delta$ of them do.**

It is worth being explicit that the $\chi_x$ are **not** independent. The exact constraint $\sum_x p_U(x)=1$ couples them: if many strings have large probability, the rest are forced small. This coupling is real, and it does not affect (4.4) in the slightest, because linearity of expectation never requires independence. (Same move as: the expected number of fixed points of a random permutation is $1$, even though fixed points are correlated.)

#### 4.2.3 Step 2 — from the average circuit to *your* circuit

Equation (4.4) is about the histogram **averaged over circuits**. Statement (a) is about **one** circuit. These are genuinely different, and the gap is not pedantic. Here is an ensemble in which the average histogram is realized by no instance at all:

> With probability $\tfrac12$ the circuit yields $p_U(x)=1/N$ for every string. With probability $\tfrac12$ it yields $p_U(x)=2/N$ on half the strings and $p_U(x)=0$ on the other half.
>
> Averaged over circuits, the histogram has three bars: weight $\tfrac14$ at $p=0$, weight $\tfrac12$ at $p=1/N$, weight $\tfrac14$ at $p=2/N$. But every *individual* circuit's histogram is either a single spike at $1/N$ or two spikes at $0$ and $2/N$. Neither looks like the average.

Why it fails: in that ensemble the $\chi_x$ are *maximally* correlated — all strings move together — so $C_B$ swings by $O(N)$ from circuit to circuit and its mean describes nothing.

For Haar-like states the situation is the opposite. From the Gaussian picture of §3.2, the amplitudes are i.i.d. up to the single global normalization, so any two indicators $\chi_x,\chi_y$ ($x\neq y$) have correlation only $O(1/N)$. The bin count is then a sum of $N$ nearly-independent $0/1$ variables — essentially **binomial** — with mean $\mu_B\equiv N\cdot Ne^{-Nb}\delta$ and

$$\mathrm{std}_U\big[C_B\big]\approx\sqrt{\mu_B},
\qquad
\frac{\mathrm{std}_U[C_B]}{\mu_B}=\frac{1}{\sqrt{\mu_B}}=\frac{1}{\sqrt{\text{expected count in the bin}}}.\tag{4.5}$$

So each bar of a single circuit's histogram sits within a relative $1/\sqrt{\text{count}}$ of the PT prediction. Since $N=2^n$ is astronomically large, bins hold enormous numbers of strings and essentially **every** circuit — not merely the average — reproduces the PT histogram.

#### 4.2.4 Step 3 — the form actually used

XEB never needs a bin count; it needs sums of the form $\sum_x f(p_U(x))$. Define the **basis average**

$$S_f(U)\;\equiv\;\frac1N\sum_x f\big(p_U(x)\big),$$

a single number once $U$ is fixed, and a random variable over the ensemble. Steps 1 and 2, applied to $f$ instead of to an indicator, give

$$\mathbb E_U\big[S_f\big]=\langle f\rangle_{\rm PT},
\qquad
\mathrm{std}_U\big[S_f\big]=\sqrt{\frac{\mathrm{Var}_{\rm PT}(f)\,\big(1-\varrho^2\big)}{N}},
\qquad \varrho\equiv\mathrm{corr}_{\rm PT}\big(f(p),\,p\big),\tag{4.6}$$

where $\langle\cdot\rangle_{\rm PT}$, $\mathrm{Var}_{\rm PT}$, $\mathrm{corr}_{\rm PT}$ are all taken against the density $Ne^{-Np}\,dp$ of §3.4.

The factor $(1-\varrho^2)$ has a clean origin: the two constraints

$$\sum_x 1=N\qquad\text{and}\qquad\sum_x p_U(x)=1$$

hold **exactly for every circuit**, so whatever part of $f$ is a linear combination of the constant function and $p$ itself contributes no fluctuation at all. Only the component of $f$ orthogonal to $\mathrm{span}\{1,p\}$ fluctuates. Sanity check: $f(p)=p$ gives $\varrho=1$ and zero error ✓.

**Summary of the logic.**

$$\underbrace{\text{(b): each string's }p_U\text{ is PT}}_{\text{fixes the mean count, via linearity}}
\;+\;
\underbrace{\text{strings nearly independent}}_{\text{fixes the }\sqrt{\text{count}}\text{ fluctuation}}
\;\Longrightarrow\;
\underbrace{\text{(a): one circuit's histogram is PT}}_{\text{what XEB uses}}$$

Randomness over $U$ appears only as a proof device — to show that the *typical* instance is good. The conclusion is a statement about one deterministic list of $N$ numbers. This is the same logical move as "time average = ensemble average" in statistical mechanics, and the same content as Berry's conjecture / ETH for chaotic eigenstates.

**Worked number.** For the output entropy, $f(p)=p\log p$; evaluating (4.6) against $Ne^{-Np}dp$ gives $\mathrm{std}_U\approx0.54\cdot2^{-n/2}$ nats $\approx0.78\cdot2^{-n/2}$ bits — the supplement's quoted $\approx0.75\cdot2^{-n/2}$, and the basis of the "4-sigma" convergence criterion of Suppl. §V. Verified numerically in §15.

### 4.3 Where (a) fails even at large $N$

**Failure mode: empty bins.** From (4.5), a bar tracks the PT curve only if the bin holds many strings, i.e. $N\cdot Ne^{-Nb}\delta\gg1$. Far out in the tail it holds none. Concretely, the *largest* of the $N$ output probabilities of a single circuit satisfies

$$N\,p_{\max}\;\approx\;\log N,$$

i.e. no string has probability more than about $\log N$ times the mean value $1/N$. So beyond $p\sim(\log N)/N$ a single circuit's histogram **has no support whatsoever**, and cannot reproduce the smooth PT tail no matter how large $N$ is.

**Consequence for moments.** For $f(p)=p^k$ (the IPRs of §5), the integral $\langle p^k\rangle$ is dominated by $p\approx k/N$, so the relevant part of the tail must be populated: one needs $Ne^{-k}\gg1$, i.e. $k\ll\log N$. Quantitatively, (4.6) gives relative error

$$\frac{\mathrm{std}_U\big[S_{p^k}\big]}{\langle p^k\rangle}\;\approx\;2^{k}\,2^{-n/2},$$

so (a) holds only while $k\ll n/2$. At $n=42$, $k=10$ this is $\sim5\times10^{-4}$ — consistent with the tight error bars in Fig. 2c, and an explanation of why the paper tests up to $k=10$ and stops there.

**Not a problem for XEB.** Every function XEB actually integrates — $p\log p$, $\log p$, $\log^2p$, $p\log^2p$ — is dominated by $p$ of order $1/N$, i.e. by the bulk of the histogram where bins are heavily populated. The tail defect never enters.

> **▸ Small $N$ — statement (a) collapses entirely.**
> From (4.6), the instance-to-instance spread of the ideal-circuit XEB value is $\approx0.54\,N^{-1/2}$, to be compared with the signal $1-1/N$:
>
> | $n$ | $N$ | ideal $\alpha=1-1/N$ | instance std (measured) | ratio |
> |---|---|---|---|---|
> | 1 | 2 | 0.500 | 0.189 | 38% |
> | 2 | 4 | 0.750 | 0.193 | 26% |
> | 3 | 8 | 0.875 | 0.156 | 18% |
> | 6 | 64 | 0.984 | 0.066 | 7% |
> | 14 | 16384 | 0.99994 | 0.0042 | 0.4% |
>
> At $n\le3$ a *single circuit instance* tells you almost nothing. **This is not fatal**, because (a) enters only as an approximation to the ensemble statement, and $\mathbb{E}_U[(\mathrm{C1})]=1-1/N$ holds *exactly* at every $N$ (proved in §6.4 without using (a) at all). So the fix is: **average over many circuit instances**, $\sim(\text{std}/\epsilon)^2$ of them. At $n=2$, $\epsilon=0.005$ needs $\sim1500$ instances. At small $N$ instance-averaging, not shot-noise, is the dominant cost.

---

## 5. Certifying PT numerically

Two consequences of (4.1) plus the table, both measured in Fig. 2b,c and both prerequisites for §6:

**Output entropy** (exact finite-$N$ form in parentheses):

$$-\sum_xp_U(x)\log p_U(x)=-N\langle p\log p\rangle=\log N-1+\gamma\qquad(=\mathcal H_N-1).\tag{5.1}$$

The entropy sits exactly $1-\gamma\approx0.4228$ nats below the maximum $\log N$. Fig. 2b shows convergence to this line with depth.

**Inverse participation ratios.** $\mathrm{IPR}^{(k)}\equiv\sum_xp_U(x)^k$:

$$\mathrm{IPR}^{(k)}=N\langle p^k\rangle=\frac{k!}{N^{k-1}}.\tag{5.2}$$

Fig. 2c shows $\mathrm{IPR}^{(k)}N^{k-1}/k!\to1$ for $k=2,\dots,10$, all converging at roughly the same depth — $d\sim20$ cycles for up to $7\times6$ qubits, growing *sublinearly* in $n$, far faster than the best rigorous bounds.

**What is assumed vs. checked.** Step 1 of §4.2 needs $\mathbb{E}_U\big[f(p_U(x))\big]$ to match the Haar value; Step 2 needs the two-string quantity $\mathbb{E}_U\big[f(p_U(x))f(p_U(y))\big]$ to match Haar as well. For $f(p)=p$ the latter is exactly the **approximate 2-design** (anticoncentration) property — the one thing with rigorous depth bounds. The paper proves nothing; it *measures* (5.1) and (5.2) on the actual instances. That measurement is the certification.

---

## 6. Constructing the score function; proof that α = F

### 6.1 The choice

$$o_U(x)=H_0+\log p_U(x),\tag{6.1}$$

with $H_0$ a circuit-independent additive constant **to be fixed by (C2)** in §6.2. This satisfies the design constraints of §1.2: it depends only on $U$ through the classically computable $p_U$, not on $\rho$, and it is state-specific as required by gap 3 of §1.5.

Define the **cross entropy** between any distribution $p_A$ and the ideal distribution:

$$H(p_A,p_U)\equiv-\sum_xp_A(x)\log p_U(x).\tag{6.2}$$

The asymmetry is deliberate: samples from $p_A$ (device), scores $\log p_U$ (ideal circuit, computed classically). With (6.1)–(6.2), Eq. (1.2) reads

$$\alpha=\sum_xp_{\exp}(x)\big[H_0+\log p_U(x)\big]=H_0-H(p_{\exp},p_U)\equiv\Delta H(p_{\exp}),\tag{6.3}$$

defining the **cross-entropy difference** $\Delta H$ — hence the name of the method.

### 6.2 Enforcing (C2): the role of chaos, and the value of $H_0$

By (6.1) and (1.5), (C2) reads $\sum_xp_\sigma(x)[H_0+\log p_U(x)]=0$, i.e. using $\sum_xp_\sigma(x)=1$,

$$\textbf{(C2)}\iff H_0=H(p_\sigma,p_U).\tag{6.4}$$

$p_\sigma$ is unknown, so (6.4) can only be imposed on ensemble average, and only under:

> **Assumption A.** $p_\sigma$ is statistically uncorrelated with $p_U$.

Under Assumption A the ensemble average of (6.2) factorizes:

$$\mathbb{E}_U\big[H(p_\sigma,p_U)\big]=-\sum_x\mathbb{E}_U[p_\sigma(x)]\;\mathbb{E}_U[\log p_U(x)].$$

**Key step.** By PT-usage **(b)**, $\mathbb{E}_U[\log p_U(x)]=-(\log N+\gamma)$ — *the same constant for every $x$*. It factors out of the sum, and $\sum_x\mathbb{E}_U[p_\sigma(x)]=1$ closes the calculation:

$$\mathbb{E}_U\big[H(p_\sigma,p_U)\big]=\log N+\gamma
\quad\Longrightarrow\quad
\boxed{\;H_0=\log N+\gamma\;}\qquad(\text{exactly }\mathcal H_{N-1}).\tag{6.5}$$

Three remarks.

*Only normalization is used, not positivity.* Every step — expanding (6.4), factorizing, pulling out the constant — uses $\sum_xp_\sigma(x)=1$ and $\mathbb{E}[AB]=\mathbb{E}[A]\mathbb{E}[B]$. Neither requires $p_\sigma(x)\ge0$. See §10.

*Stronger than the paper's first route.* Suppl. Eqs. (10)–(12) assume $\mathbb{E}_U[p_\sigma(x)]=1/N$, i.e. that the error output is *flat*. PT makes flatness unnecessary (Suppl. Eqs. 19–20): the error state may be arbitrarily structured; it need only be uncorrelated with $p_U$ and normalized.

*A coincidence worth noting.* $H_0$ numerically equals $H(p_{\rm unif},p_U)=-\frac1N\sum_x\log p_U(x)$. Not an accident: it is the same constant $-\mathbb{E}[\log p]$ under either PT reading, which is why one constant serves both.

**Justification of Assumption A.** This is where quantum chaos enters, and it is verified, not assumed. Fig. 1a: insert a **single** random $X$ or $Z$ error anywhere in the circuit and compute $\lvert\Delta H\rvert$ between the perturbed and ideal distributions. For $d\gtrsim25$–$30$ it falls to the $2^{-n/2}$ floor — the level expected for two genuinely *independent* PT distributions. Physically: PT states are near-maximally entangled, so no single-qubit error is a small perturbation of the output distribution. Hypersensitivity to perturbation is the defining signature of chaos, and it is exactly what Assumption A needs. Fig. 1b is the same statement from the other side: as the Pauli error rate grows, the density of $Np$ interpolates from $e^{-Np}$ toward $\delta(Np-1)$.

### 6.3 Verifying (C1)

With $H_0$ now pinned, (C1) is no longer free. Using PT-usage **(a)** and (5.1):

$$\langle\psi_d\rvert O_U\lvert\psi_d\rangle=H_0-H(p_U,p_U)
=(\log N+\gamma)-(\log N-1+\gamma)=1.\tag{6.6}$$

**(C1) holds with no rescaling.** The gap between $H_0$ and the PT entropy is exactly unity. So the normalization "$\Delta H=1$ ideal, $\Delta H=0$ uncorrelated" is a *consequence* of PT, not a convention.

### 6.4 The result, exactly at any $N$

Do (6.6) and (C2) with the exact constants of §3.3 rather than the large-$N$ ones.

**(C1) exactly.** $H_0=\mathcal H_{N-1}$ and $H(p_U,p_U)=\mathcal H_N-1$, so with $\mathcal H_N=\mathcal H_{N-1}+1/N$:

$$\textbf{(C1)}=\mathcal H_{N-1}-(\mathcal H_N-1)=\boxed{\;1-\frac1N\;}\tag{6.7}$$

**(C2) exactly.** Take the error component to be structureless — Haar-random in the orthogonal complement of $\lvert\psi_d\rangle$, which is also what a depolarizing channel gives after orthogonalization. Then $\mathbb{E}[\sigma_U]=\Pi^\perp/(N-1)$ with $\Pi^\perp=\mathbb 1-\lvert\psi_d\rangle\langle\psi_d\rvert$, so

$$\mathbb{E}[p_\sigma(x)]=\frac{1-p_U(x)}{N-1}
\;\Longrightarrow\;
H(p_\sigma,p_U)=\frac{N H_0-H(p_U,p_U)}{N-1},$$

$$\textbf{(C2)}=H_0-H(p_\sigma,p_U)=\frac{H(p_U,p_U)-H_0}{N-1}=\frac{-(1-1/N)}{N-1}=\boxed{\;-\frac1N\;}\tag{6.8}$$

The origin of the $-1/N$: "orthogonal to $\lvert\psi_d\rangle$" removes one of $N$ dimensions, which makes $p_\sigma$ slightly *anti*correlated with $p_U$. Negligible at large $N$; total at $N=2$, where the complement is a single ray and $p_\sigma(x)=1-p_U(x)$ exactly. **The validity of Assumption A is controlled by $1/N$ — and here the correction is analytic.**

**Combining** via §1.4:

$$\alpha=F\Big(1-\frac1N\Big)+(1-F)\Big(-\frac1N\Big)
\;\Longrightarrow\;
\boxed{\;F=\alpha+\frac1N\;}\tag{6.9}$$

Exact at every $N$. Checks: $F=1\Rightarrow\alpha=1-1/N$ ✓ (that is (C1)); $\rho=\mathbb 1/N\Rightarrow F=1/N,\ \alpha=0$ ✓ (uniform sampling must give zero). At $N=2^{49}$ the shift is invisible — this is the paper's $\alpha=F$. Verified by Monte Carlo in §15.

> **▸ Small $N$ — the reference values.**
>
> | $n$ | $N$ | (C1) $=1-1/N$ | (C2) $=-1/N$ | correction $F-\alpha$ |
> |---|---|---|---|---|
> | 1 | 2 | 0.500 | $-0.500$ | 0.500 |
> | 2 | 4 | 0.750 | $-0.250$ | 0.250 |
> | 3 | 8 | 0.875 | $-0.125$ | 0.125 |
> | 49 | $5.6\times10^{14}$ | $1-2\times10^{-15}$ | $-2\times10^{-15}$ | $2\times10^{-15}$ |
>
> At $n=1$ the correction is $0.5$ — enormous, but **exactly known and exactly correctable**. Nothing about the derivation breaks; only the numbers change. §13 shows that in the interleaved protocol you don't even need (6.9), because the constant is absorbed by a fit parameter.

### 6.5 PT-free fallback

Without invoking PT for the reference values, the same three lines give Suppl. Eq. (17):

$$F=\frac{H_{\rm unc}-\mathbb{E}_U[H(p_{\exp},p_U)]}{H_{\rm unc}-\mathbb{E}_U[H(p_U,p_U)]},
\qquad H_{\rm unc}\equiv\mathbb{E}_U[H(p_\sigma,p_U)],\tag{6.10}$$

with both reference values obtained numerically for small circuits. PT is exactly what sets $H_{\rm unc}=\log N+\gamma$ and the denominator to $1$. **For circuits too shallow to have reached PT (failing §5), use (6.10).**

### 6.6 Why $\log p_U$? The likelihood-ratio meaning

(C1)–(C2) can be satisfied by other score functions (see §14). The log is distinguished: $\Delta H$ is the log-likelihood-ratio statistic.

Let $S=\{x_1,\dots,x_m\}$ be i.i.d. from the *ideal* circuit and $\Pr_U(S)$ its probability under $U$. Then $\log\Pr_U(S)=\sum_j\log p_U(x_j)$ is a sum of $m$ i.i.d. terms with mean $-(\log N-1+\gamma)$ (the PT entropy) and $O(1)$ variance, so by the CLT

$$\log\Pr_U(S)=-m(\log N-1+\gamma)+O(\sqrt m).$$

For $S_{\rm unc}$ drawn from anything uncorrelated with $p_U$, the per-term mean is $\frac1N\sum_x\log p_U(x)=-(\log N+\gamma)$, so $\log\Pr_U(S_{\rm unc})=-m(\log N+\gamma)+O(\sqrt m)$. Subtracting:

$$\mathbb{E}_U\big[\log\Pr_U(S)-\log\Pr_U(S_{\rm unc})\big]=m.\tag{6.11}$$

A genuine $m$-sample from $U$ is $e^m$ times more likely under $U$ than an uncorrelated one: an $m$-sample is a unique fingerprint of the circuit, and $\Delta H$ is the **per-sample log-likelihood advantage**, normalized to 1 for the ideal circuit. The "1" in (6.6) and the "$m$" in (6.11) are the same $1-\gamma$ versus $\gamma$ offset.

---

## 7. The estimator

From (6.3), $H(p_{\exp},p_U)=-\mathbb{E}_{x\sim p_{\exp}}[\log p_U(x)]$ is an ordinary expectation. Replace it by the sample mean:

$$\boxed{\;\hat\alpha=\log N+\gamma+\frac1m\sum_{j=1}^{m}\log p_U\!\big(x_j^{\exp}\big)\;}
\qquad\Big(\text{exactly: }\mathcal H_{N-1}+\tfrac1m\textstyle\sum_j\log p_U\Big)\tag{7.1}$$

This is (1.1) with score (6.1) and constant (6.5). Unbiased for $F-1/N$ (exactly), hence for $F$ up to $O(2^{-n})$, plus the $O(2^{-n/2})$ instance-to-instance error of §4.2. Requires exact classical evaluation of $m$ ideal probabilities.

---

## 8. Variance and sample complexity

### 8.1 The sharpened ansatz

The mean of $\hat\alpha$ needed only Assumption A. The *variance* needs the full one-shot distribution, so adopt

$$\rho_K=f\,\lvert\psi_d\rangle\langle\psi_d\rvert+(1-f)\frac{\mathbb 1}{N}
\;\Longrightarrow\;
p_{\exp}(x)=f\,p_U(x)+\frac{1-f}{N},\tag{8.1}$$

with $f$ the depolarizing parameter. Then $F=f+(1-f)/N$ and, from (6.9), $\alpha=f(1-1/N)$.

What is actually required is weaker than $\sigma_U=\mathbb 1/N$: it is $\mathbb{E}[p_\sigma(x)\mid p_U(x)=p]=1/N$ for all $p$ — *conditional* independence, not flatness. Assumption A supplied the first moment; this supplies the full conditional law. Justification: PT states are near-maximally entangled, so one Pauli error destroys the correlation with ideal sampling entirely — no partial credit.

### 8.2 Computing the variance

Let $z\equiv-\log p_U(x)$ with $x\sim p_{\exp}$, and $L\equiv\log N+\gamma$; so $\hat\alpha=L-\frac1m\sum_jz_j$. Using (4.1), (8.1), and the table:

$$\mathbb{E}[z]=f\underbrace{\big(-N\langle p\log p\rangle\big)}_{L-1}+(1-f)\underbrace{\big(-\langle\log p\rangle\big)}_{L}=L-\alpha,$$

reproducing (6.9) (Suppl. Eq. 6). Second moment:

$$\mathbb{E}[z^2]=f\,N\langle p\log^2p\rangle+(1-f)\langle\log^2p\rangle
=f\Big[(L-1)^2+\tfrac{\pi^2}6-1\Big]+(1-f)\Big[L^2+\tfrac{\pi^2}6\Big]
=L^2-2\alpha L+\tfrac{\pi^2}6 .$$

Subtract $\mathbb{E}[z]^2=L^2-2\alpha L+\alpha^2$:

$$\boxed{\;\kappa^2\equiv\mathrm{Var}(z)=\frac{\pi^2}{6}-\alpha^2\;}\tag{8.2}$$

**All $\log N$ terms cancel identically: $\kappa$ is independent of $n$** (Suppl. Eq. 7). Numerically $\kappa=0.803$ at $\alpha=1$ and $\kappa=\pi/\sqrt6=1.283$ at $\alpha=0$. Hence

$$\hat\alpha=\alpha\pm\frac{\kappa}{\sqrt m},\qquad m\approx\frac{\kappa^2}{(\delta\alpha)^2}.$$

Resolving $\delta\alpha=0.01$ needs $m\sim10^4$ shots whether $n=20$ or $n=70$. Resolving a *small* $\alpha$ at fixed relative precision needs $m\gtrsim\kappa^2/\alpha^2$. (The $m^{-1/2}$ scaling also needs the $O(2^{-n})$ correlations of §3.2 to be negligible — they are.)

**Exact finite-$N$ variance.** Repeating the computation with (3.3)–(3.5) instead of the large-$N$ table:

$$\boxed{\;\kappa^2=\mathcal H^{(2)}_{N-1}-\alpha^2-\frac{2\alpha}{N}\;}\tag{8.3}$$

which reduces to (8.2) as $\mathcal H^{(2)}_{N-1}\to\pi^2/6$.

> **▸ Small $N$ — shot noise is *not* the problem.**
>
> | $n$ | $N$ | $\mathcal H^{(2)}_{N-1}$ | $\kappa$ at ideal ($\alpha=1-\tfrac1N$) | $\kappa$ at $\alpha=0$ | contrast $1-\tfrac1N$ | SNR/shot |
> |---|---|---|---|---|---|---|
> | 1 | 2 | 1.0000 | 0.500 | 1.000 | 0.500 | 1.00 |
> | 2 | 4 | 1.3611 | 0.651 | 1.167 | 0.750 | 1.15 |
> | 3 | 8 | 1.5118 | 0.726 | 1.230 | 0.875 | 1.21 |
> | $\infty$ | — | 1.6449 | 0.803 | 1.283 | 1.000 | 1.25 |
>
> Per-shot signal-to-noise at $n=1$ is $1.00$ versus $1.25$ at $n=\infty$ — essentially unchanged. **Shot count is never the small-$N$ obstacle**; instance-averaging (§4.2 box) and Assumption A (§11) are.

### 8.3 Why this evades the concentration obstruction

Closing gap 3 of §1.5. For a Haar-random state, the outcome of any *fixed* measurement concentrates exponentially about its Hilbert-space average (Levy's lemma), so any fixed observable carries exponentially little information and would need $m\sim2^n$ shots. XEB escapes by using a **state-specific global measurement**: by construction (6.1), $O_U$ is built out of $p_U$ itself and therefore tracks the state. Equation (8.2) is the quantitative statement that its per-shot fluctuations are $O(1)$ rather than $O(2^{-n/2})$.

The cost is displaced onto the classical side: $m$ exact evaluations of $p_U(x_j^{\exp})$. That limits the circuit size one can *directly verify*; larger fidelities must be extrapolated from fewer qubits, lower depth, or mostly-Clifford variants. **At $n\le3$ this cost is nil**, which is one reason XEB is attractive for gate calibration.

---

## 9. The full one-shot distribution

Ansatz (8.1) predicts the entire distribution of the score, not just its mean. The density of $p_U(x)$ for $x\sim\rho_K$ is

$$\Pr_\alpha(p)=\sum_xp_{\exp}(x)\delta\big(p_U(x)-p\big)
=\Big[\alpha p+\frac{1-\alpha}{N}\Big]\underbrace{\sum_x\delta\big(p_U(x)-p\big)}_{N\cdot Ne^{-Np}}
=N^2e^{-Np}\Big(\alpha p+\frac{1-\alpha}{N}\Big),\tag{9.1}$$

Suppl. Eq. (72). In $u=Np$: $\Pr(u)=e^{-u}[\alpha u+1-\alpha]$. Changing to $\zeta\equiv\log u=\log(Np_U(x))$, $du=e^\zeta d\zeta$:

$$\boxed{\;\Pr_\alpha(\zeta)=e^{\,\zeta-e^{\zeta}}\big(1+\alpha(e^{\zeta}-1)\big)\;}\tag{9.2}$$

the main-text formula, plotted in Fig. 4b. (The paper writes it with "$z=\log p_U(x)$"; it is in fact for the *rescaled* variable — the figure axis is correct, the sentence is sloppy.) Limits: $\alpha=0$ gives the Gumbel $e^{\zeta-e^\zeta}$ (log of an $\mathrm{Exp}(1)$); $\alpha=1$ gives $e^{2\zeta-e^\zeta}$. Consistency:

$$\mathbb{E}[\zeta]=(1-\alpha)\!\!\int_0^\infty\!\! e^{-u}\log u\,du+\alpha\!\!\int_0^\infty\!\! ue^{-u}\log u\,du=(1-\alpha)(-\gamma)+\alpha(1-\gamma)=\alpha-\gamma,$$

and $\zeta=\log N+\log p_U(x)$ gives $H(p_{\exp},p_U)=\log N+\gamma-\alpha$ ✓.

Two uses. (i) (9.2) is a complete one-parameter likelihood, so $\alpha$ can be fitted by maximum likelihood on the histogram rather than through the mean. (ii) More importantly, the model becomes **falsifiable**: if the measured histogram of $\log(Np_U(x))$ leaves this one-parameter family, the errors are not behaving as assumed and $\hat\alpha$ is not the fidelity. Fig. 4b's agreement with digital-error-model simulations is the paper's evidence that the ansatz holds.

---

## 10. On the positivity of $\sigma_U$

A recurring confusion: §6.2 uses $\sum_xp_\sigma(x)=1$, which might look like an assertion that $\sigma_U$ is a density matrix. It is not.

$$\mathrm{Tr}\,\sigma_U=1\;\Longrightarrow\;\sum_x\langle x\rvert\sigma_U\lvert x\rangle=\sum_xp_\sigma(x)=1 .$$

Positivity would be the strictly stronger $p_\sigma(x)\ge0$ for each $x$, never needed. A signed vector can perfectly well sum to 1. Audit of §6.2: expanding (6.4) uses normalization; factorizing uses $\mathbb{E}[AB]=\mathbb{E}[A]\mathbb{E}[B]$, which has no sign requirement; pulling out the constant uses normalization again. Structurally unsurprising: (C2) is $\mathrm{Tr}(\sigma_UO_U)=0$, a *linear functional* of a Hermitian unit-trace operator, and linear functionals do not care about the positive cone.

**When positivity does hold.** Block-decompose $\rho$ with respect to $\lvert\psi_d\rangle\oplus\lvert\psi_d\rangle^\perp$:

$$\rho=\begin{pmatrix}F & v^\dagger\\ v& M\end{pmatrix}
\;\Longrightarrow\;
(1-F)\sigma_U=\begin{pmatrix}0& v^\dagger\\ v& M\end{pmatrix},$$

$v$ being the coherences between the ideal state and the error subspace. A PSD matrix with a zero diagonal entry must have that entire row and column zero, so

$$\boxed{\;\sigma_U\succeq0\iff v=0\;}$$

i.e. positivity holds **iff $\rho$ has no coherence between $\lvert\psi_d\rangle$ and its complement.** Two instances:

- *Depolarizing*, $\rho=f\lvert\psi_d\rangle\langle\psi_d\rvert+(1-f)\mathbb 1/N$: $\sigma_U=\Pi^\perp/(N-1)$, positive, a genuine distribution.
- *Coherent*, $\rho=\lvert\phi\rangle\langle\phi\rvert$ with $\lvert\phi\rangle=\sqrt F\lvert\psi_d\rangle+\sqrt{1-F}\lvert\psi_d^\perp\rangle$: in that $2\times2$ block $(1-F)\sigma_U=\begin{psmallmatrix}0&c\\c&1\end{psmallmatrix}$ with $c=\sqrt{F(1-F)}$, eigenvalues $\tfrac12(1\pm\sqrt{1+4c^2})$ — one negative for every $0<F<1$. Non-positivity is *generic* for coherent errors, not a pathology.

Positivity of $\rho$ still bounds how negative $p_\sigma$ can be: $p_{\exp}(x)\ge0$ gives $p_\sigma(x)\ge-\frac{F}{1-F}p_U(x)$.

**Where positivity is used.** Only in the variance ansatz (8.1), which is a genuine mixture — a separate, stronger, physically motivated model needed for $\kappa^2$ and $\Pr_\alpha(\zeta)$, not for $\alpha=F$.

**The connection worth internalizing.** Non-positivity and failure of Assumption A are logically independent but co-occur, both driven by $v\neq0$: coherences between $\lvert\psi_d\rangle$ and the error subspace make $\sigma_U$ non-positive *and* make $p_\sigma$ correlated with $p_U$. So the regime where you would worry about positivity is the regime where XEB actually fails — but it fails because of Assumption A, not because of positivity.

---

## 11. Assumption audit, ordered by fragility

1. **Assumption A** (§6.2) — load-bearing; it gives (C2). Fails at shallow depth (Fig. 1a: single-error $\lvert\Delta H\rvert$ above the $2^{-n/2}$ floor for $d\lesssim20$). Fails badly for **coherent** errors: a systematic over-rotation implements $U'\approx U$, so $p_{U'}$ is correlated with $p_U$ at leading order. **Worst case, and it is not exotic:** a $Z$ rotation in the final layer commutes with the measurement, leaves $p_{\exp}=p_U$ *exactly*, so XEB reports $\alpha=1-1/N$ while $F$ may be far below 1. Verified numerically in §15. XEB is a fidelity meter for stochastic errors in chaotic circuits, not universally.
2. **PT convergence** (§3–5) — needed for $H_0$, for (C1), and for $\kappa^2$. Certified by (5.1)–(5.2), satisfied at $d\gtrsim20$. Otherwise use (6.10) or the exact constants of §3.3.
3. **Statement (a)** (§4) — needed to make (C1) an *instance* statement. Fails at small $N$; fix by instance-averaging.
4. **Conditional independence (8.1)** — needed for $\kappa$ and $\Pr_\alpha(\zeta)$, **not** for $\alpha=F$. Checked against (9.2).
5. **Near-independence of $\{p_U(x_j)\}$ across shots** (§3.2) — $O(2^{-n})$; needed for the CLT and $m^{-1/2}$.
6. **Exact classical $p_U(x_j^{\exp})$** (§7) — cannot be cheaply approximated: an approximate $\tilde p_U$ decorrelated from $p_U$ drives $\hat\alpha$ toward 0, mimicking infidelity.
7. **Finite $m$** (§8) — $\kappa/\sqrt m$ plus $O(m^{-1/2})$ bias from the logarithm.

---

## 12. Summary: what changes at small $N$

| ingredient | large $N$ | exact at any $N$ | status at $n=1,2,3$ |
|---|---|---|---|
| one-point law of $p_U(x)$, over $U$ | $Ne^{-Np}$ | $\mathrm{Beta}(1,N-1)$ | **exact, fine** |
| $H_0$ | $\log N+\gamma$ | $\mathcal H_{N-1}$ | **exact, fine** |
| ideal entropy | $\log N-1+\gamma$ | $\mathcal H_N-1$ | **exact, fine** |
| (C1) | $1$ | $1-1/N$ | 0.50 / 0.75 / 0.875 |
| (C2) | $0$ | $-1/N$ | $-0.50/-0.25/-0.125$ |
| relation | $\alpha=F$ | $F=\alpha+1/N$ | **exact, correctable** |
| per-shot variance | $\pi^2/6-\alpha^2$ | $\mathcal H^{(2)}_{N-1}-\alpha^2-2\alpha/N$ | SNR barely degraded |
| statement (a), single instance | error $2^{-n/2}$ | — | **broken**; average over instances |
| Assumption A for stochastic error | good | — | **exact** for structureless $\sigma_U$ |
| Assumption A for coherent error | poor | — | **poor**; the real limitation |

**Net:** the structure is dimension-free; the constants shift by known amounts; the statistical cost moves from shots to circuit instances; and the genuine limitation at small $N$ is coherent error, which XEB structurally cannot see. §13 is the protocol that neutralizes the first three and diagnoses the fourth.

---

## 13. Interleaved XEB: measuring a non-Clifford gate against a reference

This is the practical payoff. The goal: measure the fidelity of one specific two-qubit gate $G$ — say $\sqrt{\mathrm{iSWAP}}$ or a general $\mathrm{fSim}(\theta,\phi)$ — which is **non-Clifford**, on $n=2$ qubits ($N=4$).

### 13.1 Why standard RB cannot do this

Randomized benchmarking works by: (i) drawing a sequence of $m$ elements from a *group* $\mathcal G$ (the Cliffords); (ii) appending the inverse of the accumulated product, so the ideal net operation is the identity; (iii) invoking the **group twirl**, which converts the average error channel into a depolarizing channel, whose depolarizing parameter decays as $p^m$; (iv) fitting $A p^m+B$.

Interleaved RB inserts the gate under test between Clifford elements — but step (ii) then requires computing an inverse, which requires $G$ to lie in the group (or in a group you can invert within). A generic non-Clifford $G$ does not, and the twirl in step (iii) no longer applies. There are workarounds (dihedral RB, non-uniform RB, character RB) but each requires its own group structure tailored to $G$.

**XEB requires no group at all.** It requires only:

1. a random circuit ensemble that anticoncentrates (reaches PT on $\mathrm{SU}(N)$),
2. classical computation of $p_U(x)$ — trivial for $N=4$,
3. linearity in $\rho$ — §1.3.

The gate under test may be anything. That is precisely why XEB became the standard calibration tool for the fSim family.

### 13.2 The protocol

Fix the two qubits. Define two sequence families, each parametrized by the number of cycles $m_c$:

- **Test sequences.** $m_c$ cycles, each = [random single-qubit gate on each qubit] followed by [the gate $G$ under test]. Single-qubit gates drawn from a set that twirls well on $\mathrm{SU}(2)$ (e.g. the $\pi/2$ rotations about random axes in the $XY$ plane, or Haar on SU(2)).
- **Reference sequences.** $m_c$ cycles of the random single-qubit gates only, no $G$.

For each family, each $m_c$, and each of many circuit instances: run $m$ shots, compute $p_U$ classically for the observed strings, form $\hat\alpha$ via (7.1), and average over instances. This yields two decay curves,

$$\hat\alpha_{\rm test}(m_c)=A_{\rm test}\;p_{\rm test}^{\,m_c},
\qquad
\hat\alpha_{\rm ref}(m_c)=A_{\rm ref}\;p_{\rm ref}^{\,m_c}.\tag{13.1}$$

Fit each to an exponential. Then

$$\boxed{\;p_G=\frac{p_{\rm test}}{p_{\rm ref}},\qquad
e_G=\Big(1-\frac1N\Big)(1-p_G)=\tfrac34\,(1-p_G)\ \ \text{for }N=4\;}\tag{13.2}$$

with $e_G$ the average gate infidelity of $G$. (The factor $(N-1)/N$ is the standard conversion from depolarizing parameter to average gate infidelity: for $\mathcal E(\rho)=p\rho+(1-p)\mathbb 1/N$, $\;1-F_{\rm avg}=(1-p)(N-1)/N$.)

### 13.3 Why the exponential form is right

From §8.1, a per-cycle depolarizing channel with parameter $p_{\rm cycle}$ composes multiplicatively, so after $m_c$ cycles the surviving polarization is $p_{\rm cycle}^{m_c}$ and, by (6.9),

$$\alpha(m_c)=\Big(1-\frac1N\Big)\,p_{\rm cycle}^{\,m_c}.$$

Equivalently in the paper's language, §8's digital error model gives $\log\alpha$ affine in the gate count, hence linear in $m_c$:

$$\log\hat\alpha(m_c)=\log A-\big(r_1\bar g_1+r_2\bar g_2\big)m_c ,\tag{13.3}$$

with $r_1,r_2$ the one- and two-qubit Pauli error rates and $\bar g_1,\bar g_2$ the per-cycle gate counts. Slope $\to$ per-cycle error; intercept $\to$ everything else.

### 13.4 Why this removes SPAM — the central point

The measured decay prefactor $A$ absorbs, in one lump, everything that is *not* per-cycle:

- **state-preparation error** (imperfect $\lvert00\rangle$),
- **readout / measurement error** (assignment fidelity),
- the $(1-1/N)$ normalization of (6.7) — at $N=4$ a $25\%$ effect that would otherwise have to be corrected by hand,
- residual bias from imperfect PT convergence at the smallest $m_c$,
- any error localized in the first or last layer.

None of these depends on $m_c$. Only $p$ — the **slope on a semi-log plot** — carries the per-cycle error. Fitting the exponential and reading $p$ therefore discards SPAM entirely. Taking the ratio in (13.2) then also cancels the single-qubit gate errors and the reference's own SPAM, leaving $G$.

This is exactly why RB is SPAM-free, and exactly what tomography cannot do. Standard state or process tomography estimates $\rho$ or the process matrix from data that also contains SPAM, and cannot separate "the gate was bad" from "the readout was bad": a systematic readout bias shows up as a systematic error in the reconstructed channel, with no internal signal distinguishing the two. **Gate set tomography** solves this by treating preparations, gates, and measurements as one self-consistent set with a gauge freedom — at the cost of far more data, a nonlinear fit, and gauge-fixing subtleties. XEB gets SPAM-independence for free from the single structural fact that SPAM is $m_c$-independent while gate error compounds.

Two practical corollaries specific to small $N$:

- **The $1/N$ correction of (6.9) becomes irrelevant.** You never need $F=\alpha+1/N$ in the interleaved protocol, because the constant $1-1/N$ is inside $A$. This dissolves the largest-looking small-$N$ complication.
- **The instance-averaging cost of §4 remains.** Instance spread is a genuine random error, not a bias, so it averages down as $1/\sqrt{\#\text{instances}}$; at $n=2$, budget $\sim10^3$ instances per depth. Reusing the *same* random single-qubit sequences in the test and reference families correlates the two estimates and cancels part of this spread in the ratio.

### 13.5 What the ratio does and does not isolate

**Does:** the incoherent, stochastic error attributable to inserting $G$ once per cycle, cleanly separated from SPAM and (to the extent the interleaving is faithful) from single-qubit gate error.

**Does not:**

1. **Coherent error in $G$.** If $G$ is systematically over-rotated by $\theta$, the error is unitary. Two symptoms: (i) infidelity accumulates as $\theta^2m_c^2$ rather than $\theta^2m_c$, so (13.1) is the wrong fit function and the data shows curvature on a semi-log plot; (ii) worse, if the coherent error happens to be diagonal in the measurement basis it is **completely invisible** — $p_{\exp}=p_U$ exactly and XEB reports full fidelity (§15 quantifies this). Curvature in the fit is the diagnostic; its absence is not proof of absence.
2. **Which error.** XEB returns one scalar, not a channel. No Pauli decomposition, no leakage identification, no crosstalk map.
3. **Anything, if $G$ is weakly entangling.** The whole derivation presumes the interleaved ensemble reaches PT on $\mathrm{SU}(4)$. Random $\mathrm{SU}(2)\otimes\mathrm{SU}(2)$ plus a strong entangler (CZ, iSWAP, $\sqrt{\mathrm{iSWAP}}$) does so in a few cycles. A nearly-trivial $G$ never anticoncentrates the ensemble; then (b) fails, the constants of §6.4 are wrong, and the extracted $e_G$ is meaningless. Check §5's diagnostics on the *simulated* ideal ensemble before trusting the measurement.

### 13.6 The companion measurement: separating coherent from incoherent

Because §9 gives the *full* one-shot distribution, not just its mean, the same data carries more than $\hat\alpha$. The variance of the measured probabilities gives the **purity** of the output state (this is "speckle purity benchmarking"): for a PT-distributed ideal state, the spread of $p_{\exp}$ across bit-strings scales with $\mathrm{Tr}\rho^2$, so

$$F_{\rm inc}\ \text{from purity}\qquad\text{vs.}\qquad F\ \text{from XEB}.$$

Purity is insensitive to coherent error (a unitary error preserves $\mathrm{Tr}\rho^2$); XEB fidelity is degraded by both. **The gap between them is the coherent error budget.** Running both on the same shots is the standard way to plug the hole in §13.5(1), and it costs no extra data.

### 13.7 When to use what, at $n\le3$

| | XEB | Clifford RB | GST |
|---|---|---|---|
| works on non-Clifford $G$ | **yes** | no | yes |
| SPAM-free | **yes** (via fit) | yes | yes (self-consistent) |
| returns full error channel | no | no | **yes** |
| sees coherent error | poorly | partly | **yes** |
| classical cost | trivial at $N\le8$ | trivial | large nonlinear fit |
| data cost | moderate (instances) | low | **high** |
| typical use | fast calibration loops, fSim tuning | Clifford gate benchmarking | error-model diagnosis |

**Verdict.** XEB is the right tool at $n\le3$ when you need a fast, SPAM-free scalar for a non-Clifford gate — e.g. closing a calibration loop on fSim angles. Pair it with purity benchmarking to bound the coherent contribution. When you need to know *what* the error is, use GST: at $n\le3$ tomographic methods are affordable ($4^n\le64$ settings), which is precisely the regime XEB was not designed for.

---

## 14. Log-XEB versus linear XEB

Everything above is **log-XEB**, score $o_U(x)=\log N+\gamma+\log p_U(x)$. Later hardware papers mostly report **linear XEB**:

$$o_U^{\rm lin}(x)=Np_U(x)-1,\qquad F_{\rm XEB}=N\big\langle p_U(x)\big\rangle_{\rm samples}-1 .$$

Check the two conditions of §1.4 with the table of §3.4:

- **(C1):** $\sum_xp_U(x)[Np_U(x)-1]=N\cdot N\langle p^2\rangle-1=N^2(2/N^2)-1=1$ ✓
- **(C2):** $\sum_xp_\sigma(x)[Np_U(x)-1]$, whose ensemble average under Assumption A is $N\cdot\frac1N-1=0$ ✓

Same target, different $f(p_U)$; per-shot variance $\approx1+2\alpha-\alpha^2$ instead of $\pi^2/6-\alpha^2$. The log version is the log-likelihood-ratio statistic (§6.6); the linear version is somewhat better behaved at small $\alpha$ and is what (13.1)–(13.2) are usually written with. The derivation structure — linearity in $\rho$, two normalization conditions, PT to evaluate them, chaos to kill the error term — is identical.

---

## 15. Appendix: numerical verification

Monte Carlo over Haar-random states / Haar-random unitaries, using the exact estimator (7.1) with $H_0=\mathcal H_{N-1}$.

**(i) Exact constants.** Confirms (6.7).

| $n$ | $N$ | $H_0=\mathcal H_{N-1}$ | $\log N+\gamma$ | entropy $\mathcal H_N-1$ | (C1) computed | $1-1/N$ |
|---|---|---|---|---|---|---|
| 1 | 2 | 1.000000 | 1.270363 | 0.500000 | 0.500000 | 0.500000 |
| 2 | 4 | 1.833333 | 1.963510 | 1.083333 | 0.750000 | 0.750000 |
| 3 | 8 | 2.592857 | 2.656657 | 1.717857 | 0.875000 | 0.875000 |
| 10 | 1024 | 7.508199 | 7.508687 | 6.509176 | 0.999023 | 0.999023 |
| 20 | $10^6$ | 14.440159 | 14.440159 | 13.440160 | 0.999999 | 0.999999 |

**(ii) $F=\alpha+1/N$ for depolarizing error.** Random $f\in[0,1]$, $\rho=f\lvert\psi\rangle\langle\psi\rvert+(1-f)\mathbb 1/N$.

| $n$ | mean$(F-\alpha)$ | $1/N$ | instance std |
|---|---|---|---|
| 1 | 0.49896 | 0.50000 | 0.214 |
| 2 | 0.25042 | 0.25000 | 0.186 |
| 3 | 0.12531 | 0.12500 | 0.145 |
| 6 | 0.01558 | 0.01562 | 0.054 |

Exact on average at every $N$; the instance std is the §4 effect and requires instance-averaging.

**(iii) Instance spread of the ideal-circuit XEB value.** Confirms (4.6) with prefactor $\approx0.54$.

| $n$ | mean | exact $1-1/N$ | std | $0.54\,N^{-1/2}$ |
|---|---|---|---|---|
| 1 | 0.5065 | 0.5000 | 0.189 | 0.382 |
| 2 | 0.7506 | 0.7500 | 0.193 | 0.270 |
| 3 | 0.8743 | 0.8750 | 0.156 | 0.191 |
| 6 | 0.9834 | 0.9844 | 0.0660 | 0.0675 |
| 10 | 0.9996 | 0.9990 | 0.0170 | 0.0169 |
| 14 | 0.99991 | 0.99994 | 0.00420 | 0.00422 |

The asymptotic formula becomes accurate by $n\approx6$ and is exact by $n=14$.

**(iv) Coherent error is invisible.** Device implements $V U$ with $V=\exp(-i\tfrac{\theta}{2}\sum_jZ_j)$, $\theta=0.3$ — diagonal, so $p_{\exp}=p_U$ exactly.

| $n$ | true $\langle F\rangle$ | XEB reports $\alpha+1/N$ | overestimate |
|---|---|---|---|
| 1 | 0.9853 | 1.0038 | $+0.019$ |
| 2 | 0.9645 | 1.0079 | $+0.043$ |
| 3 | 0.9418 | 0.9959 | $+0.054$ |
| 6 | 0.8756 | 0.9994 | $+0.124$ |

XEB reports essentially perfect fidelity while the true fidelity falls to 0.88. This is the failure mode of §11.1 and §13.5(1), and the reason for the purity companion measurement of §13.6. Note it gets *worse* with $n$, because more qubits accumulate more diagonal phase — it is not a small-$N$ artifact.
