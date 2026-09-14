# Accuracy vs time in the group-commutator controlled swap

How the synthesized controlled-swap interaction trades accuracy against gate time, and how
that compares with ordinary Trotterization of $H = H_A + H_B$.

---

## 1. Notation

| symbol | meaning |
|---|---|
| $g$ | native sideband (g1e0) coupling rate |
| $dt$ | length of one short sideband pulse |
| $\epsilon = g\,dt$ | **area** of one short pulse (dimensionless) |
| $\tau = 8\,dt$ | duration of one $W_{\rm sym}$ section (8 pulses) |
| $N$ | number of sections |
| $\Theta$ | target conditional-beamsplitter angle (e.g. $\pi/4$ at $n^*$) |
| $T_{\rm phys} = N\tau$ | physical duration of the whole train |

A full swap is area $\pi/2$ and a $\pi$ rotation, so a pulse of area $\epsilon$ rotates the
g1e0 two-level system by $2\epsilon$. When the two legs differ, everything below holds with
$\epsilon \to \bar\epsilon = \sqrt{\epsilon_A\epsilon_B}$.

---

## 2. The central result

One $W_{\rm sym}$ section is $W(-\epsilon)W(+\epsilon)$. Its symmetry cancels the odd-order
term, so per section

$$\theta_{\rm sig} = 2\,\epsilon_A\epsilon_B \;=\; 2\epsilon^{2},
\qquad
\delta_{\rm err} = c\,\epsilon^{4} + \mathcal{O}(\epsilon^{6}).$$

Over $N$ sections the signal is $\Theta = 2N\epsilon^{2}$ and the error is $Nc\,\epsilon^{4}$,
so

$$\boxed{\;\frac{\text{error}}{\text{signal}} \;=\; \frac{c}{2}\,\epsilon^{2}
\quad\text{— independent of } N.\;}$$

**Relative accuracy is set by $\epsilon$ alone.** Adding sections at fixed $\epsilon$ buys a
larger angle, never a better one.

Now the speed. The synthesized rate is

$$J \;=\; \frac{\theta_{\rm sig}}{\tau} \;=\; \frac{2\epsilon^{2}}{8\,\epsilon/g}
\;=\; \frac{g\,\epsilon}{4},
\qquad\qquad
T_{\rm phys} \;=\; \frac{\Theta}{J} \;=\; \frac{4\Theta}{g\,\epsilon}.$$

$$\boxed{\;\frac{J}{g} = \frac{\epsilon}{4}\;}$$

The synthesized interaction is weaker than the native one **by the factor $\epsilon$** — the
price of buying it at second order. Accuracy depends on $\epsilon$; speed depends on the
*product* $g\epsilon$.

---

## 3. Consequences

### 3.1 Improving accuracy by $k$ at fixed $g$

$$\epsilon \to \frac{\epsilon}{\sqrt{k}}
\quad\Longrightarrow\quad
N = \frac{\Theta}{2\epsilon^{2}} \to N\,k,
\qquad
T_{\rm phys} = \frac{8N\epsilon}{g} \to T_{\rm phys}\sqrt{k}.$$

**$N$ grows by $k$, the physical time by $\sqrt{k}$.** (A common slip is to assign the
$\sqrt{k}$ to $N$; it belongs to the time.)

### 3.2 Improving accuracy by $k$ at fixed *time*

Since the error sees only $\epsilon$ but the speed sees $g\epsilon$, raising $g$ is free
accuracy-wise:

$$\epsilon \to \frac{\epsilon}{\sqrt{k}},\qquad
g \to g\sqrt{k}
\quad\Longrightarrow\quad
dt \to \frac{dt}{k},\qquad N \to N k,\qquad T_{\rm phys}\ \text{unchanged}.$$

**So yes — $g$ must change** if the improvement is to be free in time. At fixed $g$ you pay
$\sqrt{k}$.

### 3.3 The accuracy–speed trade

$$J \;\propto\; \sqrt{\text{relative error}}\qquad\text{(at fixed }g\text{)}$$

Speed and accuracy are the *same dial*, because both are controlled by $\epsilon$.

---

## 4. Comparison with ordinary Trotterization

For $H = H_A + H_B$ with interaction scale $\Lambda$ and a symmetrized (Strang) step
$e^{-iH_A dt/2}e^{-iH_B dt}e^{-iH_A dt/2}$:

$$\delta_{\rm err} = \mathcal{O}\!\big((\Lambda dt)^{3}\big),
\qquad
\frac{\text{error}}{\text{signal}} \sim (\Lambda dt)^{2}.$$

Also second order. Accuracy by $k$ needs $\Lambda dt \to \Lambda dt/\sqrt{k}$, hence
$N \to N\sqrt{k}$, and with a gate-length floor $t_0$ the time grows as $N t_0 \to \sqrt{k}$.

**The exponent is identical. Symmetrization removes the advantage.**

| | ordinary Strang | group commutator |
|---|---|---|
| signal per step | $\Lambda\,dt$ — **first order** | $2\epsilon^{2}$ — **second order** |
| leading error | $(\Lambda dt)^{3}$ | $\epsilon^{4}$ |
| relative error | $(\Lambda dt)^{2}$ | $\epsilon^{2}$ |
| $N$ for accuracy $\times k$ | $\times\sqrt{k}$ | $\times k$ |
| $T_{\rm phys}$, **no** gate floor | **unchanged** | $\times\sqrt{k}$ |
| $T_{\rm phys}$, floor $t_0$ | $\times\sqrt{k}$ | $\times\sqrt{k}$ |
| overhead vs native rate | $t_0/dt$ | $4/\epsilon$ |
| per-step cost | constant $t_0$ (wasted as $\lambda\to0$) | $\propto\epsilon$ (no waste) |

### 4.1 Where the difference actually is

**The order of the kept term, not the order of the error.**

- Ordinary Trotter keeps $H_A dt$ and $H_B dt$ at **first** order. If gate length scales with
  gate angle, refining $N$ is *free*: the steps shrink in proportion, $T_{\rm phys} = T$
  exactly. A floor $t_0$ is what creates the overhead $t_0/dt$.
- The commutator keeps $2\epsilon^{2}$ at **second** order, so the overhead $4/\epsilon$ is
  paid *always*, floor or no floor.

Formally the two overheads have the same structure under $\epsilon \leftrightarrow dt/t_0$.

### 4.2 Both are uncapped — the difference is where the waste is

Set the two side by side at the *same endpoint*: reach a target angle $\Theta$ at relative
error $\eta$.

**A — first-order kept term, floored gate duration.**
The per-gate rotation angle $\lambda$ is **freely choosable** — you may simulate an arbitrarily
small $\Lambda$ — but a gate costs $t_0$ of wall time *however small its angle is*. A Strang
step is 3 gates, so $3t_0$ per step regardless of $\lambda$.

**B — second-order kept term, no floor on the small parameter (ours).**
$\epsilon = g\,dt$, and a pulse costs $dt = \epsilon/g$ — the cost **shrinks with** $\epsilon$.

| | **A**: 1st-order kept, floored gate | **B**: 2nd-order kept, unfloored |
|---|---|---|
| free knob | gate angle $\lambda$ | pulse area $\epsilon$ |
| kept per step | $\lambda$ | $2\epsilon^{2}$ |
| leading error / step | $\lambda^{3}$ | $c\,\epsilon^{4}$ |
| relative error $\eta$ | $\lambda^{2}$ | $\tfrac{c}{2}\epsilon^{2}$ |
| knob at error $\eta$ | $\lambda=\sqrt{\eta}$ | $\epsilon=\sqrt{2\eta/c}$ |
| steps for angle $\Theta$ | $N_A=\Theta/\lambda \propto \eta^{-1/2}$ | $N_B=\Theta/2\epsilon^{2} \propto \eta^{-1}$ |
| wall time **per step** | $3t_0$ — **constant** | $8\epsilon/g \propto \eta^{1/2}$ |
| **total $T_{\rm phys}$** | $\dfrac{3t_0\,\Theta}{\sqrt{\eta}}$ | $\dfrac{2\sqrt{2c}\,\Theta}{g\sqrt{\eta}}$ |
| scaling in $\eta$ | $\eta^{-1/2}$ | $\eta^{-1/2}$ |
| accuracy limit | **none** — shrink $\lambda$ | **none** — shrink $\epsilon$ |

**Neither scheme caps accuracy, and both cost $\eta^{-1/2}$.** They reach it by opposite routes,
and that is the whole content of the comparison:

- **A wastes per step but needs few steps.** Each gate burns a full $t_0$ to deliver a rotation
  $\lambda \to 0$, so the useful fraction of its wall time is $\lambda/(\Lambda t_0) \to 0$. But
  the kept term is first order, so only $N_A \propto \eta^{-1/2}$ steps are needed.
- **B has no waste per step but needs many steps.** Pulse time is strictly proportional to the
  work done. But the kept term is second order, so $N_B \propto \eta^{-1}$ — quadratically more.

$$N_A \propto \eta^{-1/2}\ \ \text{with constant cost}
\qquad\text{vs}\qquad
N_B \propto \eta^{-1}\ \ \text{with cost} \propto \eta^{1/2}$$

The two effects cancel **exactly**, which is why the exponents match.

### 4.3 The crossover

Since both are $\eta^{-1/2}$, the ratio is a pure constant — independent of $\eta$ *and*
$\Theta$:

$$\frac{T_{\rm phys}^{B}}{T_{\rm phys}^{A}}
= \frac{2\sqrt{2c}}{3\,g\,t_0}
\qquad\Longrightarrow\qquad
\boxed{\;B\ \text{wins}\iff g\,t_0 \gtrsim 1\;}$$

Read it as a comparison of two times: the hardware floor $t_0$ against the native interaction
time $1/g$.

- $t_0 \gg 1/g$ — a floor-length gate is a large rotation, so A is throwing away most of every
  gate. **B wins**: its pulse time tracks the work done.
- $t_0 \ll 1/g$ — the floor is negligible, A's per-step waste vanishes, and its first-order
  kept term is decisive. **A wins.**

Equivalently, $g\,t_0$ is the rotation a minimum-length gate would produce. The commutator
scheme's unconditional $4/\epsilon$ overhead is worth paying exactly when the competing scheme
throws away more than that in floor time.

> $c \approx 2.5$ (fitted, operator norm) and the $3$ in $3t_0$ are both $\mathcal{O}(1)$ and
> convention-dependent, so the criterion is $g\,t_0 \sim 1$, not a sharp $1.5$. If a Strang step
> needs a different gate count, only that constant moves.

---

## 5. Verdict

**There is no scaling advantage, in either direction.** Both schemes cost $\eta^{-1/2}$ in time
and neither caps accuracy. The second-order kept term does not make the commutator scheme
asymptotically worse, and the first-order kept term does not make ordinary Trotter
asymptotically better — because a floored gate duration is exactly the handicap that cancels a
first-order kept term:

$$\underbrace{N_A \propto \eta^{-1/2}}_{\text{few steps}} \times
\underbrace{3t_0}_{\text{constant cost}}
\;\;\sim\;\;
\underbrace{N_B \propto \eta^{-1}}_{\text{many steps}} \times
\underbrace{8\epsilon/g \propto \eta^{1/2}}_{\text{shrinking cost}}$$

So the whole question is a **prefactor**, and it is decided by one dimensionless number:

$$g\,t_0 \;=\; \frac{\text{hardware floor}}{\text{native interaction time}}
\qquad\Longrightarrow\qquad
g\,t_0 \gtrsim 1 \ \Rightarrow\ \text{commutator wins.}$$

Two standing points that sit outside the scaling argument:

1. The target interaction here has **no native implementation**, so for this Hamiltonian there
   is no competing scheme to be slower than — the comparison above is with a hypothetical
   machine that has $H_A$ and $H_B$ natively.
2. The commutator's $4/\epsilon$ overhead is paid unconditionally, whereas A's $t_0$ waste
   vanishes if the floor is ever lifted. If gate durations became proportional to their angles,
   A would win outright.

---

## 6. Numerical check

Exact propagator on the two excitation sectors, $N$ chosen first so
$\epsilon = \sqrt{\Theta/2N}$ and the achieved angle is exactly $\Theta = \pi/4$. Error is the
operator-norm distance to the converged synthesis ($\epsilon = 0.015$).

```
     N      eps   error angle  angle/eps^2   T_phys (g=1)
     2   0.4431     2.145e-01       1.0923           7.09
     8   0.2216     5.183e-02       1.0559          14.18
    32   0.1108     1.266e-02       1.0316          28.36
   128   0.0554     2.980e-03       0.9713          56.72
   256   0.0392     1.371e-03       0.8941          80.21

error angle ~ eps^2.07        T_phys ~ eps^-1.00
=> improving accuracy by k costs k^0.48 in time
```

$\text{error}/\epsilon^{2} \approx 1.0$ across a 11× range in $\epsilon$, confirming
$\text{error} \propto \epsilon^{2}$ at fixed $\Theta$, and $k^{0.48} \approx \sqrt{k}$.

> The coefficient is $\mathcal{O}(1)$ and convention-dependent (operator norm over the full
> 8-dimensional space). The **exponents** are the result.

A caution when fitting this yourself: **infidelity is the square of the error angle**, so a
naive fidelity fit reads $\epsilon^{4}$, not $\epsilon^{2}$. Fit an angle (operator norm), not
$1-F$. Also choose $N$ first and let $\epsilon$ follow — rounding $N = \Theta/2\epsilon^{2}$ to
an integer perturbs the achieved angle and dominates the residual.

Reproduce with `scaling3.py` (next to this file), or from
`cswap_trotter_runtime_sim.ipynb` using its `synthesized()` construction.

---

## 7. Expected state for an $\lvert N,0\rangle$ cavity prep

What the sequence produces when cavity 1 is loaded with $N$ photons instead of one. Throughout
this section $N$ is the **photon number** and $n$ is the number of $W_{\rm sym}$ sections
(the runtime's "Trotter steps") — a deliberate reuse of $N$ from §1, flagged here because the
photon number is what varies in this section and the section count only enters through $\Theta$.

### 7.1 The sequence

| step | operation |
|---|---|
| 0 | prep cavities in $\lvert N,0\rangle$, qubit in $\lvert g\rangle$ |
| 1 | qubit $\pi/2$ $\Rightarrow$ $\lvert +x\rangle = (\lvert g\rangle + \lvert e\rangle)/\sqrt2$ |
| 2 | $n$ sections of $W_{\rm sym}$, synthesizing $U(\Theta)$ |
| 3 | closing qubit $\pi/2$ about $\sigma^y$ |
| 4 | readout: qubit, then photon counting on both cavities |

Step 2 synthesizes the conditional beamsplitter

$$U(\Theta) \;=\; \exp\!\big[-i\,\Theta\,\sigma^z (a^\dagger b + a b^\dagger)\big],
\qquad
\sigma^z = P_g - P_e,
\qquad
\Theta = n\,\theta_{\rm step} = 2n\,\epsilon_A\epsilon_B .$$

### 7.2 Step 1 — qubit into the equal superposition

$$\lvert \psi_1\rangle \;=\; \tfrac{1}{\sqrt2}\big(\lvert g\rangle + \lvert e\rangle\big)\otimes\lvert N,0\rangle .$$

### 7.3 Step 2a — split $U$ into its two branches

$\sigma^z$ is diagonal, so $U$ is block diagonal on the qubit:

$$U(\Theta) \;=\; P_g\otimes B(+\Theta) \;+\; P_e\otimes B(-\Theta),
\qquad
B(\theta) \equiv \exp\!\big[-i\theta\,(a^\dagger b + a b^\dagger)\big].$$

$B(\theta)$ is an ordinary (unconditional) beamsplitter of angle $\theta$. **The qubit state
selects the sign of the beamsplitter angle** — that is the entire content of the gate.

### 7.4 Step 2b — a beamsplitter acting on $\lvert N,0\rangle$

Work in the Heisenberg picture. $B$ rotates the mode operators,

$$B(\theta)\,a^\dagger\,B^\dagger(\theta) \;=\; a^\dagger\cos\theta \;-\; i\,b^\dagger\sin\theta .$$

Write the Fock state as a creation operator acting on vacuum, $\lvert N,0\rangle = \frac{(a^\dagger)^N}{\sqrt{N!}}\lvert 0,0\rangle$, and use $B\lvert 0,0\rangle = \lvert 0,0\rangle$:

$$B(\theta)\lvert N,0\rangle
\;=\; \frac{1}{\sqrt{N!}}\big(B a^\dagger B^\dagger\big)^{N} \lvert 0,0\rangle
\;=\; \frac{1}{\sqrt{N!}}\big(a^\dagger\cos\theta - i\,b^\dagger\sin\theta\big)^{N}\lvert 0,0\rangle .$$

$a^\dagger$ and $b^\dagger$ commute, so the binomial theorem applies directly. Using
$(a^\dagger)^{N-k}(b^\dagger)^{k}\lvert 0,0\rangle = \sqrt{(N-k)!\,k!}\;\lvert N-k,k\rangle$:

$$\boxed{\;B(\theta)\lvert N,0\rangle \;=\; \sum_{k=0}^{N}\sqrt{\binom{N}{k}}\;
(\cos\theta)^{N-k}\,(-i\sin\theta)^{k}\;\lvert N-k,\,k\rangle\;}$$

The photons split **independently and binomially**: each one transfers with probability
$\sin^2\theta$. (Verified against the exact propagator to $2.6\times10^{-16}$ for $N\le5$.)

### 7.5 Step 2c — recombine the two branches

Apply 7.4 with $\theta=+\Theta$ on the $g$ branch and $\theta=-\Theta$ on the $e$ branch. Only
the $(-i\sin\theta)^k$ factor is sign-sensitive, so the two branches differ by exactly
$(-1)^k$. Writing $c=\cos\Theta$, $s=\sin\Theta$:

$$\lvert\psi_2\rangle = \frac{1}{\sqrt2}\sum_{k=0}^{N}\sqrt{\binom{N}{k}}\,c^{\,N-k}s^{\,k}\,i^{\,k}
\Big[(-1)^{k}\lvert g\rangle + \lvert e\rangle\Big]\lvert N-k,k\rangle .$$

The bracket is $\sqrt2\,\lvert +x\rangle$ for even $k$ and $-\sqrt2\,\lvert -x\rangle$ for odd
$k$; absorbing that sign into $i^k \to (-i)^k$ gives the compact form

$$\boxed{\;\lvert\psi_2\rangle \;=\; \sum_{k=0}^{N}\sqrt{\binom{N}{k}}\;
(\cos\Theta)^{N-k}(\sin\Theta)^{k}\,(-i)^{k}\;
\big\lvert (-1)^{k}x \big\rangle \otimes \lvert N-k,\,k\rangle\;}$$

**The qubit is entangled with the *parity* of the number of transferred photons**, not with the
number itself. (Verified: state distance $4\times10^{-16}$ from the exact propagator.)

### 7.6 Step 3 — the closing $\pi/2$

The closing rotation maps $\lvert +x\rangle\to\lvert e\rangle$, $\lvert -x\rangle\to\lvert g\rangle$:

$$\lvert\psi_3\rangle \;=\; \sum_{k=0}^{N}\sqrt{\binom{N}{k}}\,c^{\,N-k}s^{\,k}\,(-i)^{k}\;
\lvert q_k\rangle\,\lvert N-k,\,k\rangle,
\qquad
q_k = \begin{cases} e, & k \text{ even}\\ g, & k \text{ odd}\end{cases}$$

This is the measured state. It is a **perfectly correlated** classical mixture over $k$ after
readout: every $\lvert N-k,k\rangle$ appears with exactly one qubit outcome.

### 7.7 Step 4 — what the measurements read

Set $p=\sin^2\Theta$, $q=\cos^2\Theta$, and $u \equiv q-p = \cos 2\Theta$.

**Photon partition** — binomial, independent of the qubit:

$$P(k) = \binom{N}{k} q^{\,N-k} p^{\,k},
\qquad \langle n_2\rangle = Np, \qquad \langle n_1\rangle = Nq .$$

**Qubit fringe** — sum the even-$k$ terms using $\sum_k \binom{N}{k}q^{N-k}(\pm p)^k = (q\pm p)^N$:

$$\boxed{\;P(e) = \frac{(q+p)^N + (q-p)^N}{2} = \frac{1 + \cos^{N}(2\Theta)}{2},
\qquad P(g) = \frac{1-\cos^{N}(2\Theta)}{2}\;}$$

**Conditional photon number** — from $\sum_k k\binom{N}{k}q^{N-k}(pz)^k = Npz\,(q+pz)^{N-1}$
evaluated at $z=\pm1$ and halved:

$$\langle n_2\rangle_e = \frac{N p\,\big(1-u^{\,N-1}\big)}{1+u^{\,N}},
\qquad
\langle n_2\rangle_g = \frac{N p\,\big(1+u^{\,N-1}\big)}{1-u^{\,N}} .$$

All four boxed/displayed results were checked against the exact three-body propagator for
$N=1\ldots4$ at $\Theta = 0.3,\ \pi/6,\ \pi/4$ — agreement to machine precision.

### 7.8 Limits and sanity checks

| case | result |
|---|---|
| $N=1$ | $P(e)=\cos^2\Theta$; state is $\cos\Theta\lvert e\rangle\lvert1,0\rangle - i\sin\Theta\lvert g\rangle\lvert0,1\rangle$ — the known single-photon answer |
| $\Theta\to0$ | $P(e)\simeq 1-N\Theta^2$, $\langle n_2\rangle_e \simeq N(N-1)\Theta^4$ |
| $\Theta=\pi/4$ (at $n^*$) | $u=0$: 50:50 splitter, $P(k)=\binom{N}{k}/2^N$, $P(e)=\tfrac12$, $\langle n_2\rangle_e = N/2$ for $N\ge2$ |
| $\Theta=\pi/2$ | $u=-1$: full swap to $\lvert 0,N\rangle$; $P(e)=1$ for even $N$, $0$ for odd $N$ |

### 7.9 The one experimentally important consequence

$$P(e) = \tfrac12\big[1 + \cos^{N}(2\Theta)\big]
\qquad\textbf{not}\qquad
\tfrac12\big[1 + \cos(2N\Theta)\big].$$

Loading $N$ photons does **not** give an $N$-fold faster qubit fringe. It gives a fringe whose
*contrast decays* as $\cos^N$ — at $\Theta=0.31$, $N=5$ reads $P(e)=0.679$ where an $N$-fold
speed-up would read $0.0004$. The reason is 7.5: the qubit tracks only the **parity** of the
transferred photons, and a binomial distribution's parity bias washes out as $\cos^N(2\Theta)$
once more than one photon can move. For even $N$ the fringe never even crosses $\tfrac12$.

The consequence for calibration: **the qubit fringe is not a usable phase reference for $N>1$.**
Calibrate $\theta_{\rm step}$ at $N=1$, where the fringe is full-contrast $\cos^2\Theta$, and
use the $N$-photon prep only to read the cavity-side observables ($P(k)$, $\langle n_2\rangle$),
which retain full binomial structure at every $N$.

### 7.10 The state written out, $N = 1 \ldots 8$

Applying 7.6 term by term, with $c \equiv \cos\Theta$, $s \equiv \sin\Theta$ and
$\Theta = 2n\,\epsilon_A\epsilon_B$. Each line is the **full state after the closing
$\pi/2$** — the thing the readout sees. Before that pulse the same lines hold with
$\lvert e\rangle \to \lvert +x\rangle$ and $\lvert g\rangle \to \lvert -x\rangle$.

**$N = 1$**

$$\lvert\psi\rangle = c\,\lvert e\rangle\lvert 1,0\rangle - is\,\lvert g\rangle\lvert 0,1\rangle$$

**$N = 2$**

$$\lvert\psi\rangle = c^{2}\,\lvert e\rangle\lvert 2,0\rangle - \sqrt{2}ics\,\lvert g\rangle\lvert 1,1\rangle - s^{2}\,\lvert e\rangle\lvert 0,2\rangle$$

**$N = 3$**

$$\lvert\psi\rangle = c^{3}\,\lvert e\rangle\lvert 3,0\rangle - \sqrt{3}ic^{2}s\,\lvert g\rangle\lvert 2,1\rangle - \sqrt{3}cs^{2}\,\lvert e\rangle\lvert 1,2\rangle + is^{3}\,\lvert g\rangle\lvert 0,3\rangle$$

**$N = 4$**

$$\lvert\psi\rangle = c^{4}\,\lvert e\rangle\lvert 4,0\rangle - 2ic^{3}s\,\lvert g\rangle\lvert 3,1\rangle - \sqrt{6}c^{2}s^{2}\,\lvert e\rangle\lvert 2,2\rangle + 2ics^{3}\,\lvert g\rangle\lvert 1,3\rangle + s^{4}\,\lvert e\rangle\lvert 0,4\rangle$$

**$N = 5$**

$$\lvert\psi\rangle = c^{5}\,\lvert e\rangle\lvert 5,0\rangle - \sqrt{5}ic^{4}s\,\lvert g\rangle\lvert 4,1\rangle - \sqrt{10}c^{3}s^{2}\,\lvert e\rangle\lvert 3,2\rangle + \sqrt{10}ic^{2}s^{3}\,\lvert g\rangle\lvert 2,3\rangle + \sqrt{5}cs^{4}\,\lvert e\rangle\lvert 1,4\rangle - is^{5}\,\lvert g\rangle\lvert 0,5\rangle$$

**$N = 6$**

$$\lvert\psi\rangle = c^{6}\,\lvert e\rangle\lvert 6,0\rangle - \sqrt{6}ic^{5}s\,\lvert g\rangle\lvert 5,1\rangle - \sqrt{15}c^{4}s^{2}\,\lvert e\rangle\lvert 4,2\rangle + 2\sqrt{5}ic^{3}s^{3}\,\lvert g\rangle\lvert 3,3\rangle + \sqrt{15}c^{2}s^{4}\,\lvert e\rangle\lvert 2,4\rangle - \sqrt{6}ics^{5}\,\lvert g\rangle\lvert 1,5\rangle - s^{6}\,\lvert e\rangle\lvert 0,6\rangle$$

**$N = 7$**

$$\lvert\psi\rangle = c^{7}\,\lvert e\rangle\lvert 7,0\rangle - \sqrt{7}ic^{6}s\,\lvert g\rangle\lvert 6,1\rangle - \sqrt{21}c^{5}s^{2}\,\lvert e\rangle\lvert 5,2\rangle + \sqrt{35}ic^{4}s^{3}\,\lvert g\rangle\lvert 4,3\rangle + \sqrt{35}c^{3}s^{4}\,\lvert e\rangle\lvert 3,4\rangle - \sqrt{21}ic^{2}s^{5}\,\lvert g\rangle\lvert 2,5\rangle - \sqrt{7}cs^{6}\,\lvert e\rangle\lvert 1,6\rangle + is^{7}\,\lvert g\rangle\lvert 0,7\rangle$$

**$N = 8$**

$$\lvert\psi\rangle = c^{8}\,\lvert e\rangle\lvert 8,0\rangle - 2\sqrt{2}ic^{7}s\,\lvert g\rangle\lvert 7,1\rangle - 2\sqrt{7}c^{6}s^{2}\,\lvert e\rangle\lvert 6,2\rangle + 2\sqrt{14}ic^{5}s^{3}\,\lvert g\rangle\lvert 5,3\rangle + \sqrt{70}c^{4}s^{4}\,\lvert e\rangle\lvert 4,4\rangle - 2\sqrt{14}ic^{3}s^{5}\,\lvert g\rangle\lvert 3,5\rangle - 2\sqrt{7}c^{2}s^{6}\,\lvert e\rangle\lvert 2,6\rangle + 2\sqrt{2}ics^{7}\,\lvert g\rangle\lvert 1,7\rangle + s^{8}\,\lvert e\rangle\lvert 0,8\rangle$$

Every one of these was checked against the exact three-body propagator at
$\Theta = 0.23,\ \pi/7,\ \pi/4,\ 1.05$ — worst state distance $6.8\times10^{-16}$.

Reading them: the cavity kets march through the binomial ladder $\lvert N,0\rangle \to
\lvert 0,N\rangle$ with amplitudes $\sqrt{\binom{N}{k}}c^{N-k}s^k$, while the qubit label
**alternates $e,g,e,g,\ldots$ with $k$** and the phase runs through the four-cycle
$+,\,-i,\,-,\,+i$. Those are the only two things that change with $N$.


### 7.11 Fringe and its harmonic content

Squaring the $\lvert e\rangle$ terms above gives $P(e)$ directly, and expanding
$\cos^N(2\Theta)$ with $\cos^N x = 2^{-N}\sum_j \binom{N}{j}\cos[(N-2j)x]$ gives its
harmonics:

| $N$ | $P(e)$ from the amplitudes | $= \tfrac12\big[1+\cos^N 2\Theta\big]$ expanded in harmonics |
|---|---|---|
| 1 | $c^{2}$ | $\tfrac{1}{2} + \tfrac{1}{2}\cos 2\Theta$ |
| 2 | $c^{4} + s^{4}$ | $\tfrac{3}{4} + \tfrac{1}{4}\cos 4\Theta$ |
| 3 | $c^{6} + 3c^{2}s^{4}$ | $\tfrac{1}{2} + \tfrac{1}{8}\cos 6\Theta + \tfrac{3}{8}\cos 2\Theta$ |
| 4 | $c^{8} + 6c^{4}s^{4} + s^{8}$ | $\tfrac{11}{16} + \tfrac{1}{16}\cos 8\Theta + \tfrac{1}{4}\cos 4\Theta$ |
| 5 | $c^{10} + 10c^{6}s^{4} + 5c^{2}s^{8}$ | $\tfrac{1}{2} + \tfrac{1}{32}\cos 10\Theta + \tfrac{5}{32}\cos 6\Theta + \tfrac{5}{16}\cos 2\Theta$ |
| 6 | $c^{12} + 15c^{8}s^{4} + 15c^{4}s^{8} + s^{12}$ | $\tfrac{21}{32} + \tfrac{1}{64}\cos 12\Theta + \tfrac{3}{32}\cos 8\Theta + \tfrac{15}{64}\cos 4\Theta$ |
| 7 | $c^{14} + 21c^{10}s^{4} + 35c^{6}s^{8} + 7c^{2}s^{12}$ | $\tfrac{1}{2} + \tfrac{1}{128}\cos 14\Theta + \tfrac{7}{128}\cos 10\Theta + \tfrac{21}{128}\cos 6\Theta + \tfrac{35}{128}\cos 2\Theta$ |
| 8 | $c^{16} + 28c^{12}s^{4} + 70c^{8}s^{8} + 28c^{4}s^{12} + s^{16}$ | $\tfrac{163}{256} + \tfrac{1}{256}\cos 16\Theta + \tfrac{1}{32}\cos 12\Theta + \tfrac{7}{64}\cos 8\Theta + \tfrac{7}{32}\cos 4\Theta$ |

Two things are visible only in this form:

1. **The fast harmonic exists but is exponentially small.** The $\cos 2N\Theta$ term —
   the one an $N$-fold speed-up would need — is present at every $N$, with amplitude
   exactly $2^{-N}$: $\tfrac12,\tfrac14,\tfrac18,\ldots,\tfrac{1}{256}$ at $N=8$. It is
   swamped by the slow harmonics, whose weights are the same binomial coefficients.
2. **Odd and even $N$ behave differently.** For odd $N$ the offset is exactly $\tfrac12$
   and the fringe still reaches $P(e)=0$; for even $N$ the offset is $>\tfrac12$
   ($\tfrac34, \tfrac{11}{16}, \tfrac{21}{32}, \tfrac{163}{256}$) and the fringe never
   crosses $\tfrac12$. This is the parity of 7.5 showing up as a DC offset: at
   $\Theta=\pi/2$ all $N$ photons have moved, and $k=N$ is even.

### 7.12 The $\sqrt{n}$: where the $N$-photon rate goes wrong

**The commutator is fine.** With $H_A = g_A(a\sigma^+e^{i\phi_A} + \text{h.c.})$ and
$H_B = g_B(b\sigma^+e^{i\phi_B} + \text{h.c.})$ at $\phi_A - \phi_B = \pi/2$,

$$[H_A, H_B] \;=\; -i\,g_Ag_B\,\sigma^z\big(a^\dagger b + a b^\dagger\big)$$

**exactly** — verified numerically to $0.00$, and it commutes with the total photon number. No
$\sqrt{n}$, no number operator. The target really is the linear beamsplitter at every photon
number, and $\Theta = 2n_{\rm Trotter}(gt)^2 = 2n\,\epsilon_A\epsilon_B$ is the right leading
order for all $N$.

**The $\sqrt{n}$ is in the individual legs, not in their commutator.** The sidebands drive
$\lvert g,n\rangle\!\leftrightarrow\!\lvert e,n-1\rangle$, matrix element $g\sqrt{n}$, so each
*leg* is stronger by $\sqrt{n}$ even though $[H_A,H_B]$ is not. The natural reading is that the
expansion parameter of $W_{\rm sym}$ becomes $\epsilon\sqrt{n}$, so that what the group
commutator discards grows with photon number while the kept term does not.

> **Status of that reading.** It is a plausible account of the measured law below, NOT a
> derivation, and it has not been verified. Attempts to pin down the operator form of the
> neglected term did not produce a trustworthy answer: a Hilbert-Schmidt decomposition over the
> full truncated space is dominated by Fock levels the experiment never reaches (the
> "unexplained" fraction grew from 3.0% to 13.8% as the truncation went from $n_{\rm cav}=3$ to
> 7, i.e. it was measuring the truncation); restricting to one excitation sector makes several
> natural candidate operators degenerate, so their fitted coefficients are not individually
> meaningful; and a sector-restricted process fidelity came out at 0.33 for $N=1$, where the
> train is known to be the beamsplitter to 1.5%, so that metric is mis-specified. **The law
> below is measured. The mechanism above is a conjecture.** Do not cite it as established.

What IS established is the empirical law -- a fractional error growing linearly in the photon
number:

$$\boxed{\;\frac{n^*(N)}{n^*(1)} \;\simeq\; 1 + \epsilon^2\,(N-1)\;}$$

Dividing the slowdown measured on the exact train by $\epsilon^2(N-1)$:

| $\epsilon$ | $N=2$ | $N=3$ | $N=5$ | $N=8$ |
|---|---|---|---|---|
| 0.050 | 0.985 | 0.987 | 0.990 | 0.996 |
| 0.100 | 0.984 | 0.992 | 1.007 | 1.030 |
| 0.150 | 0.984 | 1.000 | 1.035 | 1.093 |
| 0.213 | 0.985 | 1.016 | 1.093 | 1.227 |

A constant $\approx 1$, drifting up only where $\epsilon^2 N$ stops being small. The
coefficient is 1 and the law is $\epsilon^2(N-1)$.

**At the deployed areas** ($\epsilon_A = 0.2132$, $\epsilon_B = 0.2112$, $\epsilon^2 = 0.045$):

| $N$ | $n^*$ (exact train) | vs $N=1$ | predicted $1+\epsilon^2(N-1)$ |
|---|---|---|---|
| 1 | 8.85 | 1.000 | 1.000 |
| 2 | 9.25 | 1.045 | 1.045 |
| 4 | 10.12 | 1.143 | 1.135 |
| 8 | 12.28 | 1.388 | 1.315 |

**Confirmed on hardware.** Run `260911/231424` (`|8,0>`, 1073 shots/pt) crosses
$\langle n_1\rangle = \langle n_2\rangle$ at $n^* = 12.45$ against the train's **12.28** — 1.4%.
The closed form alone would have said 8.72, making a correct gate look 43% wrong. The `|1,0>`
runs sit at 9.53 against 8.85, a 7.7% residual that is ordinary calibration error.

The distribution narrows for the same reason: at its own $n^*$, $N=8$ gives
$[0,\,0.006,\,0.068,\,0.228,\,0.330,\,0.244,\,0.100,\,0.022,\,0.002]$ where
$\binom{8}{j}/2^8$ says $[0.004,\,0.031,\,0.109,\,0.219,\,0.273,\ldots]$.

So: §7's closed forms are exact at $N=1$ and are the right leading order at any $N$. For $N>1$
the plots build their reference by **running the train** (`train_reference` in
`controlled_swap_trotter_test.py`), which is exact at every photon number and costs nothing --
the train conserves $M = n_1+n_2+q$, so a lossless reference lives in a $2N+1$-state sector (17
states at $N=8$ rather than 200), verified against a full-space solve to $1.3\times10^{-15}$.

### 7.13 What this costs

1. **$n^*$ is photon-number dependent.** Calibrate at $N=1$, use it at $N=8$, and the gate is
   under-rotated by 39%. Each photon number needs its own $n^*$ — or it must be read off the
   data, as the $\langle n_1\rangle = \langle n_2\rangle$ crossing does.

2. **It is not a state-independent CSWAP.** A true controlled swap acts identically on any
   cavity state. Here each Fock component rotates at its own rate, so a superposition over
   photon number — a coherent state, or $\lvert0\rangle+\lvert1\rangle$ — does not swap: its
   components arrive at different angles and the superposition dephases. The gate is a genuine
   conditional beamsplitter **within one photon number**, which is exactly the regime the
   single-photon herald tests.

3. **The $\epsilon^2$ accuracy knob of §2 is really $\epsilon^2 N$.** §2's
   error/signal $= (c/2)\epsilon^2$ is independent of the number of *sections*, but not of the
   photon number. Reaching a target accuracy with $N$ photons needs
   $\epsilon \propto 1/\sqrt{N}$, and since $T_{\rm phys}\propto 1/\epsilon$ at fixed $\Theta$,
   the time cost grows as $\sqrt{N}$. A mild penalty, not a wall — but it belongs in the §4
   comparison, where it is currently absent.
