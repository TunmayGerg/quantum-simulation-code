# Strategic CD cancellation study

The target is

$$
U_{\mathrm{CD}}(1)=\exp[\sigma_x(a^\dagger-a)] .
$$

Every fidelity below uses all 18 inputs, qubit $g/e$ times Fock states
$n=0,\ldots,8$. The coherent process score is

$$
F=\frac{\left|\sum_{j=1}^{18}\langle T_j|W_j\rangle\right|^2}{18^2}.
$$

The native screening propagates the complete sampled two-level-qubit/oscillator
Hamiltonian. A screening score is not an SDK/RFDC or cutoff certification. The
coordinator independently verifies exported controls before using a gate as a
recommendation. No new gate in this study beat the accepted 370 ns, 99.2278%
constant-pump reference. These are finite, reproducible searches, not a proof of
global optimality.

## Constraints and variables

The sideband has one fixed complex plateau value throughout each gate, including
all internal dwells. Only the two outer 5 ns ramps change its magnitude. Its
coupling obeys $g\leq\pi/(2\times0.200\,\mathrm{us})$. Each qubit pulse occupies
30 ns; a $\pi/N$ pulse divides the envelope amplitude by integer $N$.

The target displacement remains exactly $\beta=1$ in every quantum objective.
When a constant coupling is fitted, it corrects the finite-order displacement
gain; it does not redefine the target. Comparisons of native scores at different
couplings must allow the duration to change, since keeping duration fixed would
change the intended displacement.

## Why the first promising construction failed

Define the identity block

$$
B=R_X(\alpha)R_\psi(\gamma)R_{\psi+\pi}(\gamma)R_{-X}(\alpha),
\qquad \alpha=\pi/N_\alpha,\quad\gamma=\pi/N_\gamma,
$$

where the displayed order is the time order. The full pulse word is
$B,X_\pi,B,X_\pi$. A 5 ns dwell after the first bridge $X_\pi$ balances the
combined bare-JC area from the two outer ramps.

Writing $Q(t)$ for the qubit-only propagator, the first averaged sideband
coefficient is

$$
M=\int dt\,s(t)Q^\dagger(t)\sigma_+Q(t).
$$

The two copies of $B$ are related by qubit $X$ conjugation, which projects $M$
onto a complex $\sigma_x$ coefficient. One constant pump phase sets the desired
oscillator direction. This is a correct first-order CD construction: 54 native
finite-Hann cases had unwanted Pauli coefficients below $2.16\times10^{-16}$ us
and zero-coupling qubit closure errors below $3.31\times10^{-15}$.

It is not a high-fidelity short gate. The second Magnus term survives. The
independent reviewer found conditional number and squeezing terms in it. Across
36 low-parameter fits at the maximum coupling, the best 315 ns native fidelity
was only 65.1285%. Correct first-order averaging does not establish useful
finite-strength accuracy.

## Time symmetry and fractional-angle geometry

For a halfword $P$, the second half uses the inverse rotations in reverse order.
If the actual waveform also satisfies $q(T-t)=-q(t)$, then
$Q(T-t)=Q(t)$ and the toggling-frame Hamiltonian is time symmetric. Its second
Magnus term vanishes. The current start-of-sample Hann array is not exactly
palindromic: reversing only the phase labels leaves a small second-order term.
The explicitly symmetric studies therefore use a sample-centered Hann envelope
and symmetric outer ramps.

For controls exclusively about $\pm X$, the unwanted first-order force is the
angular-path centroid

$$
\int dt\,\exp[i\theta(t)] .
$$

The fractional-angle study enumerates half-path steps $\pm\pi$ and
$\pm\pi/2$. A two-constraint linear program finds positive dwell weights at
angular vertices that close this centroid. The second half retraces the path.
The actual DAC-rounded residual is stored in every result. This explicitly
separates a meaningful geometric seed from an unconstrained control search.

The independent reviewer also established why this approach cannot cancel all
third-order error. In the oscillator principal cubic symbol, a longitudinal
error coefficient for an $X$-only path is proportional to

$$
-\frac14\int_0^T dt\,
\left|\int_0^t ds\,e^{i\theta(s)}\right|^2 .
$$

It has one sign. Timing can reduce it, but an additional qubit $X$ twirl cannot
remove it because it preserves both the desired $\sigma_x$ force and the
unwanted $\sigma_x$ nonlinear force. Mixed axes are needed to escape this
particular obstruction. The reviewer checked the functional against the native
cubic coefficient to $8.5\times10^{-21}$ us$^3$; see
`../independent_review/same_axis_cubic.json`.

The mixed-axis refinement fits five halfword axes and one constant coupling,
with four real first-moment constraints. Quibit closure and second-order time
symmetry come from the inverse construction. Its best native score was 97.4386%
at 330 ns, so it remains rejected.

## Completed comparisons

| Concept | Compared | Best native fidelity | Duration | Outcome |
|---|---:|---:|---:|---|
| Inverse $B$ plus $X$ twirl | 36 fits; 4,056 gate evaluations | 65.1285% | 315 ns | First order correct; second order survives |
| Symmetric eight-$\pi$ words and dwells | 108 gates | 96.2835% | 265 ns | No improvement to 99% |
| All proper ten-$\pi$ sign paths and ramp-balancing positions | 630 gates | 98.0103% | 315 ns | Correct first-order force; nonlinear residual |
| Same best ten-$\pi$ path, one constant gain calibration | 1 scalar fit | 98.2330% | 315 ns | Exported as a rejected reference |
| Retraced fractional-angle centroid paths | 512 paths; 165 feasible gates | 97.6954% | 330 ns | No improvement |
| Eleven pulses with two $\pi/2$ and nine $\pi$ rotations | 28,160 closed paths; 1,012 scored gates below 370 ns | below 97% | below 370 ns | No improvement |
| Exact-first-force, all-$\pi$ mixed-axis retracing | 72 constrained seed fits | no feasible strength within cap | 310–340 ns | Maximum found force coefficient too small |
| Mixed-axis refinement of fractional-angle geometric seeds | 12 fits; 3,168 gate evaluations | 97.4386% | 330 ns | No improvement |

Each gate evaluation propagates 18 input columns. The detailed JSON files retain
the actual word, dwell samples, integer angle fractions, pump phase, coupling,
constraints and optimizer result. `best_proper10.npz` and
`best_mixed_geometric.npz` contain raw 800 MS/s Hamiltonian coefficients, with
matching pulse metadata JSON. Their fidelities must not be relabeled as final
played-waveform fidelities.

## Why canonical fourth-order Suzuki is slow here

For $A=H_{\rm JC}$ and $B=XAX$, the Pauli frames also provide
$-A=ZAZ$ and $-B=YAY$ while the physical sideband stays fixed. The ideal
fourth-order composition is

$$
S_4(t)=S_2(pt)S_2(qt)S_2(pt),\quad
S_2(t)=e^{-iAt/2}e^{-iBt}e^{-iAt/2},
$$

$$
p=\frac{1}{2-2^{1/3}},\qquad q=1-2p.
$$

After merging adjacent $A$ intervals, the absolute dwell time for $\beta=1$
is at least 777.598 ns at the allowed maximum coupling. Six 30 ns frame-change
pulses and the outer ramps increase the floor to 967.598 ns, before correcting
the unwanted first-order force introduced during those finite pulses. Thus this
canonical signed-dwell family cannot beat the 370 ns reference. This is a cost
bound for this explicit composition, not for all possible controls.

The ideal seven-exponential product was checked at three scales. Halving the
scale changed the matrix error by factors 30.9334 and 31.7301, approaching the
expected fifth-order local-error factor 32. The full audit is in
`signed_jc_fourth_order_cost.json`.

## Reproduction

From the project root, use `../../acadia_env/bin/python code/strategic_cd.py`
with one of `validate`, `inverse`, `balanced`, `allsigns`, `centroid`,
`fractional11`, `mixed`, `refine`, or `fourth`. The default cutoff for screening
is 28–32 oscillator levels; final gate certification must separately converge
both target and output columns through the actual SDK/RFDC waveform.

## Complete-waveform correction and the phase-control extension

The sample-symmetry caveat above concerned isolated words. A direct quantum
Cauchy derivative of the COMPLETE original370ns gate including ramps gives
`||Omega2/g^2||=5.44e-16`; its complete second-order term cancels. The ramp
first moment and cubic/higher terms remain. Phase-label palindromes alone
still do not prove symmetry for a general waveform. Full evidence is in
`../phase_arc_cd/whole_gate_magnus_check.json`.

The phase-enabled follow-up uses10ns constant-radius IQ arcs at qubit zeros.
Its mirrored370ns candidate reaches99.3052%; a relaxed version reaches99.3575%
with one declared final qubitZ frame. Neither is faster. Signed JC plateaus,
full phase windings, compensatedY paths and inverse-pair removal did not
produce a shorter99% gate in the tested families. The populated current study
is `../../notebooks/phase_controlled_cd.ipynb`.
