# Independent research review

These checks distinguish a correct leading generator from a useful fast gate.
They do not certify a new gate above 99%; native fidelity on all 18 input
columns remains the deciding test.

## CD: why the first ten-pulse idea fails

For the inverse loop B, the closed sequence B, X, B, X projects its first
interaction-frame JC moment onto sigma_x. A five-ns dwell in the X frame
balances the five-ns total area of the two outer sideband ramps. The resulting
force has two oscillator quadratures; one constant sideband phase can orient
it to the desired displacement. This phase is chosen once per gate recipe.

I integrated the qubit conjugation exactly with a Rodrigues formula, separately
from the synthesis agent's Gauss quadrature. For N=1,2,3,4 and inner phase 0 or
pi/4, all eight words close the qubit and have no unwanted first-order force.
Whole-word native Cauchy derivatives agree with this result over 2048 complex
matrix amplitudes, maximum error below 1.1e-15. Closure errors are below 1.6e-15.

This does **not** cancel the second Magnus term. For N=1, phase pi/4, a native
polynomial decomposition contains conditional number rotation and squeezing:
the coefficients of `-i sigma_z (n+1/2)` and `-i sigma_y (n+1/2)` are
-0.0046125 us^2 and +0.002621055 us^2. They grow as g^2 and cannot be removed
by correcting displacement amplitude or a common final qubit Z frame. This
explains why the concept can have the right average force yet poor fidelity.

The next useful hypothesis is **time symmetry of the actual interaction-frame
Hamiltonian**. If the control ends in a scalar qubit operation, replaying its
negative time reverse gives Q_reverse(t)=Q(T-t), so the two halves have an
even interaction-frame Hamiltonian and its second Magnus term vanishes.
This is stronger than merely mirroring phase labels. The CD agent is testing
short eight-pulse constructions, symmetric dwells, and variable half-word axes.

There is an important discrete-sample correction: the current 24-sample Hann
is sampled at the start of each interval. Repeating the familiar
`X X -X -X` pattern twice leaves a small second term
`-i * 7.5e-5 us^2 * sigma_z * i(adag^2-a^2)`. Explicitly reversing the envelopes
in the mirrored half removes this sampling asymmetry. The new sequence should
claim exact second-order cancellation only after checking its actual samples.

Third-order errors include a sigma_x photon-number-dependent force. An X
twirl preserves this error, just as it preserves the target force. Another
X twirl therefore cannot remove it. The useful remaining controls are changes
to the trajectory and dwell allocation, or a properly implemented signed
higher-order composition whose bridge overhead is included.

There is a stronger reason to explore axes beyond X. For X-only qubit control,
the principal oscillator polynomial of the interaction-frame Hamiltonian is

$$
H_I(t)=\frac{g}{2}\left[\sigma_x X+
P\{\sigma_y\cos\theta(t)-\sigma_z\sin\theta(t)\}\right].
$$

Define $\mathbf b(t)=(\cos\theta(t),-\sin\theta(t))$ and
$\mathbf F(t)=\int_0^t\mathbf b(s)\,ds$. When the first unwanted force cancels,
$\mathbf F(T)=0$. The longitudinal cubic Magnus term contains

$$
\Omega_3^{\parallel}=-i g^3 c_3\,\sigma_x\operatorname{Weyl}(XP^2),
\qquad
c_3=-\frac14\int_0^T|\mathbf F(t)|^2\,dt <0.
$$

The reduction follows by integrating the ordered triple-integral factor
$2\mathbf b_1\cdot\mathbf b_3-\mathbf b_1\cdot\mathbf b_2-
\mathbf b_2\cdot\mathbf b_3$, which equals
$-3\int|\mathbf F|^2$ after integration. Oscillator commutators affect lower
polynomial degrees and do not change this principal cubic coefficient.
Thus finite X-only trajectories can reduce this error but cannot cancel it
exactly. For the native eight-pulse word, the positive integral is
2.8995006672e-5 us^3; the predicted and independently extracted coefficient
are both -7.24875166807e-6 us^3, differing by 8.47e-21. This gives a direct
Magnus motivation for the synthesis agent's mixed-axis search.

## Squeezing: what the phase cycle must cancel

A native whole-word Cauchy derivative check agrees with the cached BCH series
through cubic order for two different eight-pulse words: 576 complex
coefficient amplitudes, maximum difference 1.4e-16. The signs of those cached
terms are therefore independently supported.

The fixed-pump Z-bridge proposal retains a substantial cubic generator, and
the native fidelity screen rejects it as a fast solution. A g-sign cycle is
motivated because it preserves the desired second-order squeeze and reverses
the ideal block's cubic error. Its finite phase transitions must be included:
an upper/lower arc pair can cancel its first raw force, yet retain a
second-order conditional number/Stark term. The squeezing agent confirmed
this unwanted second-order term numerically. A second cycle with opposite arc
chirality is the next falsifiable cancellation test; useful squeeze should add
while the arc-induced number term changes sign.

Low-photon polynomial fits must include enough input states to identify the
operators: a^3 annihilates n=0,1,2, so fitting only those columns cannot separate
the coefficients of a^3 and adag^3. Include at least n=3 for that decomposition.
This is separate from the final 18-column fidelity test.

The squeezing agent subsequently identified the corresponding obstacle:
all sixteen proper positive-squeeze eight-pulse blocks have the same-sign
quartic nonlinear-squeezing coefficient multiplying
$\sigma_z(a^{\dagger3}a-a^\dagger a^3)$. Positive mixtures and the tested
parity/Z cycles preserve it. Their balanced phase-transition family reaches
99% only around 15.7 us, so it is rejected as a speed improvement over the
verified 4.81-us fixed-pump reference. See the agent's
`runs/strategic_squeezing/quartic_operators.json` for this separate evidence.

## Constant-radius IQ is approximately constant after the RFDC

The independent test programs a ten-ns quintic phase arc with eight DAC
intervals and exactly constant complex magnitude. It exercises SDK int16
packing and the documented 8x RFDC interpolation filters. Over 224 fine
samples around the transition, the played magnitude is 0.9997095 to 1.0004524
times the flat reference. Thus the transition's ripple is about 0.045%, and
the model must propagate the actual I/Q instead of pretending the output is
exactly constant-radius.

With five-ns outer amplitude ramps, the largest overall FIR amplitude is
1.00669335 times the nominal plateau. At the 200-ns played-strength cap, the
nominal swap should therefore be at least 201.33867 ns for this waveform,
or the common gate amplitude should be reduced accordingly. The relevant
maximum must be recomputed for each actual full waveform.

The ideal two-level JC model does not certify the real coupler's transient
Hamiltonian or analog transfer function. This branch is permitted by the
user's explicit simplified-model instruction, with these assumptions stated.

## Reproducible evidence

- `code/review_hypotheses.py`: exact first moments, native Cauchy comparisons,
  operator decomposition, and SDK/RFDC phase-arc test.
- `cd_moments.json`: eight gates, 2048 first-derivative amplitude comparisons.
- `bch_native.json`: two gates, 576 coefficient amplitude comparisons.
- `cd_error_operators.json`: two native second/third-order decompositions.
- `phase_arc.json`: 136 programmed complex samples and 224 central played
  magnitude comparisons; arrays saved separately for figures.
- `same_axis_cubic.json`: two native cubic coefficients independently
  confirmed against the positive-functional expression above.
- `runs/reference_strengths/summary.json`: five genuine fixed-beta recipes,
  with weaker coupling and longer duration; see that folder's `STATUS.md`.

No board jobs or runtime/calibration files were changed. These are concept
checks and independently cross-checked numerical statements, not a claim that
the present finite search has exhausted all possible controls.

## Correction: isolated-word versus complete-gate second order

The earlier sampled-Hann warning above used an isolated flat-window word.
It does not establish a nonzero second Magnus term for the complete370ns
reference. A later independent full-quantum Cauchy calculation includes its
actual ramps, decoded samples and zero padding. It finds
`||Omega2/g^2|| = 5.4366898665687e-16` for the reference and
`5.176891856773749e-16` for the new mirrored phase candidate. Their joint
qubit-odd/pump-even reflection centers align after accounting for the sample
origin; the common linear-phase interpolation preserves that symmetry.
The complete original gate therefore already cancels second order, while
its first-order ramp contribution and cubic/higher errors remain. See
`../phase_arc_cd/whole_gate_magnus_check.json`. This supersedes any inference
above that the complete baseline inherits the isolated-word residual.
