# Independent review

Completed first review; available for candidate replay and challenges. Full
discussion and evidence are in `REVIEW.md`.

- Ten-pulse CD first moment is correct: eight words, 2048 native derivative
  amplitudes, maximum discrepancy below 1.1e-15. Its substantial second-order
  conditional number/squeeze error explains the poor fast-gate fidelity.
- Exact interaction-frame time symmetry is the next CD hypothesis. Warned the
  synthesis agent that start-sampled Hann pulses need explicit sample reversal.
- Native BCH signs checked: two whole words, 576 coefficient amplitudes,
  maximum error 1.4e-16 through cubic order.
- A 10 ns constant-radius IQ arc has central FIR radius ratio
  0.9997095..1.0004524. Outer-ramp FIR peak ratio 1.00669335 requires cap backoff.
- Questioned the squeezing sign cycle's retained arc-induced number term;
  synthesis agent confirmed it and is testing opposite chirality cycles.
- Derived a sign-definite principal cubic error for every finite X-only CD
  trajectory with zero transverse first moment. Two native coefficients agree
  with the positive-functional formula to 8.47e-21. This gives a reason for
  exploring mixed axes, rather than more X twirls.
- Completed the coordinator's five fixed-beta strength/time references in
  `runs/reference_strengths/summary.json`: 370..970 ns, process fidelities
  99.2277..99.9673%, all 18 columns at converged 48/64-level cutoffs.

Falsifiers are explicit: a non-scalar final qubit operation, an unwanted first
force component, a cubic coefficient that does not match native derivatives, or
an actual played coupling above the 200 ns swap cap. A phase palindrome is not
accepted as proof of a formal Trotter order.

Files owned here: `code/review_hypotheses.py` and `runs/independent_review/`.
No hardware jobs, runtime edits, or calibration changes are involved.
