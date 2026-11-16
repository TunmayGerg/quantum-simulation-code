# Strategic CD search status

Updated 2026-11-15. Owner: `/root/strategic_cd`.

The finite searches are complete. No new candidate beat the accepted constant
sideband reference: beta1, 370 ns, about99.2278% coherent gate fidelity on all18
inputs. Do not promote the shorter rejected candidates as99% gates.

The central conceptual result was independently challenged and confirmed:
first-moment X twirling makes a proper CD generator but leaves large second
Magnus terms. Exact interaction-frame time symmetry removes second order;
however X-only trajectories retain a sign-definite longitudinal cubic error.
Mixed-axis fractional-angle constrained searches did not remove that error
well enough in the tested short families. Canonical signed-JC fourth-order
Suzuki has an ideal dwell floor777.598ns before180ns of physical frame pulses.

Completed numerical evidence:

- 54 first-Magnus cases: unwanted Pauli coefficient<=2.16e-16us;
  qubit closure<=3.31e-15.
- 36 inverse-block/twirl fits,4056 native gate evaluations/73008 input columns:
  best65.1285%,315ns.
- 108 proper eight-pulse sign/timing gates/1944 columns: best96.2835%,265ns.
- 630 proper ten-pulse sign/timing gates/11340 columns: best98.0103%,315ns.
  One constant coupling calibration raises this to98.2330%,still rejected.
- 512 fractional half-paths,165 feasible full gates/2970 columns: best97.6954%.
- 28160 closed eleven-envelope paths,1012 centroid-matched scored gates/18216
  columns below370ns: none reached97%.
- 72 exact-first-force all-pi mixed-axis constrained fits: no permitted-strength
  solution found at310–340ns; this is a finite-search result,not a no-go proof.
- 12 mixed-axis refinements,3168 native gate evaluations/57024 columns:
  best97.4386%,330ns,first-axis residual7.49e-7.
- Canonical ideal S4 operator checked at3 scales; error-halving factors
  30.9334 and31.7301 approach32. Finite bridge forces were explicitly included
  in the diagnostic and show the ideal word is not already a valid finite gate.

Raw rejected examples and pulse metadata are exported:
`best_proper10.npz/json` and `best_mixed_geometric.npz/json`.
These are native screening scores; the coordinator owns the independent
SDK/RFDC/cutoff verification before any final-use label.

See `README.md` here for equations, the cancellation argument, what each
optimizer changed, and the finite search limits. Detailed data and logs are
retained. No board job was started. No shared numerical model was edited.
