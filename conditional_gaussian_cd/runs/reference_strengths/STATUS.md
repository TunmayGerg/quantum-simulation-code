# Fixed-beta strength reference

Completed five fixed-word CD recipes: `(X X -X -X)^K`, K=3,4,5,6,8.
Each pi envelope is 30 ns. Only one global constant coupling is calibrated
against the same beta=1 target on all 18 inputs; sideband plateau phase and
amplitude stay fixed. Gate duration grows from 370 to 970 ns as coupling falls.

Process and log activity checked after launch and final completion. Results:

- K=3: 370 ns, nominal swap 283.18085 ns, played peak swap 281.35690 ns,
  process fidelity 0.99227705, worst basis fidelity 0.97999562.
- K=4: 490 ns, nominal swap 378.23755 ns, played peak swap 375.78159 ns,
  process fidelity 0.99716692, worst basis fidelity 0.99289555.
- K=5: 610 ns, nominal swap 472.95035 ns, played peak swap 469.77360 ns,
  process fidelity 0.99864086, worst basis fidelity 0.99672228.
- K=6: 730 ns, nominal swap 567.63993 ns, played peak swap 563.73289 ns,
  process fidelity 0.99923433, worst basis fidelity 0.99821082.
- K=8: 970 ns, nominal swap 756.40468 ns, played peak swap 751.11207 ns,
  process fidelity 0.99967257, worst basis fidelity 0.99928891.

Each result converged targets and outputs at 48/64 levels and compared 2304
complex amplitudes against the independent order-16 backend. Across five
recipes: 90 final input columns, 11520 independent complex amplitudes, and
1782 design input-column evaluations. Largest target column change 1.90e-16;
largest output column change 1.99e-27. Four-pulse leading force independently
checked over 256 complex derivative amplitudes, max discrepancy 1.67e-16.
Full metadata, search histories, and 800 MS/s control NPZ files are written
per recipe. `summary.json` contains all completed results.
