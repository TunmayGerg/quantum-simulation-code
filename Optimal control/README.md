# Optimal control

`Grape.py` and `parameter_oc.py` have been replaced by a single module,
**`optimal_control.py`**.

```
OptimalControl        all the shared machinery
  |- GrapeOC          piecewise-constant pulse ansatz   (was GrapeLBFGS)
  |- ParameterOC      parameterized gate sequence       (was ParameterOC)
```

Everything that does not depend on *how parameters become propagators* lives in
the base class: fidelity, projector, gauge freedom, adiabatic frame, penalties,
gradients, all optimizer drivers, progress reporting, cancellation, async
running, saving and loading. A subclass only defines the ansatz.

| file | what it is |
|---|---|
| `optimal_control.py` | the module |
| `test_optimal_control.py` | 33 tests — `python test_optimal_control.py` |
| `benchmark_optimizers.py` | optimizer shoot-out — `python benchmark_optimizers.py` |
| `helpful_functions.py` | unchanged physics helpers |

---

## Quick start

```python
from optimal_control import GrapeOC, ParameterOC, available_methods, load_run

available_methods(verbose=True)          # what you can pass as method=

opt = GrapeOC(H_drift, [H_ctrl], U_target, dt, n_steps,
              cost_type="projected", projector=P, gauge_ops=[n_r, n_q])
opt.add_smoothness_penalty(1e-5)

out = opt.optimize(pulses0=p0, theta0=t0,
                   pulse_bounds=[(lo, hi)],
                   method="trust-constr", maxiter=4000)

print(out["fidelity"], out["pulses_opt"], out["theta_opt"])
```

### Running in a notebook without blocking it

```python
handle = opt.run_async(pulses0=p0, theta0=t0, method="trust-constr")
...
handle.stop()          # graceful cancel, keeps the best iterate seen
out = handle.wait()    # blocks until finished, returns the result dict
handle.best_fidelity   # live, readable at any time
```

`optimize(..., target_fidelity=1-1e-10)` stops on its own once it is good enough.

### Asking for a fidelity

Four equivalent forms:

```python
opt.fidelity(out)                                    # the result dict
opt.fidelity(out["pulses_opt"], theta=out["theta_opt"])
opt.fidelity(x=out["x_opt"])                         # the packed vector
opt.fidelity()                                       # best point seen so far
```

`unitary()` and `raw_unitary()` take the same arguments.

### Saving

Saving is explicit — you call it — and **never silently overwrites**:

```python
path = opt.save("Grape_results/run2", out, extra={...}, notes="...")
```

If `Grape_results/run2` already exists, `run2_1` is written instead (then
`run2_2`, ...) and the path actually used is printed and returned. Pass
`override=True` to overwrite deliberately. Layout:

```
run2/results.npz     arrays: x_opt, pulses_opt/rotation_opt+phase_opt, theta_opt, history
run2/metadata.json   readable run configuration, fidelity, method, timings
run2/qobjs/*.qu      QuTiP operators (target, projector, gauge ops, drift, controls, ...)
```

Read back with `load_run(folder)`. The old single-`.npz` files already in
`Grape_results/` and `ParameterOc_results/` still load through `load_qobjs`,
`load_pulses_theta`, `load_rotation_phase_theta` and `load_parameters_dict`.

### Adding a gate

Any gate whose phase rotates the drive axis fits `FrameGate`:
`U(r, φ) = exp(-i r · D(φ) K₀ D(φ)†)` with `D(φ) = exp(i s φ N)`, `N` diagonal.
Both built-ins (`"bs"`, `"r"`) are of this form.

```python
opt.register_gate("mygate", K0, phase_diag=N_operator, phase_sign=-1.0)
```

---

## Which optimizer should I use?

Measured by `benchmark_optimizers.py` — **20 s wall clock per run, 5 random
seeds**, median infidelity `1-F` (lower is better). Wall clock is the fair
currency because one "iteration" means wildly different amounts of work across
these methods.

**The answer depends on the problem, and the split is sharp.**

### Rugged landscape, random start — the gate-sequence ansatz
*(parity gate, 10 beamsplitters, d=24, 22 parameters)*

| method | median 1-F | best 1-F |
|---|---|---|
| **dual_annealing** | **2.8e-3** | 1.5e-3 |
| ils | 3.2e-3 | 1.6e-3 |
| basinhopping | 4.0e-3 | 2.3e-3 |
| multistart | 4.2e-3 | 2.3e-3 |
| L-BFGS-B | 8.2e-2 | 2.3e-3 |
| TNC | 8.2e-2 | 4.3e-2 |
| BFGS / CG | 4.4e-1 | 5.7e-3 |
| differential_evolution | 5.9e-1 | 4.7e-1 |

A **global method that still uses the analytic gradient** wins by roughly **30×
in infidelity** over a single local descent, for the same wall time. The purely
derivative-free `differential_evolution` is not competitive.

### Smooth landscape, good physical start — GRAPE
*(dressed cross-Kerr, 60 steps, d=50, 62 parameters, flat-detuning start)*

| method | median 1-F | best 1-F |
|---|---|---|
| **trust-constr** | **1.8e-4** | 7.9e-5 |
| BFGS † | 2.0e-4 | 3.6e-5 |
| TNC | 2.3e-4 | 1.5e-4 |
| SLSQP | 4.6e-4 | 5.7e-5 |
| multistart / ils / basinhopping | ~1.3e-3 | 3.4e-4 |
| L-BFGS-B | 1.5e-3 | 3.4e-4 |
| CG | 7.5e-3 | 5.0e-4 |
| dual_annealing | 6.4e-1 | 3.3e-3 |
| differential_evolution | 6.4e-1 | 5.5e-1 |

† **BFGS and CG ignore bounds.** They score well only because they are allowed
to leave the physical detuning range. Do not use them when `pulse_bounds`
matters.

Here the restart-based globals give **no** benefit — a single descent already
consumes the whole budget, so restarting just throws work away — and
`dual_annealing` actively wanders away from a good physical starting point.
`trust-constr` beats the previous default `L-BFGS-B` by about **8×**.

### Rules of thumb

- Good physical starting guess → **`trust-constr`** (or `TNC`/`SLSQP`).
- Random start / many local minima → **`dual_annealing`**, then `ils`,
  `basinhopping`, `multistart`.
- Never `differential_evolution` on these problems: gradient-free search does
  not pay in 20–60 dimensions when an exact gradient is available.
- `shgo` is **refused above 15 variables** — its simplicial complex grows
  exponentially and, because that work happens outside the objective, `cancel()`
  cannot interrupt it. Pass `method_options={"max_dim": N}` to force it.

### Why there is no Gaussian-process / Bayesian option

Bayesian optimization with a GP surrogate was evaluated and deliberately left
out. It cannot consume the exact gradient we already have without a
gradient-enhanced kernel, whose matrix is `n(d+1) × n(d+1)` — at d = 22–62 that
costs far more than simply running more gradient descents. And BO only pays off
when a single evaluation takes minutes; here one cost+gradient evaluation is
**0.4 ms** (gate ansatz) to **60 ms** (GRAPE), so tens of thousands of gradient
steps fit in the time a GP needs for a few hundred function calls.

`method="cma"` (CMA-ES) is available if you `pip install cma`; every
derivative-free global gets an automatic gradient-based polish afterwards.

---

## Performance

All three changes below were benchmarked before being adopted, and each is
**exact** — no approximation was introduced.

| change | speedup | why |
|---|---|---|
| analytic `FrameGate` derivatives | **28–32×** per gate | cached eigendecomposition of `K₀` + a diagonal phase frame replaces `expm` + two `expm_frechet` calls |
| spectral GRAPE slice derivative | **2.4–4×** | eigendecomposition of the Hermitian `H_n` and the Daleckii–Krein formula instead of `expm_frechet`; the expensive factor is reused across all control channels |
| `cost_and_grad` fused | **2×** | the old code propagated once for the cost and again for the gradient; SciPy is now driven with `jac=True` |
| backward co-state recursion | O(d²) per parameter | `dc = Tr(L_j dU_j)` instead of building `dU_final` per parameter |
| `Tr(A@B)` → contraction | 3–4× on each trace | `np.trace(A @ B)` does a full O(d³) product to keep only the diagonal |
| commuting-gauge fast path | O(d²) | number-operator gauges are diagonalized once at construction, not `expm_frechet`'d every call |

End-to-end, at the sizes in the notebooks:

- **ParameterOC** (d=24, 10 gates): 16.6 ms → **0.53 ms** per cost+gradient (**~31×**)
- **GRAPE** (d=50, 100 steps): 306 ms → **62 ms** per cost+gradient (**~5×**)

`derivative="frechet"` and `derivative="approx"` remain available on `GrapeOC`;
`"spectral"` falls back to `"frechet"` automatically for non-Hermitian
generators.

### Progress reporting

`progress="line"` prints a self-updating line with the current and best fidelity.
The cost is **59 ns per evaluation** (one `perf_counter()` call and a float
compare) against a 0.4–60 ms evaluation, and printing is throttled to
`progress_every` seconds so it does not scale with iteration count — measured
overhead is below the timing noise. Use `"log"` from a background thread
(`run_async` defaults to it) and `"none"` to silence it.

---

## Testing

```
python test_optimal_control.py          # 33 tests, ~6 s
python test_optimal_control.py -k grad  # subset
```

What is covered:

- **Gate correctness** — `FrameGate` reproduces the original
  `_get_unitary_from_string` unitary *and* both derivatives to 1e-11, and every
  gate is verified unitary.
- **Gradients** — analytic vs central finite differences for every combination
  of cost type × gauge × phases × adiabatic frame × penalties × derivative mode,
  including a deliberately degenerate spectrum (the Daleckii–Krein limit).
- **Gauge fast paths** — diagonal, commuting and general paths all agree.
- **Optimizers** — every advertised method runs and never returns a worse
  fidelity than its start; multistart reports each start; `target_fidelity`
  stops early.
- **Cancellation** — `cancel()` unwinds cleanly and keeps the best iterate; a
  stale flag never poisons the next run; `run_async` start/stop/wait.
- **Saving** — collision handling (`run` → `run_1` → `run_2`), override, and a
  full round trip where the fidelity is rebuilt from the saved files alone.
- **Argument forms** — all four ways of passing parameters agree, and the error
  paths raise.
- **Performance** — the analytic and spectral paths are checked to match their
  reference implementations *before* being timed against them.

The two tests that compared against `Grape.py` / `parameter_oc.py` passed
(cost, gradient and fidelity agreeing to 1e-9) before those files were retired;
they now skip with a message and will run again if the old files are dropped
back into this folder.

---

## Notes on the notebooks

`parameter_optimal_control.ipynb`, `Grape_optimal_control.ipynb` and
`Grape_optimal_control_adiabatic.ipynb` were converted to the new API and each
was executed top to bottom. Two incidental fixes along the way:

- The `ipywidgets` cancel-button plumbing in `Grape_optimal_control.ipynb` was
  replaced by `run_async` / `handle.stop()`.
- `Grape_optimal_control_adiabatic.ipynb` had a pre-existing indexing bug in its
  final plotting cell: `pop[n, :, lvl]` indexed by photon *number* where `pop` is
  indexed by *position* in `photon_nums`, so it raised `IndexError` for
  `photon_nums = [4,5,6,7,8]`. Now uses `enumerate`.
