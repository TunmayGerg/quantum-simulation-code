"""
Which optimizer finds the best global optimum?

Runs every method in optimal_control.py on the two problems that actually matter
here, under a *matched wall-clock budget*, repeated over several random seeds,
and reports the distribution of the best infidelity reached.

    python benchmark_optimizers.py                 # both problems, default budget
    python benchmark_optimizers.py --budget 20     # 20 s per (method, seed)
    python benchmark_optimizers.py --problem grape

Wall clock (not iteration count) is the fair currency: the methods differ wildly
in how much work one "iteration" is, and the thing the user cares about is
"best fidelity per minute of waiting".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpful_functions import dchi_H, projector_onto_states  # noqa: E402
from optimal_control import GrapeOC, ParameterOC, available_methods  # noqa: E402


# =============================================================================
# Problem definitions -- deliberately the same physics as the notebooks
# =============================================================================

def build_parameter_problem(Nc_big=12, Nq_big=2, Nc_sub=8, num_apply=10):
    """Parity gate from a chain of beamsplitters (parameter_optimal_control.ipynb)."""
    a_c = qt.tensor(qt.destroy(Nc_big), qt.qeye(Nq_big))
    a_q = qt.tensor(qt.qeye(Nc_big), qt.destroy(Nq_big))
    n_c, n_q = a_c.dag() * a_c, a_q.dag() * a_q
    target = (-1j * np.pi * (n_c * n_q)).expm()
    projector = projector_onto_states(
        [qt.tensor(qt.fock(Nc_big, i), qt.fock(Nq_big, j))
         for i in range(Nc_sub) for j in range(Nq_big)])

    def factory():
        return ParameterOC(["bs"], target, Nc_big, Nq_big, num_apply=num_apply,
                           cost_type="projected", projector=projector,
                           gauge_ops=[n_c, n_q])

    def start(rng, opt):
        return dict(rotation0=rng.uniform(0.0, np.pi, size=opt.n_gates),
                    phase0=rng.uniform(0.0, 2 * np.pi, size=opt.n_gates),
                    theta0=rng.uniform(0.0, 2 * np.pi, size=2))

    return dict(name=f"ParameterOC parity, {num_apply} bs gates, "
                     f"d={Nc_big*Nq_big}, {2*num_apply+2} params",
                factory=factory, start=start,
                fixed=dict(rotation_bounds=(0.0, np.pi),
                           phase_bounds=(0.0, 2 * np.pi),
                           gauge_bounds=(0.0, 2 * np.pi)))


def build_grape_problem(Nc_big=10, Nq_big=5, Nr_sub=5, n_steps=60):
    """Dressed-frame cross-Kerr gate by detuning control (Grape_optimal_control.ipynb)."""
    g = 1 * 2 * np.pi
    alpha = -160 * 2 * np.pi
    dressed_detuning = 140 * 2 * np.pi
    drive = {"N_c": Nc_big, "N_q": Nq_big, "detuning": dressed_detuning, "gbs": g}
    res = dchi_H(drive, alpha)

    evecs = res["sorted_dict"]["evecs_qobj"]
    w_r = res["dressed_operators"]["w_r_dressed"]
    w_q = res["dressed_operators"]["w_q_dressed"]
    chi = res["chi_dict"]["chi_list"][0]
    a_r = res["Hamiltonian_dict"]["a_r"]
    a_q = res["Hamiltonian_dict"]["a_q"]
    n_r, n_q = a_r.dag() * a_r, a_q.dag() * a_q

    H_d = g * (a_q.dag() * a_r + a_q * a_r.dag()) + (alpha / 2) * (a_q.dag() ** 2 * a_q ** 2)
    H_c = n_q
    target = evecs * (-1j * np.pi * (n_r * n_q)).expm() * evecs.dag()
    gauge_ops = [evecs * n_r * evecs.dag(), evecs * n_q * evecs.dag()]
    projector = projector_onto_states(
        [res["sorted_dict"]["evecs_sorted"][i][j]
         for i in range(Nr_sub) for j in range(2)])
    T_gate = np.pi / chi
    dt = T_gate / n_steps

    def factory():
        return GrapeOC(H_d, [H_c], target, dt, n_steps, cost_type="projected",
                       projector=projector, gauge_ops=gauge_ops)

    def start(rng, opt):
        # physically motivated start: flat detuning + noise, gauge from the
        # dressed frequencies (this is what the notebook does)
        return dict(pulses0=dressed_detuning * np.ones((n_steps, 1))
                    + rng.normal(scale=20 * 2 * np.pi, size=(n_steps, 1)),
                    theta0=np.array([w_r * T_gate, w_q * T_gate]))

    return dict(name=f"GrapeOC cross-Kerr, {n_steps} steps, "
                     f"d={Nc_big*Nq_big}, {n_steps+2} params",
                factory=factory, start=start,
                fixed=dict(pulse_bounds=[(1.0 * 2 * np.pi, 1000.0 * 2 * np.pi)],
                           theta_bounds=[(None, None), (None, None)]))


# =============================================================================
# Budgeted runner
# =============================================================================

def run_one(problem, method, seed, budget_s):
    """Run one (method, seed) with a wall-clock budget; return the best infidelity.

    The budget is enforced through a wrapper on cost_and_grad that trips the
    optimizer's own cancel flag, so every method stops the same way and the best
    point seen so far is always what gets returned.
    """
    opt = problem["factory"]()
    rng = np.random.default_rng(seed)
    kwargs = dict(problem["start"](rng, opt))
    kwargs.update(problem["fixed"])

    t_end = time.perf_counter() + budget_s
    real = opt.cost_and_grad
    real_cost = opt.cost

    def budgeted_cg(x):
        if time.perf_counter() > t_end:
            opt.cancel()
        return real(x)

    def budgeted_c(x):
        if time.perf_counter() > t_end:
            opt.cancel()
        return real_cost(x)

    opt.cost_and_grad = budgeted_cg      # type: ignore[method-assign]
    opt.cost = budgeted_c                # type: ignore[method-assign]

    # Generous per-method knobs; the wall clock is the real limiter.
    mopts = {"niter": 10_000, "popsize": 12, "n": 256,
             "perturb": 0.3, "sigma0": 0.5}
    t0 = time.perf_counter()
    try:
        out = opt.optimize(method=method, n_starts=10_000, maxiter=100_000,
                           progress="none", seed=seed, store_history=False,
                           method_options=mopts,
                           scipy_options={"gtol": 1e-12, "ftol": 1e-14, "maxls": 60},
                           **kwargs)
        F = out["fidelity"]
    except ImportError as exc:
        return {"skip": f"not installed ({exc})"}
    except ValueError as exc:            # e.g. shgo refusing a high-dimensional problem
        return {"skip": str(exc)}
    except Exception as exc:                                    # noqa: BLE001
        return {"skip": f"{type(exc).__name__}: {exc}"}
    return {"infidelity": 1.0 - F, "fidelity": F,
            "elapsed": time.perf_counter() - t0,
            "n_eval": out["n_evaluations"]}


def bench(problem, methods, seeds, budget_s):
    print("=" * 92)
    print(f"PROBLEM: {problem['name']}")
    print(f"budget = {budget_s:.0f} s per (method, seed), {len(seeds)} seeds")
    print("=" * 92)
    print(f"{'method':<24} {'best 1-F':>11} {'median 1-F':>12} {'worst 1-F':>11} "
          f"{'evals/run':>10} {'hit 1e-6':>9}")
    print("-" * 92)

    table = {}
    for m in methods:
        raw = [run_one(problem, m, s, budget_s) for s in seeds]
        rows = [r for r in raw if r is not None and "skip" not in r]
        if not rows:
            reason = next((r["skip"] for r in raw if r and "skip" in r), "no result")
            print(f"{m:<24}   SKIPPED: {reason[:60]}")
            continue
        inf = np.array([r["infidelity"] for r in rows])
        ev = np.mean([r["n_eval"] for r in rows])
        hits = int(np.sum(inf < 1e-6))
        table[m] = inf
        print(f"{m:<24} {inf.min():11.3e} {np.median(inf):12.3e} {inf.max():11.3e} "
              f"{ev:10.0f} {hits:5d}/{len(inf)}")

    if table:
        print("-" * 92)
        best = min(table, key=lambda k: float(np.median(table[k])))
        best_any = min(table, key=lambda k: float(np.min(table[k])))
        print(f"best median infidelity : {best}  ({np.median(table[best]):.3e})")
        print(f"best single run        : {best_any}  ({np.min(table[best_any]):.3e})")
    print()
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=15.0,
                    help="wall-clock seconds per (method, seed)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--problem", choices=("parameter", "grape", "both"), default="both")
    ap.add_argument("--methods", type=str, default="",
                    help="comma-separated subset of methods")
    args = ap.parse_args()

    methods = ([m.strip() for m in args.methods.split(",") if m.strip()]
               or [m for m in available_methods() if m not in ("local",)])
    seeds = list(range(args.seeds))

    problems = []
    if args.problem in ("parameter", "both"):
        problems.append(build_parameter_problem())
    if args.problem in ("grape", "both"):
        problems.append(build_grape_problem())

    for p in problems:
        bench(p, methods, seeds, args.budget)


if __name__ == "__main__":
    main()
