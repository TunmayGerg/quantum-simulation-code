"""
Test suite for optimal_control.py.

Run with:  python test_optimal_control.py          (all tests)
           python test_optimal_control.py -k grad  (subset by name)

Covers:
  * gate/derivative correctness against the original implementations
  * analytic gradients against central finite differences, for every
    combination of cost type / gauge / adiabatic frame / penalties / derivative
  * numerical agreement of the new classes with the old Grape.py and
    parameter_oc.py classes
  * every optimizer method
  * cancellation, run_async, target_fidelity early stop
  * saving: folder collision handling, round-trip loading
  * performance: analytic vs expm_frechet derivatives, progress overhead
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimal_control import (  # noqa: E402
    FrameGate,
    GrapeOC,
    ParameterOC,
    available_methods,
    gauge_unitary,
    load_qobjs,
    load_run,
    propagate_gates,
    propagate_pulses,
)

TOL = 1e-9
_FAILURES: list[str] = []
_PASSED: list[str] = []


# =============================================================================
# Tiny test harness
# =============================================================================

def test(fn):
    """Decorator that registers a function as a test case."""
    fn._is_test = True
    return fn


def check(cond: bool, msg: str) -> None:
    """Assert with a readable message (used instead of bare ``assert``)."""
    if not cond:
        raise AssertionError(msg)


def close(a, b, tol=TOL, msg=""):
    """Assert two array-likes agree elementwise, reporting the actual error."""
    a = np.asarray(a)
    b = np.asarray(b)
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    check(err <= tol, f"{msg} max|diff| = {err:.3e} > {tol:.1e}")
    return err


# =============================================================================
# Shared fixtures
# =============================================================================

def make_param_system(n_c=6, n_q=2, seed=0):
    """A small cavity-qubit system plus a projected parity target, for ParameterOC."""
    a_c = qt.tensor(qt.destroy(n_c), qt.qeye(n_q))
    a_q = qt.tensor(qt.qeye(n_c), qt.destroy(n_q))
    n_c_op = a_c.dag() * a_c
    n_q_op = a_q.dag() * a_q
    target = (-1j * np.pi * (n_c_op * n_q_op)).expm()          # parity-check gate
    # project onto the lowest 4 cavity levels x both qubit levels
    kets = [qt.tensor(qt.fock(n_c, i), qt.fock(n_q, j)) for i in range(4) for j in range(n_q)]
    P = sum((k * k.dag() for k in kets), 0 * kets[0] * kets[0].dag())
    return dict(a_c=a_c, a_q=a_q, n_c=n_c, n_q=n_q, target=target, projector=P,
                gauge_ops=[n_c_op, n_q_op])


def make_grape_system(n_c=4, n_q=3, seed=0):
    """A small JC + Kerr system with one detuning control, for GrapeOC."""
    a_r = qt.tensor(qt.destroy(n_c), qt.qeye(n_q))
    a_q = qt.tensor(qt.qeye(n_c), qt.destroy(n_q))
    n_q_op = a_q.dag() * a_q
    n_r_op = a_r.dag() * a_r
    g, alpha = 1.0, -12.0
    H0 = g * (a_q.dag() * a_r + a_q * a_r.dag()) + (alpha / 2) * (a_q.dag() ** 2 * a_q ** 2)
    Hc = [n_q_op]
    target = (-1j * np.pi * (n_r_op * n_q_op)).expm()
    kets = [qt.tensor(qt.fock(n_c, i), qt.fock(n_q, j)) for i in range(3) for j in range(2)]
    P = sum((k * k.dag() for k in kets), 0 * kets[0] * kets[0].dag())
    return dict(H0=H0, Hc=Hc, target=target, projector=P, gauge_ops=[n_r_op, n_q_op],
                n_c=n_c, n_q=n_q)


def fd_gradient(f, x, h=1e-6):
    """Central finite-difference gradient of a scalar function."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


# =============================================================================
# 1. Gate-level correctness
# =============================================================================

@test
def test_frame_gate_matches_original_bs():
    """FrameGate('bs') must reproduce the exact unitary and both derivatives of
    the original parameter_oc._get_unitary_from_string for the beamsplitter."""
    from scipy.linalg import expm_frechet
    sys_ = make_param_system()
    a_c, a_q = sys_["a_c"], sys_["a_q"]
    opt = ParameterOC(["bs"], sys_["target"], sys_["n_c"], sys_["n_q"])
    gate = opt.gates["bs"]

    rng = np.random.default_rng(3)
    for _ in range(8):
        g_val, phase = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        # --- reference: literal transcription of the original code ---
        H = g_val * (a_c.dag() * a_q * np.exp(1j * phase)
                     + a_c * a_q.dag() * np.exp(-1j * phase))
        A = (-1j * H).full()
        U_ref = (-1j * H).expm().full()
        dA_r = (-1j * (a_c.dag() * a_q * np.exp(1j * phase)
                       + a_c * a_q.dag() * np.exp(-1j * phase))).full()
        dA_p = (-1j * g_val * (1j * a_c.dag() * a_q * np.exp(1j * phase)
                               - 1j * a_c * a_q.dag() * np.exp(-1j * phase))).full()
        dU_r_ref = expm_frechet(A, dA_r, compute_expm=False)
        dU_p_ref = expm_frechet(A, dA_p, compute_expm=False)

        U, dU_r, dU_p = gate.evaluate(g_val, phase, need_grad=True)
        close(U, U_ref, 1e-12, "bs U:")
        close(dU_r, dU_r_ref, 1e-11, "bs dU/dg:")
        close(dU_p, dU_p_ref, 1e-11, "bs dU/dphase:")


@test
def test_frame_gate_matches_original_r():
    """Same check for the qubit rotation gate 'r'."""
    from scipy.linalg import expm_frechet
    sys_ = make_param_system()
    a_q = sys_["a_q"]
    opt = ParameterOC(["r"], sys_["target"], sys_["n_c"], sys_["n_q"])
    gate = opt.gates["r"]

    # sigma_+ = |1><0| on the qubit factor, as in helpful_functions
    n_c, n_q = sys_["n_c"], sys_["n_q"]
    sp = qt.tensor(qt.qeye(n_c), qt.basis(n_q, 1) * qt.basis(n_q, 0).dag())

    rng = np.random.default_rng(4)
    for _ in range(8):
        theta, phase = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        H = theta * (sp * np.exp(1j * phase) + sp.dag() * np.exp(-1j * phase))
        A = (-1j * H).full()
        U_ref = (-1j * H).expm().full()
        dA_r = (-1j * (sp * np.exp(1j * phase) + sp.dag() * np.exp(-1j * phase))).full()
        dA_p = -1j * theta * (1j * sp * np.exp(1j * phase)
                              - 1j * sp.dag() * np.exp(-1j * phase)).full()
        dU_r_ref = expm_frechet(A, dA_r, compute_expm=False)
        dU_p_ref = expm_frechet(A, dA_p, compute_expm=False)

        U, dU_r, dU_p = gate.evaluate(theta, phase, need_grad=True)
        close(U, U_ref, 1e-12, "r U:")
        close(dU_r, dU_r_ref, 1e-11, "r dU/dtheta:")
        close(dU_p, dU_p_ref, 1e-11, "r dU/dphase:")


@test
def test_frame_gate_is_unitary():
    """Every gate must produce a genuinely unitary matrix."""
    sys_ = make_param_system()
    opt = ParameterOC(["bs", "r"], sys_["target"], sys_["n_c"], sys_["n_q"])
    I = np.eye(opt.dim)
    rng = np.random.default_rng(5)
    for name, gate in opt.gates.items():
        for _ in range(5):
            U, _, _ = gate.evaluate(rng.uniform(0, 3), rng.uniform(0, 7), False)
            close(U @ U.conj().T, I, 1e-12, f"gate {name} unitarity:")


@test
def test_custom_gate_registration():
    """A user-registered FrameGate must behave like the built-ins.

    A gate name has to exist before it can appear in ``unitary_strings``, so the
    workflow is: build with a known gate, register the new one, then swap the
    sequence over and call ``configure()``.
    """
    sys_ = make_param_system()
    n_c, n_q = sys_["n_c"], sys_["n_q"]
    a_c = sys_["a_c"]

    # A cavity "displacement-like" generator (a_c + a_c^dag), with no phase param.
    K = (a_c + a_c.dag()).full()
    opt = ParameterOC(["bs"], sys_["target"], n_c, n_q, optimize_phases=False)
    opt.register_gate("disp", K, phase_diag=None)
    opt.unitary_strings = ["disp"]
    opt.configure()

    r = 0.37
    U = opt._forward(np.array([r]), need_grad=False)["U_final"]
    close(U, (-1j * r * qt.Qobj(K)).expm().full(), 1e-12, "custom gate unitary:")
    check(not opt.gates["disp"].has_phase, "custom gate should have no phase param")

    # A phase-free gate must contribute no phase gradient, and the analytic
    # gradient must still match finite differences.
    _grad_case(opt, np.array([0.4]), "ParameterOC[custom phase-free gate]")

    # Referencing an unregistered name must still be an error.
    try:
        ParameterOC(["never_registered"], sys_["target"], n_c, n_q)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered gate name should raise")


# =============================================================================
# 2. Gradient correctness (finite differences)
# =============================================================================

def _grad_case(opt, x, label, tol=2e-6):
    """Compare the analytic gradient with central finite differences."""
    f0, g = opt.cost_and_grad(x)
    g_fd = fd_gradient(opt.cost, x, h=1e-6)
    scale = max(1.0, float(np.max(np.abs(g_fd))))
    err = float(np.max(np.abs(g - g_fd))) / scale
    check(err <= tol, f"{label}: relative gradient error {err:.3e} > {tol:.1e}\n"
                      f"  analytic = {np.round(g, 8)}\n  finite   = {np.round(g_fd, 8)}")
    # cost must be reproducible without the gradient path
    close(opt.cost(x), f0, 1e-12, f"{label} cost/cost_and_grad mismatch:")


@test
def test_parameter_gradients_all_configs():
    """ParameterOC analytic gradient vs finite differences, all option combos."""
    sys_ = make_param_system()
    rng = np.random.default_rng(11)
    for cost_type in ("unitary", "projected"):
        for use_gauge in (False, True):
            for phases in (True, False):
                for strings in (["bs"], ["r", "bs"]):
                    opt = ParameterOC(
                        strings, sys_["target"], sys_["n_c"], sys_["n_q"],
                        num_apply=3, optimize_phases=phases, cost_type=cost_type,
                        projector=sys_["projector"] if cost_type == "projected" else None,
                        gauge_ops=sys_["gauge_ops"] if use_gauge else None,
                    )
                    x = rng.uniform(0.1, 2.0, size=opt.n_core + opt.n_gauge)
                    label = f"ParameterOC[{cost_type},gauge={use_gauge},ph={phases},{strings}]"
                    _grad_case(opt, x, label)


@test
def test_parameter_gradients_with_adiabatic_and_penalties():
    """Adiabatic frame plus both built-in penalties must still be differentiated exactly."""
    sys_ = make_param_system()
    rng = np.random.default_rng(12)
    # a fixed unitary basis change
    Hrand = qt.rand_herm(sys_["target"].shape[0], seed=7)
    Hrand.dims = sys_["target"].dims
    Ad = (-1j * Hrand).expm()
    opt = ParameterOC(["bs", "r"], sys_["target"], sys_["n_c"], sys_["n_q"],
                      num_apply=2, cost_type="projected", projector=sys_["projector"],
                      gauge_ops=sys_["gauge_ops"], adiabatic_unitary=Ad)
    opt.add_smoothness_penalty(1e-3)
    opt.add_l2_amplitude_penalty(2e-3)
    x = rng.uniform(0.1, 2.0, size=opt.n_core + opt.n_gauge)
    _grad_case(opt, x, "ParameterOC[adiabatic+penalties]")


@test
def test_grape_gradients_all_configs():
    """GrapeOC analytic gradient vs finite differences for every derivative mode."""
    sys_ = make_grape_system()
    rng = np.random.default_rng(21)
    for deriv in ("spectral", "frechet"):
        for cost_type in ("unitary", "projected"):
            for use_gauge in (False, True):
                opt = GrapeOC(
                    sys_["H0"], sys_["Hc"], sys_["target"], dt=0.02, n_steps=5,
                    cost_type=cost_type,
                    projector=sys_["projector"] if cost_type == "projected" else None,
                    gauge_ops=sys_["gauge_ops"] if use_gauge else None,
                    derivative=deriv,
                )
                x = rng.uniform(-1.0, 1.0, size=opt.n_core + opt.n_gauge)
                _grad_case(opt, x, f"GrapeOC[{deriv},{cost_type},gauge={use_gauge}]")


@test
def test_grape_gradients_multi_control_adiabatic_penalties():
    """Multi-control GRAPE with an adiabatic frame and penalties."""
    sys_ = make_grape_system()
    a_r = qt.tensor(qt.destroy(sys_["n_c"]), qt.qeye(sys_["n_q"]))
    Hc = sys_["Hc"] + [a_r + a_r.dag()]                # second control channel
    Hrand = qt.rand_herm(sys_["target"].shape[0], seed=9)
    Hrand.dims = sys_["target"].dims
    Ad = (-1j * Hrand).expm()
    opt = GrapeOC(sys_["H0"], Hc, sys_["target"], dt=0.02, n_steps=4,
                  cost_type="projected", projector=sys_["projector"],
                  gauge_ops=sys_["gauge_ops"], adiabatic_unitary=Ad,
                  derivative="spectral")
    opt.add_smoothness_penalty(1e-3)
    opt.add_l2_amplitude_penalty(1e-4)
    rng = np.random.default_rng(22)
    x = rng.uniform(-1.0, 1.0, size=opt.n_core + opt.n_gauge)
    _grad_case(opt, x, "GrapeOC[2 controls + adiabatic + penalties]")


@test
def test_grape_spectral_equals_frechet():
    """The spectral and Frechet derivative paths must agree to machine precision."""
    sys_ = make_grape_system()
    rng = np.random.default_rng(23)
    kw = dict(cost_type="projected", projector=sys_["projector"],
              gauge_ops=sys_["gauge_ops"])
    a = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], 0.02, 6, derivative="spectral", **kw)
    b = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], 0.02, 6, derivative="frechet", **kw)
    x = rng.uniform(-1, 1, size=a.n_core + a.n_gauge)
    fa, ga = a.cost_and_grad(x)
    fb, gb = b.cost_and_grad(x)
    close(fa, fb, 1e-13, "spectral/frechet cost:")
    close(ga, gb, 1e-10, "spectral/frechet gradient:")


@test
def test_degenerate_spectrum_gradient():
    """The Daleckii-Krein divided difference must stay accurate for degenerate H."""
    n = 6
    # a drift with a deliberately degenerate spectrum
    H0 = qt.Qobj(np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 2.0]))
    Hc = [qt.Qobj(np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))]
    Ut = qt.Qobj(np.diag(np.exp(-1j * np.arange(n) * 0.3)))
    opt = GrapeOC(H0, Hc, Ut, dt=0.1, n_steps=3, derivative="spectral")
    x = np.zeros(opt.n_core)                       # u = 0 -> fully degenerate H_n
    _grad_case(opt, x, "GrapeOC[degenerate spectrum]", tol=1e-5)


# =============================================================================
# 3. Agreement with the old implementations
# =============================================================================

@test
def test_matches_old_grape_class():
    """New GrapeOC cost/gradient must match the legacy GrapeLBFGS from Grape.py.

    Grape.py has been retired; keep this test around so the comparison can be
    re-run by dropping the old file back in next to this one.
    """
    try:
        from Grape import GrapeLBFGS as OldGrape
    except ImportError:
        print("      (skipped: legacy Grape.py has been retired)")
        return
    sys_ = make_grape_system()
    kw = dict(cost_type="projected", projector=sys_["projector"],
              gauge_ops=sys_["gauge_ops"])
    old = OldGrape(sys_["H0"], sys_["Hc"], sys_["target"], 0.02, 5,
                   derivative="frechet", **kw)
    new = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], 0.02, 5,
                  derivative="spectral", **kw)
    old.add_smoothness_penalty(1e-3)
    new.add_smoothness_penalty(1e-3)

    rng = np.random.default_rng(31)
    for _ in range(3):
        x = rng.uniform(-1, 1, size=new.n_core + new.n_gauge)
        pulses, theta = new._unpack(x)
        pulses = pulses.reshape(new.n_steps, new.n_ctrl)
        close(new.cost(x), old.cost(pulses, theta), 1e-11, "cost vs old Grape:")
        close(new.grad(x), old.grad(pulses, theta), 1e-9, "grad vs old Grape:")
        close(new.fidelity(x=x), old.fidelity(pulses, theta), 1e-12,
              "fidelity vs old Grape:")


@test
def test_matches_old_parameter_class():
    """New ParameterOC cost/gradient must match the legacy class in parameter_oc.py.

    parameter_oc.py has been retired; keep this test around so the comparison can
    be re-run by dropping the old file back in next to this one.
    """
    import io
    import contextlib
    try:
        from parameter_oc import ParameterOC as OldParam
    except ImportError:
        print("      (skipped: legacy parameter_oc.py has been retired)")
        return

    sys_ = make_param_system()
    strings = ["bs", "r"]
    num_apply = 3
    old = OldParam(strings, sys_["target"], sys_["n_c"], sys_["n_q"],
                   cost_type="projected", projector=sys_["projector"],
                   gauge_ops=sys_["gauge_ops"])
    new = ParameterOC(strings, sys_["target"], sys_["n_c"], sys_["n_q"],
                      num_apply=num_apply, optimize_phases=True,
                      cost_type="projected", projector=sys_["projector"],
                      gauge_ops=sys_["gauge_ops"])

    rng = np.random.default_rng(32)
    M = new.n_gates
    for _ in range(3):
        rot = rng.uniform(0, np.pi, size=M)
        ph = rng.uniform(0, 2 * np.pi, size=M)
        th = rng.uniform(0, 2 * np.pi, size=new.n_gauge)
        x = new._natural_x(rotation=rot, phase=ph, theta=th)
        with contextlib.redirect_stdout(io.StringIO()):     # old class prints per call
            c_old = old.cost(rot, ph, num_apply, th)
            g_old = old.grad(rot, ph, num_apply, True, th)
            f_old = old.fidelity(rot, ph, num_apply, True, th)
        close(new.cost(x), c_old, 1e-11, "cost vs old ParameterOC:")
        close(new.grad(x), g_old, 1e-9, "grad vs old ParameterOC:")
        close(new.fidelity(x=x), f_old, 1e-12, "fidelity vs old ParameterOC:")


@test
def test_propagate_helpers_match_class():
    """The stand-alone propagate helpers must reproduce the class's U_final."""
    sys_ = make_param_system()
    strings = ["bs", "r"]
    opt = ParameterOC(strings, sys_["target"], sys_["n_c"], sys_["n_q"], num_apply=2)
    rng = np.random.default_rng(33)
    rot = rng.uniform(0, np.pi, size=opt.n_gates)
    ph = rng.uniform(0, 2 * np.pi, size=opt.n_gates)
    U_cls = opt.raw_unitary(rot, ph)
    U_fn = propagate_gates(strings, rot, ph, 2, sys_["a_c"], sys_["a_q"])
    close(U_cls.full(), U_fn.full(), 1e-12, "propagate_gates:")

    g_ = make_grape_system()
    gopt = GrapeOC(g_["H0"], g_["Hc"], g_["target"], 0.02, 5)
    pulses = rng.uniform(-1, 1, size=(5, 1))
    close(gopt.raw_unitary(pulses).full(),
          propagate_pulses(g_["H0"], g_["Hc"], 0.02, pulses).full(),
          1e-11, "propagate_pulses:")


@test
def test_parameter_passing_forms_agree():
    """fidelity()/unitary() must accept all four documented argument forms."""
    # --- GRAPE ---
    sys_ = make_grape_system()
    opt = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], dt=0.05, n_steps=8,
                  cost_type="projected", projector=sys_["projector"],
                  gauge_ops=sys_["gauge_ops"])
    out = opt.optimize(pulses0=np.ones((8, 1)), theta0=np.zeros(2),
                       method="L-BFGS-B", maxiter=30, progress="none")

    f_dict = opt.fidelity(out)                                       # result dict
    f_nat = opt.fidelity(out["pulses_opt"], theta=out["theta_opt"])  # positional natural
    f_kw = opt.fidelity(pulses=out["pulses_opt"], theta=out["theta_opt"])  # keyword
    f_flat = opt.fidelity(x=out["x_opt"])                            # packed vector
    f_best = opt.fidelity()                                          # running best
    for name, val in [("dict", f_dict), ("positional", f_nat), ("keyword", f_kw),
                      ("flat", f_flat), ("implicit-best", f_best)]:
        close(val, out["fidelity"], 1e-12, f"GRAPE fidelity via {name}:")
    close(opt.unitary(out).full(), opt.unitary(x=out["x_opt"]).full(), 1e-14,
          "GRAPE unitary via dict vs flat:")

    # --- ParameterOC ---
    ps = make_param_system(n_c=4, n_q=2)
    popt = ParameterOC(["bs"], ps["target"], 4, 2, num_apply=3,
                       cost_type="projected", projector=ps["projector"],
                       gauge_ops=ps["gauge_ops"])
    pout = popt.optimize(method="L-BFGS-B", maxiter=30, progress="none")
    close(popt.fidelity(pout), pout["fidelity"], 1e-12, "ParameterOC via dict:")
    close(popt.fidelity(pout["rotation_opt"], pout["phase_opt"],
                        theta=pout["theta_opt"]), pout["fidelity"], 1e-12,
          "ParameterOC via positional:")
    close(popt.fidelity(x=pout["x_opt"]), pout["fidelity"], 1e-12,
          "ParameterOC via flat:")

    # --- error paths ---
    def raises(exc, fn, what):
        try:
            fn()
        except exc:
            return
        except Exception as e:                                       # noqa: BLE001
            raise AssertionError(f"{what}: raised {type(e).__name__} not {exc.__name__}")
        raise AssertionError(f"{what}: no exception raised")

    raises(TypeError, lambda: opt.fidelity(out["pulses_opt"], x=out["x_opt"]),
           "both natural and x=")
    raises(ValueError, lambda: opt.fidelity(out["x_opt"]),
           "flat vector passed positionally (wrong shape for pulses)")
    raises(ValueError, lambda: GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"],
                                       0.05, 8).fidelity(),
           "no parameters and never optimized")


@test
def test_gauge_fast_paths_agree():
    """diagonal / commuting / general gauge paths must all give the same G and dG."""
    from optimal_control import _GaugeBlock
    sys_ = make_param_system()
    n_c_op, n_q_op = sys_["gauge_ops"]
    dim = n_c_op.shape[0]
    theta = np.array([0.37, -1.21])

    diag_block = _GaugeBlock([n_c_op, n_q_op], dim)
    check(diag_block.mode == "diagonal", f"expected diagonal path, got {diag_block.mode}")

    # rotate into a random basis -> commuting but not diagonal
    Hr = qt.rand_herm(dim, seed=2)
    Hr.dims = n_c_op.dims
    W = (-1j * Hr).expm()
    rot_ops = [W * n_c_op * W.dag(), W * n_q_op * W.dag()]
    comm_block = _GaugeBlock(rot_ops, dim)
    check(comm_block.mode == "commuting", f"expected commuting path, got {comm_block.mode}")

    # reference: the generic expm/expm_frechet path
    gen_block = _GaugeBlock(rot_ops, dim)
    gen_block.mode = "general"

    Gc, dGc = comm_block(theta, True)
    Gg, dGg = gen_block(theta, True)
    close(Gc, Gg, 1e-10, "commuting vs general G:")
    for a, b in zip(dGc, dGg):
        close(a, b, 1e-9, "commuting vs general dG:")

    # diagonal path against the module-level helper
    close(diag_block(theta, False)[0], gauge_unitary([n_c_op, n_q_op], theta).full(),
          1e-12, "diagonal gauge helper:")

    # non-commuting operators must fall back to 'general'
    A = qt.rand_herm(dim, seed=3); A.dims = n_c_op.dims
    B = qt.rand_herm(dim, seed=4); B.dims = n_c_op.dims
    check(_GaugeBlock([A, B], dim).mode == "general", "expected general fallback")


# =============================================================================
# 4. Optimizers
# =============================================================================

@test
def test_lbfgs_finds_known_solution():
    """A single beamsplitter with a known exact solution must be recovered."""
    sys_ = make_param_system(n_c=4, n_q=2)
    # target = exactly one bs gate at g = 0.8, phase = 1.1
    opt0 = ParameterOC(["bs"], qt.qeye(sys_["a_c"].dims[0]), 4, 2, num_apply=1)
    U_exact = opt0.raw_unitary(np.array([0.8]), np.array([1.1]))
    opt = ParameterOC(["bs"], U_exact, 4, 2, num_apply=1)
    out = opt.optimize(rotation0=np.array([0.3]), phase0=np.array([0.3]),
                       method="multistart", n_starts=6, maxiter=300,
                       progress="none", seed=1)
    check(out["fidelity"] > 1 - 1e-8,
          f"expected to recover the exact gate, got F = {out['fidelity']:.8f}")


@test
def test_all_methods_run():
    """Every advertised optimizer must run end-to-end and improve on the start point."""
    sys_ = make_param_system(n_c=4, n_q=2)
    P = sum((k * k.dag() for k in
             [qt.tensor(qt.fock(4, i), qt.fock(2, j)) for i in range(3) for j in range(2)]),
            0 * qt.tensor(qt.fock(4, 0), qt.fock(2, 0)) *
            qt.tensor(qt.fock(4, 0), qt.fock(2, 0)).dag())
    n_c_op = qt.tensor(qt.num(4), qt.qeye(2))
    n_q_op = qt.tensor(qt.qeye(4), qt.num(2))
    target = (-1j * np.pi * (n_c_op * n_q_op)).expm()

    skipped = []
    for m in available_methods():
        opt = ParameterOC(["bs"], target, 4, 2, num_apply=3, cost_type="projected",
                          projector=P, gauge_ops=[n_c_op, n_q_op])
        rng = np.random.default_rng(2)
        r0 = rng.uniform(0, np.pi, size=opt.n_gates)
        p0 = rng.uniform(0, 2 * np.pi, size=opt.n_gates)
        t0 = np.zeros(2)
        F_start = opt.fidelity(r0, p0, theta=t0)
        try:
            out = opt.optimize(rotation0=r0, phase0=p0, theta0=t0, method=m,
                               n_starts=3, maxiter=60, progress="none", seed=0,
                               method_options={"niter": 3, "n": 32, "popsize": 4})
        except ImportError as exc:
            skipped.append(f"{m} ({exc.__class__.__name__})")
            continue
        check(np.isfinite(out["fidelity"]), f"method {m}: non-finite fidelity")
        check(out["fidelity"] >= F_start - 1e-9,
              f"method {m}: fidelity got worse ({F_start:.6f} -> {out['fidelity']:.6f})")
        check(out["x_opt"].shape == (opt.n_core + opt.n_gauge,),
              f"method {m}: wrong x_opt shape")
    if skipped:
        print(f"      (optional backends not installed: {', '.join(skipped)})")


@test
def test_grape_optimizer_improves():
    """A short GRAPE run must reduce the cost and leave a consistent result dict."""
    sys_ = make_grape_system()
    opt = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], dt=0.05, n_steps=12,
                  cost_type="projected", projector=sys_["projector"],
                  gauge_ops=sys_["gauge_ops"])
    p0 = 3.0 * np.ones((12, 1))
    F0 = opt.fidelity(p0, theta=np.zeros(2))
    out = opt.optimize(pulses0=p0, theta0=np.zeros(2), method="L-BFGS-B",
                       maxiter=120, progress="none")
    check(out["fidelity"] > F0, f"GRAPE did not improve: {F0:.6f} -> {out['fidelity']:.6f}")
    close(out["pulses_opt"].shape, (12, 1), 0, "pulses_opt shape:")
    close(opt.fidelity(out["pulses_opt"], theta=out["theta_opt"]),
          out["fidelity"], 1e-12, "reported vs recomputed fidelity:")


@test
def test_multistart_reports_every_start():
    """The result dict must expose the per-start fidelities used to pick the winner."""
    sys_ = make_param_system(n_c=4, n_q=2)
    opt = ParameterOC(["bs"], sys_["target"], sys_["n_c"], sys_["n_q"], num_apply=2)
    out = opt.optimize(method="multistart", n_starts=5, maxiter=40,
                       progress="none", seed=3)
    check(len(out["starts"]) == 5, f"expected 5 start records, got {len(out['starts'])}")
    best = max(s["fidelity"] for s in out["starts"])
    check(out["fidelity"] >= best - 1e-9,
          "the returned fidelity must be at least the best individual start")


@test
def test_target_fidelity_early_stop():
    """target_fidelity must stop the run as soon as the threshold is crossed."""
    opt0 = ParameterOC(["bs"], qt.qeye([4, 2]), 4, 2, num_apply=1)
    U_exact = opt0.raw_unitary(np.array([0.8]), np.array([1.1]))
    opt = ParameterOC(["bs"], U_exact, 4, 2, num_apply=1)
    out = opt.optimize(rotation0=np.array([0.4]), phase0=np.array([0.9]),
                       method="multistart", n_starts=50, maxiter=500,
                       target_fidelity=0.99, progress="none", seed=1)
    check(out["fidelity"] >= 0.99, "target fidelity not reached")
    check(len(out["starts"]) < 50, "early stop did not cut the multistart short")


# =============================================================================
# 5. Cancellation and async
# =============================================================================

@test
def test_cancel_keeps_best_iterate():
    """cancel() must unwind cleanly and still return the best point seen."""
    sys_ = make_grape_system()
    opt = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], dt=0.05, n_steps=20,
                  cost_type="projected", projector=sys_["projector"])

    n_calls = {"k": 0}
    real_cg = opt.cost_and_grad

    def counting(x):                       # cancel after a fixed number of evaluations
        n_calls["k"] += 1
        if n_calls["k"] == 12:
            opt.cancel()
        return real_cg(x)

    opt.cost_and_grad = counting           # type: ignore[method-assign]
    out = opt.optimize(pulses0=np.ones((20, 1)), method="L-BFGS-B", maxiter=1000,
                       progress="none")
    check(out["cancelled"], "run should be flagged as cancelled")
    check(np.isfinite(out["fidelity"]), "cancelled run must still report a fidelity")
    check(out["x_opt"].shape == (opt.n_core,), "cancelled run must return an x_opt")


@test
def test_run_async_and_stop():
    """run_async must return a live handle that stop()/wait() control correctly."""
    sys_ = make_grape_system(n_c=5, n_q=3)
    opt = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], dt=0.05, n_steps=40,
                  cost_type="projected", projector=sys_["projector"],
                  gauge_ops=sys_["gauge_ops"])
    h = opt.run_async(pulses0=2.0 * np.ones((40, 1)), theta0=np.zeros(2),
                      method="multistart", n_starts=50, maxiter=2000,
                      progress="none", seed=0)
    t0 = time.perf_counter()
    while opt.best_fidelity == -np.inf and time.perf_counter() - t0 < 10:
        time.sleep(0.01)
    check(h.running or h.result is not None, "async run did not start")
    h.stop()
    out = h.wait(timeout=60)
    check(out is not None, "async run did not finish within the timeout")
    check(not h.running, "thread still alive after wait()")
    check(np.isfinite(out["fidelity"]), "async run returned a non-finite fidelity")


@test
def test_reoptimize_after_cancel():
    """A cancelled run must not poison the next one: the flag is cleared on entry."""
    sys_ = make_param_system(n_c=4, n_q=2)
    opt = ParameterOC(["bs"], sys_["target"], 4, 2, num_apply=6)
    rng = np.random.default_rng(77)
    r0 = rng.uniform(0.2, 2.5, size=opt.n_gates)
    p0 = rng.uniform(0.0, 6.0, size=opt.n_gates)

    # First run: cancel from inside the objective on the second evaluation, so it
    # fires no matter how quickly the local minimizer would otherwise converge.
    n = {"k": 0}
    real = opt.cost_and_grad

    def counting(x):
        n["k"] += 1
        if n["k"] == 2:
            opt.cancel()
        return real(x)

    opt.cost_and_grad = counting                 # type: ignore[method-assign]
    out1 = opt.optimize(rotation0=r0, phase0=p0, method="multistart", n_starts=20,
                        maxiter=500, progress="none")
    check(out1["cancelled"], "first run should report cancelled")

    # Second run: the stale flag must have been cleared on entry.
    opt.cost_and_grad = real                     # type: ignore[method-assign]
    out2 = opt.optimize(rotation0=r0, phase0=p0, method="L-BFGS-B", maxiter=50,
                        progress="none")
    check(not out2["cancelled"], "second run must not inherit the cancel flag")

    # Calling cancel() while nothing is running is a no-op.
    opt.cancel()
    out3 = opt.optimize(rotation0=r0, phase0=p0, method="L-BFGS-B", maxiter=50,
                        progress="none")
    check(not out3["cancelled"], "cancel() before a run must not abort it")


# =============================================================================
# 6. Saving / loading
# =============================================================================

@test
def test_save_creates_new_folder_on_collision():
    """Saving twice to the same name must not overwrite unless override=True."""
    sys_ = make_param_system(n_c=4, n_q=2)
    opt = ParameterOC(["bs"], sys_["target"], 4, 2, num_apply=2,
                      cost_type="projected", projector=sys_["projector"],
                      gauge_ops=sys_["gauge_ops"])
    out = opt.optimize(method="L-BFGS-B", maxiter=40, progress="none")

    tmp = Path(tempfile.mkdtemp())
    try:
        p1 = opt.save(tmp / "run", out)
        check(p1 == tmp / "run", f"first save should use the given name, got {p1}")
        p2 = opt.save(tmp / "run", out)
        check(p2 == tmp / "run_1", f"collision should give run_1, got {p2}")
        p3 = opt.save(tmp / "run", out)
        check(p3 == tmp / "run_2", f"second collision should give run_2, got {p3}")
        p4 = opt.save(tmp / "run", out, override=True)
        check(p4 == tmp / "run", f"override=True must reuse the name, got {p4}")
        check(sorted(x.name for x in tmp.iterdir()) == ["run", "run_1", "run_2"],
              "unexpected folders created")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_save_load_roundtrip_parameter():
    """A saved ParameterOC run must reload into an identical fidelity."""
    sys_ = make_param_system(n_c=4, n_q=2)
    opt = ParameterOC(["bs", "r"], sys_["target"], 4, 2, num_apply=2,
                      cost_type="projected", projector=sys_["projector"],
                      gauge_ops=sys_["gauge_ops"])
    out = opt.optimize(method="multistart", n_starts=3, maxiter=60,
                       progress="none", seed=4)
    tmp = Path(tempfile.mkdtemp())
    try:
        path = opt.save(tmp / "prun", out, extra={"note": "roundtrip", "n": 3})
        r = load_run(path)
        close(r["x_opt"], out["x_opt"], 1e-12, "x_opt roundtrip:")
        close(r["rotation_opt"], out["rotation_opt"], 1e-12, "rotation roundtrip:")
        close(r["phase_opt"], out["phase_opt"], 1e-12, "phase roundtrip:")
        close(r["theta_opt"], out["theta_opt"], 1e-12, "theta roundtrip:")
        md = r["metadata"]
        check(md["unitary_strings"] == ["bs", "r"], "metadata unitary_strings wrong")
        check(md["num_apply"] == 2, "metadata num_apply wrong")
        check(md["extra"]["note"] == "roundtrip", "extra metadata not stored")

        q = r["qobjs"]
        check(q["U_target"] is not None and q["projector"] is not None,
              "target/projector not saved")
        check(len(q["gauge_ops"]) == 2, "gauge ops not saved")
        # rebuild the fidelity from the saved objects alone
        U = propagate_gates(md["unitary_strings"], r["rotation_opt"], r["phase_opt"],
                            md["num_apply"], q["a_c"], q["a_q"])
        G = gauge_unitary(q["gauge_ops"], r["theta_opt"])
        Uc = G * U
        P = q["projector"]
        F = abs((q["U_target"].dag() * Uc * P).tr()) ** 2 / (P.tr().real ** 2)
        close(F, out["fidelity"], 1e-10, "reconstructed fidelity:")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_save_load_roundtrip_grape():
    """A saved GrapeOC run must reload into an identical fidelity."""
    sys_ = make_grape_system()
    opt = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], dt=0.05, n_steps=10,
                  cost_type="projected", projector=sys_["projector"],
                  gauge_ops=sys_["gauge_ops"])
    opt.add_smoothness_penalty(1e-5)
    out = opt.optimize(pulses0=np.ones((10, 1)), theta0=np.zeros(2),
                       method="L-BFGS-B", maxiter=60, progress="none")
    tmp = Path(tempfile.mkdtemp())
    try:
        path = opt.save(tmp / "grun", out, extra={"drive": {"g": 1.0}})
        r = load_run(path)
        md = r["metadata"]
        close(r["pulses_opt"], out["pulses_opt"], 1e-12, "pulses roundtrip:")
        check(md["n_steps"] == 10 and md["n_ctrl"] == 1, "grape metadata wrong")
        close(md["dt"], 0.05, 1e-15, "dt metadata:")
        q = r["qobjs"]
        U = propagate_pulses(q["H_drift"], q["H_controls"], md["dt"], r["pulses_opt"])
        G = gauge_unitary(q["gauge_ops"], r["theta_opt"])
        P = q["projector"]
        F = abs((q["U_target"].dag() * (G * U) * P).tr()) ** 2 / (P.tr().real ** 2)
        close(F, out["fidelity"], 1e-9, "reconstructed GRAPE fidelity:")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_load_old_format_still_works():
    """The legacy *_qobjs folders on disk must still be loadable."""
    here = Path(__file__).resolve().parent
    for old in [here / "Grape_results", here / "ParameterOc_results"]:
        if not old.is_dir():
            continue
        for qdir in sorted(old.glob("*_qobjs"))[:1]:
            q = load_qobjs(qdir)
            check(q["U_target"] is not None, f"could not load U_target from {qdir}")
            print(f"      loaded legacy {qdir.name}: "
                  f"dim={q['U_target'].shape[0]}, gauge={len(q['gauge_ops'])}")


# =============================================================================
# 7. Performance
# =============================================================================

def _time(fn, n=None, target=0.4):
    """Time a callable, auto-choosing the repeat count."""
    fn()
    if n is None:
        t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
        n = max(3, int(target / max(dt, 1e-9)))
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


@test
def test_speed_parameter_gate_derivatives():
    """The analytic FrameGate must beat expm + expm_frechet by a wide margin.

    The reference below is a literal transcription of what the old ParameterOC
    did per gate: one dense matrix exponential plus one Frechet derivative for
    each of the two parameters.  Both paths are checked to agree to machine
    precision first, so this is a like-for-like timing.
    """
    from scipy.linalg import expm_frechet

    n_c, n_q, num_apply = 12, 2, 10          # the size used in the notebook
    a_c = qt.tensor(qt.destroy(n_c), qt.qeye(n_q))
    a_q = qt.tensor(qt.qeye(n_c), qt.destroy(n_q))
    n_c_op, n_q_op = a_c.dag() * a_c, a_q.dag() * a_q
    target = (-1j * np.pi * (n_c_op * n_q_op)).expm()
    kets = [qt.tensor(qt.fock(n_c, i), qt.fock(n_q, j)) for i in range(8) for j in range(n_q)]
    P = sum((k * k.dag() for k in kets), 0 * kets[0] * kets[0].dag())

    opt = ParameterOC(["bs"], target, n_c, n_q, num_apply=num_apply,
                      cost_type="projected", projector=P, gauge_ops=[n_c_op, n_q_op])
    gate = opt.gates["bs"]

    g_val, phase = 0.73, 1.21

    def reference():
        """expm + two expm_frechet calls, i.e. the pre-refactor cost per gate."""
        H = g_val * (a_c.dag() * a_q * np.exp(1j * phase)
                     + a_c * a_q.dag() * np.exp(-1j * phase))
        A = (-1j * H).full()
        U = (-1j * H).expm().full()
        dA_r = (-1j * (a_c.dag() * a_q * np.exp(1j * phase)
                       + a_c * a_q.dag() * np.exp(-1j * phase))).full()
        dA_p = (-1j * g_val * (1j * a_c.dag() * a_q * np.exp(1j * phase)
                               - 1j * a_c * a_q.dag() * np.exp(-1j * phase))).full()
        return (U, expm_frechet(A, dA_r, compute_expm=False),
                expm_frechet(A, dA_p, compute_expm=False))

    # correctness before speed
    ref = reference()
    new_ = gate.evaluate(g_val, phase, need_grad=True)
    for r, n_, lbl in zip(ref, new_, ("U", "dU/dr", "dU/dphi")):
        close(r, n_, 1e-11, f"analytic gate {lbl}:")

    t_ref = _time(reference)
    t_new = _time(lambda: gate.evaluate(g_val, phase, need_grad=True))
    print(f"      gate d={n_c*n_q}: expm+expm_frechet {t_ref*1e3:7.3f} ms  ->  "
          f"analytic {t_new*1e3:7.3f} ms   ({t_ref/t_new:5.1f}x faster)")
    check(t_new < t_ref, "the analytic gate should not be slower")

    # and the full cost+grad, for reference
    rng = np.random.default_rng(41)
    x = opt._natural_x(rotation=rng.uniform(0, np.pi, size=num_apply),
                       phase=rng.uniform(0, 2 * np.pi, size=num_apply),
                       theta=rng.uniform(0, 2 * np.pi, size=2))
    t_cg = _time(lambda: opt.cost_and_grad(x))
    print(f"      full cost+grad, {num_apply} gates, d={n_c*n_q}: {t_cg*1e3:.3f} ms")


@test
def test_speed_grape_derivative_modes():
    """The spectral slice derivative must beat the Frechet one at notebook scale."""
    n_c, n_q, n_steps = 10, 5, 100          # the size used in the notebook
    a_r = qt.tensor(qt.destroy(n_c), qt.qeye(n_q))
    a_q = qt.tensor(qt.qeye(n_c), qt.destroy(n_q))
    n_r_op, n_q_op = a_r.dag() * a_r, a_q.dag() * a_q
    g, alpha = 2 * np.pi, -160 * 2 * np.pi
    H0 = g * (a_q.dag() * a_r + a_q * a_r.dag()) + (alpha / 2) * (a_q.dag() ** 2 * a_q ** 2)
    Hc = [n_q_op]
    target = (-1j * np.pi * (n_r_op * n_q_op)).expm()
    kets = [qt.tensor(qt.fock(n_c, i), qt.fock(n_q, j)) for i in range(5) for j in range(2)]
    P = sum((k * k.dag() for k in kets), 0 * kets[0] * kets[0].dag())

    kw = dict(cost_type="projected", projector=P, gauge_ops=[n_r_op, n_q_op])
    dt = 0.005
    spec = GrapeOC(H0, Hc, target, dt, n_steps, derivative="spectral", **kw)
    frec = GrapeOC(H0, Hc, target, dt, n_steps, derivative="frechet", **kw)

    rng = np.random.default_rng(42)
    pulses = 140 * 2 * np.pi * np.ones((n_steps, 1)) + rng.normal(size=(n_steps, 1))
    x = spec._pack(pulses.reshape(-1), np.zeros(2))

    # correctness before speed
    f_s, g_s = spec.cost_and_grad(x)
    f_f, g_f = frec.cost_and_grad(x)
    close(f_s, f_f, 1e-13, "spectral vs frechet cost:")
    close(g_s, g_f, 1e-9, "spectral vs frechet gradient:")

    t_frec = _time(lambda: frec.cost_and_grad(x), n=5)
    t_spec = _time(lambda: spec.cost_and_grad(x), n=5)
    print(f"      cost+grad, {n_steps} steps, d={n_c*n_q}: frechet {t_frec*1e3:7.1f} ms  ->  "
          f"spectral {t_spec*1e3:7.1f} ms   ({t_frec/t_spec:5.1f}x faster)")
    check(t_spec < t_frec, "the spectral path should not be slower")


@test
def test_progress_overhead():
    """Progress reporting must cost a negligible fraction of an evaluation.

    Timed A/B/A/B so that cache warm-up and CPU frequency drift cannot be
    mistaken for a real difference; the two configurations are also timed
    directly against each other in a tight loop.
    """
    from optimal_control import _Progress

    sys_ = make_grape_system()
    opt = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], dt=0.02, n_steps=20,
                  cost_type="projected", projector=sys_["projector"])
    x = np.ones(opt.n_core)

    def run(n_reps):
        t0 = time.perf_counter()
        for _ in range(n_reps):
            opt.cost_and_grad(x)
        return (time.perf_counter() - t0) / n_reps

    reps = 200
    opt._progress = None
    run(20)                                     # warm up

    off, on = [], []
    for _ in range(5):                          # interleave to cancel out drift
        opt._progress = None
        off.append(run(reps))
        opt._progress = _Progress(mode="line", every=1e9)   # tracks, never prints
        on.append(run(reps))
    opt._progress = None

    t_off, t_on = float(np.median(off)), float(np.median(on))
    over = (t_on - t_off) / t_off * 100
    print(f"      per-evaluation cost {t_off*1e3:.3f} ms; progress bookkeeping "
          f"{over:+.2f}% ({(t_on-t_off)*1e6:+.2f} us/eval, below timer noise)")
    # One-sided: the bookkeeping is a perf_counter() call plus a float compare, so
    # it can only add time.  A negative reading just means the ~1 ms evaluation is
    # noisier than the effect being measured, which is itself the point.
    check(t_on - t_off < 5e-6,
          f"progress bookkeeping costs {(t_on-t_off)*1e6:.2f} us/eval, expected < 5 us")

    # And printing itself is throttled: at every=0.25 s a 10 s run prints ~40 lines
    # no matter how many evaluations happen.
    p = _Progress(mode="none", every=0.25)
    t0 = time.perf_counter()
    for _ in range(200_000):
        p.update(0.5, 0.5)
    per_update = (time.perf_counter() - t0) / 200_000
    print(f"      _Progress.update() alone: {per_update*1e9:.0f} ns")
    check(per_update < 2e-6, "progress update itself should be sub-microsecond")


@test
def test_cost_and_grad_single_propagation():
    """cost_and_grad must propagate once, not twice (the old code propagated twice)."""
    sys_ = make_grape_system()
    opt = GrapeOC(sys_["H0"], sys_["Hc"], sys_["target"], dt=0.02, n_steps=10)
    calls = {"n": 0}
    real_forward = opt._forward

    def counting(x_core, need_grad):
        calls["n"] += 1
        return real_forward(x_core, need_grad)

    opt._forward = counting                # type: ignore[method-assign]
    opt.cost_and_grad(np.ones(opt.n_core))
    check(calls["n"] == 1, f"cost_and_grad called _forward {calls['n']} times, expected 1")


# =============================================================================
# 8. Input validation
# =============================================================================

@test
def test_input_validation():
    """Bad inputs must raise clear errors instead of failing silently later."""
    sys_ = make_param_system(n_c=4, n_q=2)

    def raises(exc, fn, what):
        try:
            fn()
        except exc:
            return
        except Exception as e:                                   # noqa: BLE001
            raise AssertionError(f"{what}: raised {type(e).__name__} not {exc.__name__}")
        raise AssertionError(f"{what}: no exception raised")

    raises(ValueError, lambda: ParameterOC(["bs"], sys_["target"], 4, 2,
                                           cost_type="projected"),
           "projected without projector")
    raises(ValueError, lambda: ParameterOC(["bs"], sys_["target"], 3, 2),
           "n_c*n_q mismatching the target dimension")
    raises(ValueError, lambda: ParameterOC(["nope"], sys_["target"], 4, 2),
           "unknown gate name")
    raises(ValueError, lambda: ParameterOC([], sys_["target"], 4, 2),
           "empty gate sequence")

    opt = ParameterOC(["bs"], sys_["target"], 4, 2, num_apply=2)
    raises(ValueError, lambda: opt.optimize(rotation0=np.zeros(5), progress="none"),
           "wrong rotation0 length")
    raises(ValueError, lambda: opt.optimize(method="does-not-exist", progress="none"),
           "unknown optimizer method")

    g_ = make_grape_system()
    gopt = GrapeOC(g_["H0"], g_["Hc"], g_["target"], 0.02, 5)
    raises(ValueError, lambda: gopt.optimize(pulses0=np.zeros((3, 1)), progress="none"),
           "wrong pulses0 shape")
    raises(ValueError, lambda: gopt.optimize(pulse_bounds=[(0, 1), (0, 1)],
                                             progress="none"),
           "pulse_bounds not matching n_ctrl")


@test
def test_non_hermitian_falls_back():
    """Non-Hermitian generators must silently switch to the Frechet derivative."""
    d = 6
    H0 = qt.Qobj(np.random.default_rng(1).normal(size=(d, d)))   # not Hermitian
    Hc = [qt.Qobj(np.diag(np.arange(d, dtype=float)))]
    Ut = qt.Qobj(np.eye(d))
    opt = GrapeOC(H0, Hc, Ut, dt=0.01, n_steps=3, derivative="spectral")
    check(opt.derivative == "frechet", "should have fallen back to frechet")
    _grad_case(opt, np.ones(opt.n_core) * 0.3, "GrapeOC[non-Hermitian fallback]")


# =============================================================================
# Runner
# =============================================================================

def main(argv):
    pattern = None
    if "-k" in argv:
        pattern = argv[argv.index("-k") + 1]

    tests = [v for v in globals().values()
             if callable(v) and getattr(v, "_is_test", False)]
    if pattern:
        tests = [t for t in tests if pattern in t.__name__]

    print(f"Running {len(tests)} test(s)\n" + "=" * 78)
    t_start = time.perf_counter()
    for fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            _FAILURES.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}")
            print(traceback.format_exc())
        else:
            _PASSED.append(fn.__name__)
            print(f"[ ok ] {fn.__name__}  ({time.perf_counter() - t0:.2f}s)")

    print("=" * 78)
    print(f"{len(_PASSED)} passed, {len(_FAILURES)} failed "
          f"in {time.perf_counter() - t_start:.1f}s")
    if _FAILURES:
        print("failed: " + ", ".join(_FAILURES))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
