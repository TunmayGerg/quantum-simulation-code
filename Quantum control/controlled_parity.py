"""
Transmon-controlled PARITY of ONE cavity, from the ge transmon-cavity exchange
only, optimised for minimum  g * T_total.

    U_target = |g><g| (x) 1  +  |e><e| (x) exp(i pi n)          (n = cavity Fock)

Build the transmon-controlled cavity-cavity SWAP from it as

    cSWAP = V^dag . (controlled parity on cavity 1) . V

with V a single 50:50 cavity-cavity beamsplitter, since  V^dag n_1 V = n_-  for
a_- = (a1 - a2)/sqrt(2)  and  SWAP = exp(i pi n_-).  Only one pump tone is ever
on at a time.

-------------------------------------------------------------------------------
DRIVE AND CONTROL QUBIT
-------------------------------------------------------------------------------
PRIMARY DRIVE.  One pump, activating the ge exchange between the transmon and
cavity 1 only:

    H_j = Delta_j |e><e| + g ( e^{i phi_j} a^dag |g><e| + h.c. ),   duration t_j

Segment j is three numbers: pulse area  x_j = g t_j, drive phase phi_j, and
detuning r_j = Delta_j / g.  Because only one cavity is driven the supermode
coupling enhancement is absent, so  G = g  and

    g * T_total = sum_j x_j

CONTROL QUBIT.  The ordinary transmon g-e qubit.  Both control states live in
the same driven 2x2 block {|m,g>, |m-1,e>}, so:

  * the ge exchange conserves n + n_q  ->  the propagator is block diagonal in
    2x2 blocks, one per total excitation m;
  * requiring |<m,g|U|m,g>| = 1 FORCES each block to be diagonal (no g-e mixing);
  * unitarity then fixes the e phase from the g phase,
        U^gg_m = e^{i(gamma + psi_m)},  U^ee_m = e^{i(gamma - psi_m)},
        gamma = -sum_j Delta_j t_j / 2   (independent of m).

Controlled parity needs  <n,e|U|n,e> / <n,g|U|n,g> = e^{i(pi n + c)}, and since
<n,g|U|n,g> = U^gg_n while <n,e|U|n,e> = U^ee_{n+1}, that is

        psi_{n+1} + psi_n = -pi n + c'          =>   psi_m = -pi m / 2

(the free constant c' = -pi/2 makes the profile linear).  So the design target is
the SAME "quarter parity" profile as for the supermode version:

        <m,g| U |m,g> = exp(-i pi m / 2)        (theta = -pi/2)

and the leftover unconditional factor exp(-i pi n / 2) is a plain cavity frame
rotation -- a virtual Z, free.  No beamsplitter is needed to finish the parity
itself.

FOCK RANGE.  To cover cavity Fock 0..ncut on BOTH control branches we need the
g branch for m = 0..ncut and the e branch for m = 1..ncut+1, i.e. blocks
m = 0..ncut+1, so  nmax = ncut + 1.  For ncut = 8 that is nmax = 9.

  Caveat for the sandwich: V mixes the cavities, so cavity 1 after V holds up to
  n1 + n2 photons.  A parity exact to Fock 8 therefore makes the cSWAP exact
  exactly when the TOTAL photon number is <= 8.  Use check_cswap_from_parity()
  to see this directly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from controlled_swap import ParityTrain, FixedArea, minimize_area, _seg2

__all__ = [
    "THETA", "target_nmax", "design_sequence", "load_sequence", "save_sequence",
    "single_cavity_ops", "sequence_unitary", "ideal_controlled_parity",
    "cparity_fidelity", "beamsplitter_to_dark", "check_cswap_from_parity",
    "block_report", "find_min_gT", "produce_optimum", "crosscheck_optimal_control",
    "blocks_gg_ee", "gate_fidelity_blocks",
    "descend_gT", "reduce_M", "optimise_alternating",
    "FLOOR_DEFAULT",
]

# Fidelity the sequence must reach before g*T is allowed to shrink further.
FLOOR_DEFAULT = 1 - 1e-4

THETA = -np.pi / 2          # design target slope; +pi/2 works equally well


def target_nmax(ncut: int) -> int:
    """Blocks m = 0..nmax needed to cover cavity Fock 0..ncut on both branches."""
    return ncut + 1


# =============================================================================
# 1.  design (reduced 2x2-block problem; machinery is shared with controlled_swap)
# =============================================================================
def _descend_area(nmax, M, p0, theta, use_detuning, floor=None,
                  step=0.97, retries=10, seed=0, verbose=False):
    """Shrink sum_j x_j (= g*T) at fixed segment count, warm-starting each step."""
    floor = FLOOR_DEFAULT if floor is None else floor
    best, stall = (float(p0[:M].sum()), np.asarray(p0, float).copy()), 0
    while stall < 5 and best[0] > np.pi / 2:
        A = best[0] * step
        fa = FixedArea(nmax, M, A, theta=theta, use_detuning=use_detuning)
        warm = np.concatenate([np.log(np.maximum(best[1][:M], 1e-9)), best[1][M:]])
        F, q = fa.solve(n_starts=0, seed=seed, warm=warm, maxiter=4000)
        if F < floor:
            F, q = fa.solve(n_starts=retries, seed=seed + 17 * stall, warm=warm,
                            maxiter=4000)
        if F >= floor:
            best, stall = (A, fa.expand(q)[0]), 0
            if verbose:
                print(f"      g*T = {A:8.4f}   F = {F:.12f}", flush=True)
        else:
            step = 1 - (1 - step) / 2.0
            stall += 1
            if step > 0.9997:
                break
    return best


def design_sequence(ncut: int = 8, M: int = 20, use_detuning: bool = False,
                    theta: float = THETA, minimise_gT: bool = True,
                    seed: int = 0, n_starts: int = 80, floor: float | None = None,
                    verbose: bool = False):
    """
    Return (params, gT) for a sequence realising the controlled parity exactly on
    cavity Fock 0..ncut, or None if no solution was found at this M.

    params = [x_0..x_{M-1}, phi_0.., r_0..]  (r block only if use_detuning).
    gT = sum_j x_j, because G = g for a single-cavity drive.

    Strategy: find any feasible sequence first (easy, analytic gradients on the
    2x2 blocks), then walk the total area down with warm starts.  Cold-starting
    directly at a small area is much less reliable.
    """
    floor = FLOOR_DEFAULT if floor is None else floor
    nmax = target_nmax(ncut)
    pt = ParityTrain(nmax, M, theta=theta, use_detuning=use_detuning)
    F, p = pt.solve(n_starts=n_starts, seed=seed, maxiter=2000)
    if F < floor:
        return None
    if verbose:
        print(f"    seed: g*T = {p[:M].sum():.4f}  F = {F:.12f}", flush=True)
    if not minimise_gT:
        return p, float(p[:M].sum())
    A, p = _descend_area(nmax, M, p, theta, use_detuning, floor=floor, seed=seed,
                         verbose=verbose)
    return p, float(A)


def blocks_gg_ee(params, M, nmax, use_detuning=False):
    """
    u_m = <m,g|U|m,g>  and  v_m = <m-1,e|U|m-1,e>  for m = 1..nmax, from the exact
    2x2 blocks (plus u_0 = 1, since |0,g> is dark).  No approximation.
    """
    x = np.asarray(params[:M], float)
    phi = np.asarray(params[M:2 * M], float)
    r = np.asarray(params[2 * M:3 * M], float) if use_detuning else np.zeros(M)
    sqn = np.sqrt(np.arange(1, nmax + 1, dtype=float))
    tot = np.broadcast_to(np.eye(2, dtype=complex), (nmax, 2, 2)).copy()
    for j in range(M):
        tot = _seg2(x[j], phi[j], r[j], sqn) @ tot
    u = np.ones(nmax + 1, dtype=complex)
    u[1:] = tot[:, 0, 0]                 # u_m, m = 0..nmax
    v = tot[:, 1, 1]                     # v[m-1] = <m-1,e|U|m-1,e>, m = 1..nmax
    return u, v


def gate_fidelity_blocks(params, M, ncut=8, use_detuning=False, nlam=4001):
    """
    The metric that matters: projected gate fidelity against
    |g><g| (x) 1 + |e><e| (x) exp(i pi n) on cavity Fock 0..ncut x {g,e},
    maximised over the two free virtual Zs (cavity phase lam, transmon phase).

    Computed from the exact 2x2 blocks, so it is cheap enough to drive the
    optimisation with, and agrees with cparity_fidelity() (dense propagation).
    """
    nmax = target_nmax(ncut)
    u, v = blocks_gg_ee(params, M, nmax, use_detuning=use_detuning)
    nn = np.arange(ncut + 1)
    ug = u[:ncut + 1]                    # g branch at Fock n  -> block m = n
    ve = v[:ncut + 1]                    # e branch at Fock n  -> block m = n+1
    d = ncut + 1
    lams = np.linspace(-np.pi, np.pi, nlam)
    Ag = np.abs(np.exp(-1j * np.outer(lams, nn)) @ ug)
    Ae = np.abs(np.exp(-1j * np.outer(lams + np.pi, nn)) @ ve)
    vals = (Ag + Ae) ** 2 / (2 * d) ** 2
    k = int(np.argmax(vals))

    def negF(lam):
        a = abs(np.exp(-1j * lam * nn) @ ug)
        b = abs(np.exp(-1j * (lam + np.pi) * nn) @ ve)
        return -((a + b) ** 2) / (2 * d) ** 2
    res = minimize_scalar(negF, bounds=(lams[max(k - 1, 0)],
                                        lams[min(k + 1, nlam - 1)]),
                          method="bounded", options=dict(xatol=1e-13))
    return float(max(vals[k], -res.fun))


def save_sequence(path, params, M, theta, ncut, use_detuning):
    np.savez(path, params=np.asarray(params), M=int(M), theta=float(theta),
             ncut=int(ncut), nmax=target_nmax(ncut),
             use_detuning=bool(use_detuning),
             gT=float(np.asarray(params)[:M].sum()),
             note=("Transmon-controlled PARITY of one cavity, ge exchange only. "
                   "H_j = Delta_j|e><e| + g(e^{i phi_j} a^dag |g><e| + h.c.) for "
                   "t_j.  params = [x_j = g t_j, phi_j, r_j = Delta_j/g] (r block "
                   "absent if resonant).  g*T = sum_j x_j.  Control qubit = "
                   "transmon g-e; parity is applied when the transmon is in |e>. "
                   "Free corrections afterwards: a virtual Z on the cavity and a "
                   "virtual Z on the transmon.  cSWAP = V^dag . cParity_1 . V "
                   "with V one 50:50 cavity-cavity beamsplitter."))


def load_sequence(path):
    d = np.load(path, allow_pickle=False)
    return (d["params"], int(d["M"]), float(d["theta"]), int(d["ncut"]),
            bool(d["use_detuning"]))


# =============================================================================
# 2.  independent verification by dense propagation (no block shortcut used)
# =============================================================================
def single_cavity_ops(Nc: int, Nq: int = 2):
    a = np.diag(np.sqrt(np.arange(1, Nc)), 1)
    Ic, Iq = np.eye(Nc), np.eye(Nq)
    A = np.kron(a, Iq)

    def lvl(i, j):
        m = np.zeros((Nq, Nq)); m[i, j] = 1.0
        return np.kron(Ic, m)
    return A, lvl


def sequence_unitary(params, M, Nc: int, Nq: int = 2, g: float = 1.0,
                     use_detuning: bool = False):
    """
    Dense propagator of the pulse train in the full cavity (x) transmon space.
    Built straight from H_j -- this does NOT use the 2x2 block reduction, so it
    independently validates it.
    """
    from scipy.linalg import expm
    x = np.asarray(params[:M], float)
    phi = np.asarray(params[M:2 * M], float)
    r = np.asarray(params[2 * M:3 * M], float) if use_detuning else np.zeros(M)

    A, lvl = single_cavity_ops(Nc, Nq)
    sm = lvl(0, 1)                                   # |g><e|
    Pe = lvl(1, 1)
    U = np.eye(Nc * Nq, dtype=complex)
    for j in range(M):
        Hex = g * np.exp(1j * phi[j]) * (A.conj().T @ sm)
        H = r[j] * g * Pe + Hex + Hex.conj().T
        U = expm(-1j * H * (x[j] / g)) @ U
    return U


def ideal_controlled_parity(Nc: int, Nq: int = 2, ctrl: int = 1):
    """|g><g| (x) 1 + |e><e| (x) exp(i pi n)   (parity applied for transmon = ctrl)."""
    par = np.diag((-1.0) ** np.arange(Nc)).astype(complex)
    Ic = np.eye(Nc, dtype=complex)
    out = np.zeros((Nc * Nq, Nc * Nq), dtype=complex)
    for q in range(Nq):
        blk = par if q == ctrl else Ic
        P = np.zeros((Nq, Nq)); P[q, q] = 1.0
        out += np.kron(blk, P)
    return out


def cparity_fidelity(params, M, ncut: int = 8, Nc: int | None = None, Nq: int = 2,
                     use_detuning: bool = False, ctrl: int = 1, nlam: int = 2001,
                     return_details: bool = False):
    """
    Projected gate fidelity of the pulse train against the ideal controlled parity
    on Fock 0..ncut x {g,e}, allowing only the two free corrections: a virtual Z
    on the cavity, exp(-i lam n), and a virtual Z on the transmon, exp(-i c P_e).
    """
    if Nc is None:
        Nc = ncut + 6
    U = sequence_unitary(params, M, Nc, Nq, use_detuning=use_detuning)
    Ut = ideal_controlled_parity(Nc, Nq, ctrl=ctrl)

    idx = [n * Nq + q for n in range(ncut + 1) for q in (0, ctrl)]
    P = np.zeros((Nc * Nq, Nc * Nq)); P[idx, idx] = 1.0
    d = ncut + 1

    Uc = Ut.conj().T @ U @ P                       # per-branch overlaps live on the diagonal
    nlist = np.arange(Nc)
    diag_g = np.array([Uc[n * Nq + 0, n * Nq + 0] for n in range(ncut + 1)])
    diag_e = np.array([Uc[n * Nq + ctrl, n * Nq + ctrl] for n in range(ncut + 1)])
    nn = np.arange(ncut + 1)

    lams = np.linspace(-np.pi, np.pi, nlam)
    E = np.exp(-1j * np.outer(lams, nn))
    vals = (np.abs(E @ diag_g) + np.abs(E @ diag_e)) ** 2 / (2 * d) ** 2
    k = int(np.argmax(vals))

    def negF(lam):
        e = np.exp(-1j * lam * nn)
        return -((abs(e @ diag_g) + abs(e @ diag_e)) ** 2) / (2 * d) ** 2
    res = minimize_scalar(negF, bounds=(lams[max(k - 1, 0)],
                                        lams[min(k + 1, nlam - 1)]),
                          method="bounded", options=dict(xatol=1e-13))
    F = max(vals[k], -res.fun)
    lam = res.x if -res.fun >= vals[k] else lams[k]
    if return_details:
        return float(F), float(lam), U
    return float(F)


def block_report(params, M, nmax: int, use_detuning: bool = False):
    """
    Per-block table: modulus of the diagonal element, the g-e mixing that must
    vanish, and the g/e phases.  Uses the 2x2 blocks, so comparing its verdict
    with cparity_fidelity() cross-checks the reduction.
    """
    x = np.asarray(params[:M], float)
    phi = np.asarray(params[M:2 * M], float)
    r = np.asarray(params[2 * M:3 * M], float) if use_detuning else np.zeros(M)
    sqn = np.sqrt(np.arange(1, nmax + 1, dtype=float))
    tot = np.broadcast_to(np.eye(2, dtype=complex), (nmax, 2, 2)).copy()
    for j in range(M):
        tot = _seg2(x[j], phi[j], r[j], sqn) @ tot
    rows = []
    for m in range(1, nmax + 1):
        U = tot[m - 1]
        rows.append((m, abs(U[0, 0]), abs(U[0, 1]),
                     np.angle(U[0, 0]) / np.pi, np.angle(U[1, 1]) / np.pi))
    return rows


# =============================================================================
# 3.  the beamsplitter sandwich  ->  controlled SWAP
# =============================================================================
def beamsplitter_to_dark(Nc: int, Nq: int = 2):
    """
    V with  V^dag n_1 V = n_-,  n_- = a_-^dag a_-,  a_- = (a1 - a2)/sqrt(2).
    Returns (V, max error of that identity on the n1+n2 <= Nc-1 block).
    """
    from scipy.linalg import expm
    a = np.diag(np.sqrt(np.arange(1, Nc)), 1)
    Ic, Iq = np.eye(Nc), np.eye(Nq)
    a1 = np.kron(np.kron(a, Ic), Iq)
    a2 = np.kron(np.kron(Ic, a), Iq)
    n1 = a1.conj().T @ a1
    am = (a1 - a2) / np.sqrt(2)
    nm = am.conj().T @ am

    best = None
    for s in (+1.0, -1.0):
        gen = s * np.pi / 4 * (a1.conj().T @ a2 - a2.conj().T @ a1)
        V = expm(gen)
        D = V.conj().T @ n1 @ V - nm
        keep = [((i * Nc) + j) * Nq + q for i in range(Nc) for j in range(Nc)
                for q in range(Nq) if i + j <= Nc - 1]
        err = np.max(np.abs(D[np.ix_(keep, keep)]))
        if best is None or err < best[1]:
            best = (V, err)
    return best


def check_cswap_from_parity(params, M, ncut_cav: int = 4, Nc: int = 12, Nq: int = 2,
                            use_detuning: bool = False, ctrl: int = 1):
    """
    Build  cSWAP = V^dag . (controlled parity on cavity 1) . V  from the sequence
    and score it against the ideal transmon-controlled SWAP on cavity Fock
    0..ncut_cav each.

    Allowed free corrections: a virtual Z on each control state, plus the passive
    cavity gauge exp(-i lam n_-) exp(-i mu N) (one cavity-cavity beamsplitter and
    a virtual Z per cavity) -- the same gauge convention as controlled_swap and as
    gauge_ops=[n_c, n_q] in optimal_control.

    Returns (F_cSWAP, beamsplitter identity error).
    """
    from controlled_swap import _gauge_corrected_fidelity

    # controlled parity on cavity 1, with cavity 2 a spectator
    Up = sequence_unitary(params, M, Nc, Nq, use_detuning=use_detuning)
    Up = Up.reshape(Nc, Nq, Nc, Nq)
    big = np.zeros((Nc, Nc, Nq, Nc, Nc, Nq), dtype=complex)
    for j in range(Nc):
        big[:, j, :, :, j, :] = Up
    big = big.reshape(Nc * Nc * Nq, Nc * Nc * Nq)

    Vbs, bs_err = beamsplitter_to_dark(Nc, Nq)
    U = Vbs.conj().T @ big @ Vbs

    # columns we test, and where the ideal cSWAP sends them
    ins, outs, branch = [], [], []
    for n1 in range(ncut_cav + 1):
        for n2 in range(ncut_cav + 1):
            for k, q in enumerate((ctrl, 0)):        # k=0 -> swap branch
                ins.append((n1 * Nc + n2) * Nq + q)
                outs.append(((n2 * Nc + n1) if k == 0 else (n1 * Nc + n2)) * Nq + q)
                branch.append(k)
    cols = U[:, ins]
    F, _ = _gauge_corrected_fidelity(cols, outs, np.array(branch), Nc, Nq)
    return float(F), float(bs_err)


# =============================================================================
# 4.  cross-check against the project's own optimal_control.py
# =============================================================================
def crosscheck_optimal_control(params, M, ncut: int = 8, Nc: int = 14, Nq: int = 2):
    """
    Rebuild the same propagator with optimal_control.propagate_gates(["bs"], ...)
    and compare.  Their "bs" gate is H = g (a_c^dag a_q e^{i phi} + h.c.), i.e.
    exactly one segment of this sequence, and its rotation parameter IS the pulse
    area x_j -- so the two constructions must agree to machine precision.

    Resonant sequences only: the built-in "bs" gate has no detuning parameter.

    Returns (max abs difference of the two unitaries, F_cParity from theirs).
    """
    import sys
    from pathlib import Path
    oc_dir = Path(__file__).resolve().parent.parent / "Optimal control"
    if str(oc_dir) not in sys.path:
        sys.path.insert(0, str(oc_dir))
    import qutip as qt
    from optimal_control import propagate_gates

    params = np.asarray(params, float)
    if len(params) == 3 * M and np.any(np.abs(params[2 * M:3 * M]) > 1e-12):
        raise ValueError("cross-check needs a resonant sequence (all Delta_j = 0)")
    rot, ph = params[:M], params[M:2 * M]

    a_c = qt.tensor(qt.destroy(Nc), qt.qeye(Nq))
    a_q = qt.tensor(qt.qeye(Nc), qt.destroy(Nq))
    U_theirs = propagate_gates(["bs"], rot, ph, M, a_c, a_q).full()
    U_mine = sequence_unitary(params, M, Nc, Nq, use_detuning=False)
    diff = float(np.max(np.abs(U_theirs - U_mine)))

    # gauge-optimised projected fidelity, computed from THEIR unitary
    Ut = ideal_controlled_parity(Nc, Nq, ctrl=1)
    nvec = np.arange(Nc)
    nn = np.arange(ncut + 1)
    d = ncut + 1
    ncav = np.repeat(nvec, Nq)                       # cavity occupation per index
    best = -1.0
    for lam in np.linspace(-np.pi, np.pi, 4001):
        Uc = Ut.conj().T @ (np.exp(-1j * lam * ncav)[:, None] * U_theirs)
        dg = np.array([Uc[n * Nq + 0, n * Nq + 0] for n in nn])
        de = np.array([Uc[n * Nq + 1, n * Nq + 1] for n in nn])
        best = max(best, (abs(dg.sum()) + abs(de.sum())) ** 2 / (2 * d) ** 2)
    return diff, float(best)


# =============================================================================
# 5.  one-call driver: find the minimum-gT sequence and verify it
# =============================================================================
def produce_optimum(ncut: int = 8, M_list=(12, 14, 16, 18, 20, 24),
                    thetas=(-np.pi / 2, np.pi / 2), use_detuning: bool = False,
                    seed: int = 3, out_path: str | None = None, verbose: bool = True):
    """
    Find the smallest-g*T sequence for the controlled parity on cavity Fock
    0..ncut, verify it every independent way available, and optionally save it.

    Returns a dict with the sequence and all the verification numbers.
    """
    best = find_min_gT(ncut, M_list=M_list, use_detuning=use_detuning,
                       thetas=thetas, seed=seed, verbose=verbose)
    if best is None:
        raise RuntimeError("no sequence found; widen M_list")
    p, gT, M, theta, det = best
    nmax = target_nmax(ncut)
    rep = dict(params=p, M=M, theta=theta, ncut=ncut, nmax=nmax,
               use_detuning=det, gT=gT)

    rep["F_cparity"] = {nc: cparity_fidelity(p, M, ncut=nc, use_detuning=det)
                        for nc in (ncut - 4, ncut - 2, ncut)}
    rep["blocks"] = block_report(p, M, nmax, use_detuning=det)
    rep["max_block_mixing"] = max(r[2] for r in rep["blocks"])
    rep["F_cswap"] = {}
    for nc in (2, 3, 4, 5):
        F, bserr = check_cswap_from_parity(p, M, ncut_cav=nc, Nc=12,
                                           use_detuning=det)
        rep["F_cswap"][nc] = F
    rep["bs_identity_err"] = bserr
    if not det:
        rep["oc_diff"], rep["oc_F"] = crosscheck_optimal_control(p, M, ncut=ncut)

    if verbose:
        print(f"\ng*T = {gT:.4f}   M = {M} segments   "
              f"theta = {theta/np.pi:+.2f} pi   detuning = {'on' if det else 'off'}")
        for nc, F in rep["F_cparity"].items():
            print(f"  F_cParity (Fock 0..{nc}) = {F:.10f}")
        print(f"  max g-e block mixing      = {rep['max_block_mixing']:.2e}")
        for nc, F in rep["F_cswap"].items():
            print(f"  F_cSWAP  (cavities 0..{nc}, total N<={2*nc:2d}) = {F:.10f}")
        if not det:
            print(f"  vs optimal_control.propagate_gates: max diff = "
                  f"{rep['oc_diff']:.2e}, F = {rep['oc_F']:.10f}")
    if out_path:
        save_sequence(out_path, p, M, theta, ncut, det)
        if verbose:
            print(f"  saved {out_path}")
    return rep


if __name__ == "__main__":
    produce_optimum(ncut=8, out_path="cparity_fock8.npz")


# =============================================================================
# 6.  the two-stage optimiser:  minimise g*T first, then the pulse count M
# =============================================================================
# The two objectives feed each other.  Shrinking g*T consumes fidelity budget;
# shrinking M at fixed g*T *frees* budget (fewer, larger segments describe the
# same gate better, and area-minimisation tends to leave near-zero no-op
# segments behind).  So alternating the two stages beats either one alone.

def _fixed_area_best(nmax, M, A, theta, det, warms, floor, seed, ncut,
                     cold=(0, 60, 160), maxiter=6000):
    """Maximise fidelity at fixed total area A; return (F_true, params)."""
    fa = FixedArea(nmax, M, A, theta=theta, use_detuning=det)
    gf = lambda q: gate_fidelity_blocks(q, M, ncut=ncut, use_detuning=det, nlam=8001)
    best = (0.0, None)
    for w in warms:
        _, q = fa.solve(n_starts=0, seed=seed, warm=w, maxiter=maxiter)
        pp = fa.expand(q)[0]
        F = gf(pp)
        if F > best[0]:
            best = (F, pp)
        if best[0] >= floor:
            return best
    for nst in cold:
        if nst == 0:
            continue
        _, q = fa.solve(n_starts=nst, seed=seed + nst, maxiter=maxiter)
        pp = fa.expand(q)[0]
        F = gf(pp)
        if F > best[0]:
            best = (F, pp)
        if best[0] >= floor:
            break
    return best


def _pack_log(pp, M):
    return np.concatenate([np.log(np.maximum(pp[:M], 1e-9)), pp[M:]])


def descend_gT(params, M, ncut, theta, det, floor=None, step=0.98, seed=0,
               n_jitter=6, verbose=True, ckpt_path=None):
    """
    STAGE A -- shrink g*T at fixed pulse count M.

    Walks the total area down multiplicatively, warm-starting each step, and
    accepts a step only if the TRUE projected parity fidelity on Fock 0..ncut
    stays >= floor.  Returns (gT, params).

    Pass ckpt_path to save after every accepted step, so a long run can be
    interrupted without losing progress.
    """
    floor = FLOOR_DEFAULT if floor is None else floor
    nmax = target_nmax(ncut)
    rng = np.random.default_rng(seed)
    best = (float(np.asarray(params)[:M].sum()), np.asarray(params, float).copy())
    stall = 0
    while stall < 8 and best[0] > np.pi / 2:
        A = best[0] * step
        base = _pack_log(best[1], M)
        warms = [base]
        for _ in range(n_jitter):
            w = base.copy()
            w[:M] += rng.normal(0, 0.2, M)
            w[M:2 * M] += rng.normal(0, 0.25, M)
            if det:
                w[2 * M:] += rng.normal(0, 0.6, M)
            warms.append(w)
        F, pp = _fixed_area_best(nmax, M, A, theta, det, warms, floor, seed, ncut)
        if F >= floor:
            best, stall = (A, pp), 0
            if verbose:
                print(f"    [A] g*T = {A:8.4f}   F = {F:.10f}", flush=True)
            if ckpt_path:
                save_sequence(ckpt_path, pp, M, theta, ncut, det)
        else:
            step = 1 - (1 - step) / 2.0
            stall += 1
            if step > 0.99985:
                break
    return best


def reduce_M(params, M, ncut, theta, det, floor=None, seed=0, n_jitter=10,
             M_min=4, verbose=True, ckpt_path=None):
    """
    STAGE B -- shrink the pulse count at FIXED g*T.

    Repeatedly deletes the smallest-area segment and re-optimises at the same
    total area (the softmax redistributes the deleted area), keeping the step only
    if the true fidelity still clears floor.  Returns (M, params).
    """
    floor = FLOOR_DEFAULT if floor is None else floor
    nmax = target_nmax(ncut)
    rng = np.random.default_rng(seed)
    A = float(np.asarray(params)[:M].sum())
    cur_M, cur_p = M, np.asarray(params, float).copy()
    while cur_M - 1 >= M_min:
        Mt = cur_M - 1
        x = cur_p[:cur_M]
        ph = cur_p[cur_M:2 * cur_M]
        r = cur_p[2 * cur_M:3 * cur_M] if det else np.zeros(cur_M)
        keep = np.array([i for i in range(cur_M) if i != int(np.argmin(x))])
        base = np.concatenate([np.log(np.maximum(x[keep], 1e-9)), ph[keep]]
                              + ([r[keep]] if det else []))
        warms = [base]
        for _ in range(n_jitter):
            w = base.copy()
            w[:Mt] += rng.normal(0, 0.25, Mt)
            w[Mt:2 * Mt] += rng.normal(0, 0.3, Mt)
            if det:
                w[2 * Mt:] += rng.normal(0, 0.7, Mt)
            warms.append(w)
        F, pp = _fixed_area_best(nmax, Mt, A, theta, det, warms, floor, seed, ncut,
                                 cold=(0, 80, 200))
        if F < floor:
            if verbose:
                print(f"    [B] M = {Mt:3d}: best F = {F:.10f}  -> stop", flush=True)
            break
        if verbose:
            print(f"    [B] M = {Mt:3d}   F = {F:.10f}   (g*T = {A:.4f})", flush=True)
        cur_M, cur_p = Mt, pp
        if ckpt_path:
            save_sequence(ckpt_path, cur_p, cur_M, theta, ncut, det)
    return cur_M, cur_p


def optimise_alternating(params, M, ncut=8, theta=THETA, det=True, floor=None,
                         rounds=6, seed=0, out_path=None, verbose=True):
    """
    Alternate stage A (minimise g*T) and stage B (minimise pulse count M) until
    neither improves.  g*T is the primary objective, M the secondary one.

    Returns dict(params, M, gT, F).
    """
    floor = FLOOR_DEFAULT if floor is None else floor
    p, M = np.asarray(params, float).copy(), int(M)
    gf = lambda q, m: gate_fidelity_blocks(q, m, ncut=ncut, use_detuning=det,
                                           nlam=8001)
    hist = []
    for it in range(rounds):
        gT0, M0 = float(p[:M].sum()), M
        if verbose:
            print(f"  round {it+1}: start g*T = {gT0:.4f}, M = {M0}, "
                  f"F = {gf(p, M):.10f}", flush=True)
        gT, p = descend_gT(p, M, ncut, theta, det, floor=floor,
                           seed=seed + 11 * it, verbose=verbose,
                           ckpt_path=out_path)
        M, p = reduce_M(p, M, ncut, theta, det, floor=floor,
                        seed=seed + 101 * it, verbose=verbose,
                        ckpt_path=out_path)
        hist.append((float(p[:M].sum()), M, gf(p, M)))
        if out_path:
            save_sequence(out_path, p, M, theta, ncut, det)
        improved = (p[:M].sum() < gT0 - 1e-9) or (M < M0)
        if not improved:
            if verbose:
                print("  no improvement this round -- converged", flush=True)
            break
    return dict(params=p, M=M, gT=float(p[:M].sum()), F=gf(p, M), history=hist)
