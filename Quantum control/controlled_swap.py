"""
Transmon-controlled cavity-cavity SWAP (and controlled beamsplitter) built only
from 3-wave-mixing exchange couplings:

    * cavity-cavity beamsplitter        g_bs (a1^dag a2 e^{i phi} + h.c.)
    * transmon-cavity JC on ge or ef    g_j  (a_j^dag |g><e| e^{i phi} + h.c.)
      each cavity separately, or both simultaneously.

No static dispersive shift chi is assumed: the transmon is idle unless a pump
is on.

All gates are treated as ideal (exact exchange Hamiltonians, no decoherence, no
parasitic pump products).

-------------------------------------------------------------------------------
THE STRUCTURAL FACT THIS MODULE IS BUILT ON
-------------------------------------------------------------------------------
Write the two supermodes

    a_+ = (a1 + a2)/sqrt(2)        a_- = (a1 - a2)/sqrt(2)

1)  SWAP_12 = exp(+ i pi n_-)  EXACTLY  (n_- = a_-^dag a_-).
    A SWAP is nothing but a pi phase per photon in the antisymmetric supermode.

2)  Driving BOTH transmon-cavity JC couplings simultaneously with equal
    amplitude g and relative phase phi is EXACTLY a single-mode JC on the
    bright supermode, with coupling sqrt(2) g; the orthogonal supermode is
    untouched to all orders.  Relative phase pi makes a_- the bright mode.

    ==> transmon-controlled SWAP  ==  transmon-controlled PARITY of a_-
        realised with ONE simultaneous, pi-out-of-phase, double JC drive.

    No beamsplitter sandwich is needed, and the three-body design problem
    collapses to a single-mode conditional-phase problem with 2x2 blocks.

3)  The Fock range that must be controlled is the TOTAL photon number
    n_- <= n1 + n2.  For cavities holding Fock 0..8 that is 17 levels, not 9.
    This is decomposition-independent: the BS - conditional-parity - BS sandwich
    needs exactly the same range, because after a 50:50 BS one cavity can hold
    all the photons.

-------------------------------------------------------------------------------
THE GATE, CONCRETELY  --  ge ONLY, CONTROL QUBIT = TRANSMON g-e
-------------------------------------------------------------------------------
PRIMARY DRIVE.  Two pumps on at the same time, both activating the ge
transmon-cavity exchange:

    H_j = Delta_j |e><e|
          + g ( e^{i phi_j}      a1^dag |g><e|  + h.c. )
          + g ( e^{i(phi_j + pi)} a2^dag |g><e| + h.c. )

i.e. equal amplitude g on both cavities and a relative phase of exactly pi.
That is identically a single-mode JC on a_- with coupling G = sqrt(2) g:

    H_j = Delta_j |e><e| + G ( e^{i phi_j} a_-^dag |g><e| + h.c. )

Segment j is specified by the pulse area x_j = G t_j, the common drive phase
phi_j, and the detuning r_j = Delta_j / G.  The f level is never used.

CONTROL QUBIT.  The ordinary transmon g-e qubit.  Both control states sit in the
SAME driven 2x2 block {|n,g>, |n-1,e>}, so there is no spectator level; instead,
demanding |<n,g|U|n,g>| = 1 forces every block to be diagonal, and then
unitarity fixes the e-branch phase from the g-branch phase.  Targeting

    <n,g|U|n,g> = exp(-i pi n / 2)        (a "quarter parity", theta = -pi/2)

gives g: exp(-i pi n/2) and e: exp(+i pi n/2) automatically, i.e.

    U = exp(-i pi n_- / 2)  x  (controlled-SWAP)

so the gate is finished by ONE unconditional 50:50 cavity-cavity beamsplitter.
Whether |g> or |e> is the branch that swaps is set by that final beamsplitter.
"""

from __future__ import annotations

import numpy as np
import qutip as qt
from scipy.optimize import minimize, minimize_scalar

__all__ = [
    "swap_operator", "supermode_ops", "check_swap_is_dark_parity",
    "check_supermode_reduction", "ideal_conditional_parity_fidelity",
    "ParityTrain", "FixedArea", "minimize_area",
    "full_cswap_fidelity", "dispersive_single_pulse", "THETA_GE",
]

# target slope for the ge-only scheme (see module docstring)
THETA_GE = -np.pi / 2


# =============================================================================
# 1.  structural identities
# =============================================================================
def swap_operator(Nc: int) -> qt.Qobj:
    """Exact SWAP on two Nc-level cavities (basis permutation)."""
    S = np.zeros((Nc * Nc, Nc * Nc))
    for n1 in range(Nc):
        for n2 in range(Nc):
            S[n2 * Nc + n1, n1 * Nc + n2] = 1.0
    return qt.Qobj(S, dims=[[Nc, Nc], [Nc, Nc]])


def supermode_ops(Nc: int, Nq: int = 3, rel_phase: float = np.pi):
    """a1, a2 and the bright/dark supermodes for a given relative drive phase."""
    Ic, Iq = qt.qeye(Nc), qt.qeye(Nq)
    a1 = qt.tensor(qt.destroy(Nc), Ic, Iq)
    a2 = qt.tensor(Ic, qt.destroy(Nc), Iq)
    bright = (a1 + np.exp(1j * rel_phase) * a2) / np.sqrt(2)
    dark = (np.exp(-1j * rel_phase) * a1 - a2) / np.sqrt(2)
    return a1, a2, bright, dark


def check_swap_is_dark_parity(Nc: int = 16, Nmax: int = 14) -> float:
    """max |exp(i pi n_-) - SWAP| on the n1+n2 <= Nmax block (should be ~1e-15)."""
    a1 = qt.tensor(qt.destroy(Nc), qt.qeye(Nc))
    a2 = qt.tensor(qt.qeye(Nc), qt.destroy(Nc))
    am = (a1 - a2) / np.sqrt(2)
    U = (1j * np.pi * am.dag() * am).expm()
    S = swap_operator(Nc)
    keep = [n1 * Nc + n2 for n1 in range(Nc) for n2 in range(Nc) if n1 + n2 <= Nmax]
    return float(np.max(np.abs(U.full()[np.ix_(keep, keep)]
                               - S.full()[np.ix_(keep, keep)])))


def check_supermode_reduction(Nc: int = 14, g: float = 1.0, Delta: float = 0.83,
                              T: float = 2.9, rel_phase: float = np.pi,
                              Ntest: int = 8) -> float:
    """
    max |U^dag a_dark U - a_dark| on the low-photon block, for the simultaneous
    double JC drive.  Zero <=> the orthogonal supermode is exactly decoupled.
    """
    Nq = 3
    a1, a2, _, dark = supermode_ops(Nc, Nq, rel_phase)
    Ic, Iq = qt.qeye(Nc), qt.qeye(Nq)
    sm = qt.tensor(Ic, Ic, qt.basis(Nq, 0) * qt.basis(Nq, 1).dag())
    Pe = qt.tensor(Ic, Ic, qt.basis(Nq, 1) * qt.basis(Nq, 1).dag())
    B = a1.dag() + np.exp(1j * rel_phase) * a2.dag()
    H = Delta * Pe + g * (B * sm + (B * sm).dag())
    U = (-1j * H * T).expm()
    lo = [i for i, (n1, n2, q) in enumerate(
        [(x, y, z) for x in range(Nc) for y in range(Nc) for z in range(Nq)])
        if n1 + n2 + (1 if q else 0) <= Ntest]
    D = (U.dag() * dark * U - dark).full()[np.ix_(lo, lo)]
    return float(np.max(np.abs(D)))


def ideal_conditional_parity_fidelity(Nc: int = 16, ncut: int = 7,
                                      ctrl=(0, 2)) -> float:
    """
    Fidelity of  exp(i pi n_- P_ctrl0)  against the ideal controlled-SWAP on
    cavities 0..ncut.  Should be exactly 1.
    """
    Nq = 3
    a1, a2, _, _ = supermode_ops(Nc, Nq)
    am = (a1 - a2) / np.sqrt(2)
    P0 = qt.tensor(qt.qeye(Nc), qt.qeye(Nc),
                   qt.basis(Nq, ctrl[0]) * qt.basis(Nq, ctrl[0]).dag())
    V = (1j * np.pi * am.dag() * am * P0).expm()

    S = swap_operator(Nc)
    Iq = qt.qeye(Nq)
    p0 = qt.basis(Nq, ctrl[0]) * qt.basis(Nq, ctrl[0]).dag()
    target = qt.tensor(S, p0) + qt.tensor(qt.qeye([Nc, Nc]), Iq - p0)
    kets = [qt.tensor(qt.fock(Nc, n1), qt.fock(Nc, n2), qt.fock(Nq, q))
            for n1 in range(ncut + 1) for n2 in range(ncut + 1) for q in ctrl]
    P = sum(k * k.dag() for k in kets)
    d = float(np.real(P.tr()))
    return float(abs((target.dag() * V * P).tr()) ** 2 / d ** 2)


# =============================================================================
# 2.  the reduced design problem: conditional phase profile on ONE mode
# =============================================================================
def _seg2(x, phi, r, sqn, want_grad=False):
    """
    exp(-i H t) for the 2x2 block {|n,g>, |n-1,e>} of one JC segment, vectorised
    over sqn = sqrt(n).  x = G t (pulse area), r = Delta / G (detuning).
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    vx, vy = x * sqn * cphi, -x * sqn * sphi
    vz = np.full_like(sqn, -r * x / 2.0)
    a0 = r * x / 2.0
    w = np.sqrt(vx * vx + vy * vy + vz * vz)
    small = w < 1e-12
    ws = np.where(small, 1.0, w)
    C = np.cos(w)
    S = np.where(small, 1.0 - w ** 2 / 6.0, np.sin(ws) / ws)
    pref = np.exp(-1j * a0)

    U = np.empty((sqn.size, 2, 2), dtype=complex)
    U[:, 0, 0] = C - 1j * S * vz
    U[:, 1, 1] = C + 1j * S * vz
    U[:, 0, 1] = -1j * S * (vx - 1j * vy)
    U[:, 1, 0] = -1j * S * (vx + 1j * vy)
    U *= pref
    if not want_grad:
        return U

    dS_dw = np.where(small, -w / 3.0, (C - S) / ws)
    zero = np.zeros_like(sqn)
    grads = []
    for da0, dvx, dvy, dvz in (
            (r / 2.0, sqn * cphi, -sqn * sphi, np.full_like(sqn, -r / 2.0)),
            (0.0, -x * sqn * sphi, -x * sqn * cphi, zero),
            (x / 2.0, zero, zero, np.full_like(sqn, -x / 2.0))):
        dw = np.where(small, 0.0, (vx * dvx + vy * dvy + vz * dvz) / ws)
        dC, dS = -np.sin(w) * dw, dS_dw * dw
        dU = np.empty_like(U)
        dU[:, 0, 0] = dC - 1j * (dS * vz + S * dvz)
        dU[:, 1, 1] = dC + 1j * (dS * vz + S * dvz)
        dU[:, 0, 1] = -1j * (dS * (vx - 1j * vy) + S * (dvx - 1j * dvy))
        dU[:, 1, 0] = -1j * (dS * (vx + 1j * vy) + S * (dvx + 1j * dvy))
        dU *= pref
        dU += (-1j * da0) * U
        grads.append(dU)
    return U, grads


class ParityTrain:
    """
    Idealised design problem (idle branch an exact spectator).

    Find a train of M JC segments on the bright supermode such that

        <n, active| U |n, active> = exp(i theta n)   for n = 0..nmax

    theta = pi  ->  controlled-SWAP;  general theta -> controlled-BS.

    Parameter vector p = [x_0..x_{M-1}, phi_0.., r_0..] where x = G t is the
    pulse area, phi the drive phase, r = Delta/G the detuning (omit the r block
    with use_detuning=False).  Total gate time is  sum_j x_j / G  with
    G = sqrt(2) * g_single.
    """

    def __init__(self, nmax: int, M: int, theta: float = np.pi,
                 use_detuning: bool = True):
        self.nmax, self.M, self.theta = nmax, M, theta
        self.use_detuning = use_detuning
        self.sqn = np.sqrt(np.arange(1, nmax + 1, dtype=float))
        self.z = np.exp(-1j * theta * np.arange(1, nmax + 1))
        self.d = nmax + 1
        self.npar = (3 if use_detuning else 2) * M

    def _unpack(self, p):
        M = self.M
        r = p[2 * M:3 * M] if self.use_detuning else np.zeros(M)
        return p[:M], p[M:2 * M], r

    def amplitudes(self, p):
        """u_n = <n, active| U |n, active>, n = 0..nmax."""
        x, phi, r = self._unpack(p)
        tot = np.broadcast_to(np.eye(2, dtype=complex), (self.sqn.size, 2, 2)).copy()
        for j in range(self.M):
            tot = _seg2(x[j], phi[j], r[j], self.sqn) @ tot
        u = np.ones(self.nmax + 1, dtype=complex)
        u[1:] = tot[:, 0, 0]
        return u

    def fidelity(self, p):
        u = self.amplitudes(p)
        return abs(1.0 + np.sum(self.z * u[1:])) ** 2 / self.d ** 2

    def gate_fidelity(self, p):
        """Joint (2 control states) x (nmax+1 Fock) projected gate fidelity."""
        return (np.sqrt(self.fidelity(p)) + 1.0) ** 2 / 4.0

    def cost_and_grad(self, p):
        M, nb = self.M, self.sqn.size
        x, phi, r = self._unpack(p)
        Us, dUs = [], []
        for j in range(M):
            U, gl = _seg2(x[j], phi[j], r[j], self.sqn, want_grad=True)
            Us.append(U); dUs.append(gl)
        I2 = np.broadcast_to(np.eye(2, dtype=complex), (nb, 2, 2)).copy()
        R = [I2]
        for j in range(M):
            R.append(Us[j] @ R[-1])
        L = [None] * (M + 1)
        L[M] = I2
        for j in range(M - 1, -1, -1):
            L[j] = L[j + 1] @ Us[j]
        A = 1.0 + np.sum(self.z * R[M][:, 0, 0])
        F = abs(A) ** 2 / self.d ** 2
        npp = 3 if self.use_detuning else 2
        g = np.zeros(self.npar)
        for j in range(M):
            for k in range(npp):
                dA = np.sum(self.z * (L[j + 1] @ dUs[j][k] @ R[j])[:, 0, 0])
                g[k * M + j] = 2.0 * np.real(np.conj(A) * dA) / self.d ** 2
        return 1.0 - F, -g

    def solve(self, n_starts: int = 60, seed: int = 0, maxiter: int = 1000,
              x_max: float = np.pi, r_max: float = 6.0, p0_list=None):
        rng = np.random.default_rng(seed)
        M = self.M
        bounds = [(0.0, x_max)] * M + [(-2 * np.pi, 2 * np.pi)] * M
        if self.use_detuning:
            bounds += [(-r_max, r_max)] * M
        best = (0.0, None)
        starts = list(p0_list) if p0_list else []
        starts += [np.concatenate(
            [rng.uniform(0.05, x_max, M), rng.uniform(-np.pi, np.pi, M)] +
            ([rng.uniform(-2.5, 2.5, M)] if self.use_detuning else []))
            for _ in range(n_starts)]
        for p0 in starts:
            res = minimize(self.cost_and_grad, p0, jac=True, method="L-BFGS-B",
                           bounds=bounds, options=dict(maxiter=maxiter,
                                                       ftol=1e-18, gtol=1e-14))
            F = self.fidelity(res.x)
            if F > best[0]:
                best = (F, res.x.copy())
            if best[0] > 1 - 1e-13:
                break
        return best


class FixedArea:
    """
    Same design problem as ParityTrain but with the TOTAL pulse area
    A = sum_j G t_j = G * T_gate  held fixed (softmax parameterisation), so the
    shortest gate can be found by walking A down.
    """

    def __init__(self, nmax, M, A, theta=np.pi, use_detuning=True):
        self.pt = ParityTrain(nmax, M, theta=theta, use_detuning=use_detuning)
        self.M, self.A, self.use_detuning = M, A, use_detuning

    def expand(self, q):
        M = self.M
        v = q[:M] - np.max(q[:M])
        w = np.exp(v)
        x = self.A * w / w.sum()
        return np.concatenate([x, q[M:]]), x

    def cost(self, q):
        M, A = self.M, self.A
        p, x = self.expand(q)
        c, g = self.pt.cost_and_grad(p)
        gx = g[:M]
        return c, np.concatenate([x * (gx - np.dot(gx, x) / A), g[M:]])

    def fidelity(self, q):
        return self.pt.fidelity(self.expand(q)[0])

    def solve(self, n_starts=40, seed=0, maxiter=900, r_max=8.0, warm=None):
        rng = np.random.default_rng(seed)
        M = self.M
        bounds = [(-8, 8)] * M + [(-2 * np.pi, 2 * np.pi)] * M
        if self.use_detuning:
            bounds += [(-r_max, r_max)] * M
        best = (0.0, None)
        starts = ([] if warm is None else [np.asarray(warm, float)])
        starts += [np.concatenate(
            [rng.normal(0, 0.7, M), rng.uniform(-np.pi, np.pi, M)] +
            ([rng.uniform(-2.5, 2.5, M)] if self.use_detuning else []))
            for _ in range(n_starts)]
        for q0 in starts:
            res = minimize(self.cost, q0, jac=True, method="L-BFGS-B",
                           bounds=bounds, options=dict(maxiter=maxiter,
                                                       ftol=1e-18, gtol=1e-14))
            F = self.fidelity(res.x)
            if F > best[0]:
                best = (F, res.x.copy())
            if best[0] > 1 - 1e-13:
                break
        return best


def minimize_area(nmax, M, use_detuning=True, theta=np.pi, A_start=None,
                  F_floor=1 - 1e-9, step=0.97, n_starts_first=60,
                  n_starts_retry=8, seed=0, verbose=False):
    """
    Shortest pulse train we can find at fixed segment count M: converge at a
    generous total area, then walk the area down, warm-starting each step.
    Returns (area, params) with params in ParityTrain layout, or None.
    """
    mk = lambda A: FixedArea(nmax, M, A, theta=theta, use_detuning=use_detuning)
    if A_start is None:
        A_start = 3.4 * (nmax + 1) + 8.0
    fa = mk(A_start)
    F, q = fa.solve(n_starts=n_starts_first, seed=seed)
    tries = 0
    while F < F_floor and tries < 5:
        A_start *= 1.3
        fa = mk(A_start)
        F, q = fa.solve(n_starts=n_starts_first, seed=seed + 13 * tries, warm=q)
        tries += 1
    if F < F_floor:
        return None
    bestA, bestq, stall = A_start, q, 0
    while stall < 4 and bestA > np.pi:
        A_new = bestA * step
        fa = mk(A_new)
        F, qn = fa.solve(n_starts=0, seed=seed, warm=bestq, maxiter=1500)
        if F < F_floor:
            F, qn = fa.solve(n_starts=n_starts_retry, seed=seed + 101,
                             warm=bestq, maxiter=1500)
        if F >= F_floor:
            bestA, bestq, stall = A_new, qn, 0
            if verbose:
                print(f"    area {A_new:7.3f}  F={F:.12f}", flush=True)
        else:
            step = 1 - (1 - step) / 2.0
            stall += 1
            if step > 0.9995:
                break
    return bestA, mk(bestA).expand(bestq)[0]


# =============================================================================
# 3.  end-to-end check in the full cavity x cavity x transmon space
# =============================================================================
def _dense_ops(Nc, Nq=3):
    a = np.diag(np.sqrt(np.arange(1, Nc)), 1)
    Ic, Iq = np.eye(Nc), np.eye(Nq)
    a1 = np.kron(np.kron(a, Ic), Iq)
    a2 = np.kron(np.kron(Ic, a), Iq)

    def lvl(i, j):
        m = np.zeros((Nq, Nq)); m[i, j] = 1
        return np.kron(np.kron(Ic, Ic), m)
    return a1, a2, lvl


def full_cswap_fidelity(params, M, *, use_detuning=False, G=1.0, Nc=16, Nq=3,
                        ncut=7, ctrl=(0, 2), primary="ge",
                        rel_phase=np.pi, return_amps=False):
    """
    Propagate the pulse train through the REAL two-cavity + 3-level transmon
    Hamiltonian (no supermode assumption) and score it against the ideal
    transmon-controlled SWAP on cavities 0..ncut, control states `ctrl`
    (ctrl[0] = swap branch, ctrl[1] = idle branch).

    ctrl = (swap branch, idle branch) as transmon level indices.  For the ge-only
    scheme use ctrl=(0, 1) or (1, 0); the final unconditional beamsplitter (which
    the free gauge below stands in for) decides which of the two it is.
    primary: "ge" or "ef" -- which exchange the pump makes resonant.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply

    x = np.asarray(params[:M], float)
    phi = np.asarray(params[M:2 * M], float)
    r = np.asarray(params[2 * M:3 * M], float) if use_detuning else np.zeros(M)

    a1, a2, lvl = _dense_ops(Nc, Nq)
    ge, ef = lvl(0, 1), lvl(1, 2)
    g0 = G / np.sqrt(2)                      # so the supermode coupling is G
    c1 = g0
    c2 = g0 * np.exp(1j * rel_phase)         # relative phase pi -> a_- is bright

    ins, outs, branch = [], [], []
    for na in range(ncut + 1):
        for nb in range(ncut + 1):
            for k, q in enumerate(ctrl):
                ins.append((na * Nc + nb) * Nq + q)
                outs.append(((nb * Nc + na) if k == 0 else (na * Nc + nb)) * Nq + q)
                branch.append(k)
    branch = np.array(branch)
    V = np.zeros((Nc * Nc * Nq, len(ins)), dtype=complex)
    for c, i in enumerate(ins):
        V[i, c] = 1.0

    for j in range(M):
        B = np.exp(1j * phi[j]) * (c1 * a1.conj().T + c2 * a2.conj().T)
        Hp = (B @ ge) if primary == "ge" else (B @ ef)
        E1 = r[j] * G if primary == "ge" else (r[j] * G - 0.0)
        H = Hp + Hp.conj().T + E1 * lvl(1, 1)
        V = expm_multiply(sp.csr_matrix(-1j * H * (x[j] / G)), V)

    F, lam = _gauge_corrected_fidelity(V, outs, branch, Nc, Nq)
    if return_amps:
        return F, np.array([V[outs[c], c] for c in range(len(ins))]), V, lam
    return F


def _gauge_corrected_fidelity(V, outs, branch, Nc, Nq, nlam=241):
    """
    Projected gate fidelity allowing exactly the corrections that are free after
    the gate:
      * an independent virtual-Z phase on each control state,
      * a general passive cavity gauge  exp(-i lam n_-) exp(-i mu N)
        (a cavity-cavity beamsplitter plus a virtual Z on each cavity).
    Same spirit as the gauge_ops=[n_c, n_q] freedom used in parameter_oc.
    """
    a = np.diag(np.sqrt(np.arange(1, Nc)), 1)
    Ic = np.eye(Nc)
    am = (np.kron(a, Ic) - np.kron(Ic, a)) / np.sqrt(2)
    nu, Q = np.linalg.eigh(am.conj().T @ am)          # n_- = Q diag(nu) Q^dag
    Ntot = np.add.outer(np.arange(Nc), np.arange(Nc)).ravel()   # N, Fock-diagonal

    ncol = V.shape[1]
    Vc = V.reshape(Nc * Nc, Nq, ncol)
    Vt = np.tensordot(Q.conj().T, Vc, axes=(1, 0))    # (cav_eig, Nq, ncol)

    o = np.array(outs)
    o_cav, o_q = o // Nq, o % Nq
    # <out_c| exp(-i lam n_-) |V_c> = sum_k g[k, c] exp(-i lam nu_k)
    g = Q[o_cav, :].T * Vt[:, o_q, np.arange(ncol)]    # (cav_eig, ncol)
    Nout = Ntot[o_cav].astype(int)                     # total photon number, conserved

    # group columns by (control branch, N) so exp(-i mu N) is a tiny DFT
    Nvals = np.unique(Nout)
    sel = np.stack([[(branch == k) & (Nout == N) for N in Nvals] for k in (0, 1)])

    lams = np.linspace(-np.pi, np.pi, nlam)
    mus = np.linspace(-np.pi, np.pi, nlam)
    S = np.exp(-1j * np.outer(lams, nu)) @ g           # (nlam, ncol)
    SB = np.stack([np.stack([S[:, sel[k, i]].sum(axis=1)
                             for i in range(len(Nvals))], axis=1)
                   for k in (0, 1)])                   # (2, nlam, nN)
    Em = np.exp(-1j * np.outer(mus, Nvals))            # (nmu, nN)
    A = np.abs(np.einsum('klN,mN->klm', SB, Em))       # (2, nlam, nmu)
    vals = (A[0] + A[1]) ** 2 / ncol ** 2
    il, im = np.unravel_index(int(np.argmax(vals)), vals.shape)

    def negF(z):
        t = (np.exp(-1j * z[0] * nu) @ g) * np.exp(-1j * z[1] * Nout)
        return -((abs(t[branch == 0].sum()) + abs(t[branch == 1].sum())) ** 2) / ncol ** 2

    res = minimize(negF, [lams[il], mus[im]], method="Nelder-Mead",
                   options=dict(xatol=1e-13, fatol=1e-17, maxiter=4000))
    if -res.fun >= vals[il, im]:
        return float(-res.fun), (float(res.x[0]), float(res.x[1]))
    return float(vals[il, im]), (float(lams[il]), float(mus[im]))


# =============================================================================
# 4.  the calibration-free alternative
# =============================================================================
def dispersive_single_pulse(r, nmax, optimise_duration=True):
    """
    The no-optimal-control option: ONE detuned simultaneous double-JC pulse,
    Delta = r*G, duration ~ pi*Delta/G^2.  Returns (branch fidelity, G*T).
    """
    sqn = np.sqrt(np.arange(1, nmax + 1, dtype=float))
    zz = np.exp(-1j * np.pi * np.arange(nmax + 1))

    def F_of(x):
        U = _seg2(x, 0.0, r, sqn)
        u = np.ones(nmax + 1, dtype=complex)
        u[1:] = U[:, 0, 0]
        return abs(np.sum(zz * u)) ** 2 / (nmax + 1) ** 2

    x0 = np.pi * r
    if not optimise_duration:
        return F_of(x0), x0
    res = minimize(lambda z: 1 - F_of(z[0]), [x0], method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
    return 1 - res.fun, float(res.x[0])
