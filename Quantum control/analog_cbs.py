"""
Analog transmon-controlled beamsplitter / SWAP from TWO simultaneous sidebands.
QuTiP throughout.  Kept deliberately simple: short functions, one idea each.

Tensor order everywhere:   cavity1  (x)  cavity2  (x)  transmon

The transmon is a Kerr oscillator (g, e, f, ... all present), so the sideband
couples to its lowering operator q and drives every rung at once.

Physics summary
---------------
Rotating frame (exists as a single static frame only if the two sideband
detunings are equal, d1 = d2 = Delta):

    H = Delta*nq + (alpha/2)*nq*(nq-1)
        + g1*(e^{i phi1} a1.dag() q + h.c.)
        + g2*(e^{i phi2} a2.dag() q + h.c.)

Second order, two-level transmon:

    H_eff = (sigma_z/Delta) * B.dag()*B,     B = g1*a1 + g2*e^{-i dphi}*a2

For g1 = g2 = g and dphi = pi this is (G^2/Delta)*sigma_z*n_minus with
G = sqrt(2)*g and a_minus = (a1-a2)/sqrt(2).  Since SWAP = exp(i pi n_minus),
exponentiating it gives the controlled SWAP.

Kerr ladder:   chi_k = g^2 * [ (k+1)/(-(Delta+alpha*k)) + k/(Delta+alpha*(k-1)) ]
               dchi  = chi_e - chi_g = 2*g^2*alpha / (Delta*(Delta+alpha))
Gate time:     T = pi / (2*dchi)
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import qutip as qt
from scipy.optimize import minimize

__all__ = [
    "make_ops", "rotating_frame_H", "sw_second_order", "fit_level_coefficients",
    "conditional_rate", "dchi_closed_form", "chi_closed_form",
    "swap_operator", "ideal_controlled_swap", "gate_time", "cswap_fidelity",
]


# ---------------------------------------------------------------- operators
def make_ops(Nc, Nq):
    """All the operators we need, as QuTiP Qobjs.  Access them as ops.a1 etc."""
    Ic, Iq = qt.qeye(Nc), qt.qeye(Nq)

    a1 = qt.tensor(qt.destroy(Nc), Ic, Iq)
    a2 = qt.tensor(Ic, qt.destroy(Nc), Iq)
    q = qt.tensor(Ic, Ic, qt.destroy(Nq))

    a_minus = (a1 - a2) / np.sqrt(2)
    a_plus = (a1 + a2) / np.sqrt(2)

    return SimpleNamespace(
        Nc=Nc, Nq=Nq,
        a1=a1, a2=a2, q=q,
        n1=a1.dag() * a1, n2=a2.dag() * a2, nq=q.dag() * q,
        N=a1.dag() * a1 + a2.dag() * a2,
        a_minus=a_minus, n_minus=a_minus.dag() * a_minus,
        a_plus=a_plus, n_plus=a_plus.dag() * a_plus,
        # transmon level projectors, P[k] = |k><k|
        P=[qt.tensor(Ic, Ic, qt.basis(Nq, k) * qt.basis(Nq, k).dag())
           for k in range(Nq)],
    )


def ket(ops, n1, n2, k):
    """The basis ket |n1, n2, k>."""
    return qt.tensor(qt.basis(ops.Nc, n1), qt.basis(ops.Nc, n2),
                     qt.basis(ops.Nq, k))


def flat_index(ops, n1, n2, k):
    """Position of |n1, n2, k> in the flattened basis."""
    return (n1 * ops.Nc + n2) * ops.Nq + k


def transmon_levels(ops):
    """Transmon level k of every flattened basis state."""
    return np.array([k for _ in range(ops.Nc) for _ in range(ops.Nc)
                     for k in range(ops.Nq)])


# ---------------------------------------------------------------- Hamiltonian
def rotating_frame_H(ops, g1, g2, Delta, alpha, dphi=np.pi, phi1=0.0):
    """The static rotating-frame Hamiltonian (see module docstring)."""
    H_transmon = Delta * ops.nq + 0.5 * alpha * ops.nq * (ops.nq - 1)

    exchange1 = np.exp(1j * phi1) * ops.a1.dag() * ops.q
    exchange2 = np.exp(1j * (phi1 + dphi)) * ops.a2.dag() * ops.q

    H_sb1 = g1 * (exchange1 + exchange1.dag())
    H_sb2 = g2 * (exchange2 + exchange2.dag())

    return H_transmon + H_sb1 + H_sb2


# ---------------------------------------------------------------- second order
def sw_second_order(H0, V, levels):
    """
    Second-order effective Hamiltonian.

    Solve [S, H0] = -V, which for diagonal H0 is S[i,j] = V[i,j]/(E_i - E_j),
    then H2 = (1/2)[S, V], keeping only the transmon-diagonal block.
    """
    E = np.real(np.diag(H0.full()))
    Varr = V.full()
    dE = E[:, None] - E[None, :]

    if np.any((np.abs(dE) < 1e-9) & (np.abs(Varr) > 1e-12)):
        raise RuntimeError("V connects degenerate states, so SW does not apply")

    with np.errstate(divide="ignore", invalid="ignore"):
        Sarr = np.where(np.abs(Varr) > 1e-14, Varr / dE, 0.0)
    Sarr[~np.isfinite(Sarr)] = 0.0

    S = qt.Qobj(Sarr, dims=H0.dims)
    H2 = 0.5 * (S * V - V * S)

    same_level = levels[:, None] == levels[None, :]
    return qt.Qobj(np.where(same_level, H2.full(), 0.0), dims=H0.dims)


def fit_level_coefficients(ops, g, Delta, alpha, dphi=np.pi, nmax=None):
    """
    For each transmon level k, fit H_eff restricted to that level onto

        c0 * 1   +   chi * (n1 + n2)   +   J * (e^{-i dphi} a1.dag() a2 + h.c.)

    chi == J means H_eff is proportional to n_minus, which is what makes the gate
    a SWAP.  Returns {k: dict(const, chi, J, resid)}.
    """
    if nmax is None:
        nmax = ops.Nc - 3

    H0 = Delta * ops.nq + 0.5 * alpha * ops.nq * (ops.nq - 1)
    ex1 = ops.a1.dag() * ops.q
    ex2 = np.exp(1j * dphi) * ops.a2.dag() * ops.q
    V = g * (ex1 + ex1.dag()) + g * (ex2 + ex2.dag())

    H2 = sw_second_order(H0, V, transmon_levels(ops))

    bs = np.exp(-1j * dphi) * ops.a1.dag() * ops.a2
    basis = [qt.qeye(H2.dims[0]), ops.n1 + ops.n2, bs + bs.dag()]

    H2a = H2.full()
    out = {}
    for k in range(ops.Nq):
        rows = [flat_index(ops, c1, c2, k)
                for c1 in range(ops.Nc) for c2 in range(ops.Nc)
                if c1 + c2 <= nmax]
        target = H2a[np.ix_(rows, rows)].ravel()
        design = np.stack([b.full()[np.ix_(rows, rows)].ravel() for b in basis],
                          axis=1)
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        out[k] = dict(const=complex(coef[0]), chi=complex(coef[1]),
                      J=complex(coef[2]),
                      resid=float(np.max(np.abs(design @ coef - target))))
    return out


def conditional_rate(g, Delta, alpha, Nc=6, Nq=4, dphi=np.pi):
    """(chi_e - chi_g, J_e - J_g) from the numerical second order."""
    coeffs = fit_level_coefficients(make_ops(Nc, Nq), g, Delta, alpha, dphi=dphi)
    dchi = float(np.real(coeffs[1]["chi"] - coeffs[0]["chi"]))
    dJ = float(np.real(coeffs[1]["J"] - coeffs[0]["J"]))
    return dchi, dJ


def chi_closed_form(k, g, Delta, alpha):
    """chi_k = g^2 [ (k+1)/(-(Delta+alpha k)) + k/(Delta+alpha(k-1)) ]."""
    up = (k + 1) / (-(Delta + alpha * k))
    down = k / (Delta + alpha * (k - 1)) if k >= 1 else 0.0
    return g ** 2 * (up + down)


def dchi_closed_form(g, Delta, alpha):
    """chi_e - chi_g = 2 g^2 alpha / (Delta (Delta + alpha))."""
    return 2 * g ** 2 * alpha / (Delta * (Delta + alpha))


# ---------------------------------------------------------------- the gate
def swap_operator(Nc):
    """SWAP on the two cavities, built by permuting basis labels."""
    M = np.zeros((Nc * Nc, Nc * Nc))
    for n1 in range(Nc):
        for n2 in range(Nc):
            M[n2 * Nc + n1, n1 * Nc + n2] = 1.0
    return qt.Qobj(M, dims=[[Nc, Nc], [Nc, Nc]])


def ideal_controlled_swap(ops, swap_level=0):
    """SWAP when the transmon is in swap_level, identity otherwise."""
    SWAP = swap_operator(ops.Nc)
    Pk = qt.basis(ops.Nq, swap_level) * qt.basis(ops.Nq, swap_level).dag()
    return (qt.tensor(SWAP, Pk)
            + qt.tensor(qt.qeye([ops.Nc, ops.Nc]), qt.qeye(ops.Nq) - Pk))


def gate_time(g, Delta, alpha=None):
    """
    Duration of a full controlled SWAP.
    alpha=None  -> two-level estimate  T = pi Delta / (4 g^2).
    alpha given -> Kerr result         T = pi / (2 dchi).
    """
    if alpha is None:
        return np.pi * Delta / (4 * g ** 2)
    return np.pi / (2 * dchi_closed_form(g, Delta, alpha))


def cswap_fidelity(ops, g, Delta, alpha, T, ncut=3, dphi=np.pi,
                   swap_level=0, idle_level=1, n_gauge=121):
    """
    Evolve for time T and compare with the ideal controlled SWAP on
    (Fock 0..ncut)^2 x {swap_level, idle_level}.  Leakage to other transmon
    levels counts as error.

    IMPORTANT: ncut is the physical Fock range the gate is scored on.  The
    simulation cutoff ops.Nc must be large enough that the dynamics of those
    states is not clipped: the effective beamsplitter can move every photon into
    one cavity, so use ops.Nc >= 2*ncut + 2.

    Free corrections allowed (same convention as gauge_ops=[n_c, n_q] elsewhere in
    this project): a virtual Z on each control level, and the passive cavity gauge
    exp(-i lam n_minus) exp(-i mu N) -- one unconditional beamsplitter plus a
    virtual Z per cavity.
    """
    if ops.Nc < 2 * ncut + 2:
        raise ValueError(f"ops.Nc={ops.Nc} too small to score Fock 0..{ncut}; "
                         f"need at least {2*ncut+2}")

    U = (-1j * rotating_frame_H(ops, g, g, Delta, alpha, dphi=dphi) * T).expm()
    U_target = ideal_controlled_swap(ops, swap_level)
    M = (U_target.dag() * U).full()

    # the flat indices we score, and which control branch each belongs to
    idx, branch, n_tot = [], [], []
    for level, b in ((swap_level, 0), (idle_level, 1)):
        for n1 in range(ncut + 1):
            for n2 in range(ncut + 1):
                idx.append(flat_index(ops, n1, n2, level))
                branch.append(b)
                n_tot.append(n1 + n2)
    idx = np.array(idx)
    branch = np.array(branch)
    n_tot = np.array(n_tot)
    n_states = len(idx)

    # Gauge: W(lam, mu) = exp(-i lam n_minus) exp(-i mu N).  We only need the
    # diagonal entries Tr(M W P_b) = sum_{c in b} (M W)[c, c], and because
    # exp(-i mu N) is diagonal with eigenvalue e^{-i mu (n1+n2)} on each state,
    #     (M W)[c, c] = e^{-i mu n_tot[c]} * (M exp(-i lam n_minus))[c, c]
    # Writing exp(-i lam n_minus) = Q diag(e^{-i lam nu}) Q^dag,
    #     (M exp(-i lam n_minus))[c, c] = sum_k w[c, k] e^{-i lam nu_k},
    #     w[c, k] = (M Q)[c, k] * conj(Q[c, k])
    # so each diagonal entry is a tiny sum -- no big matrix products per gauge point.
    nu, Q = np.linalg.eigh(ops.n_minus.full())
    MQ = M[idx, :] @ Q
    w = MQ * np.conj(Q[idx, :])                      # (n_states, dim)

    lams = np.linspace(-np.pi, np.pi, n_gauge)
    diag = np.exp(-1j * np.outer(lams, nu)) @ w.T    # (n_gauge, n_states)

    # group by (branch, total photon number) so the mu scan is a small sum
    n_values = np.unique(n_tot)
    grouped = np.zeros((2, len(lams), len(n_values)), dtype=complex)
    for b in (0, 1):
        for i, n in enumerate(n_values):
            sel = (branch == b) & (n_tot == n)
            grouped[b, :, i] = diag[:, sel].sum(axis=1)

    mus = np.linspace(-np.pi, np.pi, n_gauge)
    phase = np.exp(-1j * np.outer(mus, n_values))     # (n_gauge, n_values)
    amp = np.abs(np.einsum("bln,mn->blm", grouped, phase))
    fidelities = (amp[0] + amp[1]) ** 2 / n_states ** 2

    # The grid alone is not enough: a gauge angle off by half a grid spacing costs
    # ~(n_max * spacing)^2, and n_max here is 2*ncut, so refine continuously.
    il, im = np.unravel_index(int(np.argmax(fidelities)), fidelities.shape)

    def neg_fidelity(x):
        lam, mu = x
        d = np.exp(-1j * lam * nu) @ w.T              # (n_states,)
        total = 0.0
        for b in (0, 1):
            s_b = 0.0 + 0.0j
            for i, n in enumerate(n_values):
                sel = (branch == b) & (n_tot == n)
                s_b += np.exp(-1j * mu * n) * d[sel].sum()
            total += abs(s_b)
        return -(total ** 2) / n_states ** 2

    best = minimize(neg_fidelity, [lams[il], mus[im]], method="Nelder-Mead",
                    options=dict(xatol=1e-12, fatol=1e-16, maxiter=2000))
    return float(max(fidelities[il, im], -best.fun))
