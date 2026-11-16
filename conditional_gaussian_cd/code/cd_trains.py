"""Conditional squeezing and conditional rotation from a train of 30 ns qubit pi pulses under a continuous sideband.

Used by conditional_gaussian_gates.ipynb and optimise_trains.py. Units: microseconds and radians.

PHYSICAL MODEL (coherent, two-level qubit, no loss, no static dispersive shift)
    H(t) = g_pump(t) (e^{i psi} sigma+ a + h.c.)                 resonant sideband ("pump"), on during the whole train
         + Omega(t) (e^{i phi} sigma+ + h.c.)                    qubit drive: 30 ns Hann pi pulses
Both drives are played as DAC samples, 1.25 ns each (800 MS/s), and held constant within a sample, so each sample's
propagator is an exact matrix exponential. A pi pulse is 24 samples, Omega_j = (pi/30 ns) sin^2(pi j / 24),
whose area makes exactly exp(-i (pi/2) sigma_x). The sideband coupling g = pi / (2 t_swap) (full swap time t_swap)
ramps up over 4 samples at the start and down over 4 samples at the end (phase 0).

A TRAIN is N cells. Cell j = [gap/2 of sideband only] [pi pulse at qubit phase phi_j] [gap/2], with the sideband at
phase psi_j throughout the cell. Gate time = N (30 ns + gap) + 10 ns (ramps).

WHY ONE MATRIX IS ENOUGH: changing the phases of a cell is a conjugation by the diagonal D = exp(i (phi k - psi n)),
with n = photon number and k = n + |e><e| - 1/2. (The sideband conserves k, so phi only moves the qubit-drive phase;
psi only moves the sideband phase.) So every cell is D_j U0 D_j^dag with ONE precomputed matrix U0, and the gradient
of the gate with respect to every phase is exact and cheap.

TARGETS on the inputs (qubit g or e) x (Fock n = 0..n_max), n_max = 8 by default; sigma_z |g> = +|g>:
    squeeze   S_z(r) = |g><g| S(r) + |e><e| S(-r),   S(r) = exp[r/2 (a^2 - a^dag^2)],  8 dB: r = 8 ln(10) / 20
    rotation  exp(-i theta sigma_z n): the g and e branches rotate by -theta and +theta (relative 2 theta)
    cd        conditional displacement exp[beta sigma_x (a^dag - a)]: displaced by +beta / -beta for sigma_x = +1 / -1
Allowed free correction after the train (a frame update, no time): a qubit Rz, chosen optimally.

FIDELITY: F = |Tr(T^dag W)|^2 / d^2, T and W the target and simulated output columns of the d = 2 (n_max + 1) inputs
(all photon numbers of the output kept up to the cutoff). Phases between all inputs count.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import numpy as np
from scipy.linalg import expm

DT = 0.00125                         # us, one DAC sample (800 MS/s)
PI_SAMPLES = 24                      # 30 ns pi pulse
PI_TIME = PI_SAMPLES * DT
RAMP_SAMPLES = 4                     # sideband ramp up / down, 5 ns each
R_8DB = 8 * np.log(10) / 20          # 0.92103: 8 dB squeezing


def pi_samples():
    """Qubit drive amplitude of the 24 samples of one pi pulse (start-of-sample values of a Hann)."""
    return np.pi / PI_TIME * np.sin(np.pi * np.arange(PI_SAMPLES) / PI_SAMPLES) ** 2


# ------------------------------------------------------------------------------------------------------------------
# basis: (g0, e0, g1, e1, ...), index 2 n + (0 for g, 1 for e)
# ------------------------------------------------------------------------------------------------------------------
class Space:
    def __init__(self, levels, n_input=9):
        if not (isinstance(levels,(int,np.integer)) and 1 <= n_input <= levels):
            raise ValueError('Require 1 <= number of input Fock states <= output levels.')
        self.levels, self.n_input = levels, n_input
        self.n = np.repeat(np.arange(levels), 2).astype(float)          # photon number of each basis state
        self.e = np.tile([0.0, 1.0], levels)                             # 1 for the excited qubit
        self.z = 1 - 2 * self.e                                          # sigma_z: +1 for g
        self.k = self.n + self.e - 0.5                                   # conserved by the sideband
        self.inputs = [2 * m for m in range(n_input)] + [2 * m + 1 for m in range(n_input)]   # g0.., e0..

    def hamiltonian(self, pump, drive):
        """H at phase 0: pump * (sigma+ a + h.c.) + drive * sigma_x (real symmetric)."""
        size = 2 * self.levels
        H = np.zeros((size, size))
        for m in range(self.levels):
            H[2 * m, 2 * m + 1] = H[2 * m + 1, 2 * m] = drive                       # g_m <-> e_m
            if m + 1 < self.levels:
                H[2 * m + 1, 2 * m + 2] = H[2 * m + 2, 2 * m + 1] = pump * np.sqrt(m + 1)   # e_m <-> g_{m+1}
        return H


def cell_matrix(space, g, gap):
    """U0: one cell at phases 0 = half gap, the 24 pi samples (sideband on), half gap."""
    half = expm(-1j * DT * space.hamiltonian(g, 0.0)) if gap > 0 else None
    U = np.eye(2 * space.levels, dtype=complex)
    if gap > 0:
        U = np.linalg.matrix_power(half, int(round(gap / 2 / DT))) @ U
    for omega in pi_samples():
        U = expm(-1j * DT * space.hamiltonian(g, omega)) @ U
    if gap > 0:
        U = np.linalg.matrix_power(half, int(round(gap / 2 / DT))) @ U
    return U


def ramp_matrices(space, g):
    """Sideband ramp up (sin^2) and down (cos^2), 4 samples each, phase 0, no qubit drive."""
    up = np.eye(2 * space.levels, dtype=complex)
    down = np.eye(2 * space.levels, dtype=complex)
    for j in range(RAMP_SAMPLES):
        up = expm(-1j * DT * space.hamiltonian(g * np.sin(np.pi * j / 8) ** 2, 0.0)) @ up
        down = expm(-1j * DT * space.hamiltonian(g * np.cos(np.pi * j / 8) ** 2, 0.0)) @ down
    return up, down


def target_columns(space, gate, strength):
    """Target output columns (2 levels x 2 n_input) for the inputs g0.., e0.."""
    size, N_INPUT = 2 * space.levels, space.n_input
    T = np.zeros((size, 2 * N_INPUT), complex)
    if gate == "cd":
        a = np.diag(np.sqrt(np.arange(1, space.levels)), 1)
        D = expm(strength * (a.T - a))                                   # D(beta); sigma_x = +-1 eigenstates (g +- e)/sqrt2
        plus, minus = (D + D.T) / 2, (D - D.T) / 2                       # D(-beta) = D(beta)^T for real beta
        for m in range(N_INPUT):
            T[0::2, m], T[1::2, m] = plus[:, m], minus[:, m]             # input g_m
            T[0::2, N_INPUT + m], T[1::2, N_INPUT + m] = minus[:, m], plus[:, m]   # input e_m
        return T
    if gate == "squeeze":
        a = np.diag(np.sqrt(np.arange(1, space.levels)), 1)
        S = expm(strength / 2 * (a @ a - a.T @ a.T))                    # S(r); S(-r) = S(r)^T
        for m in range(N_INPUT):
            T[0::2, m] = S[:, m]                                         # g input -> g branch squeezed by S(r)
            T[1::2, N_INPUT + m] = S.T[:, m]                             # e input -> e branch squeezed by S(-r)
    elif gate == "rotation":
        for col, idx in enumerate(space.inputs):
            T[idx, col] = np.exp(-1j * strength * space.n[idx] * space.z[idx])
    else:
        raise ValueError(gate)
    return T


# ------------------------------------------------------------------------------------------------------------------
# the train
# ------------------------------------------------------------------------------------------------------------------
class Train:
    """A train of N cells with a given sideband swap time and gap; evaluates gate fidelity and its exact gradient.

    Controls x = (phi_1..phi_N, psi_1..psi_N), all in radians. A structured train uses x = A @ p (see structure())."""

    def __init__(self, gate, strength, swap, cells, gap=0.0, levels=128, n_input=9):
        if swap <= 0 or cells < 1 or int(cells) != cells or gap < 0:
            raise ValueError('Require positive swap/cell count and a nonnegative gap.')
        if abs(gap/(2*DT)-round(gap/(2*DT))) > 1e-9:
            raise ValueError('Each half gap must contain a whole number of DAC samples.')
        self.gate, self.strength, self.swap, self.cells, self.gap = gate, strength, swap, cells, gap
        self.g = np.pi / (2 * swap)
        self.space = Space(levels, n_input)
        self.d = 2 * n_input                                             # number of inputs
        self.U0 = cell_matrix(self.space, self.g, gap)
        self.up, self.down = ramp_matrices(self.space, self.g)
        self.target = target_columns(self.space, gate, strength)
        self.time = cells * (PI_TIME + gap) + 2 * RAMP_SAMPLES * DT
        self.inputs = np.eye(2 * levels, dtype=complex)[:, self.space.inputs]

    def run(self, x, keep_states=False):
        """Output columns before the qubit frame correction (and the intermediate states, for the gradient)."""
        phi, psi = x[:self.cells], x[self.cells:]
        k, n = self.space.k[:, None], self.space.n[:, None]
        state = self.up @ self.inputs
        states = [state] if keep_states else None
        for j in range(self.cells):
            D = np.exp(1j * (phi[j] * k - psi[j] * n))                   # phases of cell j as a diagonal conjugation
            state = D * (self.U0 @ (D.conj() * state))
            if keep_states:
                states.append(state)
        out = self.down @ state
        return out, states

    def qubit_frame(self, out):
        """Best qubit Rz after the gate: makes the overlaps of the output's g rows and e rows with the target have the
        same phase."""
        zg = np.vdot(self.target[0::2], out[0::2])
        ze = np.vdot(self.target[1::2], out[1::2])
        angle = np.angle(ze) - np.angle(zg)                               # actual correction is Rz(-angle)
        return np.exp(1j * angle * self.space.z / 2)[:, None], angle

    def infidelity(self, x, gradient=True):
        """1 - F and its gradient with respect to x (exact, by back-propagation through the cells)."""
        out, states = self.run(x, keep_states=gradient)
        frame, _ = self.qubit_frame(out)
        trace = np.vdot(self.target, frame * out)
        cost = 1 - abs(trace / self.d) ** 2
        if not gradient:
            return cost
        # back-propagated target: B_j = (frame * down * C_N ... C_{j+1})^dag T
        k, n = self.space.k[:, None], self.space.n[:, None]
        back = self.down.conj().T @ (np.conj(frame) * self.target)
        phi, psi = x[:self.cells], x[self.cells:]
        grad = np.zeros(2 * self.cells)
        for j in range(self.cells - 1, -1, -1):
            after = states[j + 1]                                         # C_j S_{j-1}
            before = states[j]                                            # S_{j-1}
            D = np.exp(1j * (phi[j] * k - psi[j] * n))
            # dC/dphi S = i (k C S - C k S),  dC/dpsi S = -i (n C S - C n S)
            Ck = D * (self.U0 @ (D.conj() * (k * before)))
            Cn = D * (self.U0 @ (D.conj() * (n * before)))
            d_phi = 1j * (k * after - Ck)
            d_psi = -1j * (n * after - Cn)
            grad[j] = -2 * np.real(np.conj(trace) * np.vdot(back, d_phi)) / self.d ** 2
            grad[self.cells + j] = -2 * np.real(np.conj(trace) * np.vdot(back, d_psi)) / self.d ** 2
            back = D * (self.U0.conj().T @ (D.conj() * back))                 # back through C_j
        return cost, grad

    def report(self, x):
        """Gate fidelity, worst single-input fidelity, qubit frame angle, time."""
        out, _ = self.run(x)
        frame, angle = self.qubit_frame(out)
        overlaps = np.sum(np.conj(self.target) * (frame * out), axis=0)
        return dict(fidelity=float(abs(overlaps.sum() / self.d) ** 2),
                    worst_input_fidelity=float(np.min(abs(overlaps) ** 2)),
                    qubit_frame_Rz=float(-angle),
                    time_us=float(self.time), cells=self.cells, swap_ns=1e3 * self.swap, gap_ns=1e3 * self.gap,
                    top_level_population=float(np.max(np.sum(abs(out[-8:]) ** 2, axis=0))))

    def unitary_on(self, psi, x):
        """Apply the whole gate (with its frame corrections) to any state psi (length 2 levels)."""
        out, _ = self.run(x)
        frame, _ = self.qubit_frame(out)
        phi, ph = x[:self.cells], x[self.cells:]
        k, n = self.space.k, self.space.n
        state = self.up @ psi
        for j in range(self.cells):
            D = np.exp(1j * (phi[j] * k - ph[j] * n))
            state = D * (self.U0 @ (np.conj(D) * state))
        return frame[:, 0] * (self.down @ state)


# ------------------------------------------------------------------------------------------------------------------
# structured trains: x = A p, with fewer parameters p
# ------------------------------------------------------------------------------------------------------------------
def structure(cells, kind, motif=None):
    """Matrix A (2N x P) mapping the free parameters p to the controls x = (phi, psi).
        "free"        every cell's qubit and pump phase free (P = 2N)
        "palindrome"  cell j and cell N+1-j have equal phase labels (P = N, N even);
                      exact sampled-Hamiltonian time symmetry is not assumed
        "motif"       a block of `motif` cells repeated N / motif times (P = 2 motif)
        "qubit only"  pump phase fixed at 0, qubit phases free (P = N)"""
    if kind == "free":
        return np.eye(2 * cells)
    rows = []
    if kind == "palindrome":
        assert cells % 2 == 0
        half = cells // 2
        A = np.zeros((2 * cells, 2 * half))
        for j in range(cells):
            src = j if j < half else cells - 1 - j
            A[j, src] = 1                                                # phi_j
            A[cells + j, half + src] = 1                                 # psi_j
        return A
    if kind == "motif":
        assert cells % motif == 0
        A = np.zeros((2 * cells, 2 * motif))
        for j in range(cells):
            A[j, j % motif] = 1
            A[cells + j, motif + j % motif] = 1
        return A
    if kind == "qubit only":
        A = np.zeros((2 * cells, cells))
        A[:cells, :cells] = np.eye(cells)
        return A
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------------------------
# the played waveform (for plotting): complex qubit and sideband samples vs time
# ------------------------------------------------------------------------------------------------------------------
def waveform(train, x):
    """Legacy cell waveform: time (us), qubit drive and sideband (rad/us).

    This design surrogate keeps the original phase-zero outer pump ramps.
    New constant-phase/10ns-shaped protocols explicitly construct their own
    outer ramps and transitions before SDK packing; see verify_short_gates.
    """
    phi, psi = x[:train.cells], x[train.cells:]
    half_gap = int(round(train.gap / 2 / DT))
    q, s = [], []
    for j in range(RAMP_SAMPLES):
        q.append(0j); s.append(train.g * np.sin(np.pi * j / 8) ** 2)
    for p1, p2 in zip(phi, psi):
        pump = train.g * np.exp(1j * p2)
        q += [0j] * half_gap; s += [pump] * half_gap
        q += list(pi_samples() * np.exp(1j * p1)); s += [pump] * PI_SAMPLES
        q += [0j] * half_gap; s += [pump] * half_gap
    for j in range(RAMP_SAMPLES):
        q.append(0j); s.append(train.g * np.cos(np.pi * j / 8) ** 2)
    t = DT * np.arange(len(q))
    return t, np.array(q), np.array(s)
