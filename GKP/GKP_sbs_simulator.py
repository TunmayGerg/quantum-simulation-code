import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.special import gammaln, eval_genlaguerre


class GKPsBsSimulator:
    """
    sBs stabilization + improved (S85) logical readout, written like GKPStabilizationSimulator (sharp/trim).
    It follows what the FPGA-loop runtimes play (msmt_workbench runtimes/gkp, GKP_PROTOCOL.md):

        vacuum -> sBs x n_rounds -> Z herald x n_heralds -> [magic injection, H only]
               -> sBs x n_rounds_after -> characteristic-function probe

    Conventions (same as GKPStabilizationSimulator):
        tensor order cavity (x) qubit; q = (a + a^dag)/sqrt2, p = i(a^dag - a)/sqrt2
        D(alpha) shifts q by sqrt2 Re(alpha) and p by sqrt2 Im(alpha)
        qubit |g> = basis(2,0), |e> = basis(2,1), sigma_z = |g><g| - |e><e|
        qubit rotation R(angle, phase) = exp(-i angle/2 (cos(phase) sigma_x + sin(phase) sigma_y))

    Gates (Royer, Singh, Girvin PRL 125, 260509, supplement Eqs. S67 and S85):
        small = sqrt(pi/8) sinh(Delta^2), big = sqrt(pi/2) cosh(Delta^2)   (one-branch displacements)
        sBs "x" round: exp(-i sqrt2 small p sigma_y) exp(-i sqrt2 big q sigma_x) exp(-i sqrt2 small p sigma_y)
        sBs "z" round: exp(-i sqrt2 small q sigma_x) exp(-i sqrt2 big p sigma_y) exp(-i sqrt2 small q sigma_x)
            qubit starts in g, is read and reset; no displacement -> cavity keeps the sum over both outcomes
        logical measurement along xi (S85): small CD on the perpendicular quadrature, then large CD,
            sizes |xi| sinh(Delta^2)/2 and |xi| cosh(Delta^2)/2; qubit read in Z; then D(+xi/2)
    These are the expressions checks/check_current_theory.py compares the runtime samples with.
    """

    def __init__(
        self,
        N,                       # Hilbert space dimension (integer)
        Delta,                   # finite-energy GKP width, sets the CD sizes (float)
        n_rounds=20,             # sBs rounds before the herald (integer)
        n_heralds=1,             # repeated Z heralds, all must agree (integer)
        n_rounds_after=2,        # sBs rounds after the herald / injection (integer)
        magic_input_angle=0.880, # qubit rotation before the injection CD [rad]
        magic_input_phase=0.0,   # its axis phase [rad] (calibrated on the device)
    ):
        # --- core parameters ---
        self.N = N
        self.Delta = Delta
        self.n_rounds = n_rounds
        self.n_heralds = n_heralds
        self.n_rounds_after = n_rounds_after
        self.magic_input_angle = magic_input_angle
        self.magic_input_phase = magic_input_phase

        # basic cavity operators
        self.a = qt.destroy(N) # annihilation operator
        self.q_op = (self.a + self.a.dag()) / np.sqrt(2.0) # unitless position operator
        self.p_op = 1j * (self.a.dag() - self.a) / np.sqrt(2.0) # unitless momentum operator
        self.I_cav = qt.qeye(N) # identity operator for cavity
        self.n_op = self.a.dag() * self.a # number operator

        # basic qubit operators and states
        self.sigma_x = qt.sigmax()
        self.sigma_y = qt.sigmay()
        self.sigma_z = qt.sigmaz()
        self.g = qt.basis(2, 0) # qubit ground state
        self.e = qt.basis(2, 1) # qubit excited state

        # one-branch CD sizes of an sBs round
        self.small = np.sqrt(np.pi/8) * np.sinh(Delta**2)
        self.big = np.sqrt(np.pi/2) * np.cosh(Delta**2)

        # logical displacements of the square code: Z <-> i sqrt(pi/2), X <-> sqrt(pi/2), Y <-> both
        self.xi_logical = {"Z": 1j*np.sqrt(np.pi/2), "X": np.sqrt(np.pi/2), "Y": (1+1j)*np.sqrt(np.pi/2)}

        # finite-energy codewords exp(-Delta^2 n)|0>, exp(-Delta^2 n)|1>, for logical Bloch vectors
        self.codeword_0 = self.codeword(0)
        self.codeword_1 = self.codeword(1)

        # for plots
        self.xmax = 6 # default x and p axis limit for wigner plots
        self.npts = 151 # default number of points along each axis for wigner plots

    # ------------------------------------------------------------------ displacements
    def displacement(self, alpha):
        """
        D(alpha) built from its exact Fock matrix elements, so large probe displacements stay exact
        inside the truncated space (qt.displace exponentiates the truncated a, which is not exact):
            <m|D|n> = sqrt(n!/m!) alpha^(m-n) exp(-|alpha|^2/2) L_n^(m-n)(|alpha|^2)          for m >= n
            <m|D|n> = sqrt(m!/n!) (-alpha*)^(n-m) exp(-|alpha|^2/2) L_m^(n-m)(|alpha|^2)       for m < n

        does not change self attributes
        """
        x = abs(alpha)**2
        n = np.arange(self.N)                # column index n, one whole row at a time
        D = np.zeros((self.N, self.N), dtype=complex)
        for m in range(self.N):
            low = np.minimum(m, n)
            high = np.maximum(m, n)
            k = high - low
            magnitude = np.exp(0.5*(gammaln(low + 1) - gammaln(high + 1)) - x/2) * eval_genlaguerre(low, k, x)
            phase_factor = np.where(m >= n, alpha, -np.conj(alpha))**k
            D[m, :] = magnitude * phase_factor
        return qt.Qobj(D, dims=[[self.N], [self.N]])

    def quadrature_shift(self, c, u, v):
        """
        exp(-i c (u q + v p)) written as a displacement: D(alpha), alpha = c (v - i u) / sqrt2.
        (exp(-i c p) shifts q by c; exp(-i c q) shifts p by -c.)

        does not change self attributes
        """
        alpha = c * (v - 1j*u) / np.sqrt(2.0)
        return self.displacement(alpha)

    def conditional_displacement(self, c, u, v, sigma):
        """
        exp(-i c (u q + v p) sigma) for a qubit operator sigma with eigenvalues s = +1, -1:
            sum over s of  exp(-i s c (u q + v p))  (x)  |s><s|
        Returns a cavity (x) qubit operator.

        does not change self attributes
        """
        values, vectors = sigma.eigenstates()
        U = 0
        for s, vec in zip(values, vectors):
            U = U + qt.tensor(self.quadrature_shift(s*c, u, v), vec * vec.dag())
        return U

    def qubit_axis(self, theta):
        """
        sigma_theta = cos(theta) sigma_x + sin(theta) sigma_y. For a CD this is the conditioning axis,
        which on the device is the pi-train axis.

        does not change self attributes
        """
        return np.cos(theta) * self.sigma_x + np.sin(theta) * self.sigma_y

    def qubit_rotation(self, angle, phase):
        """
        R(angle, phase) = exp(-i angle/2 sigma_phase) = cos(angle/2) - i sin(angle/2) sigma_phase.

        does not change self attributes
        """
        return np.cos(angle/2) * qt.qeye(2) - 1j * np.sin(angle/2) * self.qubit_axis(phase)

    # ------------------------------------------------------------------ Kraus operators
    def kraus(self, U, qubit_in, outcome):
        """
        Cavity Kraus operator of: qubit starts in |qubit_in>, unitary U, qubit read in Z with outcome
        'g' or 'e'.  K = <outcome| U |qubit_in>.

        does not change self attributes
        """
        if outcome == 'g':
            bra = self.g.dag()
        elif outcome == 'e':
            bra = self.e.dag()
        else:
            raise ValueError("outcome must be 'g' or 'e'")
        left = qt.tensor(self.I_cav, bra)
        right = qt.tensor(self.I_cav, qubit_in)
        K_raw = left * U * right # dims still carry a dim-1 qubit factor
        return qt.Qobj(K_raw.full(), dims=[[self.N], [self.N]])

    def sbs_unitary(self, kind):
        """
        One small-Big-small (S67).
        kind 'x': Big conditions sigma_x on q (moves p), smalls condition sigma_y on p (move q)
        kind 'z': Big conditions sigma_y on p (moves q), smalls condition sigma_x on q (move p)

        does not change self attributes
        """
        r2 = np.sqrt(2.0)
        if kind == 'x':
            small = self.conditional_displacement(r2*self.small, 0, 1, self.sigma_y) # exp(-i sqrt2 small p sigma_y)
            big = self.conditional_displacement(r2*self.big, 1, 0, self.sigma_x)     # exp(-i sqrt2 big q sigma_x)
        elif kind == 'z':
            small = self.conditional_displacement(r2*self.small, 1, 0, self.sigma_x) # exp(-i sqrt2 small q sigma_x)
            big = self.conditional_displacement(r2*self.big, 0, 1, self.sigma_y)     # exp(-i sqrt2 big p sigma_y)
        else:
            raise ValueError("kind must be 'x' or 'z'")
        return small * big * small

    def measurement_unitary(self, pauli):
        """
        Improved logical measurement (S85) along xi = xi_logical[pauli], pauli 'Z', 'X' or 'Y':
            phi = arg(xi), theta = pi/2 + phi
            q_par  = sin(phi) q - cos(phi) p      (large CD, qubit axis theta + pi)
            q_perp = cos(phi) q + sin(phi) p      (small CD first, qubit axis theta + 3 pi/2)

        does not change self attributes
        """
        xi = self.xi_logical[pauli]
        phi = np.angle(xi)
        theta = np.pi/2 + phi
        small_size = abs(xi) * np.sinh(self.Delta**2) / 2
        large_size = abs(xi) * np.cosh(self.Delta**2) / 2
        r2 = np.sqrt(2.0)
        small = self.conditional_displacement(r2*small_size, np.cos(phi), np.sin(phi), self.qubit_axis(theta + 3*np.pi/2))
        large = self.conditional_displacement(r2*large_size, np.sin(phi), -np.cos(phi), self.qubit_axis(theta + np.pi))
        return large * small # small acts first

    # ------------------------------------------------------------------ operations on the cavity
    def sbs_round(self, rho, kind):
        """
        One sBs round: qubit g, small-Big-small, qubit read and reset (no displacement).
        The cavity keeps the sum over both qubit outcomes.

        does not change self attributes
        """
        U = self.sbs_unitary(kind)
        K_g = self.kraus(U, self.g, 'g')
        K_e = self.kraus(U, self.g, 'e')
        return K_g * rho * K_g.dag() + K_e * rho * K_e.dag()

    def run_stabilization(self, rho, n_rounds, first_round=0):
        """
        n_rounds alternating sBs rounds, as the runtime plays them: round j is 'x' for even
        (first_round + j) and 'z' for odd.

        does not change self attributes
        """
        for j in range(n_rounds):
            if (first_round + j) % 2 == 0:
                rho = self.sbs_round(rho, 'x')
            else:
                rho = self.sbs_round(rho, 'z')
        return rho

    def logical_measurement(self, rho, pauli, outcome, qubit_in=None):
        """
        Improved logical measurement along pauli with the given qubit outcome ('g' or 'e'),
        followed by the correcting displacement D(+xi/2).
        qubit_in: qubit state entering the CD (default |g>).
        Returns (probability of the outcome, normalized cavity state).

        does not change self attributes
        """
        if qubit_in is None:
            qubit_in = self.g
        K = self.kraus(self.measurement_unitary(pauli), qubit_in, outcome)
        D_correction = self.displacement(self.xi_logical[pauli] / 2)
        rho_out = D_correction * K * rho * K.dag() * D_correction.dag()
        probability = float(np.real(rho_out.tr()))
        return probability, rho_out / probability

    def herald_Z(self, rho, outcome):
        """
        n_heralds repeated logical Z measurements, all with the same outcome ('g' -> |0_L>, 'e' -> |1_L>).
        Returns (probability that all heralds give this outcome, normalized state).

        does not change self attributes
        """
        probability = 1.0
        for k in range(self.n_heralds):
            p, rho = self.logical_measurement(rho, 'Z', outcome)
            probability = probability * p
        return probability, rho

    def magic_qubit_input(self, herald_outcome):
        """
        Qubit state entering the injection CD. The qubit is NOT reset after the herald readout:
        after g it gets R(angle, phase); after e it gets R(pi - angle, phase + pi), which reaches
        the same qubit state (up to a global phase).

        does not change self attributes
        """
        angle = self.magic_input_angle
        phase = self.magic_input_phase
        if herald_outcome == 'g':
            return self.qubit_rotation(angle, phase) * self.g
        elif herald_outcome == 'e':
            return self.qubit_rotation(np.pi - angle, phase + np.pi) * self.e
        else:
            raise ValueError("herald_outcome must be 'g' or 'e'")

    def inject_magic(self, rho, herald_outcome, injection_outcome):
        """
        Magic-state injection after a Z herald: magic qubit input -> logical-X-type improved measurement
        -> qubit read in Z -> D(+xi_X/2). Herald e means X_L, injection e means Z_L (neither corrected).
        Returns (probability of the injection outcome, normalized state).

        does not change self attributes
        """
        qubit_in = self.magic_qubit_input(herald_outcome)
        return self.logical_measurement(rho, 'X', injection_outcome, qubit_in=qubit_in)

    def prepare_states(self, rho_start=None):
        """
        The whole L4 sequence up to the probe, for every branch. Returns a dictionary
            name -> (probability, state)
        with names 'Z_g', 'Z_e' (herald alone), 'H_gg', 'H_ge', 'H_eg', 'H_ee' (herald, injection),
        and 'Z_all', 'H_all' (all shots: probability-weighted mixtures).

        does not change self attributes
        """
        if rho_start is None:
            rho_start = qt.fock_dm(self.N, 0) # the device starts from the cooled vacuum
        rho_stabilized = self.run_stabilization(rho_start, self.n_rounds)

        results = {}
        for h in ['g', 'e']:
            p_h, rho_h = self.herald_Z(rho_stabilized, h)
            # Z preparation: herald alone, then the later rounds (round index continues)
            results['Z_' + h] = (p_h, self.run_stabilization(rho_h, self.n_rounds_after, first_round=self.n_rounds))
            # H preparation: injection after the herald, then the later rounds
            for i in ['g', 'e']:
                p_i, rho_hi = self.inject_magic(rho_h, h, i)
                results['H_' + h + i] = (p_h * p_i, self.run_stabilization(rho_hi, self.n_rounds_after, first_round=self.n_rounds))

        # all shots of each preparation (what the unheralded map shows)
        for state, branches in [('Z', ['g', 'e']), ('H', ['gg', 'ge', 'eg', 'ee'])]:
            total = 0
            rho_all = 0
            for b in branches:
                p, rho = results[state + '_' + b]
                total = total + p
                rho_all = rho_all + p * rho
            results[state + '_all'] = (total, rho_all / total)
        return results

    def frame_after(self):
        """
        Logical Pauli applied by the rounds after the herald: an 'x' round applies Z_L, a 'z' round X_L.
        Returns (x_bit, z_bit) for X_L^x_bit Z_L^z_bit.

        does not change self attributes
        """
        x_bit, z_bit = 0, 0
        for j in range(self.n_rounds_after):
            if (self.n_rounds + j) % 2 == 0:
                z_bit = 1 - z_bit
            else:
                x_bit = 1 - x_bit
        return x_bit, z_bit

    def expected_bloch(self, name):
        """
        Ideal logical Bloch vector (X, Y, Z) of a branch name such as 'Z_g' or 'H_ge',
        including the logical Pauli of the later rounds.

        does not change self attributes
        """
        x_bit, z_bit = self.frame_after()
        state, branch = name.split('_')
        if state == 'Z':
            flip_z = (branch == 'e') != bool(x_bit)
            return np.array([0.0, 0.0, -1.0 if flip_z else 1.0])
        flip_z = (branch[0] == 'e') != bool(x_bit)   # herald e and X_L flip Z
        flip_x = (branch[1] == 'e') != bool(z_bit)   # injection e and Z_L flip X
        return np.array([-1.0 if flip_x else 1.0, 0.0, -1.0 if flip_z else 1.0]) / np.sqrt(2)

    # ------------------------------------------------------------------ characteristic function
    def chi(self, rho, xi):
        """
        chi(xi) = Tr[rho D(xi)].

        does not change self attributes
        """
        return complex((rho * self.displacement(xi)).tr())

    def probe_signal(self, rho, xi, qubit_in):
        """
        What the L4 probe measures: qubit prepared in 'g' or 'e', one CD on the fixed qubit axis 0
        (sigma_x) whose branches move by +-xi/2, qubit read in Z: P(g) - P(e).
        From g this equals +Re chi(xi), from e it equals -Re chi(xi).

        does not change self attributes
        """
        phi = np.angle(xi)
        U = self.conditional_displacement(np.sqrt(2.0)*abs(xi)/2, np.sin(phi), -np.cos(phi), self.sigma_x)
        if qubit_in == 'g':
            qubit = self.g
        else:
            qubit = self.e
        K_g = self.kraus(U, qubit, 'g')
        K_e = self.kraus(U, qubit, 'e')
        p_g = float(np.real((K_g * rho * K_g.dag()).tr()))
        p_e = float(np.real((K_e * rho * K_e.dag()).tr()))
        return p_g - p_e

    def chi_map(self, rho, re_values, im_values, from_probe=False):
        """
        Re chi on a grid (rows = Im xi, columns = Re xi).
        from_probe=False: Re Tr[rho D(xi)].  from_probe=True: the measurement (s_g - s_e)/2.

        does not change self attributes
        """
        grid = np.zeros((len(im_values), len(re_values)))
        for i in range(len(im_values)):
            for j in range(len(re_values)):
                xi = complex(re_values[j], im_values[i])
                if from_probe:
                    s_g = self.probe_signal(rho, xi, 'g')
                    s_e = self.probe_signal(rho, xi, 'e')
                    grid[i, j] = (s_g - s_e) / 2
                else:
                    grid[i, j] = self.chi(rho, xi).real
        return grid

    # ------------------------------------------------------------------ logical content
    def finite_energy_state(self, bloch):
        """
        Target state with logical Bloch vector (X, Y, Z) = bloch (unit length):
            exp(-Delta^2 n) [cos(t/2)|0> + e^{i f} sin(t/2)|1>],  t = polar angle, f = azimuth,
        normalized, with |0>, |1> the ideal position combs (the envelope acts on the whole superposition).

        does not change self attributes
        """
        x, y, z = bloch / np.linalg.norm(bloch)
        polar = np.arccos(np.clip(z, -1, 1))
        azimuth = np.arctan2(y, x)
        comb_0 = self.codeword(0, normalize=False)
        comb_1 = self.codeword(1, normalize=False)
        ket = np.cos(polar/2) * comb_0 + np.exp(1j*azimuth) * np.sin(polar/2) * comb_1
        return ket.unit()

    def codeword(self, mu, normalize=True):
        """
        exp(-Delta^2 n)|mu>, |mu> = position comb with peaks at q = (2j + mu) sqrt(pi), normalized by default.
        <n|q> from the stable Hermite recurrence.

        does not change self attributes
        """
        comb = np.zeros(self.N)
        for j in range(-8, 9):
            position = (2*j + mu) * np.sqrt(np.pi)
            psi = np.zeros(self.N)
            psi[0] = np.pi**(-0.25) * np.exp(-position**2 / 2)
            if self.N > 1:
                psi[1] = np.sqrt(2) * position * psi[0]
            for k in range(2, self.N):
                psi[k] = np.sqrt(2/k) * position * psi[k-1] - np.sqrt((k-1)/k) * psi[k-2]
            comb = comb + psi
        state = np.exp(-self.Delta**2 * np.arange(self.N)) * comb
        if normalize:
            state = state / np.linalg.norm(state)
        return qt.Qobj(state, dims=[[self.N], [1]])

    def logical_bloch(self, rho):
        """
        Logical Bloch vector (X, Y, Z) of rho inside the span of the two codewords (orthonormalized,
        they overlap by a few percent), and the weight of rho in that span.

        does not change self attributes
        """
        B = np.column_stack([self.codeword_0.full().ravel(), self.codeword_1.full().ravel()])
        Q, R = np.linalg.qr(B)
        Q = Q * np.sign(np.real(np.diag(R))) # keep the phases of |0>, |1>
        r = Q.conj().T @ rho.full() @ Q      # 2x2 logical block
        weight = float(np.real(np.trace(r)))
        r = r / weight
        bloch = np.array([2*np.real(r[0, 1]), -2*np.imag(r[0, 1]), np.real(r[0, 0] - r[1, 1])])
        return bloch, weight

    # ------------------------------------------------------------------ plots
    def plot_chi(self, ax, grid, re_values, im_values, title_str):
        """
        Re chi map on a given matplotlib axis, color scale -1..1.

        does not change self attributes
        """
        d_re = re_values[1] - re_values[0]
        d_im = im_values[1] - im_values[0]
        extent = [re_values[0] - d_re/2, re_values[-1] + d_re/2, im_values[0] - d_im/2, im_values[-1] + d_im/2]
        image = ax.imshow(grid, origin='lower', extent=extent, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xlabel(r'Re $\xi$')
        ax.set_ylabel(r'Im $\xi$')
        ax.set_title(title_str, fontsize=9)
        return image

    def plot_wigner(self, ax, rho, title_str, xmax=None, npts=None):
        """
        Wigner function on a given matplotlib axis, axes in units of sqrt(pi).

        does not change self attributes
        """
        if xmax is None:
            xmax = self.xmax
        if npts is None:
            npts = self.npts
        xlist = np.linspace(-xmax, xmax, npts)
        W = qt.wigner(rho, xlist, xlist)
        Wmax = np.max(np.abs(W))
        sqrt_pi = np.sqrt(np.pi)
        image = ax.imshow(W, origin='lower', extent=[xlist[0]/sqrt_pi, xlist[-1]/sqrt_pi, xlist[0]/sqrt_pi, xlist[-1]/sqrt_pi],
                          cmap='RdBu_r', vmin=-Wmax, vmax=Wmax, aspect='equal')
        ax.set_xlabel(r'$q/\sqrt{\pi}$')
        ax.set_ylabel(r'$p/\sqrt{\pi}$')
        ax.set_title(title_str, fontsize=9)
        return image
