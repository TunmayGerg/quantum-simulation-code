import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

class GKPStabilizationSimulator:
    """
    Encapsulate parameters, trimming, QEC cycles, and plotting
    for the GKP stabilization / sharp_trim notebook.
    """

    def __init__(
        self,
        N,                      # Hilbert space dimension (integer)
        n_stabalization_cycles, # number of QEC cycles to run to stabalize state (integer)
        rho_initial,            # initial cavity state (Qobj)
        eps_trim=None,          # default trim parameter (float)
        delta_fb=None,          # default feedback parameter (float)
        kappa = None,                  # cavity decay rate (units of 1/seconds)
        kappa_phi = None,              # cavity dephasing rate (units of 1/seconds)
        n_QEC_cycles=None,      # number of QEC cycles to run when doing lindblad evolution (integer)
        t_between_QEC = None,          # time between QEC cycles when doing lindblad evolution (seconds)
    ):
        # --- core parameters ---
        self.N = N
        self.n_stabalization_cycles = n_stabalization_cycles
        self.t_between_QEC = t_between_QEC
        self.kappa = kappa
        self.kappa_phi = kappa_phi

        # --- state ---
        self.rho_initial = rho_initial
        self.rho_logical_mixed = None # placeholder for logical mixed state
        self.rho_logical_pure = None # placeholder for logical pure state after projecting from mixed
        self.rhos_after_QEC = None # placeholder for list of states after each QEC cycle during lindblad evolution
        self.rhos_no_QEC = None # placeholder for list of states during lindblad evolution with no QEC

        # --- default trim / feedback parameters ---
        self.eps_trim_default = eps_trim
        self.delta_fb_default = delta_fb
        self.n_QEC_cycles = n_QEC_cycles

        # basic cavity operators
        self.a = qt.destroy(N) # annihilation operator
        self.q_op = (self.a + self.a.dag()) / np.sqrt(2.0) # unitless position operator
        self.p_op = 1j * (self.a.dag() - self.a) / np.sqrt(2.0) # unitless momentum operator
        self.I_cav = qt.qeye(N)  # identity operator for cavity
        self.n_op = self.a.dag() * self.a  # number operator

        # basic qubit operators
        self.sigma_x = qt.sigmax()  # Pauli Y operator for qubit
        self.sigma_y = qt.sigmay()  # Pauli Y operator for qubit
        self.sigma_z = qt.sigmaz()  # Pauli Z operator for qubit
        self.I_qubit = qt.qeye(2)  # identity operator for qubit

        # standard qubit basis states
        self.zero = qt.basis(2, 0)
        self.one  = qt.basis(2, 1)
        self.plus_x = (self.zero + self.one).unit()

        # Square GKP code position and momentum delta function spacing for logical states
        self.a_lat = 2.0 * np.sqrt(np.pi)

        # stabalizers
        self.S_x = self.disp_q(self.a_lat)
        self.S_z = self.disp_p(self.a_lat)

        self.X_logical = self.disp_q(self.a_lat/2)
        self.Z_logical = self.disp_p(self.a_lat/2)

        # for ploting wigner functions
        self.xmax = 6 # default x and p axis limit for wigner plots
        self.npts = 201 # default number of points along each axis for wigner plots
    
    def disp_q(self, delta_q):
        """
        Displacement that shifts q by Delta_q.
        q-shift: α real, Δq = √2 Re(α) -> α = Δq / √2.

        does not change self attributes
        """
        alpha = delta_q / np.sqrt(2.0)
        return qt.displace(self.N, alpha)
    
    def disp_p(self, delta_p):
        """
        Displacement that shifts p by Delta_p.
        p-shift: α imaginary, Δp = √2 Im(α) -> α = i Δp / √2.
        
        does not change self attributes
        """
        alpha = 1j * delta_p / np.sqrt(2.0)
        return qt.displace(self.N, alpha)
    
    def unitary_pow(self, U, k):
        """ 
        U^k for integer k, assuming U is unitary. Handles negative k safely.
        """
        if k >= 0:
            return U ** k
        else:
            return (U.dag()) ** (-k)   # U^{-k} = (U^\dagger)^{k}
    
    def prep_envelope_Z0_logical_state(self, num_displacements, Delta):
        """
        Prepare a GKP |Z0> logical state with num_displacements number of displacements and envelope function.
        
        does not change self attributes
        """
        envelope = (-Delta**2 * (self.n_op)).expm() # envelope function
        ket = 0 * qt.fock(self.N, 0) # start from nothing since we will sum

        for k in range(-num_displacements, num_displacements + 1):
            for l in range(-num_displacements, num_displacements + 1):
                ket = ket + envelope * self.unitary_pow(self.S_x, k) * self.unitary_pow(self.Z_logical, l) * qt.fock(self.N, 0)
        
        rho = qt.ket2dm(ket) # convert to density matrix
        return rho/rho.tr() # normalize state
    
    def conditional_displacement_kraus(self, beta_perp_op, measurment_basis, sign):
        """
        Return M_+ (measure Z = 1 project to 0 state, sign = +1) or M_- (measure Z = 1 project to 0 state, sign = +1)
        there is no feedback displacement here, just the Kraus operator from the measurement backaction.
        
        measurment_basis : 'X' or 'Y' for measuring qubit in X or Y basis
        beta_perp_op : Qobj (Hermitian) operator for conditional displacement argument
        sign         : +1 for '+1' outcome of Z, -1 for '-1' outcome of Z

        does not change self attributes
        """

        if measurment_basis == 'X':
            U1 = qt.tensor(self.I_cav, (1j*np.pi/4 * self.sigma_y).expm()) # rotate qubit x basis to z basis
        elif measurment_basis == 'Y':
            U1 = qt.tensor(self.I_cav, (-1j*np.pi/4 * self.sigma_x).expm()) # rotate qubit y basis to z basis
        else:
            raise ValueError("measurment_basis must be 'X' or 'Y'")

        U2 = (1j/2 * qt.tensor(beta_perp_op, self.sigma_z)).expm() # conditional displacement, essentially measures displacement by beta_perp_op
        U = U1 * U2 # total unitary is conditional displacement then qubit basis rotation

        if sign == +1:
            bra_left = self.zero.dag()  # <0| for +1 outcome
        elif sign == -1:    
            bra_left = self.one.dag()   # <1| for -1 outcome
        else:
            raise ValueError("sign must be +1 or -1")  
        
        left = qt.tensor(self.I_cav, bra_left)  # left operator <0| or <1| partial matrix element
        right = qt.tensor(self.I_cav, self.plus_x)  # right operator |+> qubit initial state

        M_raw = left * U * right # still has an extra dim-1 factor due to qubit space partial matrix element

        # Compress dims: make this a pure cavity operator -----
        # M_raw is N×N but its dims look like [[N,1],[N,1]].
        # Rewrap the same matrix as a single-space operator [[N],[N]].
        M_cav = qt.Qobj(M_raw.data, dims=[[self.N], [self.N]])

        return M_cav
    
    def sharp_trim(self,rho, quadrature, sharp_or_trim, eps_trim, delta_fb):
        """
        One sharp or trim round using conditional displacement Kraus operators and feedback.

        rho : input density matrix
        quadrature : 'q' or 'p' for q or p round
        sharp_or_trim : 'sharp' or 'trim' for sharp or trim round

        does not change self attributes
        """
        if quadrature == 'q':
            if sharp_or_trim == 'sharp':
                beta_perp = self.a_lat * self.q_op # β⊥ = a_lat q = 2√π q
                # feedback displacements
                D_plus  = self.disp_q(+delta_fb) # if +y measured, shift q by +delta_fb 
                D_minus = self.disp_q(-delta_fb) # if -y measured, shift q by -delta_fb  
            elif sharp_or_trim == 'trim':
                beta_perp = eps_trim * self.q_op
                # feedback displacements
                D_plus  = self.disp_q(+self.a_lat/2) # if +y measured, shift q by +a_lat/2 (this is sqrt(pi), the spacing between q or p peaks for mixed logical state)
                D_minus = self.disp_q(-self.a_lat/2) # if -y measured, shift q by -a_lat/2 
            else:
                raise ValueError("sharp_or_trim must be 'sharp' or 'trim'")
        elif quadrature == 'p':
            if sharp_or_trim == 'sharp':
                beta_perp = self.a_lat * self.p_op          # β⊥ = a_lat p = 2√π p
                # feedback displacements
                D_plus  = self.disp_p(+delta_fb) # if +y measured, shift p by +delta_fb
                D_minus = self.disp_p(-delta_fb) # if -y measured, shift p by -delta_fb  
            elif sharp_or_trim == 'trim':
                beta_perp = eps_trim * self.p_op
                # feedback displacements
                D_plus  = self.disp_p(+self.a_lat/2) # if +y measured, shift p by +a_lat/2
                D_minus = self.disp_p(-self.a_lat/2) # if -y measured, shift p by -a_lat/2  
            else:
                raise ValueError("sharp_or_trim must be 'sharp' or 'trim'")
        else:
            raise ValueError("quadrature must be 'q' or 'p'")
        
        # Kraus operators, sharp and trim use y measurment basis
        M_plus  = self.conditional_displacement_kraus(beta_perp, 'Y', +1) # measure +y
        M_minus = self.conditional_displacement_kraus(beta_perp, 'Y', -1) # measure -y

        rho_p = D_plus  * M_plus  * rho * M_plus.dag()  * D_plus.dag() # measure +y branch
        rho_m = D_minus * M_minus * rho * M_minus.dag() * D_minus.dag() # measure -y branch 
        return rho_p + rho_m # unconditioned output state
    
    def GKP_stabalization_sharp_trim_cycle(self, rho, eps_trim, delta_fb):
        """
        One GKP stabilization cycle: sharp q, trim q, sharp p, trim p
        starting from input state rho

        does not change self attributes
        """
        rho_use = rho.copy() # makes sure we don't modify input state

        rho_use = self.sharp_trim(rho_use, 'q', 'sharp', eps_trim=eps_trim, delta_fb=delta_fb)
        rho_use = self.sharp_trim(rho_use, 'q', 'trim', eps_trim=eps_trim, delta_fb=delta_fb)
        rho_use = self.sharp_trim(rho_use, 'p', 'sharp', eps_trim=eps_trim, delta_fb=delta_fb)
        rho_use = self.sharp_trim(rho_use, 'p', 'trim', eps_trim=eps_trim, delta_fb=delta_fb)
        return rho_use
    
    def run_stabalization(self, rho = None, n_cycles = None, eps_trim = None, delta_fb = None):
        """
        Run n_stabalization_cycles of GKP stabilization starting from rho_in (or self.rho_initial if None)

        does not change self attributes, you must change self.rho_logical_mixed manually if desired
        """
        if eps_trim is None:
            eps_trim = self.eps_trim_default
        if delta_fb is None:
            delta_fb = self.delta_fb_default
        if rho is None:
            rho = self.rho_initial.copy()
        if n_cycles is None:
            n_cycles = self.n_stabalization_cycles

        for k in range(n_cycles):
            rho = self.GKP_stabalization_sharp_trim_cycle(rho, eps_trim=eps_trim, delta_fb=delta_fb)
                
        return rho
    
    def GKP_pauli_measurment_helper(self, rho, beta_perp_scalar, beta_perp_quadrature, outcome, unconditional_disp_sign):
        """
        Perform logical paulli measurement with given outcome (+1 or -1) on state rho
        By changing beta_perp_op, can do logical X or Z measurements (or Y) or stabalizer measurements S_x = X^2 and S_z = Z^2
        Z logical measurement: beta_perp_scalar = a_lat/2, beta_perp_quadruature = 'q' 
        X logical measurement: beta_perp_scalar = -a_lat/2, beta_perp_quadruature = 'p'
        S_z stabalizer measurement: beta_perp_scalar = a_lat, beta_perp_quadruature = 'q'
        S_x stabalizer measurement: beta_perp_scalar = -a_lat, beta_perp_quadruature = 'p'
        Apply unconditional displacement in appropriate quadrature to correct for measurement backaction.
        
        rho : input density matrix
        beta_perp_scalar : scalar multiplier for quadrature
        beta_perp_quadruature : 'p' or 'q' for quadrature operator
        outcome : +1 or -1, measurement outcome
        unconditional_disp_sign : +1 or -1, sign of unconditional displacement in appropriate quadrature

        does not change self attributes
        """

        # Unnormalized post-measurement state
        if beta_perp_quadrature == 'q':
            beta_perp_op = beta_perp_scalar * self.q_op
            D_p_correction = self.disp_p(unconditional_disp_sign*beta_perp_scalar/2) # if measuring in q, correct the backaction on p distribution, the magnitude of the shift is beta_perp_scalar/2
        elif beta_perp_quadrature == 'p':
            beta_perp_op = beta_perp_scalar * self.p_op
            D_p_correction = self.disp_q(unconditional_disp_sign*beta_perp_scalar/2) # if measuring in p, correct the backaction on q distribution, the magnitude of the shift is beta_perp_scalar/2
        else:
            raise ValueError("beta_perp_quadruature must be 'q' or 'p'")
        
        M_measure = self.conditional_displacement_kraus(beta_perp_op, 'X', outcome) # measure X basis for logical Pauli measurement

        rho_measure_and_correct = D_p_correction* M_measure * rho * M_measure.dag() * D_p_correction.dag() # apply measurement and unconditional displacement correction
        p_outcome = np.abs(float(np.real(rho_measure_and_correct.tr())))
        
        return rho_measure_and_correct / p_outcome # normalized post-measurement state

    def GKP_pauli_measurment(self, operator, outcome, number_repeat, rho = None):
        """
        Perform logical operator measurement with given outcome (+1 or -1) on current mixed logical state
        
        operator : 'X', 'Z', 'S_x', or 'S_z' for logical X, logical Z, stabilizer S_x, or stabilizer S_z measurement
        outcome : +1 or -1, measurement outcome
        number_repeat : number of repeated measurements to perform (alternates sign of unconditional displacement each time to keep state centered)

        does not change self attributes, must update rho_logical_pure manually
        """
         
        if operator == 'Z':
            beta_perp_scalar = self.a_lat / 2  # for logical Z measurement
            beta_perp_quadruature = 'q'
        elif operator == 'X':
            beta_perp_scalar = -self.a_lat / 2  # for logical X measurement
            beta_perp_quadruature = 'p'
        elif operator == 'S_z':
            beta_perp_scalar = self.a_lat  # for stabilizer S_z measurement
            beta_perp_quadruature = 'q'
        elif operator == 'S_x':
            beta_perp_scalar = -self.a_lat  # for stabilizer S_x measurement
            beta_perp_quadruature = 'p'
        else:
            raise ValueError("operator must be 'X', 'Z', 'S_x', or 'S_z'")
        
        if rho == None:
            rho = self.rho_logical_mixed.copy() # start from current mixed logical state by default

        for k in range(number_repeat):
            rho = self.GKP_pauli_measurment_helper(rho, beta_perp_scalar, beta_perp_quadruature, outcome, unconditional_disp_sign=(-1)**k) # oscillate sign of unconditional displacement for each repeat measurement

        return rho
    
    def lindblad_step(self, rho, t_step, kappa, kappa_phi):
        """
        Evolve density matrix rho0 for time t_step under
        ρ̇ = κ D[a]ρ + κ_φ D[n]ρ   (no Hamiltonian term here).

        Returns ρ(t_step).

        does not change self attributes
        """
        # No coherent Hamiltonian for now (you can add w_c * a.dag()*a if desired)
        H = 0 * self.n_op

        c_ops = []
        if kappa != 0.0:
            c_ops.append(np.sqrt(kappa) * self.a)
        if kappa_phi != 0.0:
            c_ops.append(np.sqrt(kappa_phi) * self.n_op)

        # evolve from t=0 → t=t_step
        result = qt.mesolve(H, rho, [0.0, t_step], c_ops=c_ops, e_ops=[])
        return result.states[-1] # final state at t=t_step
    
    def QEC_time_evolution(self, rho_start, n_cycles = None, t_between_QEC = None, kappa = None, kappa_phi= None, eps_trim=None, delta_fb=None):
        """
        Perform n_cycles of GKP QEC (sharp+trim) with Lindblad evolution in between.
        rho_start : initial density matrix
        n_cycles : number of QEC cycles to perform (if None, use self.n_QEC_cycles)
        t_between_QEC : time between QEC cycles (if None, use self.t_between_QEC)
        kappa : cavity decay rate (if None, use self.kappa)
        kappa_phi : cavity dephasing rate (if None, use self.kappa_phi)
        eps_trim : trim parameter (if None, use self.eps_trim_default)
        delta_fb : feedback parameter (if None, use self.delta_fb_default)

        Returns list of density matrices just after each QEC cycle (length n_cycles + 1, including initial state) and list of times

        does not change self attributes
        """
        if n_cycles is None:
            n_cycles = self.n_QEC_cycles
        if t_between_QEC is None:
            t_between_QEC = self.t_between_QEC
        if kappa is None:
            kappa = self.kappa
        if kappa_phi is None:
            kappa_phi = self.kappa_phi
        if eps_trim is None:
            eps_trim = self.eps_trim_default
        if delta_fb is None:
            delta_fb = self.delta_fb_default
        
        times = np.arange(0, (self.n_QEC_cycles + 1) * self.t_between_QEC, self.t_between_QEC)
        states_after_QEC = [rho_start.copy()]  # will store ρ just *after* each sharp+trim, start with starting state, copy to avoid reference issues
        rho = rho_start.copy()

        for k in range(n_cycles):

            # 1) free evolution under Lindblad for time t_between_QEC
            rho = self.lindblad_step(rho, t_between_QEC, kappa=kappa, kappa_phi=kappa_phi)
            # 2) one full sharp+trim round (q-sharp, q-trim, p-sharp, p-trim)
            rho = self.GKP_stabalization_sharp_trim_cycle(rho, eps_trim=eps_trim, delta_fb=delta_fb) # 0.24, 0.11 for 1 us between QEC cycles and 0.2, 0.18 stabilization parameters

            # 3) store state right after QEC (sharp+trim)
            states_after_QEC.append(rho.copy())
        
        return states_after_QEC, times # length n_cycles + 1 list of states after each QEC cycle

    def no_QEC_time_evolution(self, rho_start, times = None, kappa = None, kappa_phi= None):
        """
        Perform time evolution with no QEC starting from rho_start.
        rho_start : initial density matrix
        times : list or array of times to evaluate the state at (if None, use times matching QEC evolution)
        kappa : cavity decay rate (if None, use self.kappa)
        kappa_phi : cavity dephasing rate (if None, use self.kappa_phi)
        Returns list of density matrices at each time in times

        does not change self attributes
        """

        # same initial state as the QEC run (make sure this matches your choice)
        if kappa is None:
            kappa = self.kappa
        if kappa_phi is None:
            kappa_phi = self.kappa_phi
        if times is None:
            times = np.arange(0, (self.n_QEC_cycles + 1) * self.t_between_QEC, self.t_between_QEC)
        
        H_noQEC = 0 * self.a.dag() * self.a      # zero Hamiltonian (or put your cavity H here)
        
        c_ops = []
        if kappa != 0.0:
            c_ops.append(np.sqrt(kappa) * self.a)
        if kappa_phi != 0.0:
            c_ops.append(np.sqrt(kappa_phi) * self.n_op)

        result_no_QEC = qt.mesolve(
            H_noQEC,
            rho_start,
            times,
            c_ops=c_ops,
            e_ops=[]
        )

        return result_no_QEC.states, times  # list of states at each time point

    def get_expectation_values(self, states, operator):
        """
        Get (real) expectation values of operator for list of states.

        states : list of density matrices
        operator : Qobj operator to get expectation value of

        does not change self attributes
        """
        expectations = []
        for rho in states:
            E = (rho * operator).tr()
            expectations.append(np.real(E))
        return expectations

    def get_q_marginal(self, rho, xmax=None, npts=None):
        """
        Get q marginal distribution from density matrix rho.

        Parameters
        ----------
        rho : Qobj
            Input density matrix.
        xmax : float, optional
            Maximum q value (in physical units, not rescaled by sqrt(pi)).
            Defaults to self.xmax if not provided.
        npts : int, optional
            Number of grid points along q (and p).
            Defaults to self.npts if not provided.

        Returns
        -------
        xlist : np.ndarray
            q-grid points.
        P_q : np.ndarray
            Approximate q-marginal ∫ W(q, p) dp at each q.
        """
        if xmax is None:
            xmax = self.xmax
        if npts is None:
            npts = self.npts

        xlist = np.linspace(-xmax, xmax, npts)
        ylist = np.linspace(-xmax, xmax, npts)
        
        W = qt.wigner(rho, xlist, ylist)  # shape (len(xlist), len(ylist))

        # integrate over p (y axis)
        P_q = np.trapezoid(W, ylist, axis=0)
        return P_q, xlist
    
    def get_p_marginal(self, rho, xmax=None, npts=None):
        """
        Get q marginal distribution from density matrix rho.

        Parameters
        ----------
        rho : Qobj
            Input density matrix.
        xmax : float, optional
            Maximum q value (in physical units, not rescaled by sqrt(pi)).
            Defaults to self.xmax if not provided.
        npts : int, optional
            Number of grid points along q (and p).
            Defaults to self.npts if not provided.

        Returns
        -------
        xlist : np.ndarray
            q-grid points.
        P_q : np.ndarray
            Approximate q-marginal ∫ W(q, p) dp at each q.
        """
        if xmax is None:
            xmax = self.xmax
        if npts is None:
            npts = self.npts

        xlist = np.linspace(-xmax, xmax, npts)
        ylist = np.linspace(-xmax, xmax, npts)
        
        W = qt.wigner(rho, xlist, ylist)  # shape (len(xlist), len(ylist))

        # integrate over p (y axis)
        P_q = np.trapezoid(W, ylist, axis=1)
        return P_q, xlist

    def plot_wigner(self, rho, title_str, xmax = None, npts = None):
        """
        Plot Wigner function of density matrix rho.
        rho : input density matrix
        xmax : maximum x and p value to plot (in physical units, not rescaled by sqrt(pi))
        npts : number of points along each axis
        title_str : string for plot title

        does not change self attributes
        """
        if xmax is None:
            xmax = self.xmax
        if npts is None:
            npts = self.npts
    
        # Define phase-space grid in physical units
        xlist = np.linspace(-xmax, xmax, npts)
        ylist = np.linspace(-xmax, xmax, npts)

        W = qt.wigner(rho, xlist, ylist) # get Wigner function values

        # make color scale symmetric around 0
        Wmax = np.max(np.abs(W))

        # Plot Wigner function with axes rescaled by sqrt(pi)
        sqrt_pi = np.sqrt(np.pi)

        plt.figure(figsize=(6, 5))
        plt.imshow(
            W,
            extent=[xlist[0] / sqrt_pi, xlist[-1] / sqrt_pi,
                    ylist[0] / sqrt_pi, ylist[-1] / sqrt_pi],
            origin='lower',
            aspect='equal',
            cmap='RdBu_r',
            vmin=-Wmax,
            vmax=+Wmax
        )

        # put ticks at every integer (…,-3,-2,-1,0,1,2,3,…)
        xmin = xlist[0] / sqrt_pi
        xmax = xlist[-1] / sqrt_pi
        ymin = ylist[0] / sqrt_pi
        ymax = ylist[-1] / sqrt_pi

        xticks = np.arange(np.ceil(xmin), np.floor(xmax) + 1, 1)
        yticks = np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1)

        plt.xticks(xticks)
        plt.yticks(yticks)

        plt.xlabel(r'$q / \sqrt{\pi}$')
        plt.ylabel(r'$p / \sqrt{\pi}$')
        plt.title(title_str)
        plt.colorbar(label='W(q, p)')
        plt.tight_layout()
        plt.show()