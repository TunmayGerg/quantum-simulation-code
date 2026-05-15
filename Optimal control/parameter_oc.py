from __future__ import annotations  # allow forward type annotations like -> GrapeLBFGS

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any  # typing helpers

from pathlib import Path
import json

import numpy as np                # numerical arrays + linear algebra glue
import qutip as qt                # QuTiP objects for quantum operators / unitaries
from scipy.optimize import minimize, differential_evolution          # SciPy optimizer (we use L-BFGS-B)
from scipy.linalg import expm, expm_frechet  # dense matrix exponential + its Fréchet derivative
from helpful_functions import *

class ParameterOC:
    """
    Goal is to optimize a set of parameters based on strings of unitaries and a target unitary.
    specific to two cavities and one qubit system. Unitary so makes hbar = 1
    """

    def __init__(self,
                 unitary_strings: List[str],
                 target_unitary: qt.Qobj,
                 n_c: int,
                 n_q: int,
                 cost_type: str = "unitary",
                 projector: Optional[qt.Qobj] = None,
                 gauge_ops: Optional[Sequence[qt.Qobj]] = None,
                 ):
        self.unitary_strings = unitary_strings
        self.Ut = target_unitary
        self.n_c = n_c
        self.n_q = n_q
        self.dim = int(target_unitary.shape[0])       # Hilbert dimension (assume square operator)
        self.n_base = len(unitary_strings)  # number of gates in ONE sequence application

        self.a_c = qt.tensor(qt.destroy(n_c), qt.qeye(n_q))   # cavity annihilation op in full Hilbert space
        self.a_q = qt.tensor(qt.qeye(n_c), qt.destroy(n_q))   # qubit annihilation op in full Hilbert space
        self.num_apply = 1  # default number of times to apply the sequence. This not used but will be modified based on what user inputs as num_apply during optimization call. This value will be the most recent num_apply used and will be stored here to be later saved.
        self.optimize_phases = True

        # ---- Interpret and validate cost selection ----
        if cost_type not in ("unitary", "projected"):       # only these two modes
            raise ValueError("cost_type must be 'unitary' or 'projected'.")
        self.cost_type = cost_type                          # store cost mode

        # ---- Projected fidelity: store projector P and precompute effective target block ----
        self.P: Optional[qt.Qobj] = None                    # default: no projector
        if self.cost_type == "projected":                   # if user wants subspace fidelity
            if projector is None:                           # must supply projector P
                raise ValueError("You must provide projector=... for cost_type='projected'.")
            if not isinstance(projector, qt.Qobj):          # must be QuTiP Qobj
                raise TypeError("projector must be a QuTiP Qobj.")
            if projector.shape != (self.dim, self.dim):     # must match full space dimension
                raise ValueError("projector has wrong shape.")
            self.P = projector                              # store P

        self.gauge_ops = list(gauge_ops) if gauge_ops is not None else []  # store Aj list
        self.n_gauge = len(self.gauge_ops)                                  # number of gauge params
                
        self._Ut_dense = self.Ut.full()                # dense target unitary (not including projection)
        self._P_dense = None if self.P is None else self.P.full()  # dense projector (or None)


    def _get_unitary_from_string(self, unitary_str: str, rotation_param: float, phase_param: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get the unitary operator corresponding to a given string identifier.

        Args:
            unitary_str (str): Identifier for the unitary.
            rotation_param (float): Rotation parameter.
            phase_param (float): Phase parameter.

        Returns:
            np.ndarray: The exponential argument of the unitary.
            np.ndarray: The corresponding unitary operator.
            derivative of the exponential argument in the untitary with respect to the rotation param
            derivative of the exponential argument in the untitary with respect to the phase param
        """
        if unitary_str == "bs":
            # Beamsplitter between cavity and qubit (experimentalist convention): H = g(a_c† a_q e^{iφ} + h.c.)
            # g=pi/2 -> full SWAP; g=pi -> identity up to signs.
            # U = exp(-iH) with A = -iH, so U = exp(A).
            g = rotation_param                                  # beamsplitter coupling strength
            phase = phase_param if phase_param is not None else 0  # beamsplitter phase angle in radians

            H = g * (self.a_c.dag() * self.a_q * np.exp(1j * phase) + self.a_c * self.a_q.dag() * np.exp(-1j * phase))
            # Returns: (A=-iH, U=exp(A), dA/dg, dA/dphase)
            return (-1j*H).full(), (-1j * H).expm().full(), (-1j* (self.a_c.dag() * self.a_q * np.exp(1j * phase) + self.a_c * self.a_q.dag() * np.exp(-1j * phase))).full(), (-1j * g * (1j*self.a_c.dag() * self.a_q * np.exp(1j * phase) - 1j* self.a_c * self.a_q.dag() * np.exp(-1j * phase))).full()  # exp argument, unitary operator for the beamsplitter, derivative wrt rotation param, derivative wrt phase param
        if unitary_str == "r":
            # Driven qubit rotation restricted to the 0-1 subspace: H = theta*(sigma_+ e^{iφ} + h.c.)
            # theta=pi/2 -> pi/2 pulse (X/2 gate); theta=pi -> full pi pulse (gives -I on 0-1 subspace).
            theta = rotation_param                              # rotation angle in radians (0 to 2pi)
            phase = phase_param if phase_param is not None else 0  # rotation axis phase angle in radians
            sigma_plus = sigma_plus_restricted_01_from_aq(self.a_q)  # raising op restricted to |0>-|1> subspace
            H = theta * (sigma_plus*np.exp(1j*phase) + sigma_plus.dag()*np.exp(-1j*phase))
            # Returns: (A=-iH, U=exp(A), dA/dtheta, dA/dphase)
            return (-1j*H).full(), (-1j * H).expm().full(), (-1j * (sigma_plus*np.exp(1j*phase) + sigma_plus.dag()*np.exp(-1j*phase))).full(), -1j*theta * (1j*sigma_plus*np.exp(1j*phase) - 1j*sigma_plus.dag()*np.exp(-1j*phase)).full()  # exp argument, unitary operator for the rotation, derivative wrt rotation param, derivative wrt phase param
        raise ValueError(f"Unknown unitary string: {unitary_str}")
    
    # =========================================================================
    # Fidelity and its differential
    # =========================================================================

    def _overlap(self, U_corr: np.ndarray) -> complex:
        """
        U_corr: the unitary you actually got from your pulses after applying any gauge/phase-frame corrections you’re allowing.

        Compute overlap:
        full-space:
            c = Tr(Ut† U_corr)
        projected:
            c = Tr(Ut† U_corr P), P is an orthogonal projector

        """
        if self._P_dense is None:                               # if not projected
            return np.trace(self._Ut_dense.conj().T @ U_corr)  # Tr(Ut† U)
        P = self._P_dense                                       # dense projector
        return np.trace(self._Ut_dense.conj().T @ U_corr @ P)  # Tr(Ut† U P)

    def _fidelity(self, c: complex) -> float:
        """
        Convert overlap c to a normalized fidelity F.

        squared:
            F = |c|^2 / d_eff^2   (smooth, common)
        """
        d: float = 0.0
        if self.cost_type == "unitary":                          # full-space fidelity
            d = float(self.dim)                                  # d = Hilbert space dimension
        elif self.cost_type == "projected":                      # projected fidelity
            d = float(np.real(np.trace(self._P_dense)))          # effective dimension = rank(P) = Tr(P)
        # F = |Tr(Ut† U)|^2 / d^2 normalizes to [0,1] for unitaries
        f = float((abs(c) ** 2) / (d * d))
        print(f"Fidelity: {f}")
        return f                    # F in [0, 1]

    def _dF_from_dc(self, c: complex, dc: complex) -> float:
        """
        Given c and its differential dc, compute dF (a real scalar).

        squared fidelity:
            F = |c|^2 / d^2
            dF = 2 Re(conj(c) dc) / d^2
        """
        if self.cost_type == "unitary":                           # full-space fidelity
            d = float(self.dim)                                   # full dimension
        elif self.cost_type == "projected":                       # projected fidelity
            d = float(np.real(np.trace(self._P_dense)))                    # effective dimension = Tr(P)

        return float((2.0 * np.real(np.conj(c) * dc)) / (d * d))  # exact differential

    def _propagate(self, rotation_params: np.ndarray, phase_params: np.ndarray, num_apply: int, post_bool: bool = True) -> Dict[str, Any]:
        """
        Build the sequence of unitaries specified by `self.unitary_strings`, then repeat that
        entire sequence `num_apply` times, and cache prefix/post products (GRAPE-style).

        This is the “parameterized-circuit” analog of your GRAPE `_propagate`:

            U_final = U_{N-1} ... U_1 U_0

        where each U_j is a gate from `get_unitary_from_string(...)`.

        Parameters
        ----------
        rotation_params:
            Array of shape (n_gates,) containing the rotation parameter for each gate
            in `self.unitary_strings` (one set of params for ONE application of the sequence).

        phase_params:
            Array of shape (n_gates,) containing the phase parameter (radians) for each gate.

        num_apply:
            How many times to apply the whole `unitary_strings` list in sequence.
            Total gate count = n_gates_total = len(unitary_strings) * num_apply.

        post_bool:
            If True, compute the backward “post” products (often needed for gradients).
            If False, post will be allocated but left mostly None (except last one = I).

        Returns
        -------
        cache : Dict[str, Any]
            A dictionary with:
              - "U_list"            : list of Qobj, length n_gates_total
              - "dA_drot_list"      : list of Qobj/None, length n_gates_total
              - "dA_dphase_list"    : list of Qobj/None, length n_gates_total
              - "A_list"            : list of Qobj, length n_gates_total
              - "prefix"            : list[Qobj], length n_gates_total + 1, prefix[0]=I
              - "post"              : list[Qobj], length n_gates_total, post[last]=I
              - "U_final"           : Qobj, full product

        Notes
        -----
        - `get_unitary_from_string` is assumed to return:
              (exp argument, U, dA/d(rotation_param), dA/d(phase_param or None))
          i.e., derivatives of the *exponential argument* A = -i H.
        """
    
        
        U_list: List[np.ndarray] = []
        dA_drot_list: List[Optional[np.ndarray]] = []
        dA_dphase_list: List[Optional[np.ndarray]] = []
        A_list: List[np.ndarray] = []

        # Build the full gate sequence by repeating unitary_strings num_apply times.
        # rotation_params and phase_params are already the full flat arrays of length n_base * num_apply,
        # so index i directly maps to the i-th gate in the repeated sequence.
        itterated_unitary_strings = []
        for _rep in range(num_apply):
            itterated_unitary_strings.extend(self.unitary_strings)

        for i, unitary_str in enumerate(itterated_unitary_strings):
            # rotation_params[i] and phase_params[i] are the parameters for the i-th gate globally
            A_i, U_i, dA_drot_i, dA_dphase_i = self._get_unitary_from_string(unitary_str=unitary_str, rotation_param=float(rotation_params[i]), phase_param=float(phase_params[i]))

            U_list.append(U_i)
            dA_drot_list.append(dA_drot_i)
            dA_dphase_list.append(dA_dphase_i)
            A_list.append(A_i)

        n_total = len(U_list)  # = n_base * num_apply

        # -------------------------
        # Prefix/post products (GRAPE-style)
        # -------------------------
        # Identity with the *same tensor dims* as the target operator.
        # target_unitary.dims looks like: [[n_a,n_b,n_q], [n_a,n_b,n_q]]
        I = np.eye(self.dim, dtype=np.complex128)

        # prefix[j] = U_{j-1} @ ... @ U_0, with prefix[0] = I
        # prefix[j] is the partial product of gates *before* gate j (right-to-left application order)
        prefix: List[np.ndarray] = [I]
        for j in range(n_total):
            prefix.append(U_list[j] @ prefix[j])   # prepend U_j: prefix[j+1] = U_j @ prefix[j]
        U_final = prefix[n_total]                   # full product U_{N-1} @ ... @ U_0

        # post[j] = U_{N-1} @ ... @ U_{j+1}, with post[N-1] = I
        # post[j] is the partial product of gates *after* gate j
        post: List[Optional[np.ndarray]] = [None] * n_total
        if n_total > 0:
            post[n_total - 1] = I                   # last gate has no successor -> identity

        if post_bool and n_total > 1:
            for j in range(n_total - 2, -1, -1):
                # post[j] = post[j+1] @ U_{j+1}: accumulate rightward from the end
                post[j] = post[j + 1] @ U_list[j + 1]

        return {
            "U_list": U_list,
            "dA_drot_list": dA_drot_list,
            "dA_dphase_list": dA_dphase_list,
            "A_list": A_list,
            "prefix": prefix,
            "post": post,
            "U_final": U_final,
            "n_total": n_total
        }

    
    def _gauge_unitary(self, theta: Optional[np.ndarray], derivative_bool: Optional[bool] = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute gauge unitary and its derivatives:
            G(theta) = exp(+i Σ_j theta_j A_j)

        Returns:
            G (dense array),
            dG_list where dG_list[j] = ∂G/∂theta_j (dense arrays). length is n_gauge. The derivative is evaluated at the given theta angles.

        Uses expm_frechet for exact derivatives of the matrix exponential.
        """
        if self.n_gauge == 0:                                 # if no gauge operators provided
            G = np.eye(self.dim, dtype=np.complex128)         # gauge is identity
            return G, []                                      # no derivatives needed

        if theta is None:                                     # if theta not given
            theta = np.zeros(self.n_gauge, dtype=float)       # default to zeros
        theta = np.asarray(theta, dtype=float)                # ensure numpy array float
        if theta.shape != (self.n_gauge,):                    # shape check
            raise ValueError(f"theta must be shape {(self.n_gauge,)}.")

        # Convert gauge operators to dense arrays once per call (still simple and explicit).
        gauge_dense = [Aj.full() for Aj in self.gauge_ops]     # Aj as dense arrays

        # Build exponent matrix for gauge:
        #   A_g = +i Σ theta_j Aj
        A_g = np.zeros((self.dim, self.dim), dtype=np.complex128)  # start at 0
        for j in range(self.n_gauge):                          # sum contributions
            A_g += 1j * theta[j] * gauge_dense[j]             # add +i theta_j Aj

        G = expm(A_g)                                         # compute gauge unitary G(theta)

        # Compute derivatives dG/dtheta_j via Fréchet derivative:
        #   d/dθ expm(A(θ)) = expm_frechet(A, dA/dθ)
        dG_list: List[np.ndarray] = []                         # list for each parameter derivative
        if derivative_bool:
            for j in range(self.n_gauge):                          # for each gauge angle
                dA = 1j * gauge_dense[j]                           # dA_g/dtheta_j = +i Aj
                dG = expm_frechet(A_g, dA, compute_expm=False)     # Fréchet derivative (exact)
                dG_list.append(dG)                                 # store derivative

        return G, dG_list                                     # return G and its derivatives
    
    # =========================================================================
    # Objective and gradient for SciPy
    # =========================================================================

    def cost(self, rotation_params: np.ndarray, phase_params: np.ndarray, num_apply: int, theta: Optional[np.ndarray]) -> float:
        """
        Compute cost for L-BFGS-B evaluated at the angles and phases
        cost = 1 - fidelity + Σ penalties
        """
        # ---- Run forward propagation and build cached products ----
        cache = self._propagate(rotation_params, phase_params, num_apply, False) # get dictionary of cached products
        U_final = cache["U_final"]                               # achieved unitary without gauge

        # ---- Build gauge unitary G(theta) and derivatives (if enabled) ----
        G, _ = self._gauge_unitary(theta, False)                  # gauge unitary

        # ---- Apply gauge freedom to define what we compare to target ----
        if self.n_gauge == 0:                                    # no gauge variables
            U_corr = U_final                                     # corrected unitary = achieved unitary
        else:
            U_corr = G @ U_final                                 # left gauge: U_corr = G U

        # ---- Compute overlap and base fidelity ----
        c = self._overlap(U_corr)                                # overlap c = Tr(Ut† U_corr) (maybe projected which gives Tr(Ut† U_corr P))``
        F = self._fidelity(c)                                    # fidelity F in [0,1] (roughly)
        cost = 1.0 - F                                           # GRAPE minimizes 1-F

        return cost                                              # return cost only
    
    def grad(self, rotation_params: np.ndarray, phase_params: np.ndarray, num_apply: int, optimize_phases: bool, theta: Optional[np.ndarray]) -> np.ndarray:
        """
        Compute gradient for L-BFGS-B evaluated at pulses and theta. 
        """
        # ---- Run forward propagation and build cached products ----
        
        cache = self._propagate(rotation_params, phase_params, num_apply)                          # get dictionary of cached products
        U_final = cache["U_final"]                               # achieved unitary without gauge
        A_list = cache["A_list"]                                 # list of exponential arguments
        d_rot_list = cache["dA_drot_list"]                       # list of derivative of exponential arguments wrt rotation params
        d_phase_list = cache["dA_dphase_list"]                   # list of derivative of exponential arguments wrt phase params                            
        prefix = cache["prefix"]                                 # prefix products for GRAPE
        post = cache["post"]                                     # post products for GRAPE
        n_total = cache["n_total"]                               # total number of gates

        # ---- Build gauge unitary G(theta) and derivatives (if enabled) ----
        G, dG_list = self._gauge_unitary(theta)                  # gauge unitary + dG/dtheta

        # ---- Apply gauge freedom to define what we compare to target ----
        if self.n_gauge == 0:                                    # no gauge variables
            U_corr = U_final                                     # corrected unitary = achieved unitary
        else:
            U_corr = G @ U_final                                 # left gauge: U_corr = G U

        # ---- Compute overlap and base fidelity ----
        c = self._overlap(U_corr)                                # overlap c = Tr(Ut† U_corr) (maybe projected which gives Tr(Ut† U_corr P))``
        
        # ---- Allocate gradient arrays ----
        grad_rot = np.zeros((n_total,), dtype=float)   # gradient wrt rotation params and phase params in order of the gate application
        grad_phase = np.zeros((n_total,), dtype=float)  # gradient wrt phase params
        grad_t = None if self.n_gauge == 0 else np.zeros((self.n_gauge,), dtype=float)  # gradient wrt theta, gauge angles

        # Helper: compute dc given a small dU_corr
        def dc_from_dUcorr(dUcorr: np.ndarray) -> complex:
            # If no projector: dc = Tr(Ut† dUcorr)
            if self._P_dense is None:
                return np.trace(self._Ut_dense.conj().T @ dUcorr)
            # If projected: dc = Tr(Ut† dUcorr P)
            P = self._P_dense
            return np.trace(self._Ut_dense.conj().T @ dUcorr @ P)

        # ---- GRAPE pulse gradients: loop over time slice n and control channel k ----
        for n in range(n_total):                             # for each time slice
            pre = prefix[n]                                       # U_{n-1}...U_0
            postn = post[n]                                       # U_{N-1}...U_{n+1}

            # Compute dU_n/d rot or phase for this slice.
            # Exact slice derivative via Fréchet derivative:
            # dUn = expm_frechet(A_n, dA)
            An = A_list[n]                                   # A_n = -i * H for this unitary
            dA_rot = d_rot_list[n]                           # dA/d(rotation_param) for this unitary
            dUn_rot = expm_frechet(An, dA_rot, compute_expm=False)   # exact d(exp(A))/du

            if optimize_phases:
                dA_phase = d_phase_list[n]                       # dA/d(phase_param) for this unitary
                dUn_phase = expm_frechet(An, dA_phase, compute_expm=False)   # exact d(exp(A))/d(phase)
                dU_phase_final = postn @ dUn_phase @ pre

            # Insert the local derivative into the full time-ordered product:
            #   U = U_{N-1}...U_{n+1} * U_n * U_{n-1}...U_0
            # so
            #   dU = post[n] * dU_n * prefix[n]
            dU_rot_final = postn @ dUn_rot @ pre                         # derivative of U_final wrt u[n,k]. we don't use prefix[n_steps] because that is U_final itself

            # Apply gauge mapping to get dU_corr.
            if self.n_gauge == 0:                               # no gauge -> same derivative
                dUcorr_rot = dU_rot_final
                if optimize_phases:
                    dUcorr_phase = dU_phase_final
            else:
                dUcorr_rot = G @ dU_rot_final                        # gauge multiplies derivative on left U_corr = G U
                if optimize_phases:
                    dUcorr_phase = G @ dU_phase_final
            # ---- Convert dU_corr to dc and then to dF ----
            dc_rot = dc_from_dUcorr(dUcorr_rot)                       # compute overlap differential dc
            dF_rot = self._dF_from_dc(c, dc_rot)                        # convert dc -> dF
            grad_rot[n] = -dF_rot                                  # cost=1-F => d(cost)=-dF. derivative wrt rotation angle of pulse n.

            if optimize_phases:
                dc_phase = dc_from_dUcorr(dUcorr_phase)                         # compute overlap differential dc
                dF_phase = self._dF_from_dc(c, dc_phase)                        # convert dc -> dF
                grad_phase[n] = -dF_phase                                # cost=1-F => d(cost)=-dF. derivative wrt phase parameter of pulse n.

        # ---- Gauge gradients: differentiate U_corr wrt theta_j ----
        if self.n_gauge > 0:
            for j in range(self.n_gauge):                           # for each gauge parameter theta_j
                dG = dG_list[j]                                     # dG/dtheta_j
                # If U_corr = G U, then dU_corr = (dG) U
                dUcorr = dG @ U_final

                dc = dc_from_dUcorr(dUcorr)                         # differential of overlap from gauge
                dF = self._dF_from_dc(c, dc)                        # convert to fidelity differential
                grad_t[j] = -dF                                     # cost=1-F => gradient is -dF

        grad_flat = self._pack(grad_rot, grad_phase, grad_t, optimize_phases=optimize_phases)         # flatten gradient for SciPy
        return grad_flat                               # return cost and gradient vector
    
    def _pack(self,
              rot: np.ndarray,
              phase: Optional[np.ndarray],
              theta: Optional[np.ndarray],
              optimize_phases: bool) -> np.ndarray:
        """
        Pack gradient components into a single flat array for SciPy.

        Order:
            [rot_params..., phase_params..., theta_params...]
        """
        parts = [rot]
        if optimize_phases:
            parts.append(phase)

        if theta is not None:
            parts.append(theta)
        return np.concatenate(parts)
    
    def _unpack(self,
            x: np.ndarray,
            num_apply: int,
            optimize_phases: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Unpack flat vector x into (rotation, phase_or_None, theta_or_None).
        """
        n_rot = num_apply * self.n_base   # number of rotation (and phase) parameters in total

        # Layout of x: [rotation_0...rotation_{n_rot-1}, (phase_0...phase_{n_rot-1}), (theta_0...theta_{n_gauge-1})]
        rotation = x[0:n_rot]             # rotation parameters
        idx = n_rot                        # running slice index

        phase = None
        if optimize_phases:
            phase = x[idx:idx + n_rot]
            idx += n_rot

        theta = None
        if self.n_gauge > 0:
            theta = x[idx:idx + self.n_gauge]

        return rotation, phase, theta
        
    def fidelity(self, rotation_params: np.ndarray, phase_params: np.ndarray, num_apply: int, optimize_phases: bool, theta: Optional[np.ndarray]) -> float:
        """
        Compute the configured fidelity F for given pulses (and gauge angles).
        """
        if optimize_phases:
            cache = self._propagate(rotation_params, phase_params, num_apply, False) # get dictionary of cached products
        else:
            cache = self._propagate(rotation_params, np.zeros_like(rotation_params), num_apply, False) # get dictionary of cached products with zero phases, default
        U_final = cache["U_final"]                               # achieved unitary without gauge

        # ---- Build gauge unitary G(theta) and derivatives (if enabled) ----
        G, _ = self._gauge_unitary(theta, False)                  # gauge unitary

        # ---- Apply gauge freedom to define what we compare to target ----
        if self.n_gauge == 0:                                    # no gauge variables
            U_corr = U_final                                     # corrected unitary = achieved unitary
        else:
            U_corr = G @ U_final                                 # left gauge: U_corr = G U

        # ---- Compute overlap and base fidelity ----
        c = self._overlap(U_corr)                                # overlap c = Tr(Ut† U_corr) (maybe projected which gives Tr(Ut† U_corr P))``
        F = self._fidelity(c)                                    # fidelity F in [0,1] (roughly)

        return F                                  # return normalized fidelity
    
    # =========================================================================
    # Optimization driver (SciPy L-BFGS-B)
    # =========================================================================
    def optimize(
        self,
        num_apply: int,
        rotation0: Optional[np.ndarray] = None,
        phase0: Optional[np.ndarray] = None,
        theta0: Optional[np.ndarray] = None,
        rotation_bounds: Optional[tuple[float, float]] = (0.0, np.pi),
        phase_bounds: Optional[tuple[float, float]] = (0.0, 2 * np.pi),
        gauge_bounds: Optional[tuple[float, float]] = (0.0, 2 * np.pi),
        optimize_phases: Optional[bool] = True,
        *,
        maxiter: int = 200,
        scipy_options: Optional[Dict[str, Any]] = None,
        store_history: bool = True,
        n_starts: int = 8,          
        seed: Optional[int] = None, 
    ) -> Dict[str, Any]:

        # ---- Default initial pulses ----
        self.num_apply = num_apply  # store most recent num_apply used
        self.optimize_phases = optimize_phases  # store most recent optimize_phases used
        n_rot = self.n_base * num_apply

        if rotation0 is None:
            rotation0 = np.zeros((n_rot,), dtype=float)
        rotation0 = np.asarray(rotation0, dtype=float)
        if rotation0.shape != (n_rot,):
            raise ValueError(f"rotation0 must be shape {(n_rot,)}.")

        if phase0 is None: 
            phase0 = np.zeros((n_rot,), dtype=float) # default zero phases
        phase0 = np.asarray(phase0, dtype=float)
        if phase0.shape != (n_rot,):
            raise ValueError(f"phase0 must be shape {(n_rot,)}.")

        # ---- Default initial gauge angles ----
        if self.n_gauge > 0:
            if theta0 is None:
                theta0 = np.zeros((self.n_gauge,), dtype=float)
            theta0 = np.asarray(theta0, dtype=float)
            if theta0.shape != (self.n_gauge,):
                raise ValueError(f"theta0 must be shape {(self.n_gauge,)}.")
        else:
            theta0 = None

        x0 = self._pack(rotation0, phase0, theta0, optimize_phases)

        # bounds for [rot..., phase..., theta...] — each group uses its own bounds tuple
        if optimize_phases:
            bounds = (
                [rotation_bounds] * n_rot        # rotation parameters
                + [phase_bounds] * n_rot          # phase parameters
                + [gauge_bounds] * self.n_gauge   # gauge parameters
            )
        else:
            bounds = (
                [rotation_bounds] * n_rot         # rotation parameters
                + [gauge_bounds] * self.n_gauge   # gauge parameters (no phases)
            )

        # SciPy expects fun/jac of flattened x
        def fun(x: np.ndarray) -> float:
            rot, phase, theta = self._unpack(x, num_apply, optimize_phases)
            if optimize_phases:
                return float(self.cost(rot, phase, num_apply, theta))
            return float(self.cost(rot, np.zeros_like(rot), num_apply, theta))

        def jac(x: np.ndarray) -> np.ndarray:
            rot, phase, theta = self._unpack(x, num_apply, optimize_phases)
            if optimize_phases:
                return self.grad(rot, phase, num_apply, optimize_phases, theta)
            return self.grad(rot, np.zeros_like(rot), num_apply, optimize_phases, theta)

        # Options for SciPy optimizer
        options = {"maxiter": int(maxiter)}
        if scipy_options:
            options.update(dict(scipy_options))

        rng = np.random.default_rng(seed)

        best_res = None
        best_history: List[Dict[str, Any]] = []
        best_x = None

        # ---- Multi-start loop: run n_starts independent L-BFGS-B optimizations ----
        # Start 0 uses the user-supplied initial point; remaining starts are uniformly random in bounds.
        # Keep only the result with the lowest final cost value.
        for s in range(int(n_starts)):
            # start 0 = user-provided x0, others random in bounds
            if s == 0:
                x0_s = x0
            else:
                lows = np.array([b[0] for b in bounds], dtype=float)
                highs = np.array([b[1] for b in bounds], dtype=float)
                x0_s = lows + (highs - lows) * rng.random(size=len(bounds))  # uniform random in bounds

            history: List[Dict[str, Any]] = []

            def callback(xk: np.ndarray) -> None:
                if not store_history:
                    return
                rot, phase, theta = self._unpack(xk, num_apply, optimize_phases)
                history.append({"rotation": rot, "phase": phase, "theta": theta, "start": s})

            res_local = minimize(
                fun=fun,
                x0=x0_s,
                jac=jac,
                method="L-BFGS-B",
                bounds=bounds,
                options=options,
                callback=callback,
            )

            if (best_res is None) or (res_local.fun < best_res.fun):
                best_res = res_local
                best_history = history
                best_x = res_local.x

        rot_opt, phase_opt, theta_opt = self._unpack(best_x, num_apply, optimize_phases)

        return {
            "result": best_res,
            "rotation_opt": rot_opt,
            "phase_opt": phase_opt,
            "theta_opt": theta_opt,
            "history": best_history,
        }
    
    def save_results(self, filename: str, results: Dict[str, Any]) -> None:
        """
        Save ParameterOC optimization results so they can be reloaded later.

        We split saved data into:
        (A) Arrays + metadata -> <stem>.npz
        (B) QuTiP operators   -> <stem>_qobjs/ folder via qutip.qsave()

        Expected `results` fields (minimum):
          - results["rotation_opt"] : ndarray shape (n_total,)
          - results["phase_opt"]    : ndarray shape (n_total,)
          - results["theta_opt"]    : ndarray shape (n_gauge,) or None
        """
        # ---------------------------------------------------------------------
        # 1) Resolve and create output paths
        # ---------------------------------------------------------------------
        out_path = Path(filename)
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        stem = out_path.stem
        qdir = out_path.parent / f"{stem}_qobjs"
        qdir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # 2) Pull required arrays from results
        # ---------------------------------------------------------------------
        rotation_opt = np.asarray(results["rotation_opt"], dtype=float)
        phase_opt = np.asarray(results["phase_opt"], dtype=float)
        theta_opt = results.get("theta_opt", None)

        # ---------------------------------------------------------------------
        # 3) Metadata as JSON (store inside the .npz)
        # ---------------------------------------------------------------------
        # Save enough info to reconstruct the class later.
        parameters_dict = {
            "unitary_strings": list(self.unitary_strings),
            "n_base": int(self.n_base),
            "n_c": int(self.n_c),
            "n_q": int(self.n_q),
            "dim": int(self.dim),
            "cost_type": str(self.cost_type),
            "n_gauge": int(self.n_gauge),
            "num_apply": int(self.num_apply),
            "optimize_phases": bool(self.optimize_phases),
        }
        parameters_json = json.dumps(parameters_dict)

        # ---------------------------------------------------------------------
        # 4) Write the .npz (arrays + JSON strings)
        # ---------------------------------------------------------------------
        npz_payload: Dict[str, Any] = {
            "rotation_opt": rotation_opt,
            "phase_opt": phase_opt,
            "parameters_json": np.array(parameters_json, dtype=str),        # safe string (no pickle)
        }

        if theta_opt is None:
            npz_payload["theta_opt"] = np.array([], dtype=float)
        else:
            npz_payload["theta_opt"] = np.asarray(theta_opt, dtype=float)

        np.savez_compressed(out_path, **npz_payload)

        # ---------------------------------------------------------------------
        # 5) Save QuTiP objects: target, projector, gauge ops
        # ---------------------------------------------------------------------
        def _qsave_safe(obj: Any, name: str) -> None:
            if obj is None:
                return
            if isinstance(obj, qt.Qobj):
                qt.qsave(obj, qdir / name)
                return
            if isinstance(obj, (list, tuple)) and all(isinstance(x, qt.Qobj) for x in obj):
                for i, x in enumerate(obj):
                    qt.qsave(x, qdir / f"{name}_{i:02d}")

        _qsave_safe(getattr(self, "Ut", None), "U_target")
        _qsave_safe(getattr(self, "P", None), "projector")
        _qsave_safe(getattr(self, "gauge_ops", None), "gauge_op")
        _qsave_safe(getattr(self, "a_c", None), "a_c")
        _qsave_safe(getattr(self, "a_q", None), "a_q")

# =========================================================================
# outside functions 
# =========================================================================

def load_parameters_dict(npz_file: str) -> dict:
    data = np.load(npz_file, allow_pickle=True)   # allow_pickle needed because dtype=object
    parameters_json = data["parameters_json"].item()  # 0-d object array -> Python str
    parameters_dict = json.loads(parameters_json)     # str -> dict
    return parameters_dict

def load_rotation_phase_theta(npz_file: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load an initial guess (rotation0, phase0, theta0) from a saved .npz file.

    Returns
    -------
    rotation0 : np.ndarray, shape (n_total,)
    phase0    : np.ndarray, shape (n_total,)
    theta0    : np.ndarray shape (n_gauge,) OR None
    """
    data = np.load(npz_file, allow_pickle=False)

    rotation0 = data["rotation_opt"]
    phase0 = data["phase_opt"]

    theta_arr = data["theta_opt"]
    theta0 = None if theta_arr.size == 0 else theta_arr

    return rotation0, phase0, theta0

def load_qobjs(run_npz_path: str) -> Dict[str, Any]:
    """
    Load QuTiP objects that were saved by save_results().

    Returns dict with keys: "U_target", "projector", "gauge_ops"
    """
    run_npz = Path(run_npz_path)
    if run_npz.suffix != ".npz":
        run_npz = run_npz.with_suffix(".npz")

    qdir = run_npz.parent / f"{run_npz.stem}_qobjs"
    out: Dict[str, Any] = {}

    # U_target is required for reconstruction, but keep it optional to be safe
    ut_file = qdir / "U_target.qu"
    out["U_target"] = qt.qload((qdir / "U_target")) if ut_file.exists() else None

    p_file = qdir / "projector.qu"
    out["projector"] = qt.qload((qdir / "projector")) if p_file.exists() else None

    a_c_file = qdir / "a_c.qu"
    out["a_c"] = qt.qload((qdir / "a_c")) if a_c_file.exists() else None

    a_q_file = qdir / "a_q.qu"
    out["a_q"] = qt.qload((qdir / "a_q")) if a_q_file.exists() else None

    gauge_files = sorted(qdir.glob("gauge_op_*.qu"))
    out["gauge_ops"] = [qt.qload(f.with_suffix("")) for f in gauge_files]

    return out
    
def get_unitary_from_string(a_c, a_q, unitary_str: str, rotation_param: float, phase_param: float) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """
    Get the unitary operator corresponding to a given string identifier.

    Args:
        unitary_str (str): Identifier for the unitary.
        rotation_param (float): Rotation parameter.
        phase_param (float): Phase parameter.

    Returns:
        qt.Qobj: The exponential argument of the unitary.
        qt.Qobj: The corresponding unitary operator.
        derivative of the exponential argument in the untitary with respect to the rotation param
        derivative of the exponential argument in the untitary with respect to the phase param
    """
    if unitary_str == "bs":
        # Example: Beamsplitter between cavity A and qubit. definition is experimentalist way g(a^dag b + a b^dag). Here g = pi/2 corresponds to full swap). At g = pi we are back to identity with some minus signs
        g = rotation_param # beamsplitter strength
        phase = phase_param if phase_param is not None else 0 # phase of the beamsplitter in radians

        H = g * (a_c.dag() * a_q * np.exp(1j * phase) + a_c * a_q.dag() * np.exp(-1j * phase))
        return -1j*H, (-1j * H).expm(), (-1j* (a_c.dag() * a_q * np.exp(1j * phase) + a_c * a_q.dag() * np.exp(-1j * phase))), (-1j * g * (1j*a_c.dag() * a_q * np.exp(1j * phase) - 1j* a_c * a_q.dag() * np.exp(-1j * phase)))  # exp arguement, unitary operator for the beamsplitter, derivative wrt rotation param, derivative wrt phase param 
    if unitary_str == "r":
        # Example: X-rotation on the qubit
        theta = rotation_param # rotation angle in radians. 0 to 2pi. pi/2 gives x gate and pi gives -I
        phase = phase_param if phase_param is not None else 0 # phase of the beamsplitter in radians.
        sigma_plus = sigma_plus_restricted_01_from_aq(a_q)
        H = theta * (sigma_plus*np.exp(1j*phase) + sigma_plus.dag()*np.exp(-1j*phase))
        return -1j*H, (-1j * H).expm(), (-1j * (sigma_plus*np.exp(1j*phase) + sigma_plus.dag()*np.exp(-1j*phase))), -1j*theta * (1j*sigma_plus*np.exp(1j*phase) - 1j*sigma_plus.dag()*np.exp(-1j*phase)) # exp arguement, unitary operator for the rotation, derivative wrt rotation param, derivative wrt phase param
    raise ValueError(f"Unknown unitary string: {unitary_str}")

def propagate(unitary_strings: list[str], rotation_params: np.ndarray, phase_params: np.ndarray, num_apply: int, a_c: qt.Qobj, a_q: qt.Qobj, optimize_phases: bool) -> qt.Qobj:
    """
    Public method to propogate given rotation and phase parameters through the sequence
    and return the final unitary.
    """
    itterated_unitary_strings = []
    for _rep in range(num_apply):
        itterated_unitary_strings.extend(unitary_strings)

    U_final = qt.qeye(a_c.dims[0]) # final unitary. start at identity

    for i, unitary_str in enumerate(itterated_unitary_strings):
        phase_param = float(phase_params[i]) if optimize_phases else 0.0
        _, U_i, _, _ = get_unitary_from_string(a_c=a_c, a_q=a_q, unitary_str=unitary_str, rotation_param=float(rotation_params[i]), phase_param=phase_param)

        U_final = U_i * U_final
    
    return U_final

def gauge_unitary(gauge_ops: List[qt.Qobj], theta: np.ndarray) -> qt.Qobj:
    """
    Compute gauge unitary and its derivatives:
        G(theta) = exp(+i Σ_j theta_j A_j)

    Returns:
        G (Qobj)
    """
    n_g = len(gauge_ops)
    theta = np.asarray(theta, dtype=float)                # ensure numpy array float
    if theta.shape != (n_g,):                    # shape check
        raise ValueError(f"theta must be shape {(len(gauge_ops),)}.")

    # Build exponent matrix for gauge:
    #   A_g = +i Σ theta_j Aj
    A_g = qt.qzero(gauge_ops[0].dims[0])  # start at 0
    for j in range(n_g):                          # sum contributions
        A_g += 1j * theta[j] * gauge_ops[j]             # add +i theta_j Aj
    G = A_g.expm()                                         # compute gauge unitary G(theta)
    return G  
    

