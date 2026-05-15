"""
QuTiP-first GRAPE (piecewise-constant controls) using SciPy L-BFGS-B.

Core idea (per time slice n):
    H_n = H0 + Σ_k u[n,k] Hk
    U_n = exp(-i dt H_n / ħ)

Total propagator (time-ordered product):
    U = U_{N-1} ... U_1 U_0

We maximize a gate fidelity (full-space or projected-subspace) using L-BFGS-B.
"""

from __future__ import annotations  # allow forward type annotations like -> GrapeLBFGS

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any  # typing helpers

from pathlib import Path
import json

import numpy as np                # numerical arrays + linear algebra glue
import qutip as qt                # QuTiP objects for quantum operators / unitaries
from scipy.optimize import minimize          # SciPy optimizer (we use L-BFGS-B)
from scipy.linalg import expm, expm_frechet  # dense matrix exponential + its Fréchet derivative

# PenaltyFn is a *type alias* (for readability) that tells us what shape a "penalty function" must have:
#   - It must be a Callable (i.e., a function) that takes two inputs:
#       1) pulses: np.ndarray
#            -> the control amplitudes u[n,k] as a real array of shape (n_steps, n_ctrl)
#               (n_steps = number of time slices, n_ctrl = number of control Hamiltonians)
#       2) theta: Optional[np.ndarray]
#            -> optional gauge parameters (shape (n_gauge,)) if you are optimizing over gauge freedom,
#               OR None if gauge freedom is not being used
#   - And it must return a 3-tuple:
#       (cost, grad_wrt_pulses, grad_wrt_theta)
#       where:
#         cost: float
#            -> the scalar penalty value added to the total objective
#               (total_cost = (1 - fidelity) + sum(penalty_costs))
#         grad_wrt_pulses: np.ndarray
#            -> gradient of that penalty cost w.r.t. pulses, same shape as pulses (n_steps, n_ctrl)
#               so we can add it directly to the GRAPE gradient
#         grad_wrt_theta: Optional[np.ndarray]
#            -> gradient of that penalty cost w.r.t. theta (shape (n_gauge,)) if theta exists,
#               OR None if theta does not exist / penalty does not depend on theta
PenaltyFn = Callable[[np.ndarray, Optional[np.ndarray]], Tuple[float, np.ndarray, Optional[np.ndarray]]]

class GrapeLBFGS:
    """
    GRAPE optimizer for unitary gate synthesis (QuTiP inputs) using SciPy L-BFGS-B.

    You provide:
      - H_drift: drift Hamiltonian H0
      - H_controls: list of control Hamiltonians Hk
      - U_target: desired target unitary Ut
      - dt, n_steps: piecewise-constant time grid

    You choose fidelity:
      - cost_type="unitary": full unitary fidelity on the whole Hilbert space
      - cost_type="projected": fidelity restricted to a subspace defined by projector P

    Optional gauge freedom:
      - gauge_ops = [A1, A2, ...] defines G(theta)=exp(+i Σ theta_j Aj)
      - gauge_side chooses U_corr = G U   (left) or U_corr = U G  (right)

    The optimizer variables are:
      - pulses u[n,k] (shape n_steps x n_ctrl)
      - optionally gauge angles theta[j] (shape n_gauge)
    """

    def __init__(
        self,
        H_drift: qt.Qobj,                      # drift Hamiltonian H0 (QuTiP operator)
        H_controls: Sequence[qt.Qobj],         # control Hamiltonians [H1, H2, ...]. (k index)
        U_target: qt.Qobj,                     # target unitary Ut. Must be same shape as H_drift even for case of projected fidelity
        dt: float,                             # time step duration Δt
        n_steps: int,                          # number of time steps N (n index)
        *, # Everything after this must be passed as a keyword argument (named), not positionally.
        cost_type: str = "unitary",            # "unitary" or "projected"
        projector: Optional[qt.Qobj] = None,   # projector P if using projected fidelity. Shape must match H_drift even if it’s a subspace projector.
        derivative: str = "frechet",           # "frechet" (exact slice derivative) or "approx"
        gauge_ops: Optional[Sequence[qt.Qobj]] = None,  # operators defining gauge unitary
        hbar: float = 1.0,                     # ħ (often set to 1)
    ):
        # ---- Store discretization parameters ----
        self.dt = float(dt)                    # ensure dt is a Python float
        self.n_steps = int(n_steps)            # ensure integer number of slices
        self.hbar = float(hbar)                # store ħ for consistency in exponent

        # ---- Store QuTiP objects (these are what the user thinks in) ----
        self.H0 = H_drift                      # drift Hamiltonian as Qobj
        self.Hc = list(H_controls)             # controls as list of Qobj
        self.Ut = U_target                     # target unitary as Qobj

        # ---- Basic dimension info ----
        self.dim = int(H_drift.shape[0])       # Hilbert dimension (assume square operator)
        self.n_ctrl = len(self.Hc)             # number of control channels

        # ---- Validate inputs are QuTiP operators and square with same dimension ----
        for op, name in [(self.H0, "H_drift"), (self.Ut, "U_target")]:
            if not isinstance(op, qt.Qobj):    # require QuTiP object
                raise TypeError(f"{name} must be a QuTiP Qobj.")
            if op.shape != (self.dim, self.dim):  # require same square dimension
                raise ValueError(f"{name} must be shape {(self.dim, self.dim)}.")
        for k, Hk in enumerate(self.Hc):       # loop over each control Hamiltonian
            if not isinstance(Hk, qt.Qobj):    # require QuTiP operator
                raise TypeError(f"H_controls[{k}] must be a QuTiP Qobj.")
            if Hk.shape != (self.dim, self.dim):  # dimension must match drift
                raise ValueError(f"H_controls[{k}] must be shape {(self.dim, self.dim)}.")
        
        # ---- Interpret and validate cost selection ----
        if cost_type not in ("unitary", "projected"):       # only these two modes
            raise ValueError("cost_type must be 'unitary' or 'projected'.")
        self.cost_type = cost_type                          # store cost mode

        # ---- Choose derivative method for slice exp() wrt control amplitude ----
        if derivative not in ("frechet", "approx"):          # exact vs approximation
            raise ValueError("derivative must be 'frechet' or 'approx'.")
        self.derivative = derivative                         # store derivative method

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

        # ---- Gauge freedom: operators Aj define G(theta)=exp(+i Σ theta_j Aj) ----
        self.gauge_ops = list(gauge_ops) if gauge_ops is not None else []  # store Aj list
        self.n_gauge = len(self.gauge_ops)                                  # number of gauge params

        # Validate gauge operators have the right shape/type.
        for j, Aj in enumerate(self.gauge_ops):                             # loop over gauge ops
            if not isinstance(Aj, qt.Qobj):                                 # must be Qobj
                raise TypeError(f"gauge_ops[{j}] must be a QuTiP Qobj.")
            if Aj.shape != (self.dim, self.dim):                            # dimension must match
                raise ValueError(f"gauge_ops[{j}] must be shape {(self.dim, self.dim)}.")

        # ---- Extra penalties (optional regularizers / constraints) ----
        self.penalties: List[PenaltyFn] = []  # holds user-added penalty functions

        # ---- For speed: cache dense numpy arrays of the operators (math is identical) ----
        # example: self.H0 is a QuTiP Qobj. Then .full() returns its dense numpy matrix (complex array). Math with numpy can be faster than with Qobj.
        self._H0 = self.H0.full()                               # dense array for drift Hamiltonian
        self._Hc = [Hk.full() for Hk in self.Hc]                # dense arrays for controls
        self._Ut_dense = self.Ut.full()                # dense target unitary (not including projection)
        self._P_dense = None if self.P is None else self.P.full()  # dense projector (or None)

    # =========================================================================
    # Penalties (optional) These are added to the cost function. They are added by the user by calling the add_..._penalty methods BEFORE optimizing
    # =========================================================================

    def add_l2_amplitude_penalty(self, lam: float) -> None:
        """
        Adds a simple amplitude penalty:
            penalty = lam * dt * Σ_{n,k} u[n,k]^2

        This discourages huge control amplitudes: reduces total power in pulses
        """
        lam = float(lam)                      # ensure float
        dt = self.dt                          # capture dt locally for closure

        # Embedded function: because it’s defined inside add_l2_amplitude_penalty, it automatically has access to the variables in that outer scope: lam, dt
        # the function carries those values with it even after add_l2_amplitude_penalty finishes. When this inner function is called later during optimization, the values of lam and dt are still available and will be those that were set when the penalty was added.

        def penalty(pulses: np.ndarray, theta: Optional[np.ndarray]):
            # penalty cost: lam * dt * sum of squares of all amplitudes
            cost = lam * dt * float(np.sum(pulses**2))
            # gradient wrt pulses: d/du (u^2) = 2u
            grad_p = 2.0 * lam * dt * pulses # matrix where entry [n,k] is derivative wrt u[n,k]
            # no dependence on theta unless you choose to add it
            grad_t = None if theta is None else np.zeros_like(theta)
            return cost, grad_p, grad_t

        self.penalties.append(penalty)        # register this penalty

    def add_smoothness_penalty(self, lam: float) -> None:
        """
        Adds a discrete smoothness penalty (discourages jagged pulses):
            penalty = lam * Σ_{n,k} (u[n+1,k] - u[n,k])^2

        This tends to make pulses more hardware-friendly.
        """
        lam = float(lam)                      # ensure float

        def penalty(pulses: np.ndarray, theta: Optional[np.ndarray]):
            diffs = pulses[1:] - pulses[:-1]  # forward difference along time index n. the end index is exclusive so :-1 goes up to including N-2 index
            cost = lam * float(np.sum(diffs**2))  # sum of squared differences (all controls)
            grad_p = np.zeros_like(pulses)    # initialize gradient array
            # For each difference (u[n+1]-u[n])^2:
            #   derivative adds -2*(u[n+1]-u[n]) to u[n]
            #   and +2*(u[n+1]-u[n]) to u[n+1]
            grad_p[:-1] -= 2.0 * lam * diffs # last time slice has no u[n+1], so we stop at N-2 index
            grad_p[1:]  += 2.0 * lam * diffs # first time slice has no u[n-1], so we start at index 1
            grad_t = None if theta is None else np.zeros_like(theta)  # no theta dependence here
            return cost, grad_p, grad_t

        self.penalties.append(penalty)        # register this penalty

    def add_custom_penalty(self, penalty_fn: PenaltyFn) -> None:
        """
        Add any custom penalty term with analytic gradients.

        penalty_fn(pulses, theta) must return:
          cost (float),
          grad_pulses shape (n_steps, n_ctrl),
          grad_theta shape (n_gauge,) or None.
        """
        self.penalties.append(penalty_fn)     # just store it; we call it inside cost_and_grad()

    # =========================================================================
    # Packing/unpacking variables for SciPy
    # 1D vector x = [u[0,0], u[0,1], u[0,2], u[0,3] ..., u[1,0], u[1,1], ... , theta[0], theta[1], ...]
    # 2D matrix elements are u[n,k]
    # =========================================================================

    def _pack(self, pulses: np.ndarray, theta: Optional[np.ndarray]) -> np.ndarray:
        """
        Flatten pulses (and optionally theta) into a 1D vector x for SciPy.
        """
        x = np.asarray(pulses, dtype=float).reshape(-1)   # flatten pulses into 1D float array
        if self.n_gauge > 0:                              # if we are also optimizing gauge angles
            if theta is None:                             # if theta not provided, assume zeros
                theta = np.zeros(self.n_gauge, dtype=float)
            theta = np.asarray(theta, dtype=float).reshape(-1)  # ensure 1D float array
            x = np.concatenate([x, theta])                # concatenate pulses and theta
        return x                                          # return flat optimization vector

    def _unpack(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Inverse of _pack(): map 1D vector x back to (pulses, theta).
        """
        x = np.asarray(x, dtype=float)                    # ensure float numpy vector
        n_p = self.n_steps * self.n_ctrl                  # number of pulse parameters total
        pulses = x[:n_p].reshape(self.n_steps, self.n_ctrl)  # first part is pulses (time x control)
        theta = None                                      # default: no theta
        if self.n_gauge > 0:                              # if gauge parameters exist
            theta = x[n_p:].reshape(self.n_gauge)         # remaining entries are theta
        return pulses, theta                              # return structured variables

    def _build_bounds(
        self,
        pulse_bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
        theta_bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None, # The “inner Optional” is there because SciPy bounds are allowed to be partially specified: for each variable you can bound the lower, the upper, both, or neither.
    ) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
        """
        Build bounds list for L-BFGS-B.

        pulse_bounds: length n_ctrl, bounds (min,max). These will be repeated at each time step.
        theta_bounds: length n_gauge, bounds for gauge angles.

        Returns a per-variable bounds list of length:
            n_steps*n_ctrl (+ n_gauge if used)
        """
        if pulse_bounds is None and theta_bounds is None:    # if user gave no bounds
            return None                                      # SciPy interprets None as unbounded

        bounds: List[Tuple[Optional[float], Optional[float]]] = []  # create empty bounds list

        # ---- pulse bounds ----
        if pulse_bounds is None:                              # no bounds for pulses only
            bounds.extend([(None, None)] * (self.n_steps * self.n_ctrl))  # list multiplication: creates n_steps*n_ctrl list entries of (None,None)
        else:
            if len(pulse_bounds) != self.n_ctrl:              # must match number of controls
                raise ValueError("pulse_bounds must have length n_ctrl.")
            for _ in range(self.n_steps):                     # repeat per-control bounds for each time slice
                bounds.extend(list(pulse_bounds))             # append bounds for all controls

        # ---- theta bounds ----
        if self.n_gauge > 0:                                  # only relevant if gauge used
            if theta_bounds is None:                          # no bounds for theta
                bounds.extend([(None, None)] * self.n_gauge)  # unbounded gauge angles
            else:
                if len(theta_bounds) != self.n_gauge:         # must match number of gauge params
                    raise ValueError("theta_bounds must have length n_gauge.")
                bounds.extend(list(theta_bounds))             # append theta bounds

        return bounds                                         # return per-variable bounds list

    # =========================================================================
    # GRAPE propagation and gauge
    # =========================================================================

    def _propagate(self, pulses: np.ndarray, post_bool: Optional[bool] = True)  -> Dict[str, Any]:
        """
        Build time-slice propagators U_n and cached products for GRAPE gradients given pulses.

        Returns a cache dict with:
          U_list[n] : U_n = exp(-i dt H_n / ħ)
          A_list[n] : A_n = -i dt H_n / ħ  (the exponent fed into expm)
          prefix[n] : U_{n-1} ... U_0 with prefix[0]=I. Length is n_steps + 1
          post[n]   : U_{N-1} ... U_{n+1} with post[N-1]=I. Length is n_steps
          U_final   : U_{N-1} ... U_0
        """
        pulses = np.asarray(pulses, dtype=float)              # ensure pulses are float array
        if pulses.shape != (self.n_steps, self.n_ctrl):       # check expected shape
            raise ValueError(f"pulses must be shape {(self.n_steps, self.n_ctrl)}.")

        U_list: List[np.ndarray] = []                         # will store each slice propagator
        A_list: List[np.ndarray] = []                         # will store each slice exponent A_n

        # ---- Build each slice unitary ----
        for n in range(self.n_steps):                         # loop over time slices
            Hn = self._H0.copy()                              # start with drift Hamiltonian
            for k in range(self.n_ctrl):                      # add each control Hamiltonian
                Hn += pulses[n, k] * self._Hc[k]              # Hn += u[n,k] * Hk
            A = (-1j * self.dt / self.hbar) * Hn              # A_n = -i dt H_n / ħ
            Un = expm(A)                                      # U_n = exp(A_n) = exp(-i dt H_n / ħ)
            A_list.append(A)                                  # store A_n for Frechet derivative later
            U_list.append(Un)                                  # store U_n

        I = np.eye(self.dim, dtype=np.complex128)             # identity operator I (dense)

        # ---- prefix products: prefix[n] = U_{n-1} ... U_0 ----
        prefix: List[np.ndarray] = [I]                        # prefix[0]=I by definition
        for n in range(self.n_steps):                         # build cumulatively
            prefix.append(U_list[n] @ prefix[n])              # prefix[n+1] = U_n * prefix[n]
        U_final = prefix[self.n_steps]                        # final unitary is prefix at end


        # ---- post products: post[n] = U_{N-1} ... U_{n+1} ----
        post: List[np.ndarray] = [None] * self.n_steps         # allocate list
        post[self.n_steps - 1] = I                             # post[N-1]=I (nothing after last slice)

        if post_bool:
            for n in range(self.n_steps - 2, -1, -1):              # fill backwards. start, stop, step. stop is -1 to include 0 index as last iteration. step is -1
                post[n] = post[n + 1] @ U_list[n + 1]              # post[n] = post[n+1] * U_{n+1}

        return {                                               # return everything needed for gradients
            "U_list": U_list,
            "A_list": A_list,
            "prefix": prefix,
            "post": post,
            "U_final": U_final,
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
            d = float(self.dim)                                  # full dimension
        elif self.cost_type == "projected":                      # projected fidelity
            d = float(np.real(np.trace(self._P_dense)))                   # effective dimension = Tr(P)
        f = float((abs(c) ** 2) / (d * d))
        print(f"Fidelity: {f}")
        return f                    # normalize by d^2

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
    
    # =========================================================================
    # Objective and gradient for SciPy
    # =========================================================================

    def cost(self, pulses: np.ndarray, theta: Optional[np.ndarray] = None) -> float:
        """
        Compute cost for L-BFGS-B evaluated at pulses and theta.
        pulses is in array format 2D while theta is 1D

        cost = 1 - fidelity + Σ penalties
        """
        # ---- Run forward propagation and build cached products ----
        cache = self._propagate(pulses, False)                   # get U_list, prefix, post, U_final, and A_list
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

        # ---- Add penalties (if any) ----
        for pen in self.penalties:                                  # loop over all penalty terms
            p_cost, _, _ = pen(                               # call penalty callback
                np.asarray(pulses, float),                          # pass pulses as float array
                None if self.n_gauge == 0 else np.asarray(theta, float),  # pass theta if used
            )
            cost += float(p_cost)                                   # add penalty to total cost
        return cost                                              # return cost only

    def grad(self, pulses: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradient for L-BFGS-B evaluated at pulses and theta.
        pulses is in array format 2D while theta is 1D

        gradient is flattened to match SciPy variable vector:
            grad = [∂cost/∂u[0,0], ∂cost/∂u[0,1], ∂cost/∂u[0,2], ..., ∂cost/∂u[1,0], ∂cost/∂u[1,1], ..., ∂cost/∂u[N-1,K-1], ∂cost/∂theta[0], ...]
        """
        # ---- Run forward propagation and build cached products ----
        cache = self._propagate(pulses)                          # get U_list, prefix, post, U_final, and A_list
        U_final = cache["U_final"]                               # achieved unitary without gauge
        U_list = cache["U_list"]                                 # list of slice unitaries
        A_list = cache["A_list"]                                 # list of slice exponent matrices A_n
        prefix = cache["prefix"]                                 # prefix products for GRAPE
        post = cache["post"]                                     # post products for GRAPE# post products for GRAPE

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
        grad_p = np.zeros((self.n_steps, self.n_ctrl), dtype=float)   # gradient wrt pulses u[n,k] in array format
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
        for n in range(self.n_steps):                             # for each time slice
            pre = prefix[n]                                       # U_{n-1}...U_0
            postn = post[n]                                       # U_{N-1}...U_{n+1}

            for k in range(self.n_ctrl):                          # for each control Hamiltonian Hk
                # Compute dU_n/du[n,k] for this slice.
                if self.derivative == "approx":
                    # Small-dt / commuting approximation:
                    # dU_n/du ≈ (-i dt Hk / ħ) U_n
                    dA = (-1j * self.dt / self.hbar) * self._Hc[k]  # derivative of exponent wrt u[n,k]
                    dUn = dA @ U_list[n]                            # approximate derivative of expm
                else:
                    # Exact slice derivative via Fréchet derivative:
                    #   dUn = expm_frechet(A_n, dA)
                    An = A_list[n]                                   # A_n = -i dt H_n / ħ
                    dA = (-1j * self.dt / self.hbar) * self._Hc[k]   # dA/du = -i dt Hk / ħ
                    dUn = expm_frechet(An, dA, compute_expm=False)   # exact d(exp(A))/du

                # Insert the local derivative into the full time-ordered product:
                #   U = U_{N-1}...U_{n+1} * U_n * U_{n-1}...U_0
                # so
                #   dU = post[n] * dU_n * prefix[n]
                dUfinal = postn @ dUn @ pre                         # derivative of U_final wrt u[n,k]. we don't use prefix[n_steps] because that is U_final itself

                # Apply gauge mapping to get dU_corr.
                if self.n_gauge == 0:                               # no gauge -> same derivative
                    dUcorr = dUfinal
                else:
                    dUcorr = G @ dUfinal                        # gauge multiplies derivative on left U_corr = G U

                dc = dc_from_dUcorr(dUcorr)                         # compute overlap differential dc
                dF = self._dF_from_dc(c, dc)                        # convert dc -> dF
                grad_p[n, k] = -dF                                  # cost=1-F => d(cost)=-dF. derivative wrt u[n,k].

        # ---- Gauge gradients: differentiate U_corr wrt theta_j ----
        if self.n_gauge > 0:
            for j in range(self.n_gauge):                           # for each gauge parameter theta_j
                dG = dG_list[j]                                     # dG/dtheta_j
                # If U_corr = G U, then dU_corr = (dG) U
                dUcorr = dG @ U_final

                dc = dc_from_dUcorr(dUcorr)                         # differential of overlap from gauge
                dF = self._dF_from_dc(c, dc)                        # convert to fidelity differential
                grad_t[j] = -dF                                     # cost=1-F => gradient is -dF

        # ---- Add penalties (if any) ----
        for pen in self.penalties:                                  # loop over all penalty terms
            _, p_gp, p_gt = pen(                               # call penalty callback
                np.asarray(pulses, float),                          # pass pulses as float array
                None if self.n_gauge == 0 else np.asarray(theta, float),  # pass theta if used
            )
            grad_p += p_gp                                          # add penalty gradient wrt pulses
            if self.n_gauge > 0 and grad_t is not None and p_gt is not None:
                grad_t += p_gt                                      # add penalty gradient wrt theta

        grad_flat = self._pack(grad_p, grad_t)                      # flatten gradient for SciPy
        return grad_flat                               # return cost and gradient vector

    # =========================================================================
    # Convenience utilities: unitary and fidelity
    # =========================================================================

    def unitary(self, pulses: np.ndarray, theta: Optional[np.ndarray] = None) -> qt.Qobj:
        """
        Compute the final corrected unitary U_corr as a QuTiP Qobj (includes gauge if enabled).
        """
        cache = self._propagate(pulses, False)                             # propagate to get U_final
        U_final = cache["U_final"]                                  # dense final unitary

        G, _ = self._gauge_unitary(theta, False)                           # gauge unitary (ignore derivatives)
        if self.n_gauge == 0:                                       # if no gauge
            U_corr = U_final                                        # corrected = final
        else:
            U_corr = G @ U_final                                  # apply gauge

        return qt.Qobj(U_corr, dims=self.H0.dims)                   # wrap dense matrix back into Qobj

    def fidelity(self, pulses: np.ndarray, theta: Optional[np.ndarray] = None) -> float:
        """
        Compute the configured fidelity F for given pulses (and gauge angles).
        """
        U_corr = self.unitary(pulses, theta).full()                 # compute corrected unitary, get dense matrix
        c = self._overlap(U_corr)                                   # overlap with target (maybe projected)
        return self._fidelity(c)                                    # return normalized fidelity

    # =========================================================================
    # Optimization driver (SciPy L-BFGS-B)
    # =========================================================================

    def optimize(
        self,
        pulses0: Optional[np.ndarray] = None,                       # initial guess for pulses as array
        theta0: Optional[np.ndarray] = None,                        # initial guess for gauge angles
        *,
        maxiter: int = 200,                                         # max L-BFGS iterations
        pulse_bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,  # per-control bounds
        theta_bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,  # gauge bounds
        scipy_options: Optional[Dict[str, Any]] = None,             # extra scipy.minimize options
        store_history: bool = True,                                 # store fidelity trace over iterations
    ) -> Dict[str, Any]:
        """
        Run L-BFGS-B and return:
          {
            "result": OptimizeResult,
            "pulses_opt": pulses array,
            "theta_opt": theta array or None,
            "history": list of dicts (if store_history)
          }
        """
        # ---- Default initial pulses ----
        if pulses0 is None:                                         # if user didn't provide initial guess
            pulses0 = np.zeros((self.n_steps, self.n_ctrl), dtype=float)  # start from zero controls
        pulses0 = np.asarray(pulses0, dtype=float)                  # ensure numpy float array
        if pulses0.shape != (self.n_steps, self.n_ctrl):            # verify shape
            raise ValueError(f"pulses0 must be shape {(self.n_steps, self.n_ctrl)}.")

        # ---- Default initial gauge angles ----
        if self.n_gauge > 0:                                        # if gauge parameters exist
            if theta0 is None:                                      # no initial guess given
                theta0 = np.zeros((self.n_gauge,), dtype=float)     # start at zero gauge
            theta0 = np.asarray(theta0, dtype=float)                # ensure numpy float
            if theta0.shape != (self.n_gauge,):                     # verify shape
                raise ValueError(f"theta0 must be shape {(self.n_gauge,)}.")
        else:
            theta0 = None                                           # no theta variables if gauge disabled

        x0 = self._pack(pulses0, theta0)                            # flatten initial parameters
        bounds = self._build_bounds(pulse_bounds=pulse_bounds, theta_bounds=theta_bounds)  # build L-BFGS bounds

        history: List[Dict[str, Any]] = []                          # will store progress if requested

        # SciPy expects:
        #   fun(x) -> scalar cost
        #   jac(x) -> gradient vector
        def fun(x: np.ndarray) -> float:
            pulses, theta = self._unpack(x)                         # map x back to (pulses, theta)
            c = self.cost(pulses, theta)                # compute cost (and grad, ignored)
            return float(c)                                         # return scalar cost

        def jac(x: np.ndarray) -> np.ndarray:
            pulses, theta = self._unpack(x)                         # map x back to (pulses, theta)
            g = self.grad(pulses, theta)                # compute gradient vector
            return g                                                # return gradient for L-BFGS

        # Callback runs after each iteration (useful for logging).
        def callback(xk: np.ndarray) -> None:
            if not store_history:                                   # if user doesn't want history
                return
            pulses, theta = self._unpack(xk)                        # unpack current iterate
            history.append({"pulses": pulses, "theta": theta})                  # store it

        # Options for SciPy optimizer
        options = {"maxiter": int(maxiter)}                         # set iteration limit
        if scipy_options:                                           # allow user overrides
            options.update(dict(scipy_options))                     # merge dictionaries

        # Run L-BFGS-B optimization.
        res = minimize(
            fun=fun,                                                # objective function
            x0=x0,                                                  # initial guess
            jac=jac,                                                # analytic gradient
            method="L-BFGS-B",                                      # bounded quasi-Newton
            bounds=bounds,                                          # variable bounds
            callback=callback,                                      # optional logging hook
            options=options,                                        # SciPy control options
        )

        pulses_opt, theta_opt = self._unpack(res.x)                 # unpack optimized variables
        return {                                                    # return everything user likely wants
            "result": res,
            "pulses_opt": pulses_opt,
            "theta_opt": theta_opt,
            "history": history,
        }
    

    def save_results(self, filename: str, results: Dict[str, Any]) -> None:
        """
        Save GRAPE optimization results so they can be reloaded later.

        We deliberately split the saved data into TWO parts because they have different “best” formats:

        (A) Arrays / numbers / metadata  ->  NumPy .npz (compressed, fast, portable)
        (B) QuTiP Qobj operators         ->  qutip.qsave() files (preserves dims/type cleanly)

        Files created:

        1) <stem>.npz
            - pulses_opt, theta_opt
            - parameters_json: a JSON string of the run configuration

        2) <stem>_qobjs/ folder
            - H_drift.qu
            - H_control_00.qu, H_control_01.qu, ...
            - U_target.qu
            - projector.qu (optional)
            - gauge_op_00.qu, gauge_op_01.qu, ... (optional)

        Expected `results` fields (minimum):
        - results["pulses_opt"] : ndarray of shape (n_steps, n_ctrl)
        - results["theta_opt"]  : ndarray (n_gauge,) or None

        Note:
        - We do not attempt to store *every* Python object in `results` (like callbacks, functions, etc.).
            Only arrays + small metadata are saved in npz. Operators are saved as Qobj files.
        """

        # -------------------------------------------------------------------------
        # 1) Resolve and create output paths
        # -------------------------------------------------------------------------

        # Convert filename to a Path object so we can easily manipulate suffixes,
        # parents, etc. Path is nicer than string concatenation.
        out_path = Path(filename)

        # Ensure the main file ends with ".npz".
        # If user passes "run1" we will save "run1.npz".
        # If they pass "run1.npy" we'll change to "run1.npz".
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")

        # Create parent directories if they don’t exist (e.g. "results/").
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # We'll name the Qobj folder based on the file stem:
        #   results/run1.npz  ->  results/run1_qobjs/
        stem = out_path.stem
        qdir = out_path.parent / f"{stem}_qobjs"
        qdir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------------
        # 2) Pull required arrays from `results`
        # -------------------------------------------------------------------------

        # Convert to numpy arrays explicitly.
        # This avoids issues if something is passed as a list or a view.
        pulses_opt = np.asarray(results["pulses_opt"])   # shape (n_steps, n_ctrl)

        # theta_opt can be None (if no gauge parameters) or an array (n_gauge,)
        theta_opt = results.get("theta_opt", None)

        # -------------------------------------------------------------------------
        # 3) Save run parameters as JSON (inside the .npz)
        # -------------------------------------------------------------------------

        # Store the key numerical settings of the optimizer so future-you remembers
        # how the run was configured.
        #
        # JSON is nice because:
        #  - human-readable
        #  - portable
        #  - doesn’t rely on Python pickling
        parameters_dict = {
            "n_steps": getattr(self, "n_steps", None),
            "n_ctrl": getattr(self, "n_ctrl", None),
            "dt": getattr(self, "dt", None),
            "hbar": getattr(self, "hbar", None),
            "cost_type": getattr(self, "cost_type", None),
            "derivative": getattr(self, "derivative", None),
            "n_gauge": getattr(self, "n_gauge", 0),
        }

        # Convert dictionary -> JSON string
        parameters_json = json.dumps(parameters_dict)

        # -------------------------------------------------------------------------
        # 6) Pack and write the .npz (arrays + JSON metadata)
        # -------------------------------------------------------------------------

        # np.savez_compressed expects key=value pairs.
        # Every value should be array-like.
        npz_payload: Dict[str, Any] = {
            "pulses_opt": pulses_opt,             # final pulses
            "parameters_json": np.array(parameters_json, dtype=object),  # JSON string as 0-d object array
        }

        # theta_opt: if it’s None, we store an empty float array so loading code
        # can check "size == 0" and interpret that as None.
        if theta_opt is None:
            npz_payload["theta_opt"] = np.array([], dtype=float)
        else:
            npz_payload["theta_opt"] = np.asarray(theta_opt, dtype=float)

        # Write compressed .npz file.
        np.savez_compressed(out_path, **npz_payload)

        # -------------------------------------------------------------------------
        # 7) Save QuTiP Qobj operators using qutip.qsave()
        # -------------------------------------------------------------------------

        def _qsave_safe(obj: Any, name: str) -> None:
            """
            Save a Qobj OR list/tuple of Qobj's to the qdir folder.

            - If obj is None -> do nothing
            - If obj is a single Qobj -> save to "<name>.qu"
            - If obj is a list/tuple of Qobj -> save each element to "<name>_00.qu", "<name>_01.qu", ...

            This "best-effort" behavior means save_results won't crash just because
            (say) projector wasn't used in a run.
            """
            if obj is None:
                return

            # Single operator case
            if isinstance(obj, qt.Qobj):
                qt.qsave(obj, qdir / f"{name}")
                return

            # List-of-operators case
            if isinstance(obj, (list, tuple)) and all(isinstance(x, qt.Qobj) for x in obj):
                for i, x in enumerate(obj):
                    qt.qsave(x, qdir / f"{name}_{i:02d}")

        # Save the core objects if they exist as attributes on the optimizer object.
        # These names match typical GRAPE class members.
        _qsave_safe(getattr(self, "H0", None), "H_drift")
        _qsave_safe(getattr(self, "Hc", None), "H_control")
        _qsave_safe(getattr(self, "Ut", None), "U_target")

        # Optional objects
        _qsave_safe(getattr(self, "P", None), "projector")
        _qsave_safe(getattr(self, "gauge_ops", None), "gauge_op")

# =========================================================================
# functions outside the Grape class
# =========================================================================

def load_parameters_dict(npz_file: str) -> dict:
    data = np.load(npz_file, allow_pickle=True)   # allow_pickle needed because dtype=object
    parameters_json = data["parameters_json"].item()  # 0-d object array -> Python str
    parameters_dict = json.loads(parameters_json)     # str -> dict
    return parameters_dict

def load_pulses_theta(npz_file: str):
    """
    Load an initial guess (pulses0, theta0) from a saved .npz file.

    Assumptions based on your save_results():
    - pulses_opt was saved under key "pulses_opt"
    - theta_opt was saved under key "theta_opt"
        and if theta_opt was None, you saved an *empty* float array instead.

    Returns:
    pulses0 : np.ndarray, shape (n_steps, n_ctrl)
    theta0  : np.ndarray shape (n_gauge,) OR None
    """

    # np.load reads a .npz (zip archive of arrays).
    #
    # IMPORTANT:
    #  - If your .npz contains ANY "object dtype" arrays (dtype=object),
    #    NumPy requires allow_pickle=True to load them.
    #  - In your saver, you stored "parameters_json" as dtype=object, so you
    #    may need allow_pickle=True *even if you don't use parameters_json*.
    #
    # If you remove object arrays from saving (see section below), you can set allow_pickle=False.
    data = np.load(npz_file, allow_pickle=True)

    # pulses_opt should be a numeric ndarray with shape (n_steps, n_ctrl).
    pulses0 = data["pulses_opt"]

    # theta_opt was saved as:
    #   - empty float array if theta_opt was None
    #   - float array of length n_gauge otherwise
    theta_arr = data["theta_opt"]
    theta0 = None if theta_arr.size == 0 else theta_arr

    return pulses0, theta0

def propagate(H0: qt.Qobj, Hc: List[qt.Qobj], dt: float, pulses: np.ndarray, hbar: Optional[float] = 1.0)-> qt.Qobj:
    """
    Build time-slice propagators U_n and cached products for GRAPE gradients given pulses. Works in qobj

    Returns a cache dict with:
        U_list[n] : U_n = exp(-i dt H_n / ħ)
        A_list[n] : A_n = -i dt H_n / ħ  (the exponent fed into expm)
        prefix[n] : U_{n-1} ... U_0 with prefix[0]=I. Length is n_steps + 1
        U_final   : U_{N-1} ... U_0
    """
    pulses = np.asarray(pulses, dtype=float)              # ensure pulses are float array
    U_final = qt.qeye(H0.dims[0]) # final unitary. start at identity

    n_steps = pulses.shape[0]                             # number of time slices
    # ---- Build each slice unitary ----
    for n in range(n_steps):                         # loop over time slices
        Hn = H0.copy()                              # start with drift Hamiltonian
        for k in range(len(Hc)):                    # add each control Hamiltonian
            Hn += pulses[n, k] * Hc[k]              # Hn += u[n,k] * Hk
        A = (-1j * dt / hbar) * Hn                   # A_n = -i dt H_n / ħ
        Un = A.expm()                                 # U_n = exp(A_n) = exp(-i dt H_n / ħ)
        U_final = Un * U_final                       # store U_n

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

def load_qobjs(run_npz_path: str):
    run_npz = Path(run_npz_path)
    if run_npz.suffix != ".npz":
        run_npz = run_npz.with_suffix(".npz")

    qdir = run_npz.parent / f"{run_npz.stem}_qobjs"

    out = {}

    # Saved as "H_drift" -> file is "H_drift.qu"
    out["H_drift"] = qt.qload(qdir / "H_drift")

    # Saved as "H_control_00", "H_control_01", ... -> files end with .qu
    ctrl_files = sorted(qdir.glob("H_control_*.qu"))
    out["H_controls"] = [qt.qload(f.with_suffix("")) for f in ctrl_files]

    # Optional saved objects
    ut_file = qdir / "U_target.qu"
    out["U_target"] = qt.qload(ut_file.with_suffix("")) if ut_file.exists() else None

    p_file = qdir / "projector.qu"
    out["projector"] = qt.qload(p_file.with_suffix("")) if p_file.exists() else None

    gauge_files = sorted(qdir.glob("gauge_op_*.qu"))
    out["gauge_ops"] = [qt.qload(f.with_suffix("")) for f in gauge_files]

    return out
