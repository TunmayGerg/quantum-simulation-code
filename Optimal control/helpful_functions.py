import numpy as np
import qutip as qt

def unitary_projected_fidelity(U_target, U_actual, P):
    return abs((U_target.dag()*U_actual*P).tr())**2 / ((P.tr())**2)

def norm(A):
    return ((A.dag()*A).tr())**0.5

def normalized_innerproduct(A,B):
    return (A.dag()*B).tr()/((A.dag()*A).tr() * (B.dag()*B).tr())**0.5

def normalized_innerproduct_projected(A,B,P):
    return (A.dag()*B*P).tr()/((A.dag()*A*P).tr() * (B.dag()*B*P).tr())**0.5

def projector_onto_states(states):
    # states: list of Qobj state vectors
    P = 0 * (states[0] * states[0].dag())   # zero operator with correct type/dims
    for psi in states:
        P += psi * psi.dag()
    return P

def sort_eigenvalues_eigenstates_by_excitation_number(evals, evecs, rdim, qdim, a_r, a_q):
    # takes in list of eigenvalues and list of eigenstates (Qobj)

    # sort evals and evecs from low to high first before sorting by excitation number
    idx_sort = np.argsort(evals) # indices that would sort evals
    evals = [evals[i] for i in idx_sort] # sort evals
    evecs = [evecs[i] for i in idx_sort] # sort evecs accordingly to match evals. 1D list and each element is a Qobj state vector

    # sort eigenstates by number of resonator and qubit excitations, relabeled as a 2d array
    # calculate <n_r> and <n_q> and map it to the closest integer
    # evecs_sorted[n_r][n_q] gives the eigenstate with n_r resonator excitations and n_q qubit excitations
    # first index is resonator excitations, second index is qubit excitations
    evecs_sorted = [[None for _ in range(qdim)] for _ in range(rdim)]
    evals_sorted = [[None for _ in range(qdim)] for _ in range(rdim)] # shifted for zero ground state energy 
    override_numbers = [] # list of (n_r, n_q) pairs which have two possible assignments
    evecs_qobj = np.zeros((rdim*qdim,rdim*qdim), dtype=np.complex128) # unitary operator which maps (n_r, n_q) standard ordered basis to cooresponding eigenvector of most similar photon number expectation
    evals_qobj = np.zeros((rdim*qdim,rdim*qdim), dtype=float) # diagonal operator which holds eigenvalues cooresponding to (n_r, n_q) basis 

    for i, evec in enumerate(evecs):
        n_r = round(qt.expect(a_r.dag() * a_r, evec).real)
        n_q = round(qt.expect(a_q.dag() * a_q, evec).real)
        if evecs_sorted[n_r][n_q] == None: # only assign if that state is not already assigned
            evecs_sorted[n_r][n_q] = evec
            evals_sorted[n_r][n_q] = np.real(evals[i])
            evecs_qobj[:, qdim*n_r + n_q] = np.array(evec.full().flatten()) # columns of matrix is eigenstates. This is e^{-iS} in dispersive transformation paper
            evals_qobj[qdim*n_r + n_q, qdim*n_r + n_q] = evals_sorted[n_r][n_q] # diagonal matrix entry is eigenvalue
        else:
            # this (n_r, n_q) pair has two possible assignments, store it for later resolution
            override_numbers.append((n_r, n_q, i))

    if evals_sorted[0][0] is not None:
        ground_energy = evals_sorted[0][0]
        for i in range(rdim):
            for j in range(qdim):
                if evals_sorted[i][j] is not None:
                    evals_sorted[i][j] -= ground_energy  # shift so vacuum energy is zero
                    
    evecs_qobj = qt.Qobj(evecs_qobj, dims=a_r.dims)
    evals_qobj = qt.Qobj(evals_qobj, dims=a_r.dims)
    return {'evecs_sorted': evecs_sorted, 'evals_sorted': evals_sorted, 'overrides': override_numbers, 'evecs_qobj': evecs_qobj, 'evals_qobj': evals_qobj} # return sorted eigenstates and eigenvalues along with override numbers as a dictionary

def dchi_H(drive_parameters, alpha):
    """
    drive_parameters
    """
    qdim = drive_parameters['N_q'] # qubit dimension
    rdim = drive_parameters['N_c'] # resonator dimension
    detuning = drive_parameters['detuning'] # qubit-resonator detuning
    gbs = drive_parameters['gbs']      # qubit-resonator coupling strength
    a_q = qt.tensor(qt.qeye(rdim), qt.destroy(qdim)) # qubit annihilation operator (resonator X qubit)
    a_r = qt.tensor(qt.destroy(rdim), qt.qeye(qdim)) # resonator annihilation operator (resonator X qubit)
    H0 = detuning * a_q.dag() * a_q + gbs * (a_q.dag() * a_r + a_q * a_r.dag()) # move to rotating frame where qubit and resonator rotate at same frequency of resonator frequency. Include Jaynes-Cummings interaction. Because both oscillators rotate at resonator frequency, JC interaction is time-independent.
    H0 += alpha/2 * a_q.dag() * a_q.dag() * a_q * a_q # add qubit anharmonicity term

    Hamiltonian_dict = {'H0': H0, 'a_r': a_r, 'a_q': a_q}

    evals, evecs = H0.eigenstates() # get eigenvalues and eigenstates

    sorted_dict = sort_eigenvalues_eigenstates_by_excitation_number(evals, evecs, rdim, qdim, a_r, a_q)
    evals_sorted = sorted_dict['evals_sorted']
    evecs_qobj = sorted_dict['evecs_qobj']

    a_r_dressed = evecs_qobj * a_r * evecs_qobj.dag() # transform a_r to eigenbasis. This is e^{-iS} a_r e^{iS}
    a_q_dressed = evecs_qobj * a_q * evecs_qobj.dag() # transform a_q to eigenbasis
    dressed_ops = {'a_r': a_r_dressed, 'a_q': a_q_dressed} # dictionary of dressed operators

    w_r_dressed = evals_sorted[1][0]-evals_sorted[0][0]  # dressed resonator frequency
    w_q_dressed = evals_sorted[0][1]-evals_sorted[0][0]  # dressed qubit frequency
    dressed_ops['w_r_dressed'] = w_r_dressed
    dressed_ops['w_q_dressed'] = w_q_dressed

    chi_dict = {}
    chi_list = []
    chi_list_numbers = [] # list of (n_r, n_r-1) pairs for calculating chi_list
    for i in range(len(evals_sorted)-1):
        if evals_sorted[i+1][0] is not None and evals_sorted[i][0] is not None and evals_sorted[i+1][1] is not None and evals_sorted[i][1] is not None: # check if states exist
            chi_list.append(evals_sorted[i+1][1]-evals_sorted[i][1]-(evals_sorted[i+1][0]-evals_sorted[i][0]))
            chi_list_numbers.append((i+1,i)) # resonator excitation numbers used for this chi
    chi_dict['chi_list'] = chi_list
    chi_dict['chi_list_numbers'] = chi_list_numbers
    
    results = {'sorted_dict': sorted_dict, 'chi_dict': chi_dict, 'Hamiltonian_dict': Hamiltonian_dict, 'dressed_operators': dressed_ops}

    return results

def sigma_plus_restricted_01_from_aq(a_q: qt.Qobj, q_index: int = -1) -> qt.Qobj:
    """
    Build a qubit raising operator σ⁺ = |1><0| on the subsystem corresponding to `a_q`,
    embedded in the full tensor Hilbert space, with the SAME dims/shape as `a_q`.

    Assumptions:
      - `a_q` is an operator (Qobj) living on a tensor product space.
      - The "qubit" subsystem is identified by `q_index` within `a_q.dims[0]`
        (default: last subsystem).

    Parameters
    ----------
    a_q:
        Full-space operator (e.g., tensor(I_a, I_b, destroy(n_q))).
    q_index:
        Which tensor factor is the qubit subsystem. Default -1 (last factor).

    Returns
    -------
    qt.Qobj
        Full-space σ⁺ operator with the same dims/shape as `a_q`.
    """
    if not isinstance(a_q, qt.Qobj):
        raise TypeError("a_q must be a qutip.Qobj.")
    if len(a_q.dims) != 2 or a_q.dims[0] != a_q.dims[1]:
        raise ValueError(f"a_q must be an operator with square dims. Got dims={a_q.dims}.")

    subdims = a_q.dims[0]  # e.g. [n_c, n_q]
    n_factors = len(subdims)

    # Normalize negative indexing
    if q_index < 0:
        q_index += n_factors
    if not (0 <= q_index < n_factors):
        raise IndexError(f"q_index out of range for dims {subdims}.")

    n_q = subdims[q_index]
    if n_q < 2:
        raise ValueError(f"Need at least 2 levels to define |1><0|. Got n_q={n_q}.")

    # Local σ⁺ on the qubit subspace: |1><0|
    ket0 = qt.basis(n_q, 0)
    ket1 = qt.basis(n_q, 1)
    sigma_plus_local = ket1 * ket0.dag()

    # Embed into the full space: I ⊗ ... ⊗ σ⁺ ⊗ ... ⊗ I
    factors = []
    for i, d in enumerate(subdims):
        factors.append(sigma_plus_local if i == q_index else qt.qeye(d))

    sigma_plus_full = qt.tensor(factors)

    return sigma_plus_full
