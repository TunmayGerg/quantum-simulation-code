"""Full fixed-photon-input gate fidelity for the resonant Z2 lattice.

The INPUT space has d = 8 * binomial(N + 3, 3) states. Propagation includes
all four-cavity occupations with total photon number <= photon_cutoff, and
all eight transmon states. This compact basis retains photon leakage without
allocating the mostly unused tensor product of four cavity cutoffs.

One train conserves n_left + n_right + qubit_excitation. Its small matrices
are multiplied once, then applied to the corresponding blocks of the lattice.
No projection onto the input photon number occurs between gates or steps.
"""
from functools import lru_cache
import hashlib
import json
from math import comb
import os
from pathlib import Path

import numpy as np
from scipy.linalg import expm


QUBIT_X = np.array([[0, 1], [1, 0]], complex)
QUBIT_HADAMARD = np.array([[1, 1], [1, -1]], float) / np.sqrt(2)
THREE_HADAMARDS = np.kron(np.kron(QUBIT_HADAMARD, QUBIT_HADAMARD), QUBIT_HADAMARD)


@lru_cache(maxsize=None)
def photon_configurations(total):
    """All occupations of exactly total photons on four cavities."""
    return tuple((a, b, c, total-a-b-c) for a in range(total+1)
                 for b in range(total-a+1) for c in range(total-a-b+1))


def gauss_codes(occupations):
    """Four Gauss eigenvalues as a four-bit code, in the qubit X basis."""
    parity = 1 - 2 * (np.asarray(occupations) % 2)
    codes = np.zeros((8, len(occupations)), np.int8)
    for bits in range(8):
        x = [1 - 2 * ((bits >> (2-link)) & 1) for link in range(3)]
        for site in range(4):
            eigenvalue = parity[:, site].copy()
            if site > 0:
                eigenvalue *= x[site-1]
            if site < 3:
                eigenvalue *= x[site]
            codes[bits] |= ((eigenvalue < 0).astype(np.int8) << site)
    return codes


@lru_cache(maxsize=24)
def exact_unitary(total_photons, J_t, h_over_J):
    """exp(-i H T) on exactly N photons, in the Fock / qubit-X basis.

    Here h X is diagonal, and Z flips a qubit X eigenvalue when a photon hops.
    The eight conserved Gauss sectors each have binomial(N+3,3) states.
    """
    occupations = photon_configurations(total_photons)
    count = len(occupations)
    index = {occupation: i for i, occupation in enumerate(occupations)}
    hamiltonian = np.zeros((8*count, 8*count), float)
    for bits in range(8):
        x_sum = sum(1-2*((bits >> (2-link)) & 1) for link in range(3))
        for column, occupation in enumerate(occupations):
            source = bits*count+column
            hamiltonian[source, source] = h_over_J*x_sum
            for link in range(3):
                flipped = bits ^ (1 << (2-link))
                for origin, destination in ((link, link+1), (link+1, link)):
                    if occupation[origin] > 0:
                        moved = list(occupation)
                        moved[origin] -= 1
                        moved[destination] += 1
                        target = flipped*count+index[tuple(moved)]
                        hamiltonian[target, source] += np.sqrt(occupation[origin]*(occupation[destination]+1))
    assert np.allclose(hamiltonian, hamiltonian.T, rtol=0, atol=1e-14)
    codes = gauss_codes(occupations).ravel()
    result = np.zeros_like(hamiltonian, dtype=complex)
    for code in np.unique(codes):
        sector = np.flatnonzero(codes == code)
        assert len(sector) == count
        result[np.ix_(sector, sector)] = expm(-1j*J_t*hamiltonian[np.ix_(sector, sector)])
    assert result.shape == (8*comb(total_photons+3, 3),)*2
    return result


def train_blocks(pulses, max_excitation, flip_a=False):
    """One train's unitary in each local excitation block (q, n_a, n_b)."""
    blocks = []
    for excitation in range(max_excitation+1):
        local = [(q, a, excitation-q-a) for q in (0, 1) if q <= excitation
                 for a in range(excitation-q+1)]
        index = {state: i for i, state in enumerate(local)}
        unitary = np.eye(len(local), dtype=complex)
        for leg, area, phase in pulses:
            if leg not in ("A", "B"):
                role = {"frame q": 0, "frame a": 1, "frame b": 2}[leg]
                unitary *= np.exp(1j*phase*np.array([state[role] for state in local]))[:, None]
                continue
            phase += np.pi if flip_a and leg == "A" else 0
            cavity = 1 if leg == "A" else 2
            for ground, state in enumerate(local):
                if state[0] != 0 or state[cavity] == 0:
                    continue
                paired = list(state)
                paired[0] = 1
                paired[cavity] -= 1
                excited = index[tuple(paired)]
                cosine = np.cos(area*np.sqrt(state[cavity]))
                sine = np.sin(area*np.sqrt(state[cavity]))
                g_row, e_row = unitary[ground].copy(), unitary[excited].copy()
                unitary[ground] = cosine*g_row-1j*np.exp(-1j*phase)*sine*e_row
                unitary[excited] = cosine*e_row-1j*np.exp(1j*phase)*sine*g_row
        blocks.append(unitary)
    return blocks


class PhotonSpace:
    """Compact propagation space and the indices of each link's small blocks."""

    def __init__(self, photon_cutoff):
        self.cutoff = photon_cutoff
        self.occupations = tuple(occupation for total in range(photon_cutoff+1)
                                 for occupation in photon_configurations(total))
        self.photon_index = {occupation: i for i, occupation in enumerate(self.occupations)}
        self.count = len(self.occupations)
        self.dimension = 8*self.count
        self.total = np.sum(self.occupations, axis=1)
        self.gauss = gauss_codes(self.occupations).ravel()
        self.link_indices = []
        for link in range(3):
            spectators = [site for site in range(4) if site not in (link, link+1)]
            blocks = []
            for excitation in range(photon_cutoff+1):
                local = [(q, a, excitation-q-a) for q in (0, 1) if q <= excitation
                         for a in range(excitation-q+1)]
                columns = []
                for other_bits in range(8):
                    if other_bits & (1 << (2-link)):
                        continue
                    for a in range(photon_cutoff-excitation+1):
                        for b in range(photon_cutoff-excitation-a+1):
                            column = []
                            for q, left, right in local:
                                occupation = [0]*4
                                occupation[link], occupation[link+1] = left, right
                                occupation[spectators[0]], occupation[spectators[1]] = a, b
                                bits = other_bits | (q << (2-link))
                                column.append(bits*self.count+self.photon_index[tuple(occupation)])
                            columns.append(column)
                blocks.append(np.asarray(columns, dtype=np.int64).T)
            self.link_indices.append(blocks)
        # With total photons already at the cutoff, an excited qubit cannot emit
        # another photon. Those unmatched states have identity JC evolution.

    def rotate(self, states, link, angle):
        tensor = states.reshape(8, self.count, -1)
        cosine, sine = np.cos(angle/2), np.sin(angle/2)
        mask = 1 << (2-link)
        for bits in range(8):
            if bits & mask:
                continue
            ground, excited = tensor[bits].copy(), tensor[bits | mask].copy()
            tensor[bits] = cosine*ground-1j*sine*excited
            tensor[bits | mask] = cosine*excited-1j*sine*ground

    def hop(self, states, link, blocks):
        batch = states.shape[1]
        for indices, unitary in zip(self.link_indices[link], blocks):
            packed = states[indices].reshape(len(indices), -1)
            evolved = unitary @ packed
            states[indices] = evolved.reshape(*indices.shape, batch)


@lru_cache(maxsize=3)
def propagation_space(photon_cutoff):
    return PhotonSpace(photon_cutoff)


def at_cutoff(job, photon_cutoff, batch_size=32):
    """Trace overlap, number loss and Gauss loss for ALL d input states.

    Inputs are Fock states times all eight qubit X eigenstates: a complete
    orthonormal basis, so the trace fidelity is independent of that choice.
    Gauss loss compares every output to that input's own four Gauss eigenvalues.
    """
    total = int(job.get("total_photons", 8))
    if total < 0 or photon_cutoff < total or batch_size < 1:
        raise ValueError("need N >= 0, photon_cutoff >= N and positive batch_size")
    space = propagation_space(photon_cutoff)
    inputs = photon_configurations(total)
    photon_indices = np.array([space.photon_index[occupation] for occupation in inputs])
    count = len(inputs)
    dimension = 8*count
    ideal = exact_unitary(total, float(job["J_t"]), float(job["h_over_J"]))
    input_codes = gauss_codes(inputs).ravel()
    per_link = job.get("trains_per_link") or {link: (job["train_full"], job["train_half"]) for link in range(3)}
    per_link = {int(link): trains for link, trains in per_link.items()}
    if any(isinstance(train, str) for trains in per_link.values() for train in trains if train is not None):
        raise ValueError("at_cutoff requires explicit pulse trains; use exact_trotter_metrics for exact hops")
    gates = {}
    order = int(job.get("order", 2))
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    for link, trains in per_link.items():
        for half in ((False, True) if order == 2 else (False,)):
            if any(leg not in ("A", "B") for leg, _, _ in trains[int(half)]):
                raise ValueError("lattice gate comparison requires A/B pulses with explicit phases")
            for flip in (False, True):
                gates[link, half, flip] = train_blocks(trains[int(half)], photon_cutoff, flip)
    steps = int(job["steps"])
    if steps < 1:
        raise ValueError("steps must be positive")
    theta = job["J_t"]/steps
    loc = 2*job["h_over_J"]*theta
    trace = 0j
    sector_traces = {int(code): 0j for code in np.unique(input_codes)}
    kept_number = kept_gauss = basis_fidelity = 0.0
    norm_error = boundary = 0.0
    compared = 0
    for start in range(0, dimension, batch_size):
        columns = np.arange(start, min(start+batch_size, dimension))
        states = np.zeros((space.dimension, len(columns)), complex)
        for column, source in enumerate(columns):
            x_bits, occupation = divmod(int(source), count)
            states[np.arange(8)*space.count+photon_indices[occupation], column] = THREE_HADAMARDS[:, x_bits]
        for step in range(steps):
            flip = step % 2 == 1
            for link in range(3):
                space.rotate(states, link, loc/(2 if order == 2 else 1)+(np.pi if step else 0))
            schedule = [(0, False), (2, False), (1, False)] if order == 1 else [(0, True), (2, True), (1, False), (0, True), (2, True)]
            for link, half in schedule:
                space.hop(states, link, gates[link, half, flip])
            if order == 2:
                for link in range(3):
                    space.rotate(states, link, loc/2)
        if (steps-1) % 2:
            for link in range(3):
                space.rotate(states, link, np.pi)
        norm_error = max(norm_error, float(np.max(abs(np.sum(abs(states)**2, axis=0)-1))))
        # Rotate the OUTPUT to the X basis. This changes no input or population
        # and lets Gauss preservation be counted without postselection.
        in_x = (THREE_HADAMARDS @ states.reshape(8, -1)).reshape(8, space.count, -1)
        projected = in_x[:, photon_indices].reshape(dimension, -1)
        overlaps = np.sum(ideal[:, columns].conj()*projected, axis=0)
        trace += overlaps.sum()
        for code in sector_traces:
            sector_traces[code] += overlaps[input_codes[columns] == code].sum()
        basis_fidelity += np.sum(abs(overlaps)**2)
        probability = abs(in_x)**2
        kept_number += probability[:, photon_indices].sum()
        kept_gauss += (probability.reshape(space.dimension, -1)*(space.gauss[:, None] == input_codes[columns][None, :])).sum()
        boundary = max(boundary, float(np.max(probability[:, space.total == photon_cutoff].sum(axis=(0, 1)))))
        compared += len(columns)
        if job.get("verbose", False):
            print(f"GATE_PROGRESS N={total}, cutoff={photon_cutoff}: {compared}/{dimension} inputs", flush=True)
    if compared != dimension or not np.isfinite([trace.real, trace.imag, kept_number, kept_gauss, norm_error]).all():
        raise RuntimeError(f"incomplete or nonfinite gate comparison: {compared}/{dimension} inputs")
    if norm_error > 2e-10:
        raise RuntimeError(f"norm error {norm_error:.3g} across {compared} inputs")
    return dict(infidelity=float(1-abs(trace/dimension)**2),
                basis_state_infidelity=float(1-basis_fidelity/dimension),
                number_lost=float(1-kept_number/dimension), gauss_lost=float(1-kept_gauss/dimension),
                inputs_compared=compared, photon_cutoff=photon_cutoff, propagation_dimension=space.dimension,
                norm_error=norm_error, boundary_population=boundary,
                gauss_sector_infidelities={str(code): float(1-abs(value/count)**2) for code, value in sector_traces.items()})


def run_lattice_gate(job):
    """Increase the propagation cutoff until all three physical metrics converge."""
    total = int(job.get("total_photons", 8))
    cutoff = int(job.get("photon_cutoff_start", total+3))
    tolerance = float(job.get("cutoff_tolerance", 1e-7))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("cutoff_tolerance must be positive")
    cache_file = None
    cache_key = None
    if job.get("cache_directory"):
        # A cache entry belongs to these pulse lists, physical parameters and
        # this exact implementation. It is never accepted just because a file exists.
        per_link = job.get("trains_per_link") or {link: (job["train_full"], job["train_half"]) for link in range(3)}
        specification = dict(total_photons=total, steps=int(job["steps"]), J_t=float(job["J_t"]),
                             h_over_J=float(job["h_over_J"]), order=int(job.get("order", 2)),
                             trains_per_link={str(link): trains for link, trains in per_link.items()},
                             photon_cutoff_start=cutoff, cutoff_tolerance=tolerance,
                             implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        cache_key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        directory = Path(job["cache_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        cache_file = directory/(cache_key+".json")
        if cache_file.exists():
            saved = json.loads(cache_file.read_text())
            result = saved["result"]
            if (saved.get("key") != cache_key or result["inputs_compared"] != 8*comb(total+3, 3)
                    or not result["cutoff_converged"] or len(result["cutoff_history"]) < 2
                    or result["cutoff_max_change"] > tolerance
                    or not np.isfinite([result[key] for key in ("infidelity", "gauss_lost", "number_lost")]).all()):
                raise RuntimeError(f"invalid gate comparison cache: {cache_file}")
            return result
    previous = None
    history = []
    while True:
        result = at_cutoff(job, cutoff, int(job.get("batch_size", 32)))
        history.append({key: result[key] for key in ("photon_cutoff", "propagation_dimension", "infidelity", "gauss_lost", "number_lost")})
        if previous is not None:
            differences = {key: abs(result[key]-previous[key]) for key in ("infidelity", "basis_state_infidelity", "gauss_lost", "number_lost")}
            # Check every Gauss-sector fidelity too: averaging must not hide a
            # cutoff failure confined to one sector.
            sector_difference = max(abs(result["gauss_sector_infidelities"][code]-previous["gauss_sector_infidelities"][code])
                                    for code in result["gauss_sector_infidelities"])
            change = max(*differences.values(), sector_difference)
            if change <= tolerance:
                result.update(cutoff_converged=True, cutoff_max_change=change,
                              cutoff_tolerance=tolerance, cutoff_history=history)
                if cache_file is not None:
                    temporary = cache_file.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_text(json.dumps(dict(key=cache_key, result=result), indent=1))
                    temporary.replace(cache_file)
                return result
        previous = result
        cutoff += 2


def exact_trotter_metrics(job):
    """All-input reference for ideal-hop Trotterization, without photon leakage."""
    total = int(job.get("total_photons", 8))
    inputs = photon_configurations(total)
    count = len(inputs)
    dimension = 8*count
    index = {occupation: i for i, occupation in enumerate(inputs)}
    steps, order = int(job["steps"]), int(job.get("order", 2))
    theta = job["J_t"]/steps
    # Work independently in each exact Gauss sector; each link and h X preserves it.
    codes = gauss_codes(inputs).ravel()
    ideal = exact_unitary(total, float(job["J_t"]), float(job["h_over_J"]))
    diagonal = np.repeat([sum(1-2*((bits >> (2-link)) & 1) for link in range(3)) for bits in range(8)], count)
    link_hamiltonians = []
    for link in range(3):
        h = np.zeros((dimension, dimension), float)
        for bits in range(8):
            for column, occupation in enumerate(inputs):
                for origin, destination in ((link, link+1), (link+1, link)):
                    if occupation[origin]:
                        moved = list(occupation)
                        moved[origin] -= 1; moved[destination] += 1
                        h[(bits ^ (1 << (2-link)))*count+index[tuple(moved)], bits*count+column] += np.sqrt(occupation[origin]*(occupation[destination]+1))
        link_hamiltonians.append(h)
    trace = 0j
    basis_fidelity = 0.0
    for code in np.unique(codes):
        sector = np.flatnonzero(codes == code)
        half = order == 2
        rx = np.diag(np.exp(-1j*job["h_over_J"]*theta*diagonal[sector]/(2 if half else 1)))
        outer = expm(-1j*theta/(2 if half else 1)*(link_hamiltonians[0]+link_hamiltonians[2])[np.ix_(sector, sector)])
        inner = expm(-1j*theta*link_hamiltonians[1][np.ix_(sector, sector)])
        step = rx @ outer @ inner @ outer @ rx if half else inner @ outer @ rx
        actual = np.linalg.matrix_power(step, steps)
        target = ideal[np.ix_(sector, sector)]
        overlaps = np.sum(target.conj()*actual, axis=0)
        trace += overlaps.sum()
        basis_fidelity += np.sum(abs(overlaps)**2)
    return dict(infidelity=float(1-abs(trace/dimension)**2), basis_state_infidelity=float(1-basis_fidelity/dimension),
                number_lost=0.0, gauss_lost=0.0, inputs_compared=dimension)
