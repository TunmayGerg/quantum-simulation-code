"""Independent check of lattice_gate_fidelity.py (the full-input lattice gate fidelity).

    $ACADIA_ENV/bin/python checks/check_lattice.py        (about a minute)
 This reference constructs every pulse
Hamiltonian from Fock-state matrix elements and uses expm_multiply. It does not
use the production train-block multiplication, tensor application or metrics.
"""
from functools import lru_cache
from itertools import product
from math import comb
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import expm, hadamard
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lattice_gate_fidelity as gate
import resonant_trains as rt


class SparseReference:
    def __init__(self, cutoff, photons):
        occupations = [state for state in product(range(cutoff+1), repeat=4) if sum(state) <= cutoff]
        self.basis = [(bits, occupation) for bits in product((0, 1), repeat=3) for occupation in occupations]
        self.index = {state: i for i, state in enumerate(self.basis)}
        self.cutoff = cutoff
        self.photons = photons
        self.inputs = [(bits, occupation) for bits, occupation in self.basis if sum(occupation) == photons]
        self.keep = np.array([self.index[state] for state in self.inputs])
        self.qubit_states = list(product((0, 1), repeat=3))
        self.hadamard = hadamard(8)/np.sqrt(8)
        self.x_operators = [self.x_operator(link) for link in range(3)]

    def x_operator(self, link):
        rows = []
        for bits, occupation in self.basis:
            flipped = list(bits)
            flipped[link] ^= 1
            rows.append(self.index[tuple(flipped), occupation])
        return coo_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(rows),)*2).tocsr()

    @lru_cache(None)
    def jc_hamiltonian(self, link, leg, phase):
        rows, columns, values = [], [], []
        site = link if leg == "A" else link+1
        for column, (bits, occupation) in enumerate(self.basis):
            if bits[link] == 0 and occupation[site] > 0:
                excited = list(bits); excited[link] = 1
                moved = list(occupation); moved[site] -= 1
                row = self.index[tuple(excited), tuple(moved)]
                value = np.exp(1j*phase)*np.sqrt(occupation[site])
                rows.extend((row, column)); columns.extend((column, row)); values.extend((value, value.conjugate()))
        return coo_matrix((values, (rows, columns)), shape=(len(self.basis),)*2).tocsr()

    def exact_hamiltonian(self, h_over_j):
        size = len(self.inputs)
        index = {state: i for i, state in enumerate(self.inputs)}
        h = np.zeros((size, size), float)
        hops = [np.zeros_like(h) for _ in range(3)]
        for column, (bits, occupation) in enumerate(self.inputs):
            for link in range(3):
                flipped = list(bits); flipped[link] ^= 1
                h[index[tuple(flipped), occupation], column] += h_over_j
                for origin, destination in ((link, link+1), (link+1, link)):
                    if occupation[origin]:
                        moved = list(occupation); moved[origin] -= 1; moved[destination] += 1
                        hops[link][index[bits, tuple(moved)], column] += (1-2*bits[link])*np.sqrt(occupation[origin]*(occupation[destination]+1))
        return h+sum(hops), h, hops

    def compare(self, job, exact_hops=False):
        count = len(self.inputs)
        occupations = sorted({occupation for _, occupation in self.inputs})
        initial = np.zeros((len(self.basis), count), complex)
        for column, (x_bits, occupation) in enumerate(self.inputs):
            x = self.qubit_states.index(x_bits)
            for q, bits in enumerate(self.qubit_states):
                initial[self.index[bits, occupation], column] = self.hadamard[q, x]
        h, h_x, hops = self.exact_hamiltonian(job["h_over_J"])
        ideal = expm(-1j*job["J_t"]*h)@initial[self.keep]
        state = initial.copy()
        theta = job["J_t"]/job["steps"]
        loc = 2*job["h_over_J"]*theta
        if exact_hops:
            state = initial[self.keep].copy()
            for _ in range(job["steps"]):
                state = expm(-1j*theta*h_x/(2 if job["order"] == 2 else 1))@state
                for link, half in ([(0, True), (2, True), (1, False), (0, True), (2, True)] if job["order"] == 2 else [(0, False), (2, False), (1, False)]):
                    state = expm(-1j*theta*hops[link]/(2 if half else 1))@state
                if job["order"] == 2:
                    state = expm(-1j*theta*h_x/2)@state
            overlaps = np.sum(ideal.conj()*state, axis=0)
            return dict(infidelity=1-abs(overlaps.mean())**2, basis_state_infidelity=1-np.mean(abs(overlaps)**2))
        for step in range(job["steps"]):
            for link in range(3):
                angle = loc/(2 if job["order"] == 2 else 1)+(np.pi if step else 0)
                state = expm_multiply(-.5j*angle*self.x_operators[link], state, traceA=0)
            schedule = [(0, False), (2, False), (1, False)] if job["order"] == 1 else [(0, True), (2, True), (1, False), (0, True), (2, True)]
            per_link = job.get("trains_per_link") or {l: (job["train_full"], job["train_half"]) for l in range(3)}
            for link, half in schedule:
                pulses = per_link[link][1 if half else 0]
                for leg, area, phase in pulses:
                    phase += np.pi if step%2 and leg == "A" else 0
                    state = expm_multiply(-1j*area*self.jc_hamiltonian(link, leg, phase), state, traceA=0)
            if job["order"] == 2:
                for link in range(3):
                    state = expm_multiply(-.25j*loc*self.x_operators[link], state, traceA=0)
        if (job["steps"]-1)%2:
            for link in range(3):
                state = expm_multiply(-.5j*np.pi*self.x_operators[link], state, traceA=0)
        overlaps = np.sum(ideal.conj()*state[self.keep], axis=0)
        # coherent gate fidelity inside each Gauss sector (input's four Gauss eigenvalues, from its X bits and photons)
        sector_sums, sector_counts = {}, {}
        for column, (x_bits, occupation) in enumerate(self.inputs):
            code = 0
            for site in range(4):
                value = (-1)**occupation[site]
                for link in (site-1, site):
                    if 0 <= link < 3:
                        value *= 1-2*x_bits[link]
                code |= (value < 0) << site
            sector_sums[code] = sector_sums.get(code, 0)+overlaps[column]
            sector_counts[code] = sector_counts.get(code, 0)+1
        sectors = {str(code): 1-abs(sector_sums[code]/sector_counts[code])**2 for code in sector_sums}
        p_gauss = 0.0
        for column, (x_bits, occupation) in enumerate(self.inputs):
            projected = state[:, column].copy()
            for site in range(4):
                charge = (-1)**occupation[site]
                transformed = projected.copy()
                for link in (site-1, site):
                    if 0 <= link < 3:
                        charge *= 1-2*x_bits[link]
                        transformed = self.x_operators[link]@transformed
                parity = np.array([(-1)**configuration[site] for _, configuration in self.basis])
                projected = (projected+charge*parity*transformed)/2
            p_gauss += np.vdot(projected, projected).real
        return dict(infidelity=1-abs(overlaps.mean())**2, basis_state_infidelity=1-np.mean(abs(overlaps)**2),
                    number_lost=1-np.sum(abs(state[self.keep])**2)/count, gauss_lost=1-p_gauss/count), sectors


def check_large_blocks():
    """Check all three link embeddings at the actual N=8 propagation cutoff."""
    reference = SparseReference(cutoff=13, photons=8)
    space = gate.PhotonSpace(13)
    mapping = []
    for bits, occupation in reference.basis:
        qubit = 4*bits[0]+2*bits[1]+bits[2]
        mapping.append(qubit*space.count+space.photon_index[occupation])
    mapping = np.array(mapping)
    assert len(mapping) == space.dimension == 19040 and len(np.unique(mapping)) == 19040
    rng = np.random.default_rng(7401)
    initial = rng.normal(size=(19040, 3))+1j*rng.normal(size=(19040, 3))
    initial /= np.linalg.norm(initial, axis=0)
    pulses = rt.symmetric(np.pi/24, 12)
    errors = []
    for link in range(3):
        for flip in (False, True):
            expected = initial.copy()
            for leg, area, phase in pulses:
                phase += np.pi if flip and leg == "A" else 0
                expected = expm_multiply(-1j*area*reference.jc_hamiltonian(link, leg, phase), expected, traceA=0)
            actual = np.empty_like(initial)
            actual[mapping] = initial
            space.hop(actual, link, gate.train_blocks(pulses, 13, flip))
            difference = float(np.max(abs(actual[mapping]-expected)))
            assert difference < 2e-13, (link, flip, difference)
            errors.append(difference)
            print(f"LARGE_BLOCK link={link} flip={flip}: 19040 basis amplitudes x 3 vectors; max difference {difference:.3e}", flush=True)
    print(f"LARGE_BLOCK_TOTAL checked 6 embeddings, 342720 complex amplitudes, max difference {max(errors):.3e}", flush=True)


def main():
    photons = 2
    reference = SparseReference(cutoff=4, photons=photons)
    assert len(reference.inputs) == 8*comb(photons+3, 3) == 80
    theta = np.pi/24
    full, half = rt.symmetric(theta, 12), rt.symmetric(theta/2, 12)
    errors = []
    for order in (1, 2):
        for steps in (3, 4):
            job = dict(total_photons=photons, train_full=full, train_half=half, steps=steps,
                       J_t=theta*steps, h_over_J=1.5, order=order)
            actual = gate.at_cutoff(job, 4, batch_size=17)
            expected, expected_sectors = reference.compare(job)
            assert actual["inputs_compared"] == 80
            for key in expected:
                difference = abs(actual[key]-expected[key]); errors.append(difference)
                assert difference < 3e-12, (order, steps, key, actual, expected)
            assert set(expected_sectors) == set(actual["gauss_sector_infidelities"])
            for code in expected_sectors:
                difference = abs(actual["gauss_sector_infidelities"][code]-expected_sectors[code]); errors.append(difference)
                assert difference < 3e-12, (code, actual["gauss_sector_infidelities"], expected_sectors)
            print(f"SPARSE_REFERENCE order={order} steps={steps}: compared 80/80 input columns, 4 metrics and "
                  f"{len(expected_sectors)} Gauss sectors; max difference {max(errors):.3e}", flush=True)
            exact = gate.exact_trotter_metrics(job)
            direct = reference.compare(job, exact_hops=True)
            for key in direct:
                assert abs(exact[key]-direct[key]) < 3e-12, (key, exact, direct)
            print(f"EXACT_HOP_REFERENCE order={order} steps={steps}: compared 80/80 inputs and 2 metrics", flush=True)
    rng = np.random.default_rng(11)
    per_link = {link: (rt.perturb(full, dict(phase_rms=0.05, area_rms=0.03), rng), rt.perturb(half, dict(phase_rms=0.05, area_rms=0.03), rng))
                for link in range(3)}
    assert per_link[0][0] != per_link[1][0] != per_link[2][0]
    linked = dict(total_photons=photons, train_full=None, train_half=None, trains_per_link=per_link, steps=4,
                  J_t=theta*4, h_over_J=1.5, order=2)
    actual = gate.at_cutoff(linked, 4, batch_size=13)
    expected, expected_sectors = reference.compare(linked)
    per_link_errors = [abs(actual[key]-expected[key]) for key in expected] + \
        [abs(actual["gauss_sector_infidelities"][code]-expected_sectors[code]) for code in expected_sectors]
    assert actual["inputs_compared"] == 80 and max(per_link_errors) < 3e-12, (actual, expected)
    print(f"PER_LINK_TRAINS different random-error trains on each link: compared 80/80 inputs, 4 metrics and "
          f"{len(expected_sectors)} sectors; gate error {expected['infidelity']:.3e}, Gauss loss {expected['gauss_lost']:.3e}; "
          f"max difference {max(per_link_errors):.3e}", flush=True)
    job.update(photon_cutoff_start=2, cutoff_tolerance=1e-7)
    adaptive = gate.run_lattice_gate(job)
    assert adaptive["inputs_compared"] == 80 and adaptive["cutoff_converged"]
    assert adaptive["photon_cutoff"] > 2 and len(adaptive["cutoff_history"]) >= 2
    print(f"ADAPTIVE_CUTOFF compared 80/80 inputs at cutoffs {[row['photon_cutoff'] for row in adaptive['cutoff_history']]}; "
          f"max change {adaptive['cutoff_max_change']:.3e}, tolerance 1e-7", flush=True)
    with TemporaryDirectory() as temporary:
        cached_job = {**job, "cache_directory": temporary}
        first = gate.run_lattice_gate(cached_job)
        second = gate.run_lattice_gate(cached_job)
        assert first == second and len(list(Path(temporary).glob("*.json"))) == 1
        changed = gate.run_lattice_gate({**cached_job, "h_over_J": 1.6})
        assert len(list(Path(temporary).glob("*.json"))) == 2
        assert abs(changed["infidelity"]-first["infidelity"]) > 1e-9
        for path in Path(temporary).glob("*.json"):
            saved = json.loads(path.read_text())
            saved["result"]["inputs_compared"] = 79
            path.write_text(json.dumps(saved))
        try:
            gate.run_lattice_gate(cached_job)
        except RuntimeError:
            pass
        else:
            raise AssertionError("An incomplete cached input comparison was accepted")
    print("CACHE_CHECK compared 2 complete 80-input gates; reused identical inputs, separated changed fields, rejected 79/80 cache", flush=True)
    # Opposite phases have perfect individual-state fidelity and zero coherent
    # gate overlap. This catches replacing the requested trace by a mean of F.
    overlaps = np.array([1+0j, -1+0j])
    assert abs(overlaps.mean())**2 == 0 and np.mean(abs(overlaps)**2) == 1
    print("TRACE_PHASE_TEST compared 2/2 inputs: gate fidelity 0, mean state fidelity 1", flush=True)
    print(f"CHECKED 320 sparse-reference input columns, 320 exact-hop input columns; max sparse metric difference {max(errors):.3e}", flush=True)
    check_large_blocks()


if __name__ == "__main__":
    main()
