"""Resonant sideband trains for the conditional hop exp(-i theta Z (a^dag b + b^dag a)) on one Z2 link.

Used by resonant_train_comparison.ipynb (with lattice_gate_fidelity.py for the full lattice) and by
optimise_trains.py. Ideal Hamiltonian: two-level transmon, pulse-area gates, no loss.

A TRAIN is a plain list of entries (leg, area, phase), played in that order on one link:
    leg "A" = resonant sideband transmon <-> left cavity a,  leg "B" = transmon <-> right cavity b
    one pulse = exp(-i area (e^{i phase} c sigma+ + h.c.)),  c = a or b
    ("frame q" / "frame a" / "frame b", 0, phase) = a virtual frame phase (no time): diag(1, e^{i phase}) on the
    transmon, or e^{i phase n} on a cavity. Only the free-search trains have them.
The trains compared in the notebook:
    textbook       8 pulses  B+ A+ B- A- B- A- B+ A+ (W_sym), all areas = one calibrated scale   (today's single)
    paired         textbook, then textbook with every pulse +pi: 16 pulses, one calibrated scale  (today's paired)
    half-ends      8 pulses  B+ A+ B- A- B+ A+ B- A-, areas (1,1,2,2,2,2,1,1) x one calibrated scale
    symmetric n    n = 8 / 12 / 16 pulses, signs + + - - repeated, mirror-symmetric areas, every area >= g x 5 ns
                   (fixed sideband scale), from trains/symmetric_trains.json (made by optimise_trains.py)
"-" means phase + pi. Leg B always carries an extra phase pi/2 (LEG_B_PHASE): that makes the hop Z X_ab.

Time: a pulse of area A lasts max(|A| / g + 5 ns, 10 ns): flat top with 5 ns Hann ramps (A = g (flat + 5 ns));
a pulse shorter than 10 ns would be a 10 ns Hann at lower amplitude. |A| < 1e-3 rad is dropped (0 ns).
"""
import json
import os
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(HERE, "trains")

# ---------------------------------------------------------------------------------------------------------------
# constants (times in microseconds, angles in radians)
# ---------------------------------------------------------------------------------------------------------------
G = np.pi / 2 / 0.250                    # rad/us: sideband coupling for a 250 ns full swap (g t = pi/2)
RAMP = 0.005                             # Hann ramp on each side of a sideband pulse
PI_PULSE = 0.030                         # transmon pi / R_x slot
MIN_PULSE = 0.010                        # shortest pulse
DROP_AREA = 1e-3                         # smaller areas are not played
A_MIN = G * RAMP                         # smallest area at a fixed sideband scale (zero flat time): 0.0314 rad
LEG_B_PHASE = np.pi / 2                  # leg B phase offset: makes the train's hop Z (a^dag b + b^dag a)

D = 10                                   # cavity levels for one hop on one link (inputs have <= 8 photons)
CAVITY_A = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)     # annihilation operator
N_OP = np.arange(D)
X = np.array([[0, 1], [1, 0]], complex)
SIGMA_PLUS = np.array([[0, 0], [1, 0]], complex)                     # |e><g| in the order (g, e)

# the 90 inputs of one link: transmon g or e, n_a + n_b <= 8. The state tensor is psi[input, q, n_a, n_b].
INPUTS = [(q, na, nb) for q in (0, 1) for na in range(9) for nb in range(9 - na)]
PSI_LINK = np.zeros((len(INPUTS), 2, D, D), complex)
for _j, _s in enumerate(INPUTS):
    PSI_LINK[(_j,) + _s] = 1.0


# ---------------------------------------------------------------------------------------------------------------
# gates as matrices
# ---------------------------------------------------------------------------------------------------------------
def apply(psi, U, axes):
    """Apply the matrix U to the listed axes of the tensor psi (U is indexed in the order of `axes`)."""
    k = len(axes)
    psi = np.moveaxis(psi, axes, range(k))
    shape = psi.shape
    psi = (U @ psi.reshape(U.shape[1], -1)).reshape(shape)
    return np.moveaxis(psi, range(k), axes)


def jc_gate(area, phase):
    """Resonant sideband pulse exp(-i area (e^{i phase} c sigma+ + h.c.)) on (transmon, cavity)."""
    h = np.exp(1j * phase) * np.kron(SIGMA_PLUS, CAVITY_A)
    return expm(-1j * area * (h + h.conj().T))


def exact_hop_link(theta):
    """exp(-i theta Z (a^dag b + b^dag a)) applied to the 90 link inputs (Z = +1 for g).
    Written as a 50:50 beam splitter, exp(-i theta Z n_a) exp(+i theta Z n_b), beam splitter back."""
    k = np.kron(CAVITY_A.conj().T, CAVITY_A)                                        # a^dag b
    bs = expm(np.pi / 4 * (k - k.conj().T))
    phase_a = np.diag(np.concatenate([np.exp(-1j * theta * N_OP), np.exp(+1j * theta * N_OP)]))
    phase_b = np.diag(np.concatenate([np.exp(+1j * theta * N_OP), np.exp(-1j * theta * N_OP)]))
    psi = apply(PSI_LINK, bs, [2, 3])
    psi = apply(psi, phase_a, [1, 2])
    psi = apply(psi, phase_b, [1, 3])
    return apply(psi, bs.conj().T, [2, 3])


# ---------------------------------------------------------------------------------------------------------------
# the trains
# ---------------------------------------------------------------------------------------------------------------
def pattern_train(legs, signs, areas):
    """Pulses with leg A at phase 0, leg B at pi/2, and a minus sign = + pi."""
    return [(leg, area, (LEG_B_PHASE if leg == "B" else 0.0) + (np.pi if s < 0 else 0.0))
            for leg, s, area in zip(legs, signs, areas)]


LEGS_8 = ["B", "A"] * 4
W_SYM_SIGNS = [+1, +1, -1, -1, -1, -1, +1, +1]          # textbook W_sym: B+ A+ B- A- B- A- B+ A+


def textbook(scale):
    return pattern_train(LEGS_8, W_SYM_SIGNS, [scale] * 8)


def paired(scale):
    return pattern_train(LEGS_8 * 2, W_SYM_SIGNS + [-s for s in W_SYM_SIGNS], [scale] * 16)


def half_ends(scale):
    return pattern_train(LEGS_8, [1, 1, -1, -1, 1, 1, -1, -1], scale * np.array([1, 1, 2, 2, 2, 2, 1, 1]))


def symmetric_from_half(half_areas):
    """n = 2 len(half_areas) pulses B A B A ..., signs + + - - repeated, areas mirror-symmetric."""
    half = np.asarray(half_areas, float)
    n = 2 * len(half)
    areas = np.concatenate([half, half[::-1]])
    signs = [+1 if (i // 2) % 2 == 0 else -1 for i in range(n)]
    legs = ["B" if i % 2 == 0 else "A" for i in range(n)]
    return pattern_train(legs, signs, areas)


def mirror_train(half_areas, signs):
    """n = len(signs) pulses B A B A ..., sign of pulse i from the string signs ("+" or "-"), areas mirror-symmetric
    (pulse i and pulse n-1-i have the same area). symmetric_from_half is the special case "++--" repeated."""
    half = np.asarray(half_areas, float)
    areas = np.concatenate([half, half[::-1]])
    legs = ["B" if i % 2 == 0 else "A" for i in range(len(signs))]
    return pattern_train(legs, [+1 if c == "+" else -1 for c in signs], areas)


@lru_cache(maxsize=None)
def _load(name):
    with open(os.path.join(TRAIN_DIR, name)) as f:
        return json.load(f)


def symmetric(theta, n_pulses):
    """The optimised symmetric train for this hop angle (trains/symmetric_trains.json)."""
    best = _load("symmetric_trains.json").get(f"{n_pulses} {theta:.10f}")
    if best is None:
        raise ValueError(f"no symmetric {n_pulses}-pulse train for theta = {theta}")
    return symmetric_from_half(best["half_areas"])


def pattern(theta, signs):
    """The optimised mirror-area train with this sign string for this hop angle (trains/pattern_trains.json)."""
    best = _load("pattern_trains.json").get(f"{signs} {theta:.10f}")
    if best is None:
        raise ValueError(f"no {signs} train for theta = {theta}")
    return mirror_train(best["half_areas"], signs)


def merge_neighbours(train):
    """Merge neighbouring pulses on the same leg with the same phase (within 1 deg) into one pulse.
    exp(-i A1 K) exp(-i A2 K) = exp(-i (A1 + A2) K): the unitary is unchanged, only two ramps are saved."""
    out = []
    for leg, area, phase in train:
        if (out and is_pulse(leg) and out[-1][0] == leg and area >= 0 and out[-1][1] >= 0
                and abs(np.angle(np.exp(1j * (out[-1][2] - phase)))) < np.radians(1)):
            out[-1] = (leg, out[-1][1] + area, out[-1][2])
        else:
            out.append((leg, area, phase))
    return out


def free_search_train(theta, n_pulses):
    """Best train of the FREE search (no area bound; every area and phase free, plus three frame phases) for this
    angle and pulse count, from trains/free_search.json. Pulses below 1e-3 rad are dropped, which can leave two
    pulses on the same leg next to each other; those are merged. Frame phases above 1e-3 rad are kept."""
    rows = [r for r in _load("free_search.json") if abs(r["theta"] - theta) < 1e-6 and r["n_pulses"] == n_pulses]
    if not rows:
        raise ValueError(f"no free-search {n_pulses}-pulse train for theta = {theta}")
    p = np.array(min(rows, key=lambda r: r["infidelity"])["params"])
    legs = ["B" if i % 2 == 0 else "A" for i in range(n_pulses)]
    # positive areas (a negative area is the same pulse with phase + pi), small pulses dropped, neighbours merged
    train = [(leg, abs(area), phase + (np.pi if area < 0 else 0.0))
             for leg, area, phase in zip(legs, p[:n_pulses], p[n_pulses:2 * n_pulses]) if abs(area) >= DROP_AREA]
    train = merge_neighbours(train)
    train += [(name, 0.0, phase) for name, phase in zip(("frame q", "frame a", "frame b"), p[2 * n_pulses:])
              if abs(phase) > 1e-3]
    return train


def bounded_search_trains():
    """The trains of the BOUNDED search (every area >= A_MIN), trains/bounded_search.json, as {label: train}."""
    return {label: [tuple(p) for p in row["train"]] for label, row in _load("bounded_search.json").items()}


def is_pulse(leg):
    return leg in ("A", "B")


def pulse_length(area):
    return 0.0 if abs(area) < DROP_AREA else max(abs(area) / G + RAMP, MIN_PULSE)


def train_time(train):
    return sum(pulse_length(area) for _, area, _ in train)


def lattice_time(train_full, train_half, order, steps):
    """Total time of a finite Z2 run. The net echo pi left at the end (odd number of pi's) is not played: it is
    taken into account at readout, so it costs no time.
        order 1:  each step = R_x slot + outer(theta) + inner(theta)
        order 2:  each step = R_x slot + outer(theta/2) + inner(theta) + outer(theta/2); the half R_x of neighbouring
                  steps share one slot, so s steps use s + 1 slots."""
    if order == 1:
        return steps * (PI_PULSE + 2 * train_time(train_full))
    return steps * (PI_PULSE + train_time(train_full) + 2 * train_time(train_half)) + PI_PULSE


# ---------------------------------------------------------------------------------------------------------------
# one link: one hop
# ---------------------------------------------------------------------------------------------------------------
LINK_AXES = {"qa": [1, 2], "qb": [1, 3], "q": [1], "a": [2], "b": [3]}         # axes of psi[input, q, a, b]


def gate_on_link(leg, area, phase, flip_a=False):
    """(matrix, axes role) of one train entry. flip_a: leg-A pulses at + pi (after an odd number of echo pi's)."""
    if leg == "A":
        return jc_gate(area, phase + (np.pi if flip_a else 0.0)), "qa"
    if leg == "B":
        return jc_gate(area, phase), "qb"
    if leg == "frame q":
        return np.diag([1.0, np.exp(1j * phase)]), "q"
    return np.diag(np.exp(1j * phase * N_OP)), "a" if leg == "frame a" else "b"


def play_link(train, psi=None, flip_a=False):
    psi = PSI_LINK if psi is None else psi
    for leg, area, phase in train:
        U, role = gate_on_link(leg, area, phase, flip_a)
        psi = apply(psi, U, LINK_AXES[role])
    return psi


def link_infidelity(train, theta):
    """1 - gate fidelity |Tr(U_exact^dag U_train)|^2 / 90^2 against exp(-i theta Z X_ab), 90 inputs."""
    return 1 - abs(np.vdot(exact_hop_link(theta), play_link(train))) ** 2 / len(INPUTS) ** 2


def link_transmon_flip(train):
    """Mean over the 90 inputs of P(transmon ends in the other level). A resonant sideband conserves
    n_a + n_b + (transmon excitation), so this is also the probability that the link's photon number changed."""
    psi = play_link(train)
    p = 0.0
    for j, (q, _, _) in enumerate(INPUTS):
        p += np.sum(np.abs(psi[j, 1 - q]) ** 2)
    return p / len(INPUTS)


def calibrate(make_train, theta, guess):
    """The one scale of a fixed-pattern train that minimises the one-link infidelity."""
    cost = lambda s: link_infidelity(make_train(s), theta)
    return minimize_scalar(cost, bounds=(0.7 * guess, 1.3 * guess), method="bounded", options={"xatol": 1e-8}).x


# ---------------------------------------------------------------------------------------------------------------
# one link: the controlled beam splitter exp(-i phi Z X_ab) from k steps with echo (cavity cutoff converged)
# ---------------------------------------------------------------------------------------------------------------
def hop_matrix(cavity_levels):
    """a^dag b + b^dag a on (cavity a, cavity b) with the given number of levels."""
    cavity = np.diag(np.sqrt(np.arange(1, cavity_levels)), 1)
    return np.kron(cavity.T, cavity) + np.kron(cavity, cavity.T)


@lru_cache(maxsize=32)
def link_target_columns(phi, cavity_levels):
    """exp(-i phi Z X_ab) on the 90 inputs, by direct matrix exponential, at a chosen cavity cutoff."""
    ground_gate = expm(-1j * phi * hop_matrix(cavity_levels))  # Z = +1 (g); for e the gate is the complex conjugate
    target = np.zeros((len(INPUTS), 2, cavity_levels, cavity_levels), complex)
    for column, (qubit, left, right) in enumerate(INPUTS):
        gate = ground_gate if qubit == 0 else ground_gate.conj()
        target[column, qubit] = gate[:, left * cavity_levels + right].reshape(cavity_levels, cavity_levels)
    return target


def cswap_at_cutoff(trains, k, phi, cavity_levels):
    """exp(-i phi Z X_ab) made the way ONE LINK of the lattice makes it (h = 0), at a given cavity cutoff:
    k steps; each step = [echo pi on the transmon, from step 2 on] + the step's trains on this link
        trains = [train(theta)]                     as the inner link: one hop of theta = phi / k per step
        trains = [train(theta/2), train(theta/2)]   as an outer link: two hops of theta / 2 per step
    After an odd number of pi's leg A is played at + pi; the net pi is removed at the end. A train may be the string
    "exact": the ideal hop of angle phi / k / len(trains), with the sign the flipped transmon needs.
    Returns gate error, transmon flip and worst single-input error over all 90 inputs."""
    if cavity_levels < 10 or k < 1 or isinstance(trains, str) or len(trains) == 0:
        raise ValueError("need >= 10 cavity levels, positive k, and a nonempty list of trains")
    cavity = np.diag(np.sqrt(np.arange(1, cavity_levels)), 1)
    psi = np.zeros((len(INPUTS), 2, cavity_levels, cavity_levels), complex)
    for column, state in enumerate(INPUTS):
        psi[(column,) + state] = 1
    gate_cache = {}
    for step in range(k):
        if step:
            psi = apply(psi, X, [1])                                    # echo pi
        flip = step % 2 == 1
        for train in trains:
            if isinstance(train, str):                                  # "exact": exp(-i angle Z X_ab)
                angle = (-1.0 if flip else 1.0) * phi / k / len(trains)
                for q, z_value in ((0, 1.0), (1, -1.0)):
                    U = expm(-1j * z_value * angle * hop_matrix(cavity_levels))
                    rows = psi[:, q].reshape(len(INPUTS), -1)            # one row per input, index n_a * L + n_b
                    psi[:, q] = (rows @ U.T).reshape(psi[:, q].shape)
                continue
            for leg, area, phase in train:
                phase = phase + (np.pi if flip and leg == "A" else 0)
                key = (leg, float(area), float(phase))
                if key not in gate_cache:
                    if leg in ("A", "B"):
                        coupling = np.exp(1j * phase) * np.kron(SIGMA_PLUS, cavity)
                        gate = expm(-1j * area * (coupling + coupling.conj().T))
                        axes = [1, 2 if leg == "A" else 3]
                    elif leg == "frame q":
                        gate, axes = np.diag([1, np.exp(1j * phase)]), [1]
                    elif leg in ("frame a", "frame b"):
                        gate = np.diag(np.exp(1j * phase * np.arange(cavity_levels)))
                        axes = [2 if leg == "frame a" else 3]
                    else:
                        raise ValueError(f"unknown link train entry {leg!r}")
                    gate_cache[key] = gate, axes
                gate, axes = gate_cache[key]
                psi = apply(psi, gate, axes)
    if (k - 1) % 2:
        psi = apply(psi, X, [1])                                        # remove the net echo pi
    target = link_target_columns(float(phi), cavity_levels)
    overlaps = np.sum(target.conj() * psi, axis=(1, 2, 3))
    probability = abs(psi) ** 2
    norms = probability.sum(axis=(1, 2, 3))
    flip = sum(probability[column, 1 - qubit].sum() for column, (qubit, _, _) in enumerate(INPUTS)) / len(INPUTS)
    result = dict(infidelity=float(1 - abs(overlaps.mean()) ** 2),
                  transmon_flip=float(flip), worst_basis_infidelity=float(1 - np.min(abs(overlaps) ** 2)),
                  link_inputs_compared=len(overlaps), link_norm_error=float(np.max(abs(norms - 1))))
    if len(overlaps) != 90 or not np.isfinite(list(result.values())).all() or result["link_norm_error"] > 2e-10:
        raise RuntimeError(f"incomplete or nonfinite link comparison: {result}")
    return result


def cswap_gate_metrics(trains, k, phi, cutoff_tolerance=1e-9):
    """cswap_at_cutoff with 10, 12, 14, ... cavity levels until gate error, transmon flip and worst-input error change
    by at most cutoff_tolerance (echoes between trains can push the state above 9 photons)."""
    if not np.isfinite(cutoff_tolerance) or cutoff_tolerance <= 0:
        raise ValueError("cutoff_tolerance must be positive and finite")
    previous, history, cavity_levels = None, [], 10
    while True:
        result = cswap_at_cutoff(trains, k, phi, cavity_levels)
        history.append(dict(cavity_levels=cavity_levels, infidelity=result["infidelity"],
                            transmon_flip=result["transmon_flip"], worst_basis_infidelity=result["worst_basis_infidelity"]))
        if previous is not None:
            change = max(abs(result[key] - previous[key]) for key in ("infidelity", "transmon_flip", "worst_basis_infidelity"))
            if change <= cutoff_tolerance:
                result.update(link_cavity_levels=cavity_levels, link_cutoff_max_change=change,
                              link_cutoff_tolerance=cutoff_tolerance, link_cutoff_history=history)
                return result
        previous = result
        cavity_levels += 2


# ---------------------------------------------------------------------------------------------------------------
# static calibration errors (the same wrong number every time that pulse is played)
# ---------------------------------------------------------------------------------------------------------------
def perturb(train, error, rng):
    """error = dict(area_scale=0.01 -> every area x 1.01, leg_b_phase=rad added to every leg-B pulse,
    area_rms=fraction, phase_rms=rad: independent random static error per pulse,
    area_offset=rad added to every pulse: a wrong effective ramp offset at a fixed sideband scale)."""
    out = []
    for leg, area, phase in train:
        if not is_pulse(leg):                                    # virtual frame phases are exact
            out.append((leg, area, phase))
            continue
        area = area * (1 + error.get("area_scale", 0.0)) * (1 + error.get("area_rms", 0.0) * rng.standard_normal())
        area = area + np.sign(area) * error.get("area_offset", 0.0)
        phase = phase + (error.get("leg_b_phase", 0.0) if leg == "B" else 0.0) + error.get("phase_rms", 0.0) * rng.standard_normal()
        out.append((leg, area, phase))
    return out
