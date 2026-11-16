"""Independent check of the one-link numbers (resonant_trains.link_infidelity and cswap_gate_metrics).

    $ACADIA_ENV/bin/python checks/check_link.py

Written without resonant_trains' gate code: every pulse is exp(-i H) of a full (transmon x cavity a x cavity b)
Hamiltonian built with kron, at 16 cavity levels; the echo pi is X on the transmon; the target is the matrix
exponential of Z (a^dag b + b^dag a). Only the train lists (leg, area, phase) are taken from resonant_trains.
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import resonant_trains as rt

L = 16                                                         # cavity levels
a1 = np.diag(np.sqrt(np.arange(1, L)), 1)
I2, IL = np.eye(2), np.eye(L)
sp = np.array([[0, 0], [1, 0]])                                 # |e><g|, order (g, e)
a = np.kron(np.kron(I2, a1), IL)                                # cavity a on (q, a, b)
b = np.kron(np.kron(I2, IL), a1)
SP = np.kron(np.kron(sp, IL), IL)
Z = np.kron(np.kron(np.diag([1, -1]), IL), IL)
XQ = np.kron(np.kron(np.array([[0, 1], [1, 0]]), IL), IL)
HOP = Z @ (a.conj().T @ b + b.conj().T @ a)
INPUTS = [(q, na, nb) for q in (0, 1) for na in range(9) for nb in range(9 - na)]
COLUMNS = [q * L * L + na * L + nb for q, na, nb in INPUTS]


def pulse(leg, area, phase):
    c = a if leg == "A" else b
    h = np.exp(1j * phase) * c @ SP                             # e^{i phase} c sigma+
    return expm(-1j * area * (h + h.conj().T))


def sequence(trains, k, phi):
    U = np.eye(2 * L * L, dtype=complex)
    for step in range(k):
        if step:
            U = XQ @ U
        for train in trains:
            if train == "exact":
                angle = (-1 if step % 2 else 1) * phi / k / len(trains)
                U = expm(-1j * angle * HOP) @ U
                continue
            for leg, area, phase in train:
                if leg == "A" and step % 2:
                    phase += np.pi
                U = pulse(leg, area, phase) @ U
    if (k - 1) % 2:
        U = XQ @ U
    return U


def gate_error(U, phi):
    target = expm(-1j * phi * HOP)
    overlap = np.trace(target[np.ix_(COLUMNS, COLUMNS)].conj().T @ U[np.ix_(COLUMNS, COLUMNS)])
    return 1 - abs(overlap / len(COLUMNS)) ** 2


if __name__ == "__main__":
    phi = np.pi / 4
    cases = []
    for k in (2, 3, 4, 6):
        th = phi / k
        cases += [(f"symmetric 12, inner, k={k}", [rt.symmetric(th, 12)], k),
                  (f"symmetric 16, outer, k={k}", [rt.symmetric(th / 2, 16)] * 2, k),
                  (f"paired, inner, k={k}", [rt.paired(rt.calibrate(rt.paired, th, np.sqrt(th / 4)))], k),
                  (f"exact, outer, k={k}", ["exact", "exact"], k)]
    differences = []
    for label, trains, k in cases:
        mine = gate_error(sequence(trains, k, phi), phi)
        theirs = rt.cswap_gate_metrics(trains, k, phi)["infidelity"]
        differences.append(abs(mine - theirs))
        print(f"{label:28s} independent {mine:.6e}   resonant_trains {theirs:.6e}   difference {abs(mine - theirs):.1e}")
    # one hop alone (10 cavity levels in resonant_trains is enough: one train moves at most one photon)
    for n in (8, 12, 16):
        th = np.pi / 24
        train = rt.symmetric(th, n)
        mine = gate_error(sequence([train], 1, th), th)
        theirs = rt.link_infidelity(train, th)
        differences.append(abs(mine - theirs))
        print(f"{f'one hop, symmetric {n}':28s} independent {mine:.6e}   resonant_trains {theirs:.6e}   difference {abs(mine - theirs):.1e}")
    assert len(differences) == len(cases) + 3
    assert max(differences) < 1e-10, max(differences)
    print(f"CHECKED {len(differences)} gates x 90 inputs; largest difference {max(differences):.1e}")
