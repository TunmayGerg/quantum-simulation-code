"""Claude's independent replay of a saved gate (board task C1).

    ../../acadia_env/bin/python claude_replay.py edge_n10_gap0_phase1 squeeze_8db

Independent of Codex's verifier and of the shared C++ propagator: decodes the stored int16 words with the SDK's
sample_to_complex, applies the RFDC 8x interpolation filters (AMD PG269 coefficients, written out here again),
and steps every 6.4 GS/s sample with scipy's expm_multiply on a sparse Hamiltonian
    H = q(t) |e><g| + conj(q) |g><e|  +  s(t) |e><g| a + conj(s) |g><e| a^dag      (two-level qubit, no chi)
Targets are built here from their definitions. Reports gate fidelity on qubit g/e x Fock 0..8 (18 inputs), the filtered
peak |s(t)| as an equivalent swap time, and the photon cutoff check.
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import expm
from scipy.signal import upfirdn
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply
from acadia.sample_arithmetic import sample_to_complex

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFIED = os.path.join(HERE, "..", "runs", "swap200", "verified")
OUT = os.path.join(HERE, "..", "runs", "claude", "replay")
DT = 0.00125 / 8                                   # us, after 8x interpolation
QUBIT_PI_SCALE, PUMP_SCALE_600NS = 0.43779166666666675, 0.045   # DAC scales of the 30 ns pi and of a 600 ns swap
PI_PEAK = np.pi / 0.030                            # rad/us, Hann pi peak
G_600 = np.pi / (2 * 0.600)                        # rad/us


def rfdc(x):
    halves = [[5, 0, -17, 0, 44, 0, -96, 0, 187, 0, -335, 0, 565, 0, -906, 0, 1401, 0, -2112, 0, 3145, 0, -4723, 0, 7415,
               0, -13331, 0, 41526], [-12, 0, 84, 0, -337, 0, 1008, 0, -2693, 0, 10142], [-6, 0, 54, 0, -254, 0, 1230]]
    for half, centre in zip(halves, (65536, 16384, 2048)):
        x = upfirdn(np.asarray(half + [centre] + half[::-1], float) / centre, x, up=2)
    return x


def drives(name):
    w = np.load(os.path.join(VERIFIED, f"{name}_words.npz"))
    q = np.conj(sample_to_complex(w["qubit"])) / QUBIT_PI_SCALE * PI_PEAK      # stored DAC phase = -Hamiltonian phase
    s = sample_to_complex(w["sideband"]) / PUMP_SCALE_600NS * G_600
    return rfdc(q), rfdc(s)


def operators(levels):
    n = np.arange(levels)
    a = diags(np.sqrt(n[1:]), 1, shape=(levels, levels))
    sp = csr_matrix(([1.0], ([1], [0])), shape=(2, 2))                         # |e><g|, basis (g, e)
    from scipy.sparse import kron, identity
    qubit_up = kron(sp, identity(levels)).tocsr()                             # sigma+ x 1  (index = qubit * L + n)
    pump_up = kron(sp, a).tocsr()                                             # sigma+ a
    return qubit_up, pump_up


def target(gate, levels, rz):
    n = np.arange(levels)
    a = np.diag(np.sqrt(n[1:]), 1)
    T = np.zeros((2 * levels, 18), complex)
    if gate == "cd":
        D = expm(1.0 * (a.T - a))                                            # beta = 1
        for m in range(9):
            T[:levels, m], T[levels:, m] = (D + D.T)[:, m] / 2, (D - D.T)[:, m] / 2
            T[:levels, 9 + m], T[levels:, 9 + m] = (D - D.T)[:, m] / 2, (D + D.T)[:, m] / 2
    else:
        S = expm(8 * np.log(10) / 20 / 2 * (a @ a - a.T @ a.T))
        for m in range(9):
            T[:levels, m], T[levels:, 9 + m] = S[:, m], S.T[:, m]
    z = np.r_[np.ones(levels), -np.ones(levels)]
    return T, np.exp(-1j * rz * z / 2)


def replay(name, levels):
    meta = json.load(open(os.path.join(VERIFIED, f"{name}.json")))
    q, s = drives(name)
    up_q, up_s = operators(levels)
    state = np.zeros((2 * levels, 18), complex)
    for m in range(9):
        state[m, m] = 1                       # g_m
        state[levels + m, 9 + m] = 1          # e_m
    for qj, sj in zip(q, s):
        H = qj * up_q + np.conj(qj) * up_q.T + sj * up_s + np.conj(sj) * up_s.T
        state = expm_multiply(-1j * DT * H, state)
    T, frame = target(meta["gate"], levels, meta.get("qubit_frame_Rz", 0.0))
    out = frame[:, None] * state
    overlaps = np.sum(T.conj() * out, axis=0)
    peak = np.max(np.abs(s))
    return dict(name=name, levels=levels, fidelity=float(abs(overlaps.mean()) ** 2),
                worst_input=float(np.min(abs(overlaps) ** 2)), samples=len(q), time_us=len(q) * DT,
                filtered_peak_g=float(peak), peak_equivalent_swap_ns=float(1e3 * np.pi / (2 * peak)),
                norm_error=float(np.max(abs(np.sum(abs(state) ** 2, axis=0) - 1))),
                codex_fidelity=meta["fidelity"], codex_peak_swap_ns=meta.get("actual_min_swap_ns"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:]:
        start = 64 if "cd" in name or "edge" in name else 256
        rows = [replay(name, start), replay(name, start + 32)]
        json.dump(rows, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
        for r in rows:
            print(r, flush=True)
