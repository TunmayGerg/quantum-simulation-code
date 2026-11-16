"""Claude task C3: fixed-sideband CD, a fair fixed-beta comparison of pulse families (Magnus-motivated).

    ../../acadia_env/bin/python claude_cd_magnus.py families

Model (ideal): two-level qubit, sideband g (sigma+ a + h.c.) at FIXED phase and FIXED amplitude (5 ns sin^2 ramps at the
very start and end only), 30 ns Hann pi pulses (24 samples of 1.25 ns) with phase phi, sideband ON during the pulses.
Every 1.25 ns sample is propagated exactly (matrix exponential). No chi.

Timeline: [ramp up] s_0 [pi_1] s_1 [pi_2] ... [pi_N] s_N [ramp down], s_k = sideband-only segments (ns).
    "full ends":      s_0 = s_N = s (all segments equal)
    "symmetric ends": s_0 = s_N = s/2, interior s: the segment sequence is a palindrome -> even Magnus orders cancel
Drive-sign words: all X; X X Xbar Xbar; X Xbar alternating (Xbar = phase pi). For INSTANTANEOUS pulses these are the same
toggling sequence (only 2 phi enters); any difference comes from the finite pulse.

Fixed beta: for each sequence the sideband strength g <= g_max is fitted so the gate is CD(beta = 1) = exp[beta sigma (e^{i chi}
a^dag - h.c.)] with fitted qubit axis sigma = cos(t) sx + sin(t) sy, displacement direction chi and a closing qubit Rz (frame
choices). Fidelity: coherent, 18 inputs (g/e x n = 0..8). Output: runs/claude_research/cd_magnus/families.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "runs", "claude_research", "cd_magnus")
DT = 0.00125
G_MAX = np.pi / (2 * 0.200)
LEVELS = 48
PI_SAMPLES = np.pi / 0.030 * np.sin(np.pi * np.arange(24) / 24) ** 2          # rad/us, area exactly pi/2 -> exp(-i pi/2 sx)

a = np.diag(np.sqrt(np.arange(1, LEVELS)), 1)
SP = np.array([[0, 0], [1, 0]], complex)                                       # |e><g|, order (g, e)
SX, SY, SZ = np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]).astype(complex)
JC = np.kron(SP, a)                                                            # sigma+ a   (qubit index major)
QUBIT_UP = np.kron(SP, np.eye(LEVELS))
INPUT_COLUMNS = [m for m in range(9)] + [LEVELS + m for m in range(9)]        # g_m, e_m


def hamiltonian(g, q):
    """g (sigma+ a + h.c.) + q sigma+ + conj(q) sigma-"""
    H = g * JC + q * QUBIT_UP
    return H + H.conj().T


def sequence_unitary(g, segments_ns, phases):
    """Exact propagator of: ramp up, s_0, pi(phases[0]), s_1, ..., pi(phases[-1]), s_N, ramp down."""
    U = np.eye(2 * LEVELS, dtype=complex)
    free = expm(-1j * DT * hamiltonian(g, 0.0))
    for j in range(4):
        U = expm(-1j * DT * hamiltonian(g * np.sin(np.pi * j / 8) ** 2, 0.0)) @ U
    pulse = {}
    for k, phi in enumerate(phases):
        U = np.linalg.matrix_power(free, int(round(segments_ns[k] / 1.25))) @ U
        key = round(phi % (2 * np.pi), 9)
        if key not in pulse:
            P = np.eye(2 * LEVELS, dtype=complex)
            for omega in PI_SAMPLES:
                P = expm(-1j * DT * hamiltonian(g, omega * np.exp(1j * phi))) @ P
            pulse[key] = P
        U = pulse[key] @ U
    U = np.linalg.matrix_power(free, int(round(segments_ns[-1] / 1.25))) @ U
    for j in range(4):
        U = expm(-1j * DT * hamiltonian(g * np.cos(np.pi * j / 8) ** 2, 0.0)) @ U
    return U


def cd_target(beta, axis, chi, rz):
    D = expm(beta * (np.exp(1j * chi) * a.T - np.exp(-1j * chi) * a))
    sigma = np.cos(axis) * SX + np.sin(axis) * SY
    w, V = np.linalg.eigh(sigma)                                                # sigma = +-1 projectors
    plus, minus = np.outer(V[:, 1], V[:, 1].conj()), np.outer(V[:, 0], V[:, 0].conj())
    U = np.kron(plus, D) + np.kron(minus, D.conj().T)
    R = np.kron(np.diag(np.exp(-1j * rz * np.array([1, -1]) / 2)), np.eye(LEVELS))
    return R.conj().T @ U                                                       # compare R^dag U_target with U (rz frame)


def fidelity(U, beta, axis, chi, rz):
    T = cd_target(beta, axis, chi, rz)[:, INPUT_COLUMNS]
    return abs(np.sum(T.conj() * U[:, INPUT_COLUMNS]) / 18) ** 2


def fit(job):
    """Fit g (<= G_MAX), axis, chi, rz so the sequence makes CD(beta = 1) best. Returns the result row."""
    name, segments, phases = job
    T_us = 0.010 + sum(segments) * 1e-3 + 0.030 * len(phases)
    # guess: CD rate ~ g/2 over the whole time
    g0 = min(G_MAX, 2.0 / T_us)

    def frames(U):
        """best qubit axis, displacement direction and closing Rz for a fixed unitary (cheap: only the target changes)"""
        best = None
        for axis0 in (0.0, np.pi / 2):
            for chi0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                r = minimize(lambda p: 1 - fidelity(U, 1.0, *p), [axis0, chi0, 0.0], method="Nelder-Mead",
                             options=dict(xatol=1e-9, fatol=1e-13, maxiter=2000))
                if best is None or r.fun < best.fun:
                    best = r
        return best

    from scipy.optimize import minimize_scalar
    cache = {}

    def cost_g(g):
        r = frames(sequence_unitary(g, segments, phases))
        cache[g] = r
        return r.fun

    lo = 0.3 * g0
    grid = np.linspace(lo, G_MAX, 12)                                           # coarse scan, then refine around the best
    values = [cost_g(g) for g in grid]
    i = int(np.argmin(values))
    a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(cost_g, bounds=(a_, b_), method="bounded", options=dict(xatol=1e-6))
    g = min(cache, key=lambda k: cache[k].fun)
    best = cache[g]
    return dict(name=name, pulses=len(phases), segments_ns=list(segments), time_us=T_us, g=float(g),
                swap_ns=1e3 * np.pi / (2 * g), at_cap=bool(g > 0.995 * G_MAX), infidelity=float(best.fun),
                axis_deg=float(np.degrees(best.x[0])), chi_deg=float(np.degrees(best.x[1])), rz=float(best.x[2]))


def families():
    jobs = []
    for N in (4, 6, 8, 10, 12, 16):
        words = {"all X": [0.0] * N,
                 "X X Xbar Xbar": [0.0 if (k // 2) % 2 == 0 else np.pi for k in range(N)],
                 "X Xbar alternating": [0.0 if k % 2 == 0 else np.pi for k in range(N)]}
        for s in (0.0, 5.0, 10.0, 20.0):
            for word, phases in words.items():
                for ends in ("full ends", "symmetric ends"):
                    if ends == "full ends" and word != "all X":
                        continue
                    end = s if ends == "full ends" else s / 2
                    if ends == "symmetric ends" and (s / 2) % 1.25:
                        continue
                    segments = [end] + [s] * (N - 1) + [end]
                    jobs.append((f"{word}, {ends}, gap {s:g} ns", segments, phases))
    return jobs


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = families()
    with ProcessPoolExecutor(24) as pool:
        rows = list(pool.map(fit, jobs))
    assert len(rows) == len(jobs)
    json.dump(rows, open(os.path.join(OUT, "families.json"), "w"), indent=1)
    for r in sorted(rows, key=lambda r: (r["time_us"], r["infidelity"])):
        print(f"{r['time_us']*1e3:5.0f} ns  N={r['pulses']:2d}  1-F {r['infidelity']:.2e}  swap {r['swap_ns']:6.1f} ns "
              f"{'(AT CAP: beta=1 not reached)' if r['at_cap'] else ''}  {r['name']}")
