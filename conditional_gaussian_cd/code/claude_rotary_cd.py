"""Claude task C4: rotary-frame design of a fixed-sideband CD (concept in runs/claude_research/STATUS.md).

    ../../acadia_env/bin/python claude_rotary_cd.py rank        # enumerate words, rank by the cancellation conditions
    ../../acadia_env/bin/python claude_rotary_cd.py simulate    # exact simulation of the best words, beta = 1 fixed

Physics. In the frame that follows the qubit's rotation about x by theta(t), the fixed sideband g (sigma+ a + h.c.) is
    H_I = (g/sqrt2) [ sx xhat + (sy cos theta - sz sin theta) phat ].
The CD term sx xhat is untouched by x rotations, so beta = (g/2) x (time the sideband is on). Every error is the second
term. With u = e^{i theta} and w(t) the sideband envelope (ramps), the low orders vanish when
    order 1:      C1 = int w u dt = 0
    order 2 (a):  C2a = int w (t - T/2) u dt = 0          (conditional squeeze-like error  s_perp {xhat, phat})
    order 2 (b):  C2b = Im iint_{t > t'} w w' u(t) conj(u(t')) = 0     (error  sx phat^2)
A word = K slots of 30 ns; each slot rotates the qubit about x by +pi, -pi, +pi/2, -pi/2 (same 30 ns Hann envelope at
1/N amplitude) or idles (sideband only). Total rotation must be 0 mod 2 pi. Ramps: 5 ns sin^2 up and down, no qubit drive.
"""
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_cd_magnus as base                                    # same exact model (hamiltonian, frames fit, target)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "claude_research", "rotary_cd")
DT = base.DT
G_MAX = base.G_MAX
LETTERS = {"X": np.pi, "Xbar": -np.pi, "x": np.pi / 2, "xbar": -np.pi / 2, "-": 0.0}
NAMES = list(LETTERS)
ANGLES = np.array([LETTERS[k] for k in NAMES])
HANN = np.sin(np.pi * np.arange(24) / 24) ** 2
F = np.cumsum(HANN) / HANN.sum()                                    # rotation fraction completed after each sample
F_MID = F - HANN / HANN.sum() / 2                                   # at the sample midpoint
RAMP_UP = np.sin(np.pi * np.arange(4) / 8) ** 2
RAMP_DOWN = np.cos(np.pi * np.arange(4) / 8) ** 2


def slot_tables():
    """per letter: S1 = sum u dt, S2 = sum t_local u dt, intra = Im sum_{j>k in slot} u_j conj(u_k) dt^2 (relative to the
    slot's start angle; t_local from the slot start)."""
    t_local = (np.arange(24) + 0.5) * DT
    S1, S2, intra = [], [], []
    for a in ANGLES:
        u = np.exp(1j * a * F_MID)
        S1.append(u.sum() * DT)
        S2.append((t_local * u).sum() * DT)
        cum = np.cumsum(u) - u                                      # sum over earlier samples in the slot
        intra.append(np.imag(np.sum(u * np.conj(cum))) * DT ** 2)
    return np.array(S1), np.array(S2), np.array(intra)


def conditions(words):
    """words: int array (n_words, K) of letter indices. Returns C1, C2a, C2b (arrays over words) for the sideband
    normalised to 1 (multiply C1 by g, C2 by g^2 for the actual size)."""
    S1, S2, intra = slot_tables()
    n, K = words.shape
    T = 2 * 4 * DT + K * 24 * DT
    ramp_t = (np.arange(4) + 0.5) * DT
    # ramp up (theta = 0)
    C1 = np.full(n, RAMP_UP.sum() * DT, complex)
    C2a = np.full(n, ((ramp_t - T / 2) * RAMP_UP).sum() * DT, complex)
    C2b = np.zeros(n)                                              # u = 1 throughout the ramp: no imaginary part
    theta = np.zeros(n)
    t0 = 4 * DT
    for k in range(K):
        a = ANGLES[words[:, k]]
        e = np.exp(1j * theta)
        slot1 = e * S1[words[:, k]]
        C2b += intra[words[:, k]] + np.imag(slot1 * np.conj(C1))
        C2a += e * S2[words[:, k]] + (t0 - T / 2) * slot1
        C1 += slot1
        theta += a
        t0 += 24 * DT
    e = np.exp(1j * theta)                                         # ramp down at the final angle
    down = RAMP_DOWN * DT
    for j in range(4):
        piece = e * down[j]
        C2b += np.imag(piece * np.conj(C1))
        C2a += (t0 + (j + 0.5) * DT - T / 2) * piece
        C1 += piece
    return C1, C2a, C2b, theta


def rank(K, keep=400):
    letters = len(NAMES)
    best = []
    total = letters ** K
    chunk = 200000
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total))
        words = np.stack([(idx // letters ** (K - 1 - k)) % letters for k in range(K)], axis=1)
        rot = ANGLES[words].sum(axis=1)
        ok = np.isclose(np.mod(rot + 1e-9, 2 * np.pi), 0, atol=1e-6) | np.isclose(np.mod(rot + 1e-9, 2 * np.pi), 2 * np.pi, atol=1e-6)
        words = words[ok]
        if len(words) == 0:
            continue
        C1, C2a, C2b, _ = conditions(words)
        T_eff = (K * 24 + RAMP_UP.sum() + RAMP_DOWN.sum()) * DT
        g = 2.0 / T_eff                                            # beta = 1
        score = (g * abs(C1)) ** 2 + (g ** 2 * abs(C2a)) ** 2 + (g ** 2 * C2b) ** 2
        order = np.argsort(score)[:keep]
        best += [(float(score[i]), words[i].tolist(), float(abs(C1[i])), float(abs(C2a[i])), float(C2b[i])) for i in order]
        best = sorted(best)[:keep]
    return [dict(K=K, score=s, word=[NAMES[i] for i in w], C1=c1, C2a=c2a, C2b=c2b) for s, w, c1, c2a, c2b in best]


def sequence_unitary(g, word):
    """exact propagator: ramp up, the K slots (x rotations, sideband on), ramp down."""
    U = np.eye(2 * base.LEVELS, dtype=complex)
    for w in RAMP_UP:
        U = expm(-1j * DT * base.hamiltonian(g * w, 0.0)) @ U
    cache = {}
    for letter in word:
        if letter not in cache:
            a = LETTERS[letter]
            P = np.eye(2 * base.LEVELS, dtype=complex)
            for omega in base.PI_SAMPLES:
                P = expm(-1j * DT * base.hamiltonian(g, omega * a / np.pi)) @ P     # +-pi or +-pi/2: amplitude x a/pi
            cache[letter] = P
        U = cache[letter] @ U
    for w in RAMP_DOWN:
        U = expm(-1j * DT * base.hamiltonian(g * w, 0.0)) @ U
    return U


def simulate(row):
    word = row["word"]
    T_eff = (len(word) * 24 + RAMP_UP.sum() + RAMP_DOWN.sum()) * DT
    g_guess = 2.0 / T_eff
    store = {}

    def cost(g):
        U = sequence_unitary(g, word)
        best = None
        for axis0 in (0.0, np.pi):
            for chi0 in (np.pi / 2, -np.pi / 2, 0.0, np.pi):
                r = minimize(lambda p: 1 - base.fidelity(U, 1.0, *p), [axis0, chi0, 0.0], method="Nelder-Mead",
                             options=dict(xatol=1e-9, fatol=1e-13, maxiter=2000))
                if best is None or r.fun < best.fun:
                    best = r
        store[g] = best
        return best.fun

    hi = min(G_MAX, 1.15 * g_guess)
    minimize_scalar(cost, bounds=(0.85 * g_guess, hi), method="bounded", options=dict(xatol=1e-6))
    g = min(store, key=lambda k: store[k].fun)
    return dict(row, time_ns=1e3 * (len(word) * 0.030 + 0.010), g=float(g),
                swap_ns=float(1e3 * np.pi / (2 * g)), infidelity=float(store[g].fun),
                at_cap=bool(g >= 0.999 * G_MAX))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "rank11":                                            # K = 11 with the four rotation letters only
        keep_letters = [NAMES.index(k) for k in ("X", "Xbar", "x", "xbar")]
        NAMES[:] = [NAMES[i] for i in keep_letters]
        ANGLES = ANGLES[keep_letters]
        r = rank(11)
        json.dump(r, open(os.path.join(OUT, "ranked11.json"), "w"))
        with ProcessPoolExecutor(24) as pool:
            res = list(pool.map(simulate, r[:96]))
        json.dump(res, open(os.path.join(OUT, "simulated11.json"), "w"), indent=1)
        for x in sorted(res, key=lambda x: x["infidelity"])[:12]:
            print(f"K=11 {x['time_ns']:.0f} ns  1-F {x['infidelity']:.2e}  swap {x['swap_ns']:.1f} ns  score {x['score']:.2e}  {' '.join(x['word'])}")
    if what == "rank":
        rows = []
        for K in (9, 10):
            r = rank(K)
            rows += r
            print(f"K = {K}: best scores {[round(x['score'], 6) for x in r[:5]]}; e.g. {' '.join(r[0]['word'])}", flush=True)
        json.dump(rows, open(os.path.join(OUT, "ranked.json"), "w"))
    elif what == "simulate":
        rows = json.load(open(os.path.join(OUT, "ranked.json")))
        picks = [r for K in (9, 10) for r in [x for x in rows if x["K"] == K][:120]]
        # references in the same model: today's 12-pulse rotary echo, and the 10-slot X X Xbar Xbar X X Xbar Xbar X Xbar ...
        with ProcessPoolExecutor(24) as pool:
            res = list(pool.map(simulate, picks))
        json.dump(res, open(os.path.join(OUT, "simulated.json"), "w"), indent=1)
        for r in sorted(res, key=lambda r: r["infidelity"])[:25]:
            print(f"K={r['K']} {r['time_ns']:.0f} ns  1-F {r['infidelity']:.2e}  swap {r['swap_ns']:.1f} ns  score {r['score']:.2e}  "
                  f"{' '.join(r['word'])}")
