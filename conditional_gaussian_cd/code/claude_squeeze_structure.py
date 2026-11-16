"""Claude C5 analysis: what structure does a fixed-sideband squeeze train use? (qubit-frame vector paths)

In the frame of the qubit drive alone, R(t), the fixed sideband g (sigma+ a + h.c.) = (g/sqrt2)(sx xhat + sy phat) becomes
    H_I(t) = (g w(t)/sqrt2) [ n_x(t).sigma xhat + n_y(t).sigma phat ],   n_k(t) = Bloch vector of R^dag s_k R
(w = sideband envelope: ramps). Magnus, using [A.s B, C.s D] = i(A.C)[B,D]/... + i (A x C).s {B,D}/... :
    order 1:  M_x = int w n_x dt,  M_y = int w n_y dt                 -> must vanish (no leftover CD)
    order 2:  A_xp = iint_{t>t'} w w' (n_x(t) x n_y(t') - n_y(t) x n_x(t'))  -> the squeeze sigma{xhat, phat}; must be along z,
              with |A_xp| = 2 r / g^2 for r = 8 ln10 / 20
              A_xx = iint w w' n_x(t) x n_x(t'),  A_pp = iint w w' n_y(t) x n_y(t')  -> sigma.xhat^2, sigma.phat^2 errors: must vanish
Rotation conventions are taken from cd_trains (same samples, phases, ramps), so this describes the simulated trains exactly
up to second order.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cd_trains as cd

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1.0, -1.0]).astype(complex)
SP = np.array([[0, 0], [1, 0]], complex)          # |e><g|, order (g, e) as in cd_trains


def bloch(M):
    return np.real([np.trace(SX @ M), np.trace(SY @ M), np.trace(SZ @ M)]) / 2


def paths(cells, gap, phases):
    """per 1.25 ns sample: sideband weight w and the vectors n_x, n_y (in the drive frame, sample midpoint)."""
    R = np.eye(2, dtype=complex)
    half_gap = int(round(gap / 2 / cd.DT))
    ws, nx, ny = [], [], []

    def record(w, step):
        nonlocal R
        Rm = step(0.5) @ R                                      # midpoint
        ws.append(w); nx.append(bloch(Rm.conj().T @ SX @ Rm)); ny.append(bloch(Rm.conj().T @ SY @ Rm))
        R = step(1.0) @ R

    idle = lambda f: np.eye(2, dtype=complex)
    for j in range(cd.RAMP_SAMPLES):
        record(np.sin(np.pi * j / 8) ** 2, idle)
    for phi in phases:
        for _ in range(half_gap):
            record(1.0, idle)
        for omega in cd.pi_samples():
            H = omega * (np.exp(1j * phi) * SP + np.exp(-1j * phi) * SP.conj().T)
            w_, V = np.linalg.eigh(H)
            step = lambda f, w_=w_, V=V: (V * np.exp(-1j * f * cd.DT * w_)) @ V.conj().T
            record(1.0, step)
        for _ in range(half_gap):
            record(1.0, idle)
    for j in range(cd.RAMP_SAMPLES):
        record(np.cos(np.pi * j / 8) ** 2, idle)
    return np.array(ws), np.array(nx), np.array(ny)


def terms(w, nx, ny):
    """first moments and second-order area vectors (in units of us and us^2; multiply by g/sqrt2 and g^2/2)."""
    dt = cd.DT
    wx, wy = w[:, None] * nx * dt, w[:, None] * ny * dt
    Mx, My = wx.sum(0), wy.sum(0)
    cx, cy = np.cumsum(wx, 0) - wx, np.cumsum(wy, 0) - wy             # sums over earlier samples
    A_xp = (np.cross(wx, cy) - np.cross(wy, cx)).sum(0)
    A_xx = np.cross(wx, cx).sum(0)
    A_pp = np.cross(wy, cy).sum(0)
    return Mx, My, A_xp, A_xx, A_pp


def report(result):
    j = result["job"]
    phases = np.array(result["qubit_phases"])
    w, nx, ny = paths(j["cells"], j["gap"], phases)
    Mx, My, A_xp, A_xx, A_pp = terms(w, nx, ny)
    g = np.pi / (2 * j["swap"])
    need = 2 * cd.R_8DB / g ** 2
    return dict(swap_ns=1e3 * j["swap"], cells=j["cells"], gap_ns=1e3 * j["gap"], time_us=result["time_us"],
                fidelity=result["fidelity"],
                first_order=float(g / np.sqrt(2) * max(np.linalg.norm(Mx), np.linalg.norm(My))),
                A_xp=(A_xp / need).tolist(), A_xx_rel=float(np.linalg.norm(A_xx) / need), A_pp_rel=float(np.linalg.norm(A_pp) / need))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    runs = json.load(open(os.path.join(here, "..", "runs", "claude_research", "fixed_squeeze", "runs.json")))
    best = {}
    for r in runs:
        k = (r["job"]["swap"], round(r["time_us"], 2))
        if k not in best or r["fidelity"] > best[k]["fidelity"]:
            best[k] = r
    rows = [report(r) for r in best.values()]
    json.dump(rows, open(os.path.join(here, "..", "runs", "claude_research", "fixed_squeeze", "structure.json"), "w"), indent=1)
    print("A_xp in units of the needed 2r/g^2 (should be (0, 0, +-1)); A_xx, A_pp relative to it (should be 0)")
    for r in sorted(rows, key=lambda r: (r["swap_ns"], r["time_us"])):
        ax = r["A_xp"]
        print(f"swap {r['swap_ns']:.0f} ns, {r['cells']} pi, gap {r['gap_ns']:.1f} ns, {r['time_us']:.2f} us: F {r['fidelity']:.4f}  "
              f"first order {r['first_order']:.3f}  A_xp ({ax[0]:+.2f}, {ax[1]:+.2f}, {ax[2]:+.2f})  A_xx {r['A_xx_rel']:.2f}  A_pp {r['A_pp_rel']:.2f}")
