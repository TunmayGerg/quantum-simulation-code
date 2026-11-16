"""Plots and simple measurements for conditional_gaussian_gates.ipynb.

    branch_states   apply a gate to (|g> + |e>)/sqrt2 x (cavity state) and return the cavity state of each qubit branch
    squeezing_db    squeezing of a cavity state in dB and the angle of its squeezed quadrature
    wigner_panels   Wigner functions of the g and e branches: simulated gate vs ideal target
    sequence_plot   qubit and pump phase of every pi pulse, and the envelope of the whole gate
    iq_plot         programmed DAC I/Q (steps, 800 MS/s) and what the board outputs (after the converter's 8x filter)
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import expm

import cd_trains as cd
import hardware_check as hw

GROUP_DELAY_NS = 22.65625            # delay of the RFDC interpolation filters (29, 11 and 7 taps at 1.6, 3.2, 6.4 GS/s)


def cavity_ops(levels):
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    return a


def branch_states(train, x, cavity_state):
    """Gate applied to (|g> + |e>)/sqrt2 x cavity_state; returns (g-branch, e-branch) normalised cavity kets."""
    levels = train.space.levels
    psi = np.zeros(2 * levels, complex)
    psi[0::2] = cavity_state / np.sqrt(2)
    psi[1::2] = cavity_state / np.sqrt(2)
    out = played_state(train,x,psi)
    g, e = out[0::2], out[1::2]
    return g / np.linalg.norm(g), e / np.linalg.norm(e)


def played_state(train,x,initial,frame_angle=None):
    """Actual SDK+RFDC waveform on any input, with one fixed common Z frame."""
    q,s,_=hw.played_arrays(train,x)
    if frame_angle is None:frame_angle=train.report(x)['qubit_frame_Rz']
    state=np.asarray(initial,complex)
    vector=state.ndim==1
    if vector:state=state[:,None]
    out=hw.propagate(q,s,state,cd.DT/8,order=8,fast=True)
    out*=np.exp(-1j*frame_angle*train.space.z/2)[:,None]
    return out[:,0] if vector else out


def toggling_magnus(train,x):
    """Finite-drive toggling-frame first/second Magnus terms, a diagnostic.

    In H_I=sum_mu sigma_mu(c_mu a+c_mu* a†), integrate the actual q waveform
    exactly per fine sample. Use the midpoint qubit frame for that sample's
    sideband coefficient. No equality of a truncated Magnus series with the
    full gate is assumed. Output G2 satisfies U_I=exp[-i(G1+G2+...)].
    """
    q,s,_=hw.played_arrays(train,x);dt=cd.DT/8
    paulis=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
    sp=np.array([[0,0],[1,0]],complex);Q=np.eye(2,dtype=complex)
    increments=np.empty((len(q),3),complex)
    for j,(drive,pump) in enumerate(zip(q,s)):
        magnitude=abs(drive)
        H=np.array([[0,drive.conjugate()],[drive,0]],complex)
        half=np.cos(magnitude*dt/2)*np.eye(2)-1j*(np.sin(magnitude*dt/2)/magnitude if magnitude else dt/2)*H
        mid=half@Q;B=mid.conj().T@sp@mid
        increments[j]=pump*dt*np.einsum('aij,ji->a',paulis,B)/2
        Q=half@mid
    first=np.cumsum(increments,axis=0)
    previous=first-increments
    aa=np.cumsum(np.cross(increments,previous),axis=0)
    nn=np.cumsum(np.cross(increments,previous.conj()).real,axis=0)
    return dict(time_us=np.arange(1,len(q)+1)*dt,first=first,aa=aa,nn=nn,
                final_qubit_unitary=Q)


def ideal_branches(train, cavity_state):
    """The target gate applied to the same input: (g-branch, e-branch)."""
    a = cavity_ops(train.space.levels)
    n = np.arange(train.space.levels)
    if train.gate == "squeeze":
        S = expm(train.strength / 2 * (a @ a - a.T @ a.T))
        return S @ cavity_state, S.T @ cavity_state
    return np.exp(-1j * train.strength * n) * cavity_state, np.exp(1j * train.strength * n) * cavity_state


def squeezing_db(ket):
    """Smallest quadrature variance relative to vacuum (in dB, positive = squeezed) and its angle (deg).
    Quadrature x_theta = (a e^{-i theta} + a^dag e^{i theta}) / sqrt2; vacuum variance 1/2."""
    a = cavity_ops(len(ket))
    mean_a = np.vdot(ket, a @ ket)
    mean_aa = np.vdot(ket, a @ a @ ket)
    mean_n = np.vdot(ket, a.T @ a @ ket).real
    thetas = np.linspace(0, np.pi, 3601)
    var = [0.5 + mean_n + np.real(mean_aa * np.exp(-2j * t)) - 2 * np.real(mean_a * np.exp(-1j * t)) ** 2 for t in thetas]
    i = int(np.argmin(var))
    return float(-10 * np.log10(var[i] / 0.5)), float(np.degrees(thetas[i]))


def coherent(levels, alpha):
    n = np.arange(levels)
    from scipy.special import gammaln
    return np.exp(-abs(alpha) ** 2 / 2 + n * np.log(abs(alpha) + 1e-300) - 0.5 * gammaln(n + 1)) * np.exp(1j * np.angle(alpha) * n)


def wigner_panels(train, x, cavity_state, title, extent=4.0):
    """2 x 2 Wigner functions: rows = qubit branch g / e, columns = simulated gate / ideal target."""
    sim = branch_states(train, x, cavity_state)
    ideal = ideal_branches(train, cavity_state)
    grid = np.linspace(-extent, extent, 161)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.6), constrained_layout=True)
    for row, branch in enumerate(("g", "e")):
        for col, (label, ket) in enumerate((("simulated gate", sim[row]), ("ideal target", ideal[row]))):
            W = qt.wigner(qt.Qobj(ket[:, None]), grid, grid)
            ax = axes[row, col]
            ax.contourf(grid, grid, W, 60, cmap="RdBu_r", vmin=-1 / np.pi, vmax=1 / np.pi)
            ax.set_aspect("equal")
            ax.set_title(f"qubit {branch}: {label}", fontsize=10)
            ax.set_xlabel("x = (a + a$^\\dagger$)/$\\sqrt{2}$")
            ax.set_ylabel("p = (a - a$^\\dagger$)/($i\\sqrt{2}$)")
    fig.suptitle(title)
    plt.show()
    return sim, ideal


def sequence_plot(train, x, title):
    """Top: qubit and pump phase of every pi pulse. Bottom: the drive envelopes of the whole gate."""
    t, q, s = cd.waveform(train, x)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5.5), constrained_layout=True)
    pulses = np.arange(1, train.cells + 1)
    ax1.plot(pulses, np.degrees(-x[:train.cells]) % 360, "o", ms=3, label="physical qubit DAC phase $-\\phi_j$")
    ax1.plot(pulses, np.degrees(x[train.cells:]) % 360, "s", ms=3, label="sideband (pump) phase $\\psi_j$ during cell j")
    ax1.set_xlabel("pi pulse number j"); ax1.set_ylabel("phase (deg)"); ax1.set_ylim(-5, 365); ax1.set_yticks(range(0, 361, 90))
    ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.3)
    ax2.plot(1e3 * t, abs(q) / (np.pi / cd.PI_TIME), lw=0.6, label="qubit drive |amplitude| / pi-pulse peak")
    ax2.plot(1e3 * t, abs(s) / train.g, lw=1.2, label="sideband |coupling| / g")
    ax2.set_xlabel("time (ns)"); ax2.set_ylabel("envelope"); ax2.legend(fontsize=8, loc="lower right"); ax2.grid(alpha=0.3)
    fig.suptitle(title)
    plt.show()


def iq_plot(train, x, start_pulse=10, n_pulses=4, title=""):
    """Programmed DAC I/Q (what is written to the board, 800 MS/s, held for 1.25 ns) and the closest thing to what the
    board outputs: the same words after the converter's 8x interpolation filter (6.4 GS/s), shifted back by its fixed
    22.66 ns delay so the two line up. Units: fraction of DAC full scale."""
    t, q, s = cd.waveform(train, x)
    q_scale, s_scale = hw.dac_scales(train.swap)
    _, q_prog = hw.to_words_and_back(q, np.pi / cd.PI_TIME, q_scale,qubit=True)
    _, s_prog = hw.to_words_and_back(s, train.g, s_scale)
    q_dac, s_dac = q_prog.conj() / (np.pi / cd.PI_TIME) * q_scale, s_prog / train.g * s_scale
    q_out, s_out = hw.rfdc_interpolate(q_dac), hw.rfdc_interpolate(s_dac)   # same units (each stage: x2 zeros, gain 2)
    t_out = cd.DT / 8 * np.arange(len(q_out)) - GROUP_DELAY_NS * 1e-3
    cell = cd.PI_TIME + train.gap
    t0 = cd.RAMP_SAMPLES * cd.DT + start_pulse * cell
    t1 = t0 + n_pulses * cell
    keep, keep_out = (t >= t0) & (t <= t1), (t_out >= t0) & (t_out <= t1)
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, constrained_layout=True)
    for ax, prog, out, name in ((axes[0], q_dac, q_out, "qubit drive"), (axes[1], s_dac, s_out, "sideband (pump)")):
        for part, colour in (("I", "C0"), ("Q", "C3")):
            p = prog.real if part == "I" else prog.imag
            o = out.real if part == "I" else out.imag
            ax.step(1e3 * t[keep], p[keep], where="post", color=colour, lw=1, alpha=0.6, label=f"{part} programmed (800 MS/s)")
            ax.plot(1e3 * t_out[keep_out], o[keep_out], color=colour, lw=1.8, label=f"{part} out of the board (filtered)")
        ax.set_ylabel(f"{name}\n(fraction of full scale)"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=4, loc="upper right")
    axes[1].set_xlabel("time (ns)")
    fig.suptitle(title or f"pi pulses {start_pulse + 1}-{start_pulse + n_pulses}: programmed vs board output")
    plt.show()


def pulse_diagram(train, x, title, per_row=40):
    """The whole gate as a pulse diagram, wrapped into rows of `per_row` pi pulses. In each row the upper line is the
    qubit drive (one bar per 30 ns pi pulse), the lower line the sideband (one bar per cell, constant amplitude);
    colour = phase. White space on the qubit line = gaps (sideband only)."""
    cmap = plt.get_cmap("twilight")
    colour = lambda phase: cmap((phase % (2 * np.pi)) / (2 * np.pi))
    cell = cd.PI_TIME + train.gap
    ramp = cd.RAMP_SAMPLES * cd.DT
    rows = int(np.ceil(train.cells / per_row))
    fig, axes = plt.subplots(rows, 1, figsize=(14, 1.15 * rows + 0.9), squeeze=False, constrained_layout=True)
    for r, ax in enumerate(axes[:, 0]):
        for j in range(r * per_row, min((r + 1) * per_row, train.cells)):
            start = ramp + j * cell
            ax.barh(1, 1e3 * cell, left=1e3 * start, height=0.7, color=colour(x[train.cells + j]), edgecolor="none")
            ax.barh(2, 1e3 * cd.PI_TIME, left=1e3 * (start + train.gap / 2), height=0.7, color=colour(-x[j]),
                    edgecolor="k", lw=0.3)
        if r == 0:
            ax.barh(1, 1e3 * ramp, left=0, height=0.7, color="0.7")
        if r == rows - 1:
            ax.barh(1, 1e3 * ramp, left=1e3 * (ramp + train.cells * cell), height=0.7, color="0.7")
        ax.set_yticks([1, 2], ["sideband", "qubit pi"])
        ax.set_ylim(0.4, 2.6)
        ax.set_xlim(1e3 * (r * per_row * cell), 1e3 * ((r + 1) * per_row * cell + 2 * ramp))
        ax.text(1.003, 0.5, f"pulses {r * per_row + 1}-{min((r + 1) * per_row, train.cells)}", transform=ax.transAxes,
                fontsize=8, va="center")
    axes[-1, 0].set_xlabel("time (ns)")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 360)), ax=axes[:, 0], label="phase (deg)",
                 pad=0.06, fraction=0.02)
    fig.suptitle(title)
    plt.show()
