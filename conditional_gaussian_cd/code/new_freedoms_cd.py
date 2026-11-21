"""Fast CD and conditional squeezing with the user's sideband freedoms of 2026-11-20.

WHAT IS ALLOWED (user, 2026-11-20)
  * The sideband phase may change along a 10 ns constant-radius quintic arc at ANY time
    (not only at qubit-envelope zeros), or
  * the sideband may ramp down and back up over 30 ns (a "dip"), aligned with one qubit
    pulse; its phase may jump while the amplitude is zero.
  * Otherwise the sideband amplitude (scale) is the same during and between qubit pulses.
  * |g(t)| <= pi/(2 x 200 ns) = 7.854 rad/us, including the RFDC filter overshoot.
  * Every qubit pulse is the same 30 ns Hann pi pulse; its only per-pulse freedom is the
    qubit phase. (Identical pi pulses also force an EVEN pulse count: an odd number of
    equatorial pi rotations is a net pi rotation, so the qubit could not close.)

MODEL (same as the rest of this folder; units us and rad/us)
  H(t) = q(t) sigma+ + s(t) sigma+ a + h.c.,  sigma_z = +1 on g,
  q = 30 ns Hann pi pulse x exp(i phi_j), s = g x envelope(t) x exp(i psi(t)).
  The 800 MS/s samples are packed into SDK int16 words, passed through the documented
  RFDC 8x interpolator (full tail) and propagated sample by sample (hardware_check.py).
  The gradient uses the folder's checked adjoint (waveform_adjoint.cpp).

FIGURE OF MERIT (unchanged): coherent process fidelity |Tr(T^dag V)|^2/18^2 on the
18 inputs g/e x Fock 0..8, relative phases and leakage counted. An optional final qubit
Rz is a declared virtual frame (it costs no time; it is reported). We also report the
state fidelity for the vacuum inputs |g,0> and |e,0>, because the device uses CDs mostly
from vacuum or cats.

SPEED BOUND: Heisenberg's equation gives d<a>/dt = -i s*(t) <sigma->, and |<sigma->| <= 1/2,
so |<a>(T)-<a>(0)| <= (1/2) int |s| dt for ANY qubit control and ANY sideband phase.
The CD target moves |+,0> to |+,beta>, hence beta <= (1/2) int |g| dt <= g_max T/2.
Dips only lower int|g|. `bound` checks this numerically on random controls.

Usage (from the project root, OPENBLAS_NUM_THREADS=1):
  python code/new_freedoms_cd.py reference      # replay the 370 ns reference with verify_research
  python code/new_freedoms_cd.py bound          # speed bound: derivation check on many waveforms
  python code/new_freedoms_cd.py gradient       # finite-difference check of the new model's gradient
  python code/new_freedoms_cd.py cd_search      # CD searches (see the function for the families)
  python code/new_freedoms_cd.py certify <recipe.json> ...
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from project_paths import ROOT, CODE
import cd_trains as cd
import hardware_check as hw
from shaped_pulse_optimisation import PlayedGate

OUT = ROOT / 'runs' / 'new_freedoms_2026_11_20'
OUT.mkdir(exist_ok=True, parents=True)
MAX_G = np.pi / (2 * 0.200)          # rad/us: the 200 ns vacuum-swap limit
ARC = 8                              # a 10 ns arc spans 8 DAC intervals (centre +-4 samples)
DIP = 24                             # a 30 ns dip spans 24 DAC samples (one qubit pulse)


def quintic_step(count, centre):
    """0 before centre-4, quintic 10u^3-15u^4+6u^5 over the 10 ns arc, 1 after centre+4."""
    step = np.zeros(count)
    u = np.arange(ARC + 1) / ARC
    step[centre - 4:centre + 5] = 10 * u**3 - 15 * u**4 + 6 * u**5
    step[centre + 5:] = 1.0
    return step


class FreedomGate:
    """N identical 30 ns pi pulses plus a constant-amplitude sideband with allowed phase moves.

    pulses      number N of qubit pi pulses (each 24 samples, start-of-sample Hann)
    gaps        samples of qubit-idle time BEFORE each pulse (gaps[0] = 0: the first pulse
                starts right after the 5 ns sideband ramp). Sum must keep the length a
                multiple of 4 samples (SDK requirement).
    arc_centres DAC sample indices of 10 ns phase arcs (anywhere on the plateau, no overlap)
    dips        indices of pulses under which the sideband dips (cos^2 down-up over 24 samples);
                each dip carries a free phase jump at its zero
    qubit_map   matrix A so that the qubit phases are phi = A @ x + qubit_offset (default: all free)
    frame       allow one final virtual qubit Rz (declared and reported)

    Parameters p = [x (qubit), psi0, arc steps..., dip jumps..., log g].
    """

    def __init__(self, pulses, gaps=None, arc_centres=(), dips=(), levels=32, gate='cd',
                 strength=1.0, frame=True, qubit_map=None, qubit_offset=None, n_input=9,
                 phase_map=None, rfdc=True):
        self.pulses, self.levels, self.gate, self.strength = pulses, levels, gate, strength
        self.frame = frame
        # rfdc=True: SDK words + 8x RFDC interpolation (the certified model).
        # rfdc=False: piecewise-constant 1.25 ns DAC samples (design proxy for long gates,
        # identical physics to the folder's cell model); final numbers always use rfdc=True.
        self.rfdc = rfdc
        self.dt, self.order = (cd.DT / 8, 8) if rfdc else (cd.DT, 12)
        self.gaps = np.zeros(pulses, int) if gaps is None else np.asarray(gaps, int)
        assert len(self.gaps) == pulses and np.all(self.gaps >= 0)
        self.starts = 4 + 24 * np.arange(pulses) + np.cumsum(self.gaps)
        self.count = 8 + 24 * pulses + int(self.gaps.sum())
        assert self.count % 4 == 0, 'SDK waveforms need a multiple of 4 samples'

        # Qubit: identical Hann pi pulses; qcell[i] = which pulse sample i belongs to.
        self.qshape = np.zeros(self.count, complex)
        self.qcell = np.full(self.count, -1, int)
        for j, start in enumerate(self.starts):
            self.qshape[start:start + 24] = cd.pi_samples()
            self.qcell[start:start + 24] = j

        # Sideband amplitude: 5 ns outer ramps (same convention as the reference) and dips.
        self.envelope = np.ones(self.count)
        self.envelope[:4] = np.sin(np.pi * np.arange(4) / 8) ** 2
        self.envelope[-4:] = np.cos(np.pi * np.arange(4) / 8) ** 2
        self.dips = list(dips)
        dip_centres = []
        for j in self.dips:
            start = self.starts[j]
            self.envelope[start:start + DIP] *= np.cos(np.pi * np.arange(DIP) / DIP) ** 2
            dip_centres.append(start + DIP // 2)          # amplitude exactly zero here

        # Sideband phase: psi(t) = psi0 + sum_k step_k * arc_k(t) + sum_d jump_d * [t >= dip zero].
        self.arc_centres = sorted(int(c) for c in arc_centres)
        for a, b in zip(self.arc_centres, self.arc_centres[1:]):
            assert b - a >= ARC, 'arcs must not overlap'
        for c in self.arc_centres:
            assert 8 <= c <= self.count - 9, 'arc must lie on the plateau'
            for d in dip_centres:
                assert abs(c - d) >= 4 + DIP // 2, 'arc inside a dip'
        columns = [np.ones(self.count)] + [quintic_step(self.count, c) for c in self.arc_centres]
        for d in dip_centres:
            jump = np.zeros(self.count)
            jump[d:] = 1.0
            columns.append(jump)
        self.phase_W = np.stack(columns, axis=1)
        self.dip_centres = dip_centres

        # Qubit phase parametrization.
        self.qubit_map = np.eye(pulses) if qubit_map is None else np.asarray(qubit_map, float)
        self.qubit_offset = np.zeros(pulses) if qubit_offset is None else np.asarray(qubit_offset, float)
        self.nq = self.qubit_map.shape[1]
        # Sideband phase parametrization: [psi0, arc steps..., dip jumps...] = phase_map @ y.
        self.phase_map = np.eye(self.phase_W.shape[1]) if phase_map is None else np.asarray(phase_map, float)
        assert self.phase_map.shape[0] == self.phase_W.shape[1]
        self.nphase = self.phase_map.shape[1]
        self.nparams = self.nq + self.nphase + 1

        # Largest constant g whose FILTERED peak (incl. ramp overshoot) stays below the cap.
        self.maximum_g = MAX_G * 0.997 / np.max(abs(hw.rfdc_interpolate(self.envelope)))

        # Target, inputs, adjoint library (shared with the rest of the folder).
        space = cd.Space(levels, n_input)
        self.z = space.z
        self.initial = np.eye(2 * levels, dtype=complex)[:, space.inputs]
        self.target = cd.target_columns(space, gate, strength)
        helper = PlayedGate('cd', 1, 12, False, False)  # compiles/loads waveform_adjoint.cpp
        self.backward, self.filter = helper.backward, helper.filter

    # -------------------------------------------------------------------------------------
    def split(self, p):
        p = np.asarray(p, float)
        phi = self.qubit_map @ p[:self.nq] + self.qubit_offset
        phases = self.phase_map @ p[self.nq:self.nq + self.nphase]
        return phi, phases, float(np.exp(p[-1]))

    def raw(self, p):
        """Ideal 800 MS/s samples (model units, rad/us) before SDK packing."""
        phi, phases, g = self.split(p)
        q = self.qshape.copy()
        keep = self.qcell >= 0
        q[keep] *= np.exp(1j * phi[self.qcell[keep]])
        s = g * self.envelope * np.exp(1j * (self.phase_W @ phases))
        return q, s

    def arrays(self, p, quantized=True):
        q, s = self.raw(p)
        words = None
        if quantized:
            qs, ss = hw.dac_scales(.6)
            qw, q = hw.to_words_and_back(q, np.pi / cd.PI_TIME, qs, qubit=True)
            sw, s = hw.to_words_and_back(s, np.pi / (2 * .6), ss)
            words = dict(qubit=qw, sideband=sw)
        if self.rfdc:
            qf, sf = hw.rfdc_interpolate(q), hw.rfdc_interpolate(s)
        else:
            qf, sf = np.ascontiguousarray(q), np.ascontiguousarray(s)
        return q, s, qf, sf, words

    def propagate(self, p, quantized=True):
        q, s, qf, sf, words = self.arrays(p, quantized)
        V = hw.propagate(qf, sf, self.initial, self.dt, order=self.order, fast=True)
        return V, (q, s, qf, sf, words)

    def frame_angle(self, V):
        if not self.frame:
            return 0.0
        zg = np.vdot(self.target[0::2], V[0::2])
        ze = np.vdot(self.target[1::2], V[1::2])
        return float(np.angle(zg) - np.angle(ze))

    def evaluate(self, p, gradient=True, quantized=True):
        """1-F and its gradient. The optimal final Rz has zero first derivative, so the
        gradient at fixed frame is exact (envelope theorem)."""
        V, (q, s, qf, sf, words) = self.propagate(p, quantized)
        angle = self.frame_angle(V)
        frame = np.exp(-1j * angle * self.z / 2)[:, None]
        trace = np.vdot(self.target, frame * V)
        d = self.target.shape[1]
        cost = float(1 - abs(trace / d) ** 2)
        if not gradient:
            return cost
        adjoint = np.ascontiguousarray(frame.conj() * self.target)
        grad = np.zeros((len(qf), 4), float)
        self.backward(self.levels, d, len(qf), self.order, self.dt, float(trace.real), float(trace.imag),
                      qf.ctypes.data, sf.ctypes.data, V.ctypes.data, adjoint.ctypes.data, grad.ctypes.data)
        if self.rfdc:
            # Fine-sample gradients back to DAC samples through the interpolator.
            Gq = fftconvolve(grad[:, 0] + 1j * grad[:, 1], self.filter[::-1], mode='valid')[::8]
            Gs = fftconvolve(grad[:, 2] + 1j * grad[:, 3], self.filter[::-1], mode='valid')[::8]
        else:
            Gq, Gs = grad[:, 0] + 1j * grad[:, 1], grad[:, 2] + 1j * grad[:, 3]
        qb, sb = self.raw(p)
        keep = self.qcell >= 0
        dphi = np.bincount(self.qcell[keep], weights=np.real(Gq[keep].conj() * 1j * qb[keep]),
                           minlength=self.pulses)
        dphase = self.phase_map.T @ (self.phase_W.T @ np.real(Gs.conj() * 1j * sb))
        dlogg = float(np.real(np.vdot(Gs, sb)))
        return cost, np.r_[self.qubit_map.T @ dphi, dphase, dlogg]

    def report(self, p, levels=None):
        """Fidelity numbers at this model's cutoff (design value; certification converges)."""
        V, (q, s, qf, sf, words) = self.propagate(p)
        angle = self.frame_angle(V)
        V = np.exp(-1j * angle * self.z / 2)[:, None] * V
        return scores(V, self.target) | dict(qubit_frame_Rz=angle, command_ns=1000 * self.count * cd.DT,
                                              peak_g=float(max(abs(sf))),
                                              peak_swap_ns=float(1000 * np.pi / (2 * max(abs(sf)))),
                                              g=float(np.exp(p[-1])))

    def recipe(self, p, name, extra=None):
        """Write controls + metadata for certify(); every declared operation is checked there."""
        q, s = self.raw(p)
        phi, phases, g = self.split(p)
        controls = OUT / (name + '_controls.npz')
        np.savez_compressed(controls, q=q, s=s)
        meta = dict(name=name, gate=self.gate, strength=self.strength,
                    controls_file=str(controls.relative_to(ROOT)), g_nominal=g,
                    nominal_swap_ns=1000 * np.pi / (2 * g), psi0_rad=float(phases[0]),
                    phase_arcs=[dict(centre_sample=int(c), step_rad=float(phases[1 + k]),
                                     duration_ns=10., shape='quintic constant-radius')
                                for k, c in enumerate(self.arc_centres)],
                    dips=[dict(pulse=int(j), start_sample=int(self.starts[j]), zero_sample=int(c),
                               jump_rad=float(phases[1 + len(self.arc_centres) + k]),
                               duration_ns=30., shape='cos^2 down-up')
                          for k, (j, c) in enumerate(zip(self.dips, self.dip_centres))],
                    pulses=[dict(start_sample=int(self.starts[j]), samples=24, angle_fraction=1.,
                                 phase_rad=float(phi[j])) for j in range(self.pulses)],
                    gaps_samples=self.gaps.tolist(), research_fit=extra or {})
        frame_value = (extra or {}).get('qubit_frame_Rz', 0.0)
        meta['qubit_frame_Rz'] = float(frame_value)
        path = OUT / (name + '_recipe.json')
        path.write_text(json.dumps(meta, indent=2) + '\n')
        return path


def scores(V, T):
    """Process fidelity, weakest basis input, and the two vacuum-input state fidelities."""
    d = T.shape[1]
    overlaps = np.sum(T.conj() * V, axis=0)
    n_input = d // 2
    return dict(fidelity=float(abs(overlaps.sum() / d) ** 2),
                worst_basis_fidelity=float(np.min(abs(overlaps) ** 2)),
                vacuum_g0_fidelity=float(abs(overlaps[0]) ** 2),
                vacuum_e0_fidelity=float(abs(overlaps[n_input]) ** 2),
                basis_fidelities=(abs(overlaps) ** 2).tolist())


def fit(model, initial, budget=300, stop=0.0, step_bound=None, label=''):
    """L-BFGS-B on the exact SDK/RFDC objective. g is bounded by the filtered cap."""
    history = []
    best = [np.inf, np.asarray(initial, float).copy()]
    started = time.time()

    def objective(p):
        # Design on the unquantized samples: the int16 staircase is not differentiable and
        # can stall the line search. Reports and certificates use the exact SDK words.
        value, gradient = model.evaluate(p, quantized=False)
        history.append(value)
        if value < best[0]:
            best[:] = [value, p.copy()]
        return value, gradient

    def callback(p):
        if best[0] < stop or len(history) >= budget:
            raise StopIteration

    bounds = [(None, None)] * (model.nparams - 1) + [(np.log(1.0), np.log(model.maximum_g))]
    x0 = np.asarray(initial, float).copy()
    if step_bound is not None:
        # Each arc may turn the phase by at most step_bound within its 10 ns (no fast windings).
        for i in range(model.nq + 1, model.nq + model.nphase):
            bounds[i] = (-step_bound, step_bound)
            x0[i] = np.clip(np.angle(np.exp(1j * x0[i])), -step_bound + 1e-9, step_bound - 1e-9)
    x0[-1] = min(x0[-1], np.log(model.maximum_g) - 1e-9)
    message = 'stopped by budget or target'
    try:
        result = minimize(objective, x0, jac=True, method='L-BFGS-B', bounds=bounds, callback=callback,
                          options=dict(maxiter=budget, maxfun=budget, maxcor=30, maxls=40, ftol=1e-15, gtol=1e-10))
        message = str(result.message)
    except StopIteration:
        pass
    print(f'FIT {label}: F {1 - history[0]:.5f} -> {1 - best[0]:.5f}  evaluations {len(history)}  '
          f'{time.time() - started:.0f} s  ({message})', flush=True)
    return best[1], 1 - best[0], len(history)


# =========================================================================================
# Diagnostics shared by every study
# =========================================================================================
def plus_state_moments(V, n_input=9):
    """For inputs |+,n> = (|g,n>+|e,n>)/sqrt2, return <a> of the output (the CD should give
    exactly beta for every n). A deficit growing with n is the spin-locking shortfall."""
    rows = []
    for n in range(n_input):
        psi = ((V[:, n] + V[:, n_input + n]) / np.sqrt(2)).reshape(-1, 2)
        levels = np.arange(len(psi))
        a = np.sum(np.sqrt(levels[1:])[:, None] * psi[:-1].conj() * psi[1:])
        rows.append(complex(a))
    return rows


def bound_ratio(V, sf):
    """|<a>(T)| for input |+,0> divided by (1/2) int |s| dt of the played waveform."""
    a = plus_state_moments(V, V.shape[1] // 2)[0]
    area = 0.5 * np.sum(abs(sf)) * cd.DT / 8
    return abs(a), area, abs(a) / area


# =========================================================================================
# 1. Reproduce the 370 ns reference with the folder's own verifier (unchanged code)
# =========================================================================================
def reference():
    import verify_research as vr
    vr.OUT = OUT / 'reference_replay'
    vr.OUT.mkdir(exist_ok=True)
    result = vr.certify(ROOT / 'runs/research_verified/cd_constant_reference_recipe.json')
    columns = np.load(vr.OUT / 'cd_constant_reference_columns.npz')
    extra = scores(columns['output'], columns['target'])
    moments = plus_state_moments(columns['output'])
    summary = dict(certified_fidelity=result['fidelity'], worst_basis=result['worst_basis_fidelity'],
                   levels=result['levels'], columns=result['columns'],
                   complex_amplitudes=result['complex_amplitudes'],
                   independent_max_difference=result['independent_max_difference'],
                   peak_swap_ns=result['peak_swap_ns'], command_ns=1000 * result['command_us'],
                   vacuum_g0_fidelity=extra['vacuum_g0_fidelity'], vacuum_e0_fidelity=extra['vacuum_e0_fidelity'],
                   plus_n_displacement=[[m.real, m.imag] for m in moments],
                   previously_certified=0.9922779449668537)
    (OUT / 'reference_replay.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('REFERENCE REPLAY F', result['fidelity'], 'previous 0.9922779449668537 difference',
          result['fidelity'] - 0.9922779449668537, 'vacuum g0/e0', extra['vacuum_g0_fidelity'],
          extra['vacuum_e0_fidelity'], flush=True)
    return summary


# =========================================================================================
# 2. Speed bound: |<a>(T)| <= (1/2) int |s| dt for any qubit control and sideband phase
# =========================================================================================
def bound(samples=60):
    """Random qubit phases, random arcs anywhere, random dips, random g <= cap, random
    lengths: the displacement of |+,0> never exceeds half the played sideband area.
    Also report how close the CD references come to the bound."""
    rng = np.random.default_rng(20261120)
    rows = []
    for trial in range(samples):
        pulses = int(rng.choice([4, 6, 8, 10, 12]))
        gaps = np.r_[0, 4 * rng.integers(0, 3, pulses - 1)]
        count = 8 + 24 * pulses + gaps.sum()
        # Arcs at random non-overlapping centres anywhere on the plateau.
        centres, c = [], 8 + int(rng.integers(0, 8))
        while c <= count - 9:
            if rng.random() < 0.4:
                centres.append(c)
                c += ARC + int(rng.integers(0, 12))
            else:
                c += int(rng.integers(1, 12))
        dips = sorted(set(rng.choice(pulses, size=int(rng.integers(0, 3)), replace=False).tolist()))
        dips = [j for j in dips if all(abs(cc - (4 + 24 * j + np.cumsum(gaps)[j] + 12)) >= 16 for cc in centres)]
        model = FreedomGate(pulses, gaps, centres, dips, levels=24)
        p = np.r_[rng.uniform(-np.pi, np.pi, pulses), rng.uniform(-np.pi, np.pi, model.nphase),
                  np.log(rng.uniform(2.0, model.maximum_g))]
        V, (q, s, qf, sf, words) = model.propagate(p)
        a, area, ratio = bound_ratio(V, sf)
        rows.append(dict(trial=trial, pulses=pulses, arcs=len(centres), dips=len(dips),
                         displacement=a, half_area=area, ratio=ratio))
    # The same check for the rotary CD family at the cap (how close a real CD gets).
    for pulses in [8, 10, 12]:
        model = FreedomGate(pulses, levels=24, frame=False)
        p = np.r_[np.resize([0, 0, np.pi, np.pi], pulses), -np.pi / 2, np.log(model.maximum_g * 0.999)]
        V, (q, s, qf, sf, words) = model.propagate(p)
        a, area, ratio = bound_ratio(V, sf)
        rows.append(dict(trial=f'rotary_n{pulses}_cap', pulses=pulses, arcs=0, dips=0,
                         displacement=a, half_area=area, ratio=ratio))
    worst = max(r['ratio'] for r in rows)
    assert worst <= 1 + 1e-9 and len(rows) == samples + 3
    payload = dict(waveforms=len(rows), random_waveforms=samples, max_ratio=worst, rows=rows,
                   statement='|<a>(T)| <= (1/2) int |s| dt, exact consequence of d<a>/dt = -i s* <sigma-> and |<sigma->| <= 1/2')
    (OUT / 'speed_bound_check.json').write_text(json.dumps(payload, indent=2) + '\n')
    print('SPEED BOUND:', len(rows), 'waveforms compared; max displacement/half-area', worst,
          '; rotary at cap', [round(r['ratio'], 5) for r in rows[-3:]], flush=True)
    return payload


def gradient_check():
    rng = np.random.default_rng(7)
    model = FreedomGate(8, gaps=[0, 0, 4, 0, 0, 0, 4, 0], arc_centres=[30, 47, 120, 160], dips=[3], levels=20)
    p = np.r_[rng.normal(0, .3, 8), -np.pi / 2 + rng.normal(0, .3, model.nphase), np.log(6.)]
    value, gradient = model.evaluate(p, quantized=False)
    errors = []
    for _ in range(4):
        v = rng.normal(size=len(p))
        v /= np.linalg.norm(v)
        e = 1e-5
        fd = (model.evaluate(p + e * v, False, False) - model.evaluate(p - e * v, False, False)) / (2 * e)
        errors.append(float(abs(fd - gradient @ v)))
    assert len(errors) == 4 and max(errors) < 2e-5, errors
    (OUT / 'gradient_check.json').write_text(json.dumps(dict(directions=4, errors=errors), indent=2) + '\n')
    print('GRADIENT CHECK: 4 directions; max finite-difference error', max(errors), flush=True)


# =========================================================================================
# 3. Certification of a new-freedom recipe (same standard as verify_research.certify)
# =========================================================================================
def rebuild_sideband(meta, count):
    """Expected sideband samples from the declared recipe alone (ramps, arcs, dips)."""
    envelope = np.ones(count)
    envelope[:4] = np.sin(np.pi * np.arange(4) / 8) ** 2
    envelope[-4:] = np.cos(np.pi * np.arange(4) / 8) ** 2
    phase = np.full(count, meta['psi0_rad'])
    previous = -10
    for arc in sorted(meta['phase_arcs'], key=lambda a: a['centre_sample']):
        c = arc['centre_sample']
        assert arc['duration_ns'] == 10. and arc['shape'] == 'quintic constant-radius'
        assert c - 4 >= previous and 8 <= c <= count - 9, 'overlapping or off-plateau arc'
        phase += arc['step_rad'] * quintic_step(count, c)
        previous = c + 4
    for dip in meta['dips']:
        start = dip['start_sample']
        assert dip['duration_ns'] == 30. and dip['zero_sample'] == start + DIP // 2
        envelope[start:start + DIP] *= np.cos(np.pi * np.arange(DIP) / DIP) ** 2
        phase[dip['zero_sample']:] += dip['jump_rad']
    return meta['g_nominal'] * envelope * np.exp(1j * phase), envelope


def certify(path, tolerance=1e-9):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    meta = json.loads(path.read_text())
    name = meta['name']
    data = np.load(ROOT / meta['controls_file'])
    q, s = np.asarray(data['q'], complex), np.asarray(data['s'], complex)
    count = len(q)
    assert q.shape == s.shape and count % 4 == 0
    # Qubit: every sample belongs to a declared identical 30 ns Hann pi pulse.
    expected = np.zeros(count, complex)
    covered = np.zeros(count, bool)
    for pulse in meta['pulses']:
        a, n = pulse['start_sample'], pulse['samples']
        assert n == 24 and pulse['angle_fraction'] == 1.0 and not covered[a:a + n].any()
        expected[a:a + n] = cd.pi_samples() * np.exp(1j * pulse['phase_rad'])
        covered[a:a + n] = True
    assert np.max(abs(q - expected)) < 1e-9, 'undeclared qubit waveform'
    # Sideband: constant scale except ramps/dips; phase only along declared arcs/dip jumps.
    s_expected, envelope = rebuild_sideband(meta, count)
    assert np.max(abs(s - s_expected)) < 1e-9, 'undeclared sideband waveform'
    for dip in meta['dips']:
        pulse_starts = [p['start_sample'] for p in meta['pulses']]
        assert dip['start_sample'] in pulse_starts, 'dip not aligned with a qubit pulse'
    plateau = envelope == 1.0
    # SDK words, RFDC, cap.
    qs, ss = hw.dac_scales(.6)
    qw, qd = hw.to_words_and_back(q, np.pi / cd.PI_TIME, qs, qubit=True)
    sw, sd = hw.to_words_and_back(s, np.pi / (2 * .6), ss)
    words = dict(qubit=qw, sideband=sw)
    qf, sf = hw.rfdc_interpolate(qd), hw.rfdc_interpolate(sd)
    peak = float(max(abs(sf)))
    assert max(max(abs(sd)), peak) <= MAX_G, ('peak coupling above the 200 ns swap cap', peak)
    # Filtered plateau magnitude ripple (arcs are constant radius before the filter).
    delay = np.argmax(abs(hw.rfdc_interpolate(np.r_[1.0, np.zeros(7)])))
    repeated = np.repeat(plateau, 8)
    fine_plateau = np.zeros(len(sf), bool)
    fine_plateau[delay:delay + len(repeated)] = repeated[:len(sf) - delay]
    # keep away from the edges of each plateau stretch by 40 fine samples (filter memory)
    edge = np.convolve(~fine_plateau, np.ones(81), mode='same') > 0
    inner = fine_plateau & ~edge
    ripple = [float(min(abs(sf[inner]) / abs(sd[plateau]).mean())), float(max(abs(sf[inner]) / abs(sd[plateau]).mean()))]
    angle = meta.get('qubit_frame_Rz', 0.0)
    previous, history = None, []
    strength = meta['strength']
    for levels in [64, 96, 128, 160, 192, 256, 320]:
        space = cd.Space(levels, 9)
        initial = np.eye(2 * levels, dtype=complex)[:, space.inputs]
        T = cd.target_columns(space, meta['gate'], strength)
        frame = np.exp(-1j * angle * space.z / 2)[:, None]
        V = frame * hw.propagate(qf, sf, initial, cd.DT / 8, order=8, fast=True)
        row = dict(levels=levels, fidelity=scores(V, T)['fidelity'])
        if previous is not None:
            old, oldT = previous
            pad = np.zeros_like(V); pad[:len(old)] = old
            padT = np.zeros_like(T); padT[:len(oldT)] = oldT
            row.update(output_change=float(max(np.linalg.norm(V - pad, axis=0))),
                       target_change=float(max(np.linalg.norm(T - padT, axis=0))))
        history.append(row)
        if previous is not None and max(row['output_change'], row['target_change']) < tolerance:
            break
        previous = V, T
    else:
        raise AssertionError((name, 'cutoff not converged'))
    independent = frame * hw.propagate(qf, sf, initial, cd.DT / 8, order=16)
    error = float(np.max(abs(independent - V)))
    assert error < 1e-10, error
    sc = scores(V, T)
    moments = plus_state_moments(V)
    from audit_cd import compile_words
    compiled = compile_words(words, OUT / (name + '_compiled'))
    result = dict(name=name, gate=meta['gate'], strength=strength, fidelity=sc['fidelity'],
                  worst_basis_fidelity=sc['worst_basis_fidelity'],
                  vacuum_g0_fidelity=sc['vacuum_g0_fidelity'], vacuum_e0_fidelity=sc['vacuum_e0_fidelity'],
                  basis_fidelities=sc['basis_fidelities'], accepted=bool(sc['fidelity'] >= 0.99),
                  command_ns=1000 * count * cd.DT, columns=T.shape[1], levels=levels,
                  complex_amplitudes=int(V.size), independent_max_difference=error,
                  peak_g=peak, peak_swap_ns=1000 * np.pi / (2 * peak), qubit_frame_Rz=angle,
                  filtered_plateau_magnitude_ratio=ripple, arcs=len(meta['phase_arcs']), dips=len(meta['dips']),
                  plus_n_displacement=[[m.real, m.imag] for m in moments], cutoff_history=history,
                  recipe_file=str(path.relative_to(ROOT)), compiled=compiled,
                  word_sha256=hashlib.sha256(qw.tobytes() + sw.tobytes()).hexdigest())
    np.savez_compressed(OUT / (name + '_words.npz'), **words)
    np.savez_compressed(OUT / (name + '_columns.npz'), output=V, target=T)
    with (OUT / (name + '_pulses.csv')).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pulse', 'start_ns', 'duration_ns', 'qubit_DAC_deg'])
        for j, pulse in enumerate(meta['pulses'], 1):
            writer.writerow([j, pulse['start_sample'] * 1.25, 30, round(-np.degrees(pulse['phase_rad']) % 360, 6)])
    (OUT / (name + '_certificate.json')).write_text(json.dumps(result, indent=2) + '\n')
    print('CERTIFIED', name, 'F', sc['fidelity'], 'worst', sc['worst_basis_fidelity'], 'vacuum g0/e0',
          sc['vacuum_g0_fidelity'], sc['vacuum_e0_fidelity'], 'T ns', result['command_ns'], 'levels', levels,
          '18 columns;', V.size, 'amplitudes; independent error', error, 'peak swap', result['peak_swap_ns'],
          flush=True)
    return result



# =========================================================================================
# 4. CD searches at fixed duration (beta = 1, all 18 inputs)
# =========================================================================================
def grid_arcs(model_count, first):
    """Contiguous 10 ns arcs: one every 8 samples from `first` to the end of the plateau.
    This is the MOST phase freedom the rules allow (any phase path with 10 ns resolution)."""
    return list(range(first, model_count - 8, ARC))


def mirror(j, n):
    """Index of the time-mirrored partner used by the symmetric families."""
    return min(j, n - 1 - j)


def bump_structure(n, starts, word):
    """Symmetric 'mid-pulse phase bump' family (few parameters).

    Every pulse j gets one smooth sideband-phase excursion: +eps_j along an arc centred
    10 ns into the pulse and -eps_j along an arc centred 20 ns into it, so the phase returns
    before the pulse ends. Time-mirror symmetry ties eps_(N-1-j) = eps_j and the qubit axes
    phi_(N-1-j) = phi_j + pi (the mirrored pulse undoes its partner), so the even Magnus
    orders cancel up to Hann sampling. Returns (arc centres, phase_map, qubit_map, offset)."""
    half = n // 2
    centres = []
    for j in range(n):
        centres += [int(starts[j]) + 8, int(starts[j]) + 16]
    phase_map = np.zeros((1 + 2 * n, 1 + half))
    phase_map[0, 0] = 1.0                                   # psi0
    for j in range(n):
        phase_map[1 + 2 * j, 1 + mirror(j, n)] = 1.0        # +eps up-arc
        phase_map[2 + 2 * j, 1 + mirror(j, n)] = -1.0       # -eps down-arc
    qubit_map = np.zeros((n, half))
    for j in range(n):
        qubit_map[j, mirror(j, n)] = 1.0
    offset = np.where(np.arange(n) >= half, np.pi, 0.0)
    word = np.asarray(word)
    assert np.all(word[::-1] == -word), 'symmetric families need an anti-palindromic X/-X word'
    return centres, phase_map, qubit_map, offset


def cd_job(job):
    """One fixed-duration fit (beta and Fock range from the job). Families:
       qubit_only   per-pulse qubit phases + one constant sideband phase (baseline)
       maxfree      qubit phases + contiguous 10 ns arcs on a grid (all allowed phase moves)
       dips         qubit phases + 30 ns dips with free phase jumps (no arcs)
       bumps_fixed  fixed X/-X word + one symmetric mid-pulse phase bump per pulse
       bumps_free   as bumps_fixed, qubit axes free but mirror-tied"""
    n, gaps = job['pulses'], job.get('gaps')
    count = 8 + 24 * n + (0 if gaps is None else sum(gaps))
    family = job['family']
    word = np.asarray(job['word'])[:n]
    rng = np.random.default_rng(job['seed'])
    strength, levels = job.get('strength', 1.0), job.get('levels', 32)
    if family.startswith('bumps'):
        starts = 4 + 24 * np.arange(n) + np.cumsum(np.zeros(n, int) if gaps is None else gaps)
        arcs, phase_map, qubit_map, offset = bump_structure(n, starts, word)
        if family == 'bumps_fixed':
            qubit_map, offset = np.zeros((n, 0)), np.where(word > 0, 0., np.pi)
        model = FreedomGate(n, gaps, arcs, (), levels=levels, frame=True, strength=strength,
                            qubit_map=qubit_map, qubit_offset=offset, phase_map=phase_map)
        x0 = (np.where(word[:n // 2] > 0, 0., np.pi) + rng.normal(0, .1, n // 2)) if family == 'bumps_free' else []
        y0 = np.r_[-np.pi / 2, rng.normal(0, job.get('arc_spread', 0.3), model.nphase - 1)]
    elif family == 'motif':
        # One block of m pulses (qubit phases + its 3m arcs on the 8-sample grid) repeated.
        m = job['motif']
        arcs = grid_arcs(count, job['grid_first'])
        qubit_map = np.zeros((n, m)); qubit_map[np.arange(n), np.arange(n) % m] = 1.0
        per = len([c for c in arcs if c < 4 + 24 * m + 4])     # arcs that belong to the first block
        phase_map = np.zeros((1 + len(arcs), 1 + per)); phase_map[0, 0] = 1.0
        for k in range(len(arcs)):
            phase_map[1 + k, 1 + k % per] = 1.0
        model = FreedomGate(n, gaps, arcs, (), levels=levels, frame=True, strength=strength,
                            qubit_map=qubit_map, phase_map=phase_map)
        x0 = np.where(word[:m] > 0, 0., np.pi) + rng.normal(0, .15, m)
        y0 = np.r_[-np.pi / 2, rng.normal(0, job.get('arc_spread', 0.4), model.nphase - 1)]
    else:
        arcs = grid_arcs(count, job['grid_first']) if family == 'maxfree' else []
        model = FreedomGate(n, gaps, arcs, job.get('dips', ()), levels=levels, frame=True, strength=strength)
        x0 = np.where(word > 0, 0., np.pi)                    # X for +1, -X for -1
        if job['seed']:
            x0 = x0 + rng.normal(0, .15, n)
        spread = job.get('arc_spread', 0.0) if job['seed'] else 0.0
        y0 = np.r_[-np.pi / 2, rng.normal(0, spread, model.nphase - 1)]
    g0 = min(job.get('g0', 2 * strength * 1.03 / (count * cd.DT - 0.006 - 0.015 * len(job.get('dips', ())))),
             model.maximum_g * 0.999)
    p0 = np.r_[x0, y0, np.log(g0)]
    if 'init_from' in job:                                     # continue an earlier fit of the same structure
        p0 = np.asarray(json.loads((OUT / (job['init_from'] + '.json')).read_text())['parameters'])
    p, F, evaluations = fit(model, p0, budget=job.get('budget', 300), stop=1 - job.get('stop_fidelity', 0.9965),
                            step_bound=job.get('step_bound'), label=job['name'])
    report = model.report(p)
    recipe = model.recipe(p, job['name'], dict(design_fidelity=F, evaluations=evaluations,
                                               qubit_frame_Rz=report['qubit_frame_Rz'], family=family,
                                               free_parameters=int(model.nparams)))
    row = dict(job=job, parameters=p.tolist(), evaluations=evaluations, free_parameters=int(model.nparams),
               recipe=str(recipe.relative_to(ROOT)), **report)
    (OUT / (job['name'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    print('CD JOB', job['name'], 'params', model.nparams, 'T', report['command_ns'], 'F', report['fidelity'],
          'vacuum', report['vacuum_g0_fidelity'], report['vacuum_e0_fidelity'], 'swap', report['peak_swap_ns'], flush=True)
    return row


WORD10 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1]        # best proper ten-pulse word (strategic_cd)
WORD10_SYM = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1]    # anti-palindromic: time-symmetric toggling path
WORD12 = [1, 1, -1, -1] * 3                          # the 370 ns reference word


def cd_jobs(stage):
    jobs = []
    if stage == 'n10':
        jobs.append(dict(name='n10_qubit_only', family='qubit_only', pulses=10, word=WORD10, seed=0, grid_first=8))
        for first in [8, 12]:
            for seed in [0, 1, 2]:
                jobs.append(dict(name=f'n10_maxfree_grid{first}_s{seed}', family='maxfree', pulses=10, word=WORD10,
                                 seed=seed, grid_first=first, arc_spread=0.25))
        for dips in [(4, 5), (0, 9)]:
            for seed in [0, 1]:
                jobs.append(dict(name=f'n10_dips{dips[0]}{dips[1]}_s{seed}', family='dips', pulses=10, word=WORD10,
                                 seed=seed, grid_first=8, dips=dips))
    if stage == 'n10more':
        for first, seed in [(12, 3), (12, 4), (8, 3)]:
            jobs.append(dict(name=f'n10_maxfree_grid{first}_s{seed}', family='maxfree', pulses=10, word=WORD10,
                             seed=seed, grid_first=first, arc_spread=0.4, budget=700, stop_fidelity=0.995))
        for family in ['bumps_fixed', 'bumps_free']:
            for seed in [1, 2]:
                jobs.append(dict(name=f'n10_{family}_s{seed}', family=family, pulses=10, word=WORD10_SYM,
                                 seed=seed, arc_spread=0.5, budget=400, stop_fidelity=0.995))
        for seed in [1, 2]:
            jobs.append(dict(name=f'n8_gap24_maxfree_s{seed}', family='maxfree', pulses=8, word=WORD12[:8],
                             gaps=[0, 0, 8, 0, 8, 0, 8, 0], seed=seed, grid_first=12, arc_spread=0.4, budget=500,
                             stop_fidelity=0.995))
    if stage == 'n10b':
        pi = float(np.pi)
        jobs.append(dict(name='n10_maxfree_grid8_s3_bounded', family='maxfree', pulses=10, word=WORD10, seed=3,
                         grid_first=8, init_from='n10_maxfree_grid8_s3', step_bound=pi, budget=400, stop_fidelity=0.996))
        for seed in [5, 6]:
            jobs.append(dict(name=f'n10_maxfree_grid8_s{seed}_bounded', family='maxfree', pulses=10, word=WORD10,
                             seed=seed, grid_first=8, arc_spread=0.4, step_bound=pi, budget=700, stop_fidelity=0.996))
        for m, seeds in [(2, [1, 2, 3]), (5, [1, 2])]:
            for seed in seeds:
                jobs.append(dict(name=f'n10_motif{m}_s{seed}', family='motif', motif=m, pulses=10, word=WORD10,
                                 seed=seed, grid_first=8, arc_spread=0.4, step_bound=pi, budget=500, stop_fidelity=0.996))
    if stage == 'b2':
        for n in [24, 28]:
            jobs.append(dict(name=f'b2_n{n}_maxfree_s1', family='maxfree', pulses=n, word=WORD12 * 3, strength=2.0,
                             levels=64, seed=1, grid_first=8, arc_spread=0.2, step_bound=float(np.pi), budget=500,
                             stop_fidelity=0.995))
    return jobs


# =========================================================================================
# 5. Conditional squeezing (8 dB) with the same freedoms
# =========================================================================================
def squeeze_job(job):
    """Fewer 30 ns pi pulses at stronger coupling, starting from the folder's qubit-only
    compression seeds (runs/fixed_sideband/pair_seeds.json). Families:
       qubit_only  per-pulse qubit phases, constant sideband (baseline at the same budget)
       maxfree     plus contiguous 10 ns sideband phase arcs (every allowed phase move)
    Design: DAC-rate model at `levels`; the result is re-scored on the full SDK/RFDC model."""
    n = job['pulses']
    seeds = json.loads((ROOT / 'runs/fixed_sideband/pair_seeds.json').read_text())
    if n == 160:
        ref = json.loads((ROOT / 'runs/constant_pump_gaussian/squeeze_compression_07.json').read_text())
        phi0, g0 = np.array(ref['qubit_phases']), np.pi / (2 * ref['job']['swap'])
    else:
        seed = next(r for r in seeds if r['cells'] == n)
        phi0, g0 = np.array(seed['qubit_phases']), seed['g']
    count = 8 + 24 * n
    arcs = grid_arcs(count, job.get('grid_first', 12)) if job['family'] == 'maxfree' else []
    levels = job.get('levels', 80)
    model = FreedomGate(n, None, arcs, (), levels=levels, gate='squeeze', strength=cd.R_8DB, rfdc=False)
    rng = np.random.default_rng(job['seed'])
    p0 = np.r_[phi0, 0.0, rng.normal(0, job.get('arc_spread', 0.05), model.nphase - 1), np.log(g0 * job.get('g_scale', 1.0))]
    if 'init_from' in job:                                     # continue an earlier fit of the same structure
        p0 = np.asarray(json.loads((OUT / (job['init_from'] + '.json')).read_text())['parameters'])
    p, F, evaluations = fit(model, p0, budget=job.get('budget', 300), stop=1 - job.get('stop_fidelity', 0.995),
                            label=job['name'])
    full = FreedomGate(n, None, arcs, (), levels=levels, gate='squeeze', strength=cd.R_8DB, rfdc=True)
    report = full.report(p)
    recipe = full.recipe(p, job['name'], dict(design_fidelity_dac_rate=F, evaluations=evaluations,
                                              qubit_frame_Rz=report['qubit_frame_Rz'], family=job['family']))
    row = dict(job=job, parameters=p.tolist(), evaluations=evaluations, design_fidelity_dac_rate=F,
               free_parameters=int(model.nparams), recipe=str(recipe.relative_to(ROOT)), **report)
    (OUT / (job['name'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    print('SQUEEZE JOB', job['name'], 'T', report['command_ns'], 'design F', F, 'full RFDC F', report['fidelity'],
          'swap', report['peak_swap_ns'], flush=True)
    return row


def squeeze_jobs(stage):
    jobs = []
    if stage == 'sq1':
        for n in [144, 128]:
            jobs.append(dict(name=f'sq_n{n}_qubit_only', family='qubit_only', pulses=n, seed=0, budget=150))
            jobs.append(dict(name=f'sq_n{n}_maxfree', family='maxfree', pulses=n, seed=1, budget=300))
    if stage == 'sq2':
        jobs.append(dict(name='sq_n144_maxfree_cont', family='maxfree', pulses=144, seed=1, budget=500,
                         init_from='sq_n144_maxfree'))
    return jobs


# =========================================================================================
# 6. Generalization to larger beta without refitting: repeat the certified beta=1 gate
# =========================================================================================
def repeated_gate(recipe_name, copies, levels_list=(64, 96, 128, 160, 192, 256)):
    """CD(beta)^k = CD(k beta) exactly (same sigma_x axis, displacements add). If the beta=1
    certificate is CD ~ F V with the declared virtual frame F = exp(-i a sigma_z/2), then
    CD(k) = F^k (F^-(k-1) V F^(k-1)) ... (F^-1 V F) V. Conjugating V by F^-j multiplies BOTH
    the qubit and the sideband samples by exp(-i j a): copy j is the same waveform with a
    common phase offset -j a (each copy keeps its own 5 ns outer ramps, so the junction
    passes through zero sideband amplitude). Returns converged scores for target beta=k."""
    meta = json.loads((OUT / (recipe_name + '_recipe.json')).read_text())
    cert = json.loads((OUT / (recipe_name + '_certificate.json')).read_text())
    data = np.load(ROOT / meta['controls_file'])
    a = cert['qubit_frame_Rz']
    q = np.concatenate([data['q'] * np.exp(-1j * j * a) for j in range(copies)])
    s = np.concatenate([data['s'] * np.exp(-1j * j * a) for j in range(copies)])
    qs, ss = hw.dac_scales(.6)
    qw, qd = hw.to_words_and_back(q, np.pi / cd.PI_TIME, qs, qubit=True)
    sw, sd = hw.to_words_and_back(s, np.pi / (2 * .6), ss)
    qf, sf = hw.rfdc_interpolate(qd), hw.rfdc_interpolate(sd)
    assert max(abs(sf)) <= MAX_G
    previous, history = None, []
    for levels in levels_list:
        space = cd.Space(levels, 9)
        initial = np.eye(2 * levels, dtype=complex)[:, space.inputs]
        T = cd.target_columns(space, 'cd', float(copies))
        frame = np.exp(-1j * copies * a * space.z / 2)[:, None]
        V = frame * hw.propagate(qf, sf, initial, cd.DT / 8, order=8, fast=True)
        row = dict(levels=levels, fidelity=scores(V, T)['fidelity'])
        if previous is not None:
            old, oldT = previous
            pad = np.zeros_like(V); pad[:len(old)] = old
            padT = np.zeros_like(T); padT[:len(oldT)] = oldT
            row.update(output_change=float(max(np.linalg.norm(V - pad, axis=0))),
                       target_change=float(max(np.linalg.norm(T - padT, axis=0))))
        history.append(row)
        if previous is not None and max(row['output_change'], row['target_change']) < 1e-9:
            break
        previous = V, T
    else:
        raise AssertionError('cutoff not converged')
    sc = scores(V, T)
    result = dict(source=recipe_name, copies=copies, beta=float(copies), command_ns=1000 * len(q) * cd.DT,
                  levels=levels, columns=18, complex_amplitudes=int(V.size), qubit_frame_Rz=copies * a,
                  peak_swap_ns=float(1000 * np.pi / (2 * max(abs(sf)))), cutoff_history=history,
                  **{k: v for k, v in sc.items()})
    (OUT / f'{recipe_name}_x{copies}.json').write_text(json.dumps(result, indent=2) + '\n')
    print('REPEATED', recipe_name, 'x', copies, 'beta', copies, 'T ns', result['command_ns'], 'F', sc['fidelity'],
          'vacuum', sc['vacuum_g0_fidelity'], sc['vacuum_e0_fidelity'], 'levels', levels, flush=True)
    return result


def summary():
    """Collect every fit result and certificate of this folder into summary.json."""
    rows = []
    for path in sorted(OUT.glob('*.json')):
        if path.name.endswith(('_recipe.json',)) or path.name in ('summary.json',):
            continue
        r = json.loads(path.read_text())
        if not isinstance(r, dict) or 'fidelity' not in r:
            continue
        kind = 'certificate' if path.name.endswith('_certificate.json') else (
            'repeated' if 'copies' in r else 'fit')
        rows.append(dict(file=path.name, kind=kind, name=r.get('name', r.get('job', {}).get('name', path.stem)),
                         gate=r.get('gate', 'squeeze' if path.name.startswith('sq') else 'cd'),
                         command_ns=r.get('command_ns'), fidelity=r['fidelity'],
                         worst_basis=r.get('worst_basis_fidelity'), vacuum_g0=r.get('vacuum_g0_fidelity'),
                         vacuum_e0=r.get('vacuum_e0_fidelity'), peak_swap_ns=r.get('peak_swap_ns'),
                         levels=r.get('levels'), qubit_frame_Rz=r.get('qubit_frame_Rz'),
                         free_parameters=r.get('free_parameters'), evaluations=r.get('evaluations')))
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for r in sorted(rows, key=lambda r: (r['gate'], r['command_ns'] or 0, -r['fidelity'])):
        print(f"{r['kind']:11s} {r['name']:34s} {r['command_ns']!s:>7} ns  F {r['fidelity']:.5f}  "
              f"vac {r['vacuum_g0'] or 0:.4f}/{r['vacuum_e0'] or 0:.4f}  swap {r['peak_swap_ns'] or 0:.1f}")
    return rows


def run_jobs(jobs, workers=3):
    from concurrent.futures import ProcessPoolExecutor
    print('START', len(jobs), 'fits; all 18 inputs; workers', workers, flush=True)
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(squeeze_job if jobs[0]['name'].startswith('sq_') else cd_job, jobs))
    assert len(rows) == len(jobs)
    print('DONE', len(rows), 'fits;', sum(r['evaluations'] for r in rows), 'full SDK/RFDC evaluations',
          '(each 18 input columns)', flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['reference', 'bound', 'gradient', 'certify', 'cd_search', 'summary', 'repeat'])
    parser.add_argument('--set', default='n10')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('recipes', nargs='*')
    args = parser.parse_args()
    if args.stage == 'reference':
        reference()
    elif args.stage == 'bound':
        bound()
    elif args.stage == 'gradient':
        gradient_check()
    elif args.stage == 'cd_search':
        jobs = squeeze_jobs(args.set) if args.set.startswith('sq') else cd_jobs(args.set)
        rows = run_jobs(jobs, args.workers)
        (OUT / f'cd_search_{args.set}.json').write_text(json.dumps(rows, indent=2) + '\n')
    elif args.stage == 'summary':
        summary()
    elif args.stage == 'repeat':                      # repeat <recipe name> <copies>
        repeated_gate(args.recipes[0], int(args.recipes[1]))
    elif args.stage == 'certify':
        assert args.recipes
        results = [certify(r) for r in args.recipes]
        print('CERTIFIED COUNTS', len(results), 'gates;', sum(r['columns'] for r in results), 'columns;',
              sum(r['complex_amplitudes'] for r in results), 'amplitudes', flush=True)
