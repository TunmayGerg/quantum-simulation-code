"""JC/anti-JC product formulas and finite-pulse corrected CD constructions.

This study deliberately distinguishes formal order (instantaneous switches)
from fidelity after 30 ns qubit pulses, int16 packing and the RFDC response.
Units are microseconds and radians. No board or live calibration is modified.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix

import cd_trains as cd
import hardware_check as hw
from shaped_pulse_optimisation import PlayedGate


OUT = ROOT / 'runs/higher_order_cd'
OUT.mkdir(parents=True, exist_ok=True)


class CorrectedTrain(PlayedGate):
    """30 ns pi pulses; smooth signed pump weights or complex pump plateaus.

    Signed weights are composite forward/backward interaction intervals. The
    optional reflection ties the two halves rather than fitting every sample.
    A DCG-inspired construction is not called a proved DCG unless its error
    cancellation is established separately.
    """
    def __init__(self, cells, levels=48, complex_pump=False, mirror=False, gate='cd'):
        super().__init__(gate, cells, levels, True, gate != 'cd')
        self.complex_pump = complex_pump
        self.mirror = mirror
        self.block = cells // 2 if mirror else cells

    def expand(self, parameters):
        m = self.block
        phi = parameters[:m]
        force = parameters[m:2*m].astype(complex)
        if self.complex_pump:
            force += 1j * parameters[2*m:3*m]
        if self.mirror:
            phi = np.r_[phi, phi[::-1] + np.pi]
            force = np.r_[force, force[::-1]]
        # Positive real force means the -pi/2 pump phase of the CD target.
        return phi, -1j * force

    def arrays(self, parameters, quantized=True):
        phi, pump = self.expand(parameters)
        q = self.qshape.copy()
        keep = self.qcell >= 0
        q[keep] *= np.exp(1j * phi[self.qcell[keep]])
        s = self.W @ pump
        words = None
        if quantized:
            qs, ss = hw.dac_scales(.6)
            qw, q = hw.to_words_and_back(q, np.pi/cd.PI_TIME, qs, qubit=True)
            sw, s = hw.to_words_and_back(s, np.pi/(2*.6), ss)
            words = dict(qubit=qw, sideband=sw)
        return q, s, hw.rfdc_interpolate(q), hw.rfdc_interpolate(s), words

    def evaluate(self, parameters, gradient=True, quantized=True, outputs=False):
        q, s, qf, sf, words = self.arrays(parameters, quantized)
        output = hw.propagate(qf, sf, self.initial, cd.DT/8, order=8, fast=True)
        zeta = 0.
        if self.frame:
            zg = np.vdot(self.target[0::2], output[0::2])
            ze = np.vdot(self.target[1::2], output[1::2])
            zeta = float(np.angle(zg)-np.angle(ze))
        frame = np.exp(-1j*zeta*self.z/2)[:, None]
        trace = np.vdot(self.target, frame*output)
        cost = float(1-abs(trace/18)**2)
        if outputs:
            return dict(fidelity=1-cost, qubit_frame_Rz=zeta), frame*output, self.target, words
        if not gradient:
            return cost
        adjoint = np.ascontiguousarray(frame.conj()*self.target)
        grad = np.zeros((len(qf), 4), float)
        self.backward(self.levels, 18, len(qf), 8, cd.DT/8,
                      float(trace.real), float(trace.imag), qf.ctypes.data,
                      sf.ctypes.data, output.ctypes.data, adjoint.ctypes.data,
                      grad.ctypes.data)
        Gq = fftconvolve(grad[:, 0]+1j*grad[:, 1], self.filter[::-1], mode='valid')[::8]
        Gs = fftconvolve(grad[:, 2]+1j*grad[:, 3], self.filter[::-1], mode='valid')[::8]
        phi, pump = self.expand(parameters)
        keep = self.qcell >= 0
        qb = self.qshape[keep]*np.exp(1j*phi[self.qcell[keep]])
        dphi = np.bincount(self.qcell[keep],
                          weights=np.real(Gq[keep].conj()*1j*qb), minlength=self.cells)
        G = self.W.T @ Gs
        da = np.real(G.conj()*(-1j))
        db = np.real(G.conj())
        if self.mirror:
            m = self.block
            dphi = dphi[:m]+dphi[m:][::-1]
            da = da[:m]+da[m:][::-1]
            db = db[:m]+db[m:][::-1]
        return cost, np.r_[dphi, da, db if self.complex_pump else []]


def gradient_check():
    errors = []
    for mirror in [False, True]:
        model = CorrectedTrain(6, 24, True, mirror)
        rng = np.random.default_rng(810+mirror)
        p = np.r_[rng.normal(0, .1, model.block),
                  np.full(model.block, 8.), rng.normal(0, .2, model.block)]
        _, g = model.evaluate(p, quantized=False)
        for _ in range(3):
            v = rng.normal(size=len(p)); v /= np.linalg.norm(v)
            eps = 1e-5
            fd = (model.evaluate(p+eps*v, False, False)-
                  model.evaluate(p-eps*v, False, False))/(2*eps)
            errors.append(float(abs(fd-g@v)))
    assert len(errors) == 6 and max(errors) < 2e-5, errors
    print('CORRECTED TRAIN GRADIENT:6 directions;maxerror', max(errors), flush=True)
    (OUT/'gradient_check.json').write_text(json.dumps(dict(directions=6, errors=errors), indent=2)+'\n')


def fit(job):
    started = time.monotonic()
    n = job['cells']
    model = CorrectedTrain(n, job.get('levels', 48), job['complex_pump'], job['mirror'], job.get('gate', 'cd'))
    m = model.block
    # XX-- is only the baseline. Signed Yoshida weights or Eulerian XY axes
    # supply physically motivated alternatives before the finite-pulse fit.
    phi = np.resize([0., 0., np.pi, np.pi], n)
    g = 2./(.030*n+.005)
    force = np.full(n, g)
    if job['seed'] == 'yoshida':
        p = 1/(2-2**(1/3))
        force = np.resize([p, 1-2*p, p], n)*g
        force *= n/np.sum(force/g)
    elif job['seed'] == 'xy8':
        phi = np.resize([0., np.pi/2, 0., np.pi/2, np.pi/2, 0., np.pi/2, 0.], n)
    elif job['seed'] == 'opposed_axes':
        phi = np.resize([0., np.pi/3, np.pi, 4*np.pi/3], n)
    imaginary = np.zeros(n)
    if job.get('initial') is not None:
        phi, force, imaginary = map(np.asarray, job['initial'])
    initial = np.r_[phi[:m], force[:m], imaginary[:m] if model.complex_pump else []]
    history = []
    best = [np.inf, initial.copy()]
    def cost(p):
        value, gradient = model.evaluate(p)
        history.append(value)
        if value < best[0]: best[:] = [value, p.copy()]
        if len(history) % 40 == 0:
            print('CORRECTED FIT', job['name'], 'eval', len(history), 'bestF', 1-best[0], flush=True)
        return value, gradient
    def callback(p):
        if (best[0] < job.get('stop_infidelity', .002) or len(history) >= job.get('budget', 260)
                or time.monotonic()-started > job.get('seconds', 180)):
            raise StopIteration
    # The components below correspond to <0.775 DAC amplitude each. They are
    # model limits, not a claim that this stronger pump is calibrated on device.
    bound = job.get('max_g', 45.)
    bounds = [(None, None)]*m+[(-bound, bound)]*(len(initial)-m)
    result = minimize(cost, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                      callback=callback, options=dict(maxiter=400, maxcor=25,
                      maxls=25, ftol=1e-13, gtol=1e-8))
    report, output, target, words = model.evaluate(best[1], False, outputs=True)
    phi, pump = model.expand(best[1])
    overlaps = np.sum(target.conj()*output, axis=0)
    report.update(job=job, parameters=best[1].tolist(), evaluations=len(history),
                  initial_fidelity=1-history[0], history=history,
                  qubit_phases=phi.tolist(), pump_real=pump.real.tolist(),
                  pump_imag=pump.imag.tolist(), worst_basis_fidelity=float(min(abs(overlaps)**2)),
                  command_us=len(words['qubit'])*cd.DT,
                  RFDC_window_us=len(model.arrays(best[1])[2])*cd.DT/8,
                  seconds=time.monotonic()-started, optimizer_message=str(result.message),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/(job['name']+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    print('CORRECTED RESULT', job['name'], 'F', report['fidelity'], 'T', report['command_us'],
          'eval', len(history), 'seconds', report['seconds'], flush=True)
    return report


def study(workers):
    jobs = []
    for n in [6, 8, 10]:
        for mirror, complex_pump, seed in [(True, False, 'baseline'),
                                          (False, False, 'baseline'),
                                          (False, True, 'yoshida'),
                                          (False, True, 'xy8')]:
            jobs.append(dict(name=f'corrected_n{n}_mirror{int(mirror)}_iq{int(complex_pump)}_{seed}',
                             cells=n, mirror=mirror, complex_pump=complex_pump, seed=seed))
    print('START CORRECTED CD STUDY', len(jobs), 'fits;all18inputs', flush=True)
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(fit, jobs))
    assert len(rows) == len(jobs)
    (OUT/'corrected_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('COMPLETED CORRECTED CD STUDY', len(rows), 'fits;',
          sum(r['evaluations'] for r in rows)*18, 'input columns evaluated', flush=True)


def squeeze_study(workers):
    """Continue fixed 8 dB targets with variable smooth interaction weights.

    These are corrected commutator fits, not proofs of formal DCG order.
    """
    jobs = []
    old = ROOT/'runs/constant_pump_gaussian'
    for swap in [200, 225, 250]:
        source = json.loads((old/f'shorter_squeeze_n96_swap{swap}.json').read_text())
        n = source['job']['cells']; p = np.asarray(source['parameters'])
        z = 1j*np.exp(p[-1]+1j*p[n:2*n])
        jobs.append(dict(name=f'corrected_squeeze_n96_seed{swap}', gate='squeeze',
                         cells=n, mirror=False, complex_pump=True, seed='previous_commutator_fit',
                         initial=[p[:n].tolist(), z.real.tolist(), z.imag.tolist()],
                         levels=64, budget=100, seconds=300, stop_infidelity=.004,
                         source_record=f'shorter_squeeze_n96_swap{swap}.json'))
    print('START CORRECTED SQUEEZE STUDY', len(jobs), 'fits;all18inputs', flush=True)
    with ProcessPoolExecutor(workers) as pool: rows = list(pool.map(fit, jobs))
    assert len(rows) == len(jobs)
    (OUT/'squeeze_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('COMPLETED CORRECTED SQUEEZE STUDY', len(rows), 'fits;',
          sum(r['evaluations'] for r in rows)*18, 'input columns evaluated', flush=True)


def product_coefficients(method, repeats):
    """Chronological, merged coefficients of A=JC and B=X A X.

    The target is exp[-i beta (A+B)] when A,B have unit coupling. Negative
    coefficients are negative pump amplitudes, never negative elapsed time.
    """
    second = [('A', .5), ('B', 1.), ('A', .5)]
    if method == 'first':
        step = [('A', 1.), ('B', 1.)]
    elif method == 'strang':
        step = second
    elif method == 'yoshida':
        p = 1/(2-2**(1/3))
        step = [(k, a*w) for w in [p, 1-2*p, p] for k, a in second]
    else:
        raise ValueError(method)
    merged = []
    for k, a in [(k, a/repeats) for k, a in step]*repeats:
        if merged and merged[-1][0] == k:
            merged[-1] = (k, merged[-1][1]+a)
        else:
            merged.append((k, a))
    assert abs(sum(a for k, a in merged if k == 'A')-1) < 1e-12
    assert abs(sum(a for k, a in merged if k == 'B')-1) < 1e-12
    return merged


class ProductGate(PlayedGate):
    """Literal product formula: JC stretches separated by 30 ns pi pulses.

    Pump is programmed off during the qubit pulses. Each stretch has 5 ns
    smooth turn-on/off ramps; actual filter tails are still propagated.
    """
    def __init__(self, method, repeats, swap_ns, levels=48, overlap_samples=0, sample_grain=4, complex_pump=False):
        self.coefficients = product_coefficients(method, repeats)
        flips = len(self.coefficients)-1+(self.coefficients[-1][0] == 'B')
        super().__init__('cd', flips, levels, True, False)
        g = np.pi/(2*swap_ns*.001)
        sections = []
        for _, area in self.coefficients:
            count = max(8, sample_grain*int(np.ceil((abs(area)/(g*cd.DT)+4)/sample_grain)))
            shape = np.ones(count)
            shape[:4] = np.sin(np.pi*np.arange(4)/8)**2
            shape[-4:] = np.cos(np.pi*np.arange(4)/8)**2
            sections.append((shape, area/(shape.sum()*cd.DT)))
        assert 0 <= overlap_samples <= 4
        overlaps = 2*(len(sections)-1)+(self.coefficients[-1][0] == 'B')
        raw_count = sum(len(shape) for shape, _ in sections)+24*flips-overlap_samples*overlaps
        count = 4*int(np.ceil(raw_count/4))
        self.qshape = np.zeros(count, complex)
        self.qcell = np.full(count, -1, int)
        W = np.zeros((count, len(sections)))
        cursor = 0
        for j, (shape, _) in enumerate(sections):
            W[cursor:cursor+len(shape), j] = shape
            cursor += len(shape)
            if j < flips:
                cursor -= overlap_samples
                self.qshape[cursor:cursor+24] = cd.pi_samples()
                self.qcell[cursor:cursor+24] = j
                cursor += 24
                if j < len(sections)-1:
                    cursor -= overlap_samples
        assert cursor == raw_count and count % 4 == 0
        self.W = csr_matrix(W)
        self.seed_amplitudes = np.array([a for _, a in sections])
        self.seed_phases = np.resize([0., 0., np.pi, np.pi], flips)
        self.section_count = len(sections)
        self.complex_pump = complex_pump

    def arrays(self, parameters, quantized=True):
        phi = parameters[:self.cells]
        q = self.qshape.copy(); keep = self.qcell >= 0
        q[keep] *= np.exp(1j*phi[self.qcell[keep]])
        a = parameters[self.cells:self.cells+self.section_count].astype(complex)
        if self.complex_pump:
            a += 1j*parameters[self.cells+self.section_count:]
        s = -1j*(self.W @ a)
        words = None
        if quantized:
            qs, ss = hw.dac_scales(.6)
            qw, q = hw.to_words_and_back(q, np.pi/cd.PI_TIME, qs, qubit=True)
            sw, s = hw.to_words_and_back(s, np.pi/(2*.6), ss)
            words = dict(qubit=qw, sideband=sw)
        return q, s, hw.rfdc_interpolate(q)*getattr(self, 'qubit_gain', 1.), hw.rfdc_interpolate(s), words

    def evaluate(self, parameters, gradient=True, quantized=True, outputs=False):
        q, s, qf, sf, words = self.arrays(parameters, quantized)
        output = hw.propagate(qf, sf, self.initial, cd.DT/8, order=8, fast=True)
        trace = np.vdot(self.target, output)
        cost = float(1-abs(trace/18)**2)
        if outputs: return dict(fidelity=1-cost), output, self.target, words
        if not gradient: return cost
        adjoint = np.ascontiguousarray(self.target)
        grad = np.zeros((len(qf), 4), float)
        self.backward(self.levels, 18, len(qf), 8, cd.DT/8,
                      float(trace.real), float(trace.imag), qf.ctypes.data,
                      sf.ctypes.data, output.ctypes.data, adjoint.ctypes.data,
                      grad.ctypes.data)
        Gq = fftconvolve(grad[:, 0]+1j*grad[:, 1], self.filter[::-1], mode='valid')[::8]
        Gs = fftconvolve(grad[:, 2]+1j*grad[:, 3], self.filter[::-1], mode='valid')[::8]
        keep = self.qcell >= 0
        qb = self.qshape[keep]*np.exp(1j*parameters[:self.cells][self.qcell[keep]])
        dphi = np.bincount(self.qcell[keep], weights=np.real(Gq[keep].conj()*1j*qb),
                          minlength=self.cells)
        dphi *= getattr(self, 'qubit_gain', 1.)
        da = np.real((self.W.T @ Gs).conj()*(-1j))
        db = np.real((self.W.T @ Gs).conj())
        return cost, np.r_[dphi, da, db if self.complex_pump else []]


def fit_product(job):
    model = ProductGate(job['method'], job['repeats'], job['swap_ns'],
                        overlap_samples=job.get('overlap_samples', 0),
                        sample_grain=job.get('sample_grain', 4),
                        complex_pump=job.get('complex_pump', False))
    initial = np.r_[model.seed_phases, model.seed_amplitudes]
    if model.complex_pump:
        initial = np.r_[initial, np.zeros(model.section_count)]
    if 'initial' in job:
        initial = np.asarray(job['initial'])
    history = []; best = [np.inf, initial.copy()]
    started = time.monotonic()
    def cost(p):
        if job.get('robust', False):
            samples = []
            for gain in [.97, 1., 1.03]:
                model.qubit_gain = gain
                samples.append(model.evaluate(p))
            model.qubit_gain = 1.
            value = float(np.mean([sample[0] for sample in samples]))
            gradient = np.mean([sample[1] for sample in samples], axis=0)
        else:
            value, gradient = model.evaluate(p)
        history.append(value)
        if value < best[0]: best[:] = [value, p.copy()]
        return value, gradient
    def callback(p):
        if best[0] < .001 or len(history) >= job.get('budget', 180) or time.monotonic()-started > job.get('seconds', 100):
            raise StopIteration
    fitted = minimize(cost, initial, jac=True, method='L-BFGS-B',
                      bounds=[(None, None)]*model.cells+[(-45., 45.)]*(len(initial)-model.cells),
                      callback=callback, options=dict(maxiter=180, maxcor=20,
                      maxls=25, ftol=1e-13, gtol=1e-8))
    report, output, target, words = model.evaluate(best[1], False, outputs=True)
    report.update(job=job, parameters=best[1].tolist(), initial_fidelity=1-history[0],
                  history=history, evaluations=len(history),
                  coefficients=model.coefficients, pulses=model.cells,
                  command_us=len(words['qubit'])*cd.DT,
                  RFDC_window_us=len(model.arrays(best[1])[2])*cd.DT/8,
                  seconds=time.monotonic()-started, optimizer_message=str(fitted.message))
    (OUT/(job['name']+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    print('PRODUCT RESULT', job['name'], 'initialF', 1-history[0], 'F', report['fidelity'],
          'T', report['command_us'], 'eval', len(history), flush=True)
    return report


def product_study(workers):
    jobs = [dict(name=f'product_{method}_r{r}_s{swap}', method=method, repeats=r, swap_ns=swap)
            for method, repeats in [('strang', [3, 4, 5]), ('yoshida', [1, 2])]
            for r in repeats for swap in [35, 50, 75, 100]]
    print('START PRODUCT STUDY', len(jobs), 'fits;all18inputs', flush=True)
    with ProcessPoolExecutor(workers) as pool: rows = list(pool.map(fit_product, jobs))
    assert len(rows) == len(jobs)
    (OUT/'product_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('COMPLETED PRODUCT STUDY', len(rows), 'fits;',
          18*sum(r['evaluations'] for r in rows), 'input columns evaluated', flush=True)


def overlap_study(workers):
    jobs = [dict(name=f'overlap_strang_r{r}_s{swap}', method='strang', repeats=r,
                 swap_ns=swap, overlap_samples=4, sample_grain=1)
            for r in [3, 4, 5] for swap in [35, 50, 75, 100]]
    print('START OVERLAP STUDY', len(jobs), 'fits;all18inputs', flush=True)
    with ProcessPoolExecutor(workers) as pool: rows = list(pool.map(fit_product, jobs))
    assert len(rows) == len(jobs)
    (OUT/'overlap_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('COMPLETED OVERLAP STUDY', len(rows), 'fits;',
          18*sum(r['evaluations'] for r in rows), 'input columns evaluated', flush=True)


def phasecycle_study(workers):
    """Paired transverse corrections around the symmetric product-formula seed."""
    jobs = []
    for r, swap in [(3, 35), (3, 50), (3, 75), (4, 35)]:
        base = json.loads((OUT/f'overlap_strang_r{r}_s{swap}.json').read_text())
        for angle in [.08, .30]:
            phi = np.array(base['parameters'][:2*r])
            a = np.array(base['parameters'][2*r:])
            # Antisymmetric transverse weights have zero leading sum, while
            # their ordering can alter nested commutators and finite-pulse errors.
            pattern = np.sin(np.linspace(-np.pi/2, np.pi/2, len(a)))
            phi += angle*np.sin(np.linspace(-np.pi, np.pi, len(phi)))
            p = np.r_[phi, a, angle*abs(a)*pattern]
            jobs.append(dict(name=f'phasecycle_strang_r{r}_s{swap}_a{angle:g}',
                             method='strang', repeats=r, swap_ns=swap,
                             overlap_samples=4, sample_grain=1, complex_pump=True,
                             initial=p.tolist(), budget=240, seconds=140))
    print('START PHASE-CYCLE STUDY', len(jobs), 'fits;all18inputs', flush=True)
    with ProcessPoolExecutor(workers) as pool: rows = list(pool.map(fit_product, jobs))
    assert len(rows) == len(jobs)
    (OUT/'phasecycle_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('COMPLETED PHASE-CYCLE STUDY', len(rows), 'fits;',
          18*sum(r['evaluations'] for r in rows), 'input columns evaluated', flush=True)


def dcg_study(workers):
    """Target coherent gate error AND qubit-gain sensitivity at fixed duration.

    A three-point ensemble approximates a DCG sensitivity objective. A final
    derivative measurement distinguishes reduced sensitivity from a proven
    vanishing first-order error generator. No DCG order is assumed.
    """
    base = json.loads((OUT/'overlap_strang_r4_s35.json').read_text())
    jobs = []
    for angle in [.05, .2]:
        p = np.array(base['parameters'])
        p[:8] += angle*np.sin(np.linspace(-np.pi, np.pi, 8))
        jobs.append(dict(name=f'dcg_strang_r4_a{angle:g}', method='strang', repeats=4,
                         swap_ns=35, overlap_samples=4, sample_grain=1,
                         robust=True, initial=p.tolist(), budget=120, seconds=160))
    print('START DCG-OBJECTIVE STUDY', len(jobs), 'fits;3gainpoints;all18inputs', flush=True)
    with ProcessPoolExecutor(workers) as pool: rows = list(pool.map(fit_product, jobs))
    assert len(rows) == len(jobs)
    (OUT/'dcg_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('COMPLETED DCG-OBJECTIVE STUDY', len(rows), 'fits;',
          54*sum(r['evaluations'] for r in rows), 'input columns evaluated', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['gradient', 'fit', 'product', 'squeeze', 'overlap', 'phasecycle', 'dcg'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.stage == 'gradient': gradient_check()
    elif args.stage == 'fit': study(args.workers)
    elif args.stage == 'product': product_study(args.workers)
    elif args.stage == 'squeeze': squeeze_study(args.workers)
    elif args.stage == 'overlap': overlap_study(args.workers)
    elif args.stage == 'phasecycle': phasecycle_study(args.workers)
    else: dcg_study(args.workers)
