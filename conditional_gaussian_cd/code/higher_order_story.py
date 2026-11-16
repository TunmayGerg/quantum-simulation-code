"""Small, counted calculations for the higher-order/CD correction notebook."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import numpy as np
from scipy.linalg import expm
import cd_trains as cd
import hardware_check as hw
from higher_order_cd import OUT, ROOT, product_coefficients
from verify_higher_order_cd import make_model
from verify_short_gates import arrays as original_arrays


def ideal_product(method, repeats, levels):
    space = cd.Space(levels, 9)
    state = np.eye(2*levels, dtype=complex)[:, space.inputs]
    def A(state, angle):
        out = state.copy()
        x = angle*np.sqrt(np.arange(1, levels))
        c = np.cos(x)[:, None]; s = np.sin(x)[:, None]
        out[1:-1:2] = c*state[1:-1:2]-s*state[2::2]
        out[2::2] = c*state[2::2]+s*state[1:-1:2]
        return out
    flip = np.arange(2*levels)^1
    for kind, angle in product_coefficients(method, repeats):
        state = A(state, angle) if kind == 'A' else A(state[flip], angle)[flip]
    return state, cd.target_columns(space, 'cd', 1.)


def references():
    rng = np.random.default_rng(922)
    space = cd.Space(16, 9)
    # Test the JC signs/phases against an independent dense exponential.
    pump = -.27j
    H = np.zeros((32, 32), complex)
    for n in range(15):
        H[2*n+1, 2*n+2] = pump*np.sqrt(n+1)
        H[2*n+2, 2*n+1] = pump.conjugate()*np.sqrt(n+1)
    probe = rng.normal(size=(32,18))+1j*rng.normal(size=(32,18))
    exact = expm(-1j*H)@probe
    out = probe.copy(); x = .27*np.sqrt(np.arange(1,16))
    out[1:-1:2] = np.cos(x)[:,None]*probe[1:-1:2]-np.sin(x)[:,None]*probe[2::2]
    out[2::2] = np.cos(x)[:,None]*probe[2::2]+np.sin(x)[:,None]*probe[1:-1:2]
    error = float(np.max(abs(exact-out)))
    assert probe.size == 576 and error < 1e-12
    rows = []
    for method in ['first', 'strang', 'yoshida']:
        for repeats in [1,2,3,4,6,8,12,16,24,32]:
            V, T = ideal_product(method, repeats, 64)
            large, largeT = ideal_product(method, repeats, 96)
            padded = np.zeros_like(large); padded[:len(V)] = V
            paddedT = np.zeros_like(largeT); paddedT[:len(T)] = T
            change = float(max(np.linalg.norm(large-padded, axis=0)))
            target_change = float(max(np.linalg.norm(largeT-paddedT, axis=0)))
            assert max(change, target_change) < 1e-9
            rows.append(dict(method=method, repeats=repeats,
                             fidelity=float(abs(np.vdot(largeT,large)/18)**2),
                             output_change=change, target_change=target_change))
    assert len(rows) == 30
    (OUT/'ideal_product_reference.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('IDEAL PRODUCT REFERENCE:30points;540columns;576densecheckamplitudes;maxerror', error, flush=True)


def sensitivity():
    curves = []; derivatives = []
    names = ['baseline', 'overlap_strang_r4_s35', 'phasecycle_strang_r4_s35_a0.3', 'dcg_strang_r4_a0.2']
    comparisons = 0
    cache=OUT/'sensitivity.json'
    if cache.exists():
        previous=json.loads(cache.read_text())
        curves=previous['curves'];derivatives=previous['derivatives']
        comparisons=previous['cutoff_comparison_amplitudes']
    for name in ['dcg_generator_weight0.005','dcg_generator_weight0.02']:
        if (OUT/(name+'.json')).exists():names.append(name)
    done={r['name'] for r in derivatives}
    for name in names:
        if name in done:continue
        if name == 'baseline':
            record = json.loads((ROOT/'runs/constant_pump_gaussian/cd_binary_scaled_b1.json').read_text())
            q, s, _, _, _, _ = original_arrays(record)
        else:
            record = json.loads((OUT/(name+'.json')).read_text())
            _, _, q, s, _ = make_model(record).arrays(np.asarray(record['parameters']))
        for error_type in ['qubit_gain', 'pump_gain']:
            for gain in [.96,.98,.99,.995,1.,1.005,1.01,1.02,1.04]:
                previous = None
                for levels in [64,96]:
                    sp = cd.Space(levels,9); initial = np.eye(2*levels,dtype=complex)[:,sp.inputs]
                    T = cd.target_columns(sp,'cd',1.)
                    qg = q*(gain if error_type == 'qubit_gain' else 1.)
                    sg = s*(gain if error_type == 'pump_gain' else 1.)
                    V = hw.propagate(qg,sg,initial,cd.DT/8,order=8,fast=True)
                    if previous is not None:
                        padded = np.zeros_like(V); padded[:len(previous)] = previous
                        change = float(max(np.linalg.norm(V-padded,axis=0)))
                        assert change < 1e-9
                        comparisons += V.size
                    previous = V
                curves.append(dict(name=name,error=error_type,gain=gain,
                                   fidelity=float(abs(np.vdot(T,V)/18)**2),output_change=change))
        # Differentiate the actual gate, removing only its irrelevant global phase.
        sp = cd.Space(96,9); initial = np.eye(192,dtype=complex)[:,sp.inputs]
        V0 = hw.propagate(q,s,initial,cd.DT/8,order=8,fast=True)
        eps = .001; estimates = []
        for e in [eps,eps/2]:
            plus = hw.propagate(q*(1+e),s,initial,cd.DT/8,order=8,fast=True)
            minus = hw.propagate(q*(1-e),s,initial,cd.DT/8,order=8,fast=True)
            derivative = (plus-minus)/(2*e)
            derivative -= 1j*np.imag(np.vdot(V0,derivative)/18)*V0
            estimates.append(float(np.linalg.norm(derivative)/np.sqrt(18)))
        derivatives.append(dict(name=name,qubit_gain_derivative_norm=estimates[-1],
                                derivative_step_change=abs(estimates[0]-estimates[1])))
    assert len(curves) == 18*len(names) and len(derivatives) == len(names)
    (OUT/'sensitivity.json').write_text(json.dumps(dict(points=len(curves),columns=len(curves)*18,
                cutoff_comparison_amplitudes=comparisons,curves=curves,derivatives=derivatives),indent=2)+'\n')
    print('GAIN SENSITIVITY:',len(curves),'points;',len(curves)*18,'columns;', comparisons,'cutoffcomparisonamplitudes;',
          len(derivatives),'derivatives,2steps each',flush=True)
    for row in derivatives: print('QUIBIT GAIN DERIVATIVE',row,flush=True)


if __name__ == '__main__':
    references()
    sensitivity()
