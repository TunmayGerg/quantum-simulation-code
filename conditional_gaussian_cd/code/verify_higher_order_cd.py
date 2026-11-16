"""Counted, independent checks of the new product/correction study."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cd_trains as cd
import hardware_check as hw
from audit_cd import compile_words
from higher_order_cd import OUT, CorrectedTrain, ProductGate
from shaped_pulse_optimisation import PlayedGate

VERIFIED = OUT/'verified'
VERIFIED.mkdir(exist_ok=True)


def make_model(record, levels=16):
    j = record['job']
    if 'method' in j:
        return ProductGate(j['method'], j['repeats'], j['swap_ns'], levels,
                           j.get('overlap_samples', 0), j.get('sample_grain', 4),
                           j.get('complex_pump', False))
    if j['name'].startswith('compressed_squeeze_n'):
        return PlayedGate('squeeze', j['cells'], levels, True, True)
    return CorrectedTrain(j['cells'], levels, j['complex_pump'], j['mirror'], j.get('gate', 'cd'))


def gradient_check():
    rng = np.random.default_rng(611); errors = []
    for model in [ProductGate('strang', 2, 50, 24, 4, 1),
                  CorrectedTrain(6, 24, True, False, 'squeeze')]:
        if isinstance(model, ProductGate):
            p = np.r_[model.seed_phases+.05, model.seed_amplitudes]
        else:
            p = np.r_[rng.normal(size=6), np.full(6, 5.), np.full(6, .2)]
        _, gradient = model.evaluate(p, quantized=False)
        for _ in range(3):
            v = rng.normal(size=len(p)); v /= np.linalg.norm(v); eps = 1e-5
            fd = (model.evaluate(p+eps*v, False, False)-model.evaluate(p-eps*v, False, False))/(2*eps)
            errors.append(float(abs(fd-gradient@v)))
    assert len(errors) == 6 and max(errors) < 2e-5, errors
    (OUT/'additional_gradient_check.json').write_text(json.dumps(dict(directions=6, errors=errors), indent=2)+'\n')
    print('PRODUCT/SQUEEZE GRADIENT:6 directions;maxerror', max(errors), flush=True)


def verify(name):
    record = json.loads((OUT/(name+'.json')).read_text())
    model = make_model(record)
    parameters = np.asarray(record['parameters'])
    qraw, sraw, q, s, words = model.arrays(parameters)
    gate = record['job'].get('gate', 'cd')
    strength = cd.R_8DB if gate == 'squeeze' else 1.
    frame_angle = record.get('qubit_frame_Rz', 0.)
    previous = None; history = []
    for levels in [64, 96, 128, 192, 256, 320, 384, 512]:
        space = cd.Space(levels, 9)
        initial = np.eye(2*levels, dtype=complex)[:, space.inputs]
        target = cd.target_columns(space, gate, strength)
        frame = np.exp(-1j*frame_angle*space.z/2)[:, None]
        output = frame*hw.propagate(q, s, initial, cd.DT/8, order=8, fast=True)
        item = dict(levels=levels)
        if previous is not None:
            old, oldT = previous
            padded = np.zeros_like(output); padded[:len(old)] = old
            paddedT = np.zeros_like(target); paddedT[:len(oldT)] = oldT
            item.update(output_change=float(max(np.linalg.norm(output-padded, axis=0))),
                        target_change=float(max(np.linalg.norm(target-paddedT, axis=0))))
        history.append(item)
        if previous is not None and max(item['output_change'], item['target_change']) < 1e-9:
            break
        previous = output, target
    else:
        raise AssertionError((name, 'cutoff not converged'))
    independent = frame*hw.propagate(q, s, initial, cd.DT/8, order=16)
    difference = float(np.max(abs(independent-output)))
    assert difference < 1e-10
    overlaps = np.sum(target.conj()*output, axis=0)
    fidelity = float(abs(overlaps.mean())**2)
    # Also report the user's aligned ideal device-I/Q convention. This removes
    # the digital filter, not the 5 ns/10 ns programmed envelope transitions.
    ideal = frame*hw.propagate(np.repeat(qraw, 8), np.repeat(sraw, 8), initial,
                              cd.DT/8, order=8, fast=True)
    ideal_fidelity = float(abs(np.vdot(target, ideal)/18)**2)
    result = dict(name=name, gate=gate, fidelity=fidelity,
                  ideal_device_fidelity=ideal_fidelity,
                  accepted=bool(fidelity > .99), columns=18, levels=levels,
                  independent_amplitudes=output.size, independent_max_difference=difference,
                  command_us=len(qraw)*cd.DT, RFDC_window_us=len(q)*cd.DT/8,
                  qubit_frame_Rz=frame_angle, cutoff_history=history,
                  worst_basis_fidelity=float(min(abs(overlaps)**2)),
                  word_sha256=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest(),
                  source_record=name+'.json')
    if result['accepted']:
        result['compiled'] = compile_words(words, VERIFIED/(name+'_compiled'))
        np.savez_compressed(VERIFIED/(name+'_words.npz'), **words)
        np.savez_compressed(VERIFIED/(name+'_columns.npz'), output=output, target=target)
        with (VERIFIED/(name+'_qubit_sequence.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['pulse', 'start_ns', 'duration_ns', 'physical_DAC_phase_deg'])
            for j in range(model.cells):
                indices = np.flatnonzero(model.qcell == j)
                writer.writerow([j+1, indices[0]*cd.DT*1000, 30., -np.degrees(parameters[j]) % 360])
        # Exact I/Q words supply every sideband amplitude and ramp sample.
        result['pump_peak_DAC_magnitude'] = float(np.max(abs(hw.sample_to_complex(words['sideband']))))
    (VERIFIED/(name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print('CHECKED', name, 'F', fidelity, 'ideal_deviceF', ideal_fidelity,
          'columns18;amplitudes', output.size, 'cutoff', levels,
          'independenterror', difference, 'accepted', result['accepted'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='*')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--gradient', action='store_true')
    args = parser.parse_args()
    if args.gradient:
        gradient_check(); return
    names = args.names
    if not names:
        for filename in ['corrected_summary', 'product_summary', 'overlap_summary', 'squeeze_summary',
                         'phasecycle_summary', 'dcg_summary', 'dcg_generator_summary',
                         'phase_only_summary', 'nearby_squeeze_summary']:
            path = OUT/(filename+'.json')
            if path.exists(): names.extend(r['job']['name'] for r in json.loads(path.read_text()))
        if (OUT/'compressed_squeeze_n112.json').exists(): names.append('compressed_squeeze_n112')
    assert len(names) > 0 and len(names) == len(set(names))
    print('START NEW-STUDY VERIFICATION', len(names), 'records;all18inputs', flush=True)
    with ProcessPoolExecutor(args.workers) as pool: rows = list(pool.map(verify, names))
    assert len(rows) == len(names)
    allrows = [json.loads(p.read_text()) for p in VERIFIED.glob('*.json') if p.name != 'manifest.json']
    manifest = dict(records=len(allrows), columns=18*len(allrows),
                    independent_amplitudes=sum(r['independent_amplitudes'] for r in allrows),
                    accepted=sum(r['accepted'] for r in allrows), results=allrows)
    (VERIFIED/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('FINAL NEW-STUDY COMPARISONS', len(rows), 'records;', 18*len(rows), 'columns;',
          sum(r['independent_amplitudes'] for r in rows), 'amplitudes;',
          sum(r['accepted'] for r in rows), 'above99%', flush=True)


if __name__ == '__main__': main()
