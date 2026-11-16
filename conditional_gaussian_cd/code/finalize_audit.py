"""Check saved columns, words and notebook outputs and write a counted manifest.

This does not replace the independent propagation tests. It checks that the
published tables and artifacts still describe the arrays those tests produced.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import hashlib
import json
from pathlib import Path
import numpy as np
import nbformat


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_sha(path):
    with np.load(path) as words:
        return hashlib.sha256(words['qubit'].tobytes() + words['sideband'].tobytes()).hexdigest()


def columns(path, expected_fidelity, expected_columns):
    with np.load(path) as data:
        output, target = data['output'], data['target']
    assert output.shape == target.shape and output.shape[1] == expected_columns
    overlaps = np.sum(target.conj() * output, axis=0)
    fidelity = float(abs(overlaps.mean()) ** 2)
    assert abs(fidelity - expected_fidelity) < 1e-13, (path, fidelity, expected_fidelity)
    assert np.max(abs(np.sum(abs(output)**2, axis=0)-1)) < 1e-9
    return output.size


def main():
    original = load('runs/audit/manifest.json')
    amplitudes = 0
    for row in original['results']:
        prefix = f"{row['group']}_{row['index']}"
        amplitudes += columns(ROOT / 'runs/audit' / (prefix + '_columns.npz'),
                              row['fidelity'], row['columns'])
        gate = load(f"runs/{row['group']}.json")[row['index']]
        assert word_sha(ROOT / 'runs' / gate['words_file']) == row['word_sha256']
        last = row['cutoff_history'][-1]
        assert max(last['output_column_change'], last['target_column_change']) < 1e-9
    assert len(original['results']) == 16 and amplitudes == original['independent_amplitudes'] == 90240
    print('LIBRARY ARTIFACT COMPARISONS:16gates;320columns;90240amplitudes;16word hashes')

    families = load('runs/transition_families/manifest.json')
    amplitudes = 0
    for row in families['results']:
        prefix = f"{row['kind']}_beta{row['beta']:g}"
        folder = ROOT / 'runs/transition_families'
        amplitudes += columns(folder / (prefix + '_columns.npz'), row['fidelity'], 22)
        assert word_sha(folder / (prefix + '_words.npz')) == row['word_sha256']
        last = row['cutoff_history'][-1]
        assert max(last['output_column_change'], last['target_column_change']) < 1e-9
        assert row['fidelity'] > .99 and row['independent_max_difference'] < 1e-10
    assert len(families['results']) == 40 and amplitudes == families['independent_amplitudes'] == 174592
    print('SHARED FAMILY ARTIFACT COMPARISONS:40gates;880columns;174592amplitudes;40word hashes')

    microscopic = []
    amplitudes = 0
    for kind in ['constant', 'cartesian']:
        prefix = f'programmed_{kind}_a20_c8_dt0.005'
        folder = ROOT / 'runs/coupler'
        row = load(f'runs/coupler/{prefix}.json')
        amplitudes += columns(folder / (prefix + '_columns.npz'), row['coherent_gate_fidelity'], 18)
        columns(folder / (prefix + '_frame_corrected_columns.npz'),
                row['observer_frame_calibration']['coherent_gate_fidelity'], 18)
        assert word_sha(folder / (prefix + '_words.npz')) == row['word_sha256']
        assert len(row['final_coupler_excitation']) == 18
        assert row['observer_frame_calibration']['coherent_gate_fidelity'] > .99
        microscopic.append(row)
        print('MICROSCOPIC ARTIFACT', kind, 'rawF', row['coherent_gate_fidelity'],
              'correctedF', row['observer_frame_calibration']['coherent_gate_fidelity'],
              'columns18;rawandcorrectedamplitudes11520')
    assert amplitudes == 11520
    convergence = load('runs/coupler/programmed_convergence.json')
    assert len(convergence) == 4
    assert sum(r['compared_input_columns'] for r in convergence) == 8
    assert sum(r['compared_amplitudes'] for r in convergence) == 3200
    assert max(r['maximum_output_column_change'] for r in convergence) < 1e-5

    # The new ideal-model study is independent of the archived coupler work.
    # Check every accepted short-gate certificate against its actual arrays.
    study=ROOT/'runs/constant_pump_gaussian';verified=study/'verified'
    short=[]
    for path in sorted(verified.glob('*.json')):
        if path.name=='manifest.json':continue
        row=json.loads(path.read_text());name=row['name']
        recipe=json.loads((study/row['source_record']).read_text())
        from verify_short_gates import arrays as rebuild_arrays
        _,_,rebuilt,_,_,_=rebuild_arrays(recipe)
        rebuilt_hash=hashlib.sha256(rebuilt['qubit'].tobytes()+rebuilt['sideband'].tobytes()).hexdigest()
        assert rebuilt_hash==row['word_sha256'],(name,'source recipe no longer reproduces certified words')
        g=(float(np.exp(recipe['parameters'][-1])) if 'g' in recipe and 'RFDC_window_us' in recipe
           else float(np.pi/(2*recipe['job']['swap'])))
        import hardware_check as hardware
        qs,ss=hardware.dac_scales(.6)
        row.update(g_rad_per_us=g,equivalent_swap_ns=1000*np.pi/(2*g),
                   qubit_pi_DAC_scale=qs,pump_plateau_DAC_scale=ss*g/(np.pi/(2*.6)),
                   programmed_NCO_changes=False)
        path.write_text(json.dumps(row,indent=2)+'\n')
        count=columns(verified/(name+'_columns.npz'),row['fidelity'],18)
        assert count==row['complex_amplitudes'] and row['fidelity']>.99
        assert word_sha(verified/(name+'_words.npz'))==row['word_sha256']
        last=row['cutoff_history'][-1]
        assert max(last['output_change'],last['target_change'])<1e-9
        assert row['independent_max_difference']<1e-10
        short.append(row)
    assert len(short)>=31
    rule=json.loads((study/'binary_rule_summary.json').read_text())
    rule_names={r['job']['name'] for r in rule['results']}
    rule_certificates=[r for r in short if r['name'] in rule_names]
    assert len(rule_certificates)==20
    assert min(r['fidelity'] for r in rule_certificates)>.99
    short_manifest=dict(certified_records=len(short),columns=18*len(short),
                        independent_amplitudes=sum(r['complex_amplitudes'] for r in short),
                        unique_word_hashes=len({r['word_sha256'] for r in short}),
                        results=short,source_sha256={p.name:sha(p) for p in CODE.glob('*.py')})
    (verified/'manifest.json').write_text(json.dumps(short_manifest,indent=2)+'\n')
    print('SHORT GATE ARTIFACT COUNTS',len(short),'records;',18*len(short),'columns;',
          short_manifest['independent_amplitudes'],'amplitudes;',short_manifest['unique_word_hashes'],'unique word hashes')
    print('BINARY CD FAMILY COUNTS',20,'strengths;',360,'columns;',
          sum(r['complex_amplitudes'] for r in rule_certificates),'amplitudes;minimumF',
          min(r['fidelity'] for r in rule_certificates))
    curves=json.loads((study/'strength_sensitivity.json').read_text())
    assert curves['schemes']==8 and curves['strength_points']==176 and curves['columns']==3168
    for curve in curves['results']:
        assert len(curve['results'])==22
        assert max(max(r['output_change'],r['target_change']) for r in curve['results'])<1e-8
    maps=json.loads((study/'strength_map_validation.json').read_text())
    assert len(maps)==3 and sum(r['points'] for r in maps)==63
    assert sum(r['columns'] for r in maps)==1134 and sum(r['amplitudes'] for r in maps)==352512
    for row in maps:
        points=json.loads((study/row['file']).read_text())
        assert len(points)==row['points']
        assert all(max(p['plot_cutoff_history'][-1]['output_change'],
                       p['plot_cutoff_history'][-1]['target_change'])<1e-8 for p in points)
    reference=json.loads((study/'instantaneous_commutator_reference.json').read_text())
    assert len(reference)==32 and max(max(r['output_change'],r['target_change']) for r in reference)<1e-9
    print('PLOT CHECK COUNTS:176fixed-sequence strength points;63retuned/screenedpoints;32idealcommutatorpoints')
    completion=json.loads((study/'cd_completion_curve.json').read_text())
    assert completion['independent_columns']==18 and completion['independent_amplitudes']==4608
    columns(study/'cd_400ns_columns.npz',completion['independent_fidelity'],18)
    assert completion['independent_fidelity']>.99
    assert completion['remaining_tail_input_operator_norm']<1e-6
    assert completion['independent_larger_cutoff_column_change']<1e-9
    print('ACTUAL400ns CHECK:18columns;4608amplitudes;F',completion['independent_fidelity'],
          'remainingtailoperatornorm',completion['remaining_tail_input_operator_norm'])

    notebooks = []
    specs = [('conditional_gaussian_gates.ipynb',7,6,'runs/pulse_spacing/notebook_execution.log'),
             ('fast_conditional_displacement.ipynb',8,7,'runs/pulse_spacing/notebook_execution.log'),
             ('higher_order_cd_and_squeezing.ipynb',7,6,'runs/swap200/notebook_execution.log'),
             ('coupler_phase_transients.ipynb',9,7,'runs/audit/coupler_notebook_execution.log')]
    for name, code_count, figure_count, log in specs:
        notebook = nbformat.read(notebook_path(name), 4)
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type == 'code']
        assert len(code) == code_count
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type == 'error' for c in code for o in c.outputs)
        figures = sum(any(k.startswith('image/') for k in o.get('data', {}))
                      for c in code for o in c.outputs)
        assert figures == figure_count, (name, figures)
        log_text = (ROOT / log).read_text()
        assert f'EXECUTED NOTEBOOK {{\'file\': \'{name}\'' in log_text
        assert sum(line.startswith('COMPLETED ' + name + ' cell') for line in log_text.splitlines()) == code_count
        row = dict(file=name, cells=len(notebook.cells), executed_code_cells=code_count,
                   embedded_figures=figures, errors=0, notebook_sha256=sha(notebook_path(name)),
                   execution_log=log)
        notebooks.append(row)
        print('NOTEBOOK OUTPUT CHECK', row)
    assert sum(r['executed_code_cells'] for r in notebooks) == 31
    assert sum(r['embedded_figures'] for r in notebooks) == 26
    (ROOT / 'runs/audit/notebook_execution.json').write_text(json.dumps(notebooks, indent=2) + '\n')

    source_names = ['coupler_phase_model.py', 'coupler_replay.cpp', 'calibrate_coupler.py',
                    'calibrate_programmed_family.py', 'verify_programmed_family.py',
                    'check_programmed_convergence.py', 'finalize_audit.py']
    sdk = ROOT.parent.parent / 'acadia_qmsmt/acadia_qmsmt/qmsmt.py'
    sdk_sha = sha(sdk)
    assert sdk_sha == '769a10adf1119cfeb8b29f53bbba832a2a96b73b41b4dae649479fbbbaac9713'
    final = dict(gates=2, columns=36, propagated_amplitudes=11520,
                 raw_and_frame_corrected_amplitudes_checked=23040,
                 results=microscopic, high_photon_convergence=convergence,
                 high_photon_convergence_scope='g8/e8 for each family; not an all-column cutoff certificate',
                 circuit_status='assumed normal-mode parameters; ideal analog reconstruction; no device-fidelity claim',
                 frame_calibration='fixed observable reference states, two common frames, no target/trace argument',
                 current_source_sha256={name: sha(CODE / name) for name in source_names},
                 qmsmt_unchanged_sha256=sdk_sha, notebooks=notebooks)
    (ROOT / 'runs/coupler/final_coupler_manifest.json').write_text(json.dumps(final, indent=2) + '\n')
    # Keep the familiar entry point consistent with the finalized records.
    (ROOT / 'runs/coupler/programmed_manifest_a20_c8_dt0.005.json').write_text(json.dumps(final, indent=2) + '\n')
    report=dict(short_gate_certificates=short_manifest,notebooks=notebooks,
                selected_gates=dict(cd_beta1='cd_constant_reference',squeeze_8dB='squeeze_constant_reference'),
                current_fixed_sideband_study=load('runs/research_verified/manifest.json'),
                current_minimum_swap_ns=200,
                hardware_limited_study=json.loads((ROOT/'runs/swap200/final_manifest.json').read_text()),
                historical_number_rotation_outside_current_limit='strength_rotation_swap150',
                baseline_selected_gates=dict(cd_beta1='cd_binary_scaled_b1',squeeze_8dB='strength_squeeze_swap300'),
                higher_order_study=json.loads((ROOT/'runs/higher_order_cd/final_manifest.json').read_text()),
                cd_400ns_check=completion,
                binary_cd_rule_minimum_fidelity=min(r['fidelity'] for r in rule_certificates),
                fixed_sequence_strength_points=176,retuned_or_screened_map_points=63,
                ideal_reference_points=32,qmsmt_unchanged_sha256=sdk_sha,
                scope='Finite documented searches; no global optimality or exhaustive continuous-control claim.',
                model_source_sha256={name:sha(CODE/name) for name in [
                    'cd_trains.py','hardware_check.py','fast_waveform_replay.cpp','native_waveform_check.cpp',
                    'shaped_pulse_optimisation.py','waveform_adjoint.cpp',
                    'constant_pump_gaussian.py','binary_cd.py','binary_cd_scaling.py',
                    'verify_short_gates.py','short_gate_strength_curves.py','make_story_data.py']})
    (study/'final_study_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL COUNTS:2 current notebooks,15 code cells,13 figures;2 historical appendices,16 code cells,13 figures;SDK unchanged')


if __name__ == '__main__':
    main()
