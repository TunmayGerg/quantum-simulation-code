"""Count and verify phase-enabled CD artifacts and executed notebook evidence."""
import hashlib,json
import numpy as np
import nbformat
from project_paths import ROOT,NOTEBOOKS,CODE
from verify_research import check_phase_arcs,MAX_G
import cd_trains as cd
import hardware_check as hw


def read(path):return json.loads((ROOT/path).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    names=['arc_sym_n12_magnus4','arc_lifted_n12','arc_free_n10_all5_1','central_arc_gap50_half3_word0']
    records=[];amplitudes=0;arc_count=0
    for name in names:
        r=read('runs/research_verified/'+name+'.json');recipe=read(r['recipe_file']);raw=np.load(ROOT/r['controls_file'])
        assert r['gate']=='cd' and r['strength']==1 and r['columns']==18
        assert r['independent_max_difference']<1e-10 and r['peak_swap_ns']>=200
        assert max(r['cutoff_history'][-1]['output_change'],r['cutoff_history'][-1]['target_change'])<1e-9
        arc_count+=check_phase_arcs(recipe,raw['q'],raw['s'])
        assert max(abs(abs(raw['s'][4:-4])-abs(raw['s'][4])))<1e-9
        with np.load(ROOT/'runs/research_verified'/(name+'_columns.npz')) as data:
            V,T=data['output'],data['target']
        assert V.shape==T.shape==(2*r['levels'],18)
        assert abs(abs(np.vdot(T,V)/18)**2-r['fidelity'])<1e-13
        assert abs(min(abs(np.sum(T.conj()*V,axis=0))**2)-r['worst_basis_fidelity'])<1e-13
        assert max(abs(np.sum(abs(V)**2,axis=0)-1))<1e-9
        amplitudes+=V.size;assert V.size==r['complex_amplitudes']
        with np.load(ROOT/'runs/research_verified'/(name+'_words.npz')) as words:
            digest=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest()
            assert digest==r['word_sha256']
            qs,ss=hw.dac_scales(.6)
            q=hw.sample_to_complex(words['qubit']).conj()/qs*np.pi/cd.PI_TIME
            s=hw.sample_to_complex(words['sideband'])/ss*np.pi/(2*.6)
            qf,sf=hw.rfdc_interpolate(q),hw.rfdc_interpolate(s)
            assert max(abs(sf))<=MAX_G
            delay=np.argmax(abs(hw.rfdc_interpolate(np.array([1.]))))
            assert max(abs(qf[[8*a['center_sample']+delay for a in recipe['phase_arcs']]]))<1e-12
        records.append(r)
        print('PHASE CERTIFICATE:',name,'18columns;',V.size,'amplitudes;F',r['fidelity'],
              'arcs',len(recipe['phase_arcs']),'peak_swap_ns',r['peak_swap_ns'])
    assert len(records)==4 and amplitudes==13824
    assert sum(r['accepted'] for r in records)==2
    assert min(r['command_us'] for r in records if r['accepted'])>=.370-1e-12
    qcheck=read('runs/phase_arc_cd/whole_gate_magnus_check.json')
    assert len(qcheck)==2 and max(r['exact_second_norm'] for r in qcheck)<1e-12
    assert sum(r['first_amplitudes'] for r in qcheck)==256
    assert sum(r['third_operator_coefficients'] for r in qcheck)==24
    print('COMPLETE-GATE MAGNUS:2waveforms;256first amplitudes;24cubiccoefficients;both second norms<1e-12')
    finite=read('runs/phase_arc_cd/finite_gate_error.json')
    assert len(finite)==6 and sum(r['full_columns'] for r in finite)==1152
    assert sum(r['amplitudes'] for r in finite)==237568
    last=[r for r in finite if r['levels']==128]
    assert len(last)==2 and all(r['polynomial_fit_stable'] for r in last)
    assert all(r['fit_relative_residual']>.6 for r in last)
    print('FINITE ERROR DIAGNOSTIC:6propagators;1152columns;237568amplitudes;polynomial fits leave>60% norm residual')

    n=NOTEBOOKS/'phase_controlled_cd.ipynb';book=nbformat.read(n,4);nbformat.validate(book)
    code=[c for c in book.cells if c.cell_type=='code']
    figures=sum(any(k.startswith('image/') for k in o.get('data',{})) for c in code for o in c.outputs)
    assert len(code)==6 and figures==5 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    log=(ROOT/'runs/phase_arc_cd/notebook_execution.log').read_text()
    assert sum(line.startswith('COMPLETED phase_controlled_cd.ipynb cell') for line in log.splitlines())==6
    assert 'FRESH PHASE CD CHECK:18 columns; 4608 complex amplitudes' in log
    assert 'ARC CHECK:11 centers;max played qubit amplitude at those centers 0.0' in log
    print('PHASE NOTEBOOK:6executed cells;5figures;fresh18columns/4608amplitudes;11actualplayed qubit-zero centers')

    groups={
        'n10_refinements':read('runs/phase_arc_cd/refine_n10.json'),
        'fraction_wait_patterns':read('runs/phase_arc_cd/pattern_results.json'),
        'signed_JC_n11':read('runs/phase_arc_cd/reversed_jc_results.json'),
        'rectified_XY':read('runs/rectified_xy_cd/refinement.json'),
        'central_JC_arc':read('runs/central_arc_cd/refinement.json'),
        'n12_refinements':read('runs/phase_arc_cd/refine_n12.json'),
        'half_pair_continuation':read('runs/cd_pair_continuation/half_pairs.json'),
        'prune_split':read('runs/phase_arc_cd/prune_split_results.json'),
        'lifted_refinement':[read('runs/phase_arc_cd/arc_lifted_n12.json')],
    }
    fit_count=sum(len(v) for v in groups.values());fit_columns=18*sum(r['evaluations'] for v in groups.values() for r in v)
    assert fit_count==75 and all(len(v)>0 for v in groups.values())
    winding=read('runs/winding_cd/catalogue.json');xy=read('runs/rectified_xy_cd/catalogue.json')
    assert winding['enumerated_paths']==14256 and winding['full_input_columns']==864
    assert xy['qubit_words_tested']==1980 and xy['moment_paths']==3204 and xy['full_input_columns']==1080
    scaling=read('runs/phase_cd_scaling/results.json');assert len(scaling)==5
    print('COUNTED RESEARCH:',fit_count,'full-unitary fits;',fit_columns,'18-input-column evaluations;',
          '14256 winding paths;3204XYmoment paths')
    report=dict(certified_gates=4,certified_columns=72,independent_amplitudes=amplitudes,
        phase_arcs_reconstructed=arc_count,results=records,full_unitary_fits=fit_count,
        full_fit_input_column_evaluations=fit_columns,magnus_validation=qcheck,finite_coupling_error_diagnostic=finite,
        notebook=dict(file=str(n.relative_to(ROOT)),executed_code_cells=6,embedded_figures=5,sha256=sha(n)),
        winding_catalogue_paths=14256,rectified_XY_moment_paths=3204,scaling_targets=5,
        source_sha256={p.name:sha(p) for p in CODE.glob('*') if p.suffix in ('.py','.cpp')},
        conclusion='Small fidelity improvement at370ns;no shorter99%gate and no demonstrated full cubic cancellation.')
    (ROOT/'runs/phase_arc_cd/final_audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
