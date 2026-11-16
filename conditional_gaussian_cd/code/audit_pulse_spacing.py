"""Audit saved independent spacing simulations, fixed targets and exact timing."""
import csv,hashlib,json
import numpy as np
import nbformat
from project_paths import ROOT,NOTEBOOKS,CODE
from pulse_spacing_sweep import jobs,MAX_G
import cd_trains as cd
import hardware_check as hw

OUT=ROOT/'runs/pulse_spacing'


def main():
    summary=json.loads((OUT/'summary.json').read_text());rows=summary['results'];expected=jobs()
    assert len(rows)==len(expected)==18 and {r['name'] for r in rows}=={r['name'] for r in expected}
    amplitudes=0;columns=0;timing_rows=0;maximum_error=0.
    groups={}
    for r in rows:
        gate=r['gate'];target=1. if gate=='cd' else cd.R_8DB
        coefficient=target if gate=='cd' else target/2
        assert r['target_strength']==target and r['wanted_coefficient_target']==coefficient
        assert r['relative_coefficient_error']==abs(r['wanted_coefficient_actual']/coefficient-1)
        assert r['relative_coefficient_error']<.003
        assert r['phases_reoptimized'] is False and r['fidelity_optimized'] is False
        assert r['qubit_envelope_ns']==30 and abs(r['command_us']-r['duration_us'])<1e-12
        assert abs(r['cells']*r['spacing_ns']+10-r['duration_us']*1000)<1e-8
        assert abs(r['gap_ns']+30-r['spacing_ns'])<1e-8
        if gate=='cd':assert r['qubit_frame_Rz']==0.
        last=r['cutoff_history'][-1];assert max(last['output_change'],last['target_change'])<1e-9
        with np.load(OUT/(r['name']+'_columns.npz')) as z:V,T=z['output'],z['target']
        assert V.shape==T.shape==(2*r['levels'],18)
        assert np.max(abs(np.sum(abs(V)**2,axis=0)-1))<1e-9
        assert np.max(abs(np.sum(abs(T)**2,axis=0)-1))<1e-9
        overlaps=np.sum(T.conj()*V,axis=0)
        assert abs(abs(overlaps.mean())**2-r['fidelity'])<1e-13
        assert abs(min(abs(overlaps)**2)-r['worst_basis_fidelity'])<1e-13
        assert r['independent_amplitudes']==V.size and r['independent_max_error']<1e-10
        maximum_error=max(maximum_error,r['independent_max_error']);amplitudes+=V.size;columns+=18
        with np.load(OUT/(r['name']+'_words.npz')) as words:
            assert words['qubit'].dtype==np.int16 and words['sideband'].dtype==np.int16
            digest=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest()
            assert digest==r['word_sha256']
            qs,ss=hw.dac_scales(.6);s=hw.sample_to_complex(words['sideband'])/ss*np.pi/(2*.6)
            assert max(abs(s[4:-4]-s[4]))<1e-12
            peak=float(max(abs(hw.rfdc_interpolate(s))));assert peak<=MAX_G
            assert abs(1000*np.pi/(2*peak)-r['peak_swap_ns'])<1e-10
            assert len(words['qubit'])*cd.DT==r['command_us']
        with (OUT/(r['name']+'_pulses.csv')).open() as f:table=list(csv.DictReader(f))
        assert len(table)==r['cells']
        centers=np.array([float(p['center_ns']) for p in table])
        assert max(abs(np.diff(centers)-r['spacing_ns']))<1e-8
        assert all(float(p['duration_ns'])==30 for p in table)
        assert all(float(p['start_ns'])>=5 for p in table)
        assert float(table[-1]['start_ns'])+30<=r['command_us']*1000-5+1e-8
        timing_rows+=len(table);groups.setdefault((gate,r['duration_us']),[]).append(r)
        print('SPACING AUDIT',r['name'],'spacing_ns',r['spacing_ns'],'Tns',r['command_us']*1000,
              '18columns;',V.size,'amplitudes;F',r['fidelity'],'coefficient_relative_error',r['relative_coefficient_error'])
    assert columns==summary['columns']==324 and amplitudes==summary['independent_amplitudes']==133632
    assert len(groups)==3
    for key,group in groups.items():
        assert max(r['command_us'] for r in group)-min(r['command_us'] for r in group)<1e-12
        if key[0]=='cd':assert max(r['g'] for r in group)-min(r['g'] for r in group)<1e-12
    series=json.loads((OUT/'series_validation.json').read_text())
    assert len(series)==3 and sum(r['amplitudes'] for r in series)==1536
    assert max(r['max_error'] for r in series)<1e-10
    books=[];log=(OUT/'notebook_execution.log').read_text()
    for name,code_count,figure_count in [('fast_conditional_displacement.ipynb',8,7),('conditional_gaussian_gates.ipynb',7,6)]:
        b=nbformat.read(NOTEBOOKS/name,4);nbformat.validate(b);cells=[c for c in b.cells if c.cell_type=='code']
        figures=sum(any(k.startswith('image/') for k in o.get('data',{})) for c in cells for o in c.outputs)
        assert len(cells)==code_count and figures==figure_count
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        assert sum(line.startswith('COMPLETED '+name+' cell') for line in log.splitlines())==code_count
        books.append(dict(file=name,executed_cells=code_count,figures=figure_count))
    report=dict(points=18,columns=columns,independent_complex_amplitudes=amplitudes,maximum_independent_error=maximum_error,
        exact_pulse_timing_rows=timing_rows,coefficient_validation_comparisons=1536,fixed_duration_groups=3,
        notebooks=books,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [CODE/'pulse_spacing_sweep.py',CODE/'coupling_series.py',CODE/'jc_coupling_series.cpp',CODE/'pulse_spacing_figures.py']})
    (OUT/'final_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SPACING FINAL:18points;324columns;133632independent amplitudes;',timing_rows,'timing rows;',
          '15executed cells;13figures;max independent error',maximum_error)


if __name__=='__main__':main()
