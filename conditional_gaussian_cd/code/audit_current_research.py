"""Count and cross-check the current gates, scaling data and notebook outputs.

This audits saved independent-propagation evidence; it does not pretend that
reading a certificate is another independent dynamics simulation.
"""
import ast
import hashlib
import json
import re
from pathlib import Path
import nbformat
import numpy as np
from project_paths import ROOT,CODE,NOTEBOOKS
import cd_trains as cd
import hardware_check as hw


def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_columns(folder,row):
    with np.load(folder/(row['name']+'_columns.npz')) as saved:
        V,T=saved['output'],saved['target']
    assert V.shape==T.shape and V.shape[1]==18
    assert abs(abs(np.vdot(T,V)/18)**2-row['fidelity'])<1e-13
    assert np.max(abs(np.sum(abs(V)**2,axis=0)-1))<1e-9
    assert np.max(abs(np.sum(abs(T)**2,axis=0)-1))<1e-9
    overlaps=np.sum(T.conj()*V,axis=0)
    assert abs(min(abs(overlaps)**2)-row['worst_basis_fidelity'])<1e-13
    with np.load(folder/(row['name']+'_words.npz')) as words:
        digest=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest()
        assert digest==row['word_sha256']
        assert words['qubit'].dtype==np.int16 and words['sideband'].dtype==np.int16
        scales=hw.dac_scales(.6)
        pump=hw.sample_to_complex(words['sideband'])/scales[1]*(np.pi/(2*.6))
        assert np.max(abs(pump[4:-4]-pump[4]))<1e-12
        peak=float(np.max(abs(hw.rfdc_interpolate(pump))))
        assert peak<=np.pi/(2*.200)+1e-12
    return V.size


def main():
    # All simulation sources and all notebooks must live in their requested folders.
    sources=list(ROOT.rglob('*.py'))+list(ROOT.rglob('*.cpp'))
    books=list(ROOT.rglob('*.ipynb'))
    assert sources and books
    assert all(p.is_relative_to(CODE) for p in sources),[str(p) for p in sources if not p.is_relative_to(CODE)]
    assert all(p.is_relative_to(NOTEBOOKS) for p in books)
    parsed=0
    for p in sources:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p));parsed+=1
    print('LAYOUT:',len(sources),'sources;',parsed,'Python files parsed;',len(books),'notebooks in notebooks/')
    all_books=[]
    for p in books:
        b=nbformat.read(p,4);nbformat.validate(b)
        cells=[c for c in b.cells if c.cell_type=='code']
        figures=sum(any(k.startswith('image/') for k in o.get('data',{})) for c in cells for o in c.outputs)
        assert cells and figures>0 and all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        all_books.append(dict(file=str(p.relative_to(ROOT)),executed_cells=len(cells),figures=figures))
    assert len(all_books)==10 and sum(r['executed_cells'] for r in all_books)==87
    assert sum(r['figures'] for r in all_books)==140
    print('ALL NOTEBOOKS:10 current/historical books;87 executed cells;140 embedded figures;0 saved errors')

    folder=ROOT/'runs/research_verified';names=['cd_constant_reference','squeeze_constant_reference']
    refs=[read('runs/research_verified/'+n+'.json') for n in names]
    n_amplitudes=0
    for row in refs:
        assert row['accepted'] and row['sideband_mode']=='fixed' and row['columns']==18
        assert row['peak_swap_ns']>=200 and row['independent_max_difference']<1e-10
        final=row['cutoff_history'][-1]
        assert max(final['output_change'],final['target_change'])<1e-9
        amplitudes=check_columns(folder,row);assert amplitudes==row['complex_amplitudes']
        recipe=read(row['recipe_file']);pulses=recipe['pulses'];assert pulses
        assert all(p['samples']*cd.DT>=.030-1e-14 for p in pulses)
        assert row['compiled']['arbitrary_commands']==2 and row['compiled']['triggers']==1
        n_amplitudes+=amplitudes
        print('CURRENT GATE:',row['name'],'18 columns;',amplitudes,'amplitudes;F',row['fidelity'],
              'weakest basis',row['worst_basis_fidelity'],'independent error',row['independent_max_difference'])
    assert len(refs)==2 and n_amplitudes==14976

    # Count independent evidence at fixed beta, not a fixed-time underdisplacement sweep.
    strength=read('runs/reference_strengths/summary.json');points=strength['results']
    assert len(points)==5 and strength['target_input_columns_compared']==90
    assert strength['independent_amplitudes_compared']==11520
    assert all(r['strength']==1 and r['pump_amplitude_constant'] and r['pump_phase_constant'] for r in points)
    assert np.all(np.diff([r['peak_swap_ns'] for r in points])>0)
    assert np.all(np.diff([r['command_time_us'] for r in points])>0)
    assert np.all(np.diff([r['fidelity'] for r in points])>0)
    assert max(r['independent_backend_max_amplitude_error'] for r in points)<1e-10
    print('SAME BETA:',len(points),'gates;90 columns;11520 independently compared amplitudes;F range',
          min(r['fidelity'] for r in points),max(r['fidelity'] for r in points))

    rule=read('runs/constant_pump_gaussian/binary_rule_summary.json')
    manifest=read('runs/constant_pump_gaussian/verified/manifest.json')
    rule_names={r['job']['name'] for r in rule['results']}
    scaling=[r for r in manifest['results'] if r['name'] in rule_names]
    assert len(scaling)==20 and sum(r['columns'] for r in scaling)==360
    assert min(r['fidelity'] for r in scaling)>.99
    scaling_amplitudes=sum(check_columns(ROOT/'runs/constant_pump_gaussian/verified',r) for r in scaling)
    assert scaling_amplitudes==71424
    print('BETA SCALING:20 targets;360 columns;',scaling_amplitudes,'amplitudes;minimumF',min(r['fidelity'] for r in scaling))

    open_z=[]
    for length,pairs,half_paths in [(8,855296,4096),(10,50039808,32768)]:
        r=read(f'runs/open_z_squeezing/length{length}.json')
        assert r['Z_endpoint_pairs']==pairs and r['nearest_queries']==half_paths
        assert r['zero_first_Z_endpoints']==0 and r['minimum_first_moment_norm']>1e-4
        assert r['independent_moment_amplitudes']==2048 and r['independent_moment_max_error']<1e-11
        open_z.append(r)
    print('OPEN-Z:',sum(r['Z_endpoint_pairs'] for r in open_z),
          'endpoint pairs;0 zero-first-moment paths;4096 independent coefficient amplitudes')

    log=(ROOT/'runs/pulse_spacing/notebook_execution.log').read_text();reports=[]
    for name,ncode,nfig in [('fast_conditional_displacement.ipynb',8,7),('conditional_gaussian_gates.ipynb',7,6)]:
        p=NOTEBOOKS/name;book=nbformat.read(p,4);nbformat.validate(book)
        cells=[c for c in book.cells if c.cell_type=='code']
        figures=sum(any(k.startswith('image/') for k in o.get('data',{})) for c in cells for o in c.outputs)
        assert len(cells)==ncode and figures==nfig
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        assert sum(line.startswith('COMPLETED '+name+' cell') for line in log.splitlines())==ncode
        reports.append(dict(file=str(p.relative_to(ROOT)),executed_cells=ncode,figures=nfig,sha256=sha(p)))
        print('NOTEBOOK:',name,ncode,'executed cells;',nfig,'embedded figures;0 errors')
    assert re.search(r'FRESH CD CHECK:18columns; 4608 amplitudes',log)
    assert re.search(r'FRESH SQUEEZE CHECK:2boundaryinputs; 1536 amplitudes',log)
    docs=read('runs/audit/document_checks.json');assert len(docs)==6
    sdk=ROOT.parents[1]/'acadia_qmsmt/acadia_qmsmt/qmsmt.py'
    sdk_sha=sha(sdk);assert sdk_sha=='769a10adf1119cfeb8b29f53bbba832a2a96b73b41b4dae649479fbbbaac9713'
    report=dict(current_references=refs,independent_reference_columns=36,independent_reference_amplitudes=n_amplitudes,
        fixed_beta_comparisons=5,fixed_beta_input_columns=90,fixed_beta_independent_amplitudes=11520,
        signed_beta_targets=20,signed_beta_columns=360,signed_beta_independent_amplitudes=scaling_amplitudes,
        fresh_notebook_CD_amplitudes=4608,fresh_notebook_squeeze_boundary_amplitudes=1536,
        notebooks=reports,all_current_and_historical_notebooks=all_books,
        documents=docs,open_Z_searches=open_z,qmsmt_unchanged_sha256=sdk_sha,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in CODE.glob('*') if p.suffix in ('.py','.cpp')},
        scope='Saved-artifact audit plus fresh notebook replay. Finite searches; no global optimum claim.')
    (ROOT/'runs/fixed_sideband/current_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CURRENT AUDIT:2 references;15 executed code cells;13 figures;SDK unchanged')


if __name__=='__main__':main()
