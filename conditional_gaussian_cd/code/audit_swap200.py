"""Audit the active hardware-limited recommendations and populated notebook."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import hashlib,json
from pathlib import Path
import nbformat
import numpy as np
from capped_cd_study import OUT,ROOT,MAX_G,MIN_SWAP_NS
from verify_swap200 import source_and_arrays


def main():
    manifest=json.loads((OUT/'verified/manifest.json').read_text())
    assert manifest['records']==14 and manifest['columns']==252
    assert manifest['independent_amplitudes']==56448
    assert manifest['selected']==dict(cd='edge_n10_gap0_phase1',squeeze='squeeze_8db')
    words_checked=0;accepted=0
    for r in manifest['results']:
        record,(qr,sr,q,s,w),model=source_and_arrays(r['name'])
        peak=max(max(abs(sr)),max(abs(s)))
        assert peak<=MAX_G and r['actual_min_swap_ns']>=MIN_SWAP_NS
        assert abs(peak-r['peak_g_rad_per_us'])<1e-13
        assert r['columns']==18 and r['independent_amplitudes']==2*r['levels']*18
        assert r['independent_max_difference']<1e-10
        h=r['cutoff_history'][-1];assert max(h['output_change'],h['target_change'])<1e-9
        digest=hashlib.sha256(w['qubit'].tobytes()+w['sideband'].tobytes()).hexdigest()
        assert digest==r['word_sha256']
        if r['accepted']:
            accepted+=1;assert r['fidelity']>.99
            columns=np.load(OUT/'verified'/(r['name']+'_columns.npz'))
            V,T=columns['output'],columns['target']
            assert V.shape==T.shape==(2*r['levels'],18)
            assert abs(abs(np.vdot(T,V)/18)**2-r['fidelity'])<1e-13
            saved=np.load(OUT/'verified'/(r['name']+'_words.npz'))
            for key in w:assert np.array_equal(w[key],saved[key])
            assert r['compiled']['arbitrary_commands']==2 and r['compiled']['triggers']==1
            assert r['compiled']['int16_values_compared']==4*len(w['qubit'])
            words_checked+=r['compiled']['int16_values_compared']
    notebook=nbformat.read(notebook_path('higher_order_cd_and_squeezing.ipynb'),4);nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert len(code)==7 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    figures=sum(any(k.startswith('image/') for k in o.get('data',{})) for c in code for o in c.outputs)
    assert figures==6
    gradient=json.loads((OUT/'gradient_check.json').read_text())
    assert gradient['directions']==6 and max(gradient['errors'])<2e-5
    log=(OUT/'notebook_execution.log').read_text()
    assert log.count('COMPLETED higher_order_cd_and_squeezing.ipynb cell')==7
    assert 'FRESH CD:18columns; 4608 amplitudes' in log
    assert 'FRESH SQUEEZE:2boundaryinputs; 1536 amplitudes' in log
    assert 'STRENGTH CHECK:9points;162columns; 31104' in log
    ends=json.loads((OUT/'aligned_end_checks.json').read_text())
    assert len(ends)==4 and sum(r['columns'] for r in ends)==72
    assert sum(r['amplitudes'] for r in ends)==21888
    assert all(r['fidelity_at_aligned_end']>.99 and r['remaining_tail_input_operator_norm']<5e-4 for r in ends)
    sdk=ROOT.parents[1]/'acadia_qmsmt/acadia_qmsmt/qmsmt.py'
    assert hashlib.sha256(sdk.read_bytes()).hexdigest()=='769a10adf1119cfeb8b29f53bbba832a2a96b73b41b4dae649479fbbbaac9713'
    final={k:v for k,v in manifest.items() if k!='results'}
    final.update(accepted=accepted,SDK_int16_values_checked=words_checked,
                 notebook_code_cells=7,notebook_figures=6,aligned_end_columns=72,aligned_end_amplitudes=21888,
                 notebook_sha256=hashlib.sha256((notebook_path('higher_order_cd_and_squeezing.ipynb')).read_bytes()).hexdigest(),
                 source_sha256={name:hashlib.sha256((CODE / name).read_bytes()).hexdigest() for name in
                    ['capped_cd_study.py','verify_swap200.py','build_swap200_notebook.py','audit_swap200.py']})
    (OUT/'final_manifest.json').write_text(json.dumps(final,indent=2)+'\n')
    print('SWAP200 AUDIT:14records;252columns;56448independentamplitudes;',accepted,
          'accepted;',words_checked,'SDKint16values;allraw/filteredpeaks<=',MAX_G,flush=True)
    print('POPULATED NOTEBOOK:7executedcodecells;6figures;zeroerrors;',
          'strength9points/162columns/31104amplitudes;freshCD4608andfreshsqueeze1536amplitudes',flush=True)
    print('ALIGNED ENVELOPE ENDS:4gates;72columns;21888amplitudes;allF>99%',flush=True)


if __name__=='__main__':main()
