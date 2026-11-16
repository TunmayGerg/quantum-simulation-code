"""Audit provenance, comparison counts and populated follow-up outputs."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import csv,hashlib,json
from pathlib import Path
import nbformat
import numpy as np
from higher_order_cd import ROOT,OUT
from verify_higher_order_cd import make_model


def main():
    names=[]
    for filename in ['corrected_summary','product_summary','overlap_summary','squeeze_summary',
                     'phasecycle_summary','dcg_summary','dcg_generator_summary',
                     'phase_only_summary','nearby_squeeze_summary']:
        rows=json.loads((OUT/(filename+'.json')).read_text())
        names.extend(r['job']['name'] for r in rows)
    names.append('compressed_squeeze_n112')
    assert len(names)==64 and len(set(names))==64
    results=[];accepted_amplitudes=0;compiled_words=0
    for name in names:
        source=json.loads((OUT/(name+'.json')).read_text())
        cert=json.loads((OUT/'verified'/(name+'.json')).read_text())
        assert cert['columns']==18
        assert cert['independent_amplitudes']==2*cert['levels']*18
        assert cert['independent_max_difference']<1e-10
        last=cert['cutoff_history'][-1]
        assert max(last['output_change'],last['target_change'])<1e-9
        model=make_model(source)
        _,_,_,_,words=model.arrays(np.asarray(source['parameters']))
        digest=hashlib.sha256(words['qubit'].tobytes()+words['sideband'].tobytes()).hexdigest()
        assert digest==cert['word_sha256'],name
        if cert['accepted']:
            assert cert['fidelity']>.99
            stored=np.load(OUT/'verified'/(name+'_words.npz'))
            for key in words:assert np.array_equal(stored[key],words[key])
            columns=np.load(OUT/'verified'/(name+'_columns.npz'))
            V,T=columns['output'],columns['target']
            assert V.shape==T.shape==(2*cert['levels'],18)
            assert abs(abs(np.vdot(T,V)/18)**2-cert['fidelity'])<1e-13
            accepted_amplitudes+=V.size
            assert cert['compiled']['arbitrary_commands']==2 and cert['compiled']['triggers']==1
            assert cert['compiled']['int16_values_compared']==4*len(words['qubit'])
            compiled_words+=cert['compiled']['int16_values_compared']
            parameters=np.asarray(source['parameters']);n=model.cells
            if source['job']['name'].startswith('compressed_squeeze_n'):
                with (OUT/'verified'/(name+'_sequence.csv')).open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['pulse','start_ns','pi_duration_ns','qubit_DAC_deg',
                                     'pump_plateau_DAC_deg','g_rad_per_us','pump_transition_ns'])
                    for j in range(n):
                        indices=np.flatnonzero(model.qcell==j)
                        writer.writerow([j+1,indices[0]*1.25,30.,-np.degrees(parameters[j])%360,
                                         np.degrees(parameters[n+j])%360,np.exp(parameters[-1]),10.])
            elif 'method' in source['job']:
                with (OUT/'verified'/(name+'_JC_sequence.csv')).open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['section','seed_frame','start_ns','duration_ns',
                                     'g_magnitude_rad_per_us','pump_DAC_deg','ramp_each_side_ns'])
                    m=model.section_count
                    force=parameters[n:n+m].astype(complex)
                    if model.complex_pump:force+=1j*parameters[n+m:]
                    pump=-1j*force
                    for j in range(m):
                        column=model.W.getcol(j).toarray()[:,0];indices=np.flatnonzero(column)
                        start=max(0,indices[0]-1);end=indices[-1]+1
                        writer.writerow([j+1,model.coefficients[j][0],start*1.25,(end-start)*1.25,
                                         abs(pump[j]),np.degrees(np.angle(pump[j]))%360,5.])
        if source['job'].get('magnitudes_fixed',False):
            p=np.asarray(source['parameters']);n=model.cells;m=model.section_count
            assert np.max(abs(np.hypot(p[n:n+m],p[n+m:])-source['fixed_section_magnitudes']))<1e-12
        results.append(cert)
    manifest=dict(records=len(results),columns=18*len(results),
                  independent_amplitudes=sum(r['independent_amplitudes'] for r in results),
                  accepted=sum(r['accepted'] for r in results),results=results)
    (OUT/'verified/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('NEW STUDY AUDIT:',manifest['records'],'records;',manifest['columns'],'columns;',
          manifest['independent_amplitudes'],'independentamplitudes;',manifest['accepted'],'accepted;',
          accepted_amplitudes,'savedacceptedamplitudes;',compiled_words,'SDKint16values',flush=True)
    notebook=nbformat.read(notebook_path('runs/audit/pre_swap200/higher_order_cd_and_squeezing.ipynb'),4)
    nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.outputs if o.output_type=='error']
    figures=sum(any(k.startswith('image/') for k in o.get('data',{})) for c in code for o in c.outputs)
    assert len(code)==7 and all(c.execution_count is not None for c in code)
    assert not errors and figures>=5
    print('ARCHIVED FOLLOW-UP NOTEBOOK:',len(notebook.cells),'cells;',len(code),'executedcodecells;',figures,'figures;zeroerrors',flush=True)
    sensitivity=json.loads((OUT/'sensitivity.json').read_text())
    assert sensitivity['points']==108 and sensitivity['columns']==1944
    assert all(r['output_change']<1e-9 for r in sensitivity['curves'])
    assert len(sensitivity['derivatives'])==6
    print('SENSITIVITY AUDIT:108points;1944columns;',sensitivity['cutoff_comparison_amplitudes'],'amplitudes;6derivatives',flush=True)
    sdk=ROOT.parents[1]/'acadia_qmsmt/acadia_qmsmt/qmsmt.py'
    assert hashlib.sha256(sdk.read_bytes()).hexdigest()=='769a10adf1119cfeb8b29f53bbba832a2a96b73b41b4dae649479fbbbaac9713'
    final=dict(**{k:v for k,v in manifest.items() if k!='results'},notebook_codecels=len(code),
               notebook_figures=figures,sensitivity_points=sensitivity['points'],
               source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [CODE/name for name in ['higher_order_cd.py','verify_higher_order_cd.py',
                   'higher_order_story.py','compress_squeeze_blocks.py','dcg_derivative_cd.py',
                   'phase_only_product_cd.py','build_higher_order_notebook.py']]})
    (OUT/'final_manifest.json').write_text(json.dumps(final,indent=2)+'\n')


if __name__=='__main__':main()
