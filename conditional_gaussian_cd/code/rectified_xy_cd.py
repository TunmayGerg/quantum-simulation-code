"""Dynamically corrected CD seeds with two Y pulses and rectified JC phase.

Pauli Y flips the desired sigma_x force. Flip the pump phase too to keep
that force pointing forward. Nine X envelopes plus two Y envelopes occupy
340ns including outer ramps; two pi/2 X rotations allow a closed qubit path.
This spends two pulse slots on a different finite-control trajectory to escape
the X-only cubic sign argument. First/second/third moments are calculated,
not assumed to cancel. All pump changes remain10ns circular IQ arcs at zeros.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import itertools,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from project_paths import ROOT
from phase_arc_cd import ArcCD,symbol,MAX_G,fit

OUT=ROOT/'runs/rectified_xy_cd';OUT.mkdir(exist_ok=True)
X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex)


def catalogue():
    started=time.time();model=ArcCD(11,32,False);records=[];tested=0;closed=0
    for ypos in itertools.combinations(range(11),2):
        xpos=[j for j in range(11) if j not in ypos]
        for hi,halves in enumerate([[0,8],[1,7],[3,4]]):
            for word_id,word in enumerate([np.resize([0,0,np.pi,np.pi],9),np.zeros(9),np.resize([0,np.pi],9)]):
                for ysign in itertools.product([1,-1],repeat=2):
                    phi=np.zeros(11);fraction=np.ones(11)
                    phi[xpos]=word;fraction[[xpos[j] for j in halves]]=.5
                    phi[list(ypos)]=np.array(ysign)*np.pi/2
                    Q=np.eye(2,dtype=complex)
                    for axis,angle in zip(phi,np.pi*fraction):
                        P=X*np.cos(axis)+Y*np.sin(axis)
                        Q=(np.cos(angle/2)*np.eye(2)-1j*np.sin(angle/2)*P)@Q
                    tested+=1
                    if abs(abs(np.trace(Q)/2)-1)>1e-10:continue
                    closed+=1
                    for y_phase in ['entry','midpoint','exit']:
                        psi=np.empty(11);count=0
                        for j in range(11):
                            psi[j]=np.pi*count
                            if j in ypos:
                                psi[j]+={'entry':0,'midpoint':np.pi/2,'exit':np.pi}[y_phase];count+=1
                        # The finite ramps and waveform determine the true
                        # first force, including JC during both Y pulses.
                        model=ArcCD(11,24,False,fraction)
                        p=np.r_[phi,psi,0.];q,s=model.raw(p);A,_,_=symbol(q,s)
                        global_phase=np.arctan2(-A[0,0],A[1,0]);psi+=global_phase;p[11:22]=psi
                        q,s=model.raw(p);A,B,C=symbol(q,s)
                        if A[1,0]<=0:continue
                        g=min(1/A[1,0],model.maximum_g);p[-1]=np.log(g)
                        target=np.zeros((2,3));target[1,0]=1
                        first=float(np.linalg.norm(g*A-target));second=float(g*g*np.linalg.norm(B));third=float(g**3*np.linalg.norm(C))
                        records.append(dict(y_positions=list(ypos),y_signs=list(ysign),half_x_indices=halves,word_id=word_id,
                            y_phase=y_phase,phi=phi.tolist(),fractions=fraction.tolist(),parameters=p.tolist(),
                            first_error=first,second_norm=second,third_norm=third,curvature=float(g**3*C[1,0]),
                            score=4*first+8*second+20*third))
        if len(records)%100<18:print('RECTIFIED XY catalogue',len(records),'moment paths',flush=True)
    assert tested==55*3*3*4 and closed>0
    chosen=[]
    for row in sorted(records,key=lambda r:r['score']):
        chosen.append(row)
        if len(chosen)==60:break
    for i,row in enumerate(chosen):
        m=ArcCD(11,32,False,row['fractions'])
        row['fidelity']=1-m.evaluate(np.asarray(row['parameters']),False)
        print('RECTIFIED XY FULL',i,'F',row['fidelity'],'Y',row['y_positions'],'first',row['first_error'],
              'second',row['second_norm'],'third',row['third_norm'],flush=True)
    payload=dict(qubit_words_tested=tested,closed_qubit_words=closed,moment_paths=len(records),
        full_waveforms=len(chosen),full_input_columns=18*len(chosen),results=chosen,seconds=time.time()-started)
    (OUT/'catalogue.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('RECTIFIED XY COMPLETE:',tested,'qubit words;',len(records),'moment paths;',18*len(chosen),'actual-filter columns',flush=True)


def refine():
    payload=json.loads((OUT/'catalogue.json').read_text());chosen=[]
    for row in sorted(payload['results'],key=lambda r:-r['fidelity']):
        if any(row['y_positions']==r['y_positions'] and row['half_x_indices']==r['half_x_indices'] and row['word_id']==r['word_id'] for r in chosen):continue
        chosen.append(row)
        if len(chosen)==6:break
    jobs=[]
    for i,row in enumerate(chosen):
        psi=np.asarray(row['parameters'][11:22])
        jobs.append(dict(name=f'arc_rectified_xy_n11_{i}',cells=11,symmetric=False,fractions=row['fractions'],
            parameters=row['parameters'],phase_lower=(psi-1.4).tolist(),phase_upper=(psi+1.4).tolist(),
            frame=True,budget=400,seed_reason='Two Y pulses with JC sign rectification; closed fractional-angle qubit path screened by first/second/third moments.'))
    with ProcessPoolExecutor(3) as pool:results=list(pool.map(fit,jobs))
    assert len(results)==6
    (OUT/'refinement.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RECTIFIED XY REFINEMENT:6 fits;',18*sum(r['evaluations'] for r in results),'input-column evaluations',flush=True)


if __name__=='__main__':
    import sys
    catalogue() if len(sys.argv)<2 or sys.argv[1]=='catalogue' else refine()
