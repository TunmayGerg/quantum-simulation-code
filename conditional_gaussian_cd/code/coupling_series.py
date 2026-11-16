"""Wanted first/second Magnus coefficients from the actual sampled waveform."""
import ctypes,hashlib,subprocess
from pathlib import Path
import numpy as np
from project_paths import CODE


def coefficients(q,s,dt,levels=8,order=14):
    source=CODE/'jc_coupling_series.cpp'
    library=Path('/tmp')/('jc_coupling_series_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.so')
    if not library.exists():subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    f=ctypes.CDLL(str(library)).jc_coupling_series
    f.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int]+[ctypes.c_void_p]*3
    q=np.ascontiguousarray(q);s=np.ascontiguousarray(s);d=2*levels
    C=np.zeros((3,d,d),complex)
    f(levels,len(q),dt,order,q.ctypes.data,s.ctypes.data,C.ctypes.data)
    assert np.max(abs(C[0].conj().T@C[0]-np.eye(d)))<1e-10
    normalized=np.array([C[0].conj().T@v for v in C])
    first=normalized[1];second=normalized[2]-first@first/2
    return first,second


def wanted(q,s,dt,gate):
    first,second=coefficients(q,s,dt)
    a=np.diag(np.sqrt(np.arange(1,8)),1);low=np.arange(8)
    if gate=='cd':G=np.kron(a.T-a,np.array([[0.,1.],[1.,0.]]));L=first
    else:G=np.kron(a@a-a.T@a.T,np.diag([1.,-1.]));L=second
    b=G[:,low].ravel();v=L[:,low].ravel()
    value=float(np.vdot(b,v).real/np.vdot(b,b).real)
    return value,dict(first_norm=float(np.linalg.norm(first[:,low])),second_norm=float(np.linalg.norm(second[:,low])),
        unwanted_at_calibration_order=float(np.linalg.norm(v-value*b)))
