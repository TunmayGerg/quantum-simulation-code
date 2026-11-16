"""Final check of a gate with the waveform the FPGA would actually play.

The optimiser (cd_trains.py) uses ideal DAC samples. This check changes three things and recomputes the gate fidelity:
  1. the samples are converted to the SDK's signed 16-bit I/Q words (acadia.sample_arithmetic.complex_to_sample, the
     function the SDK's load_pulse uses) at the DAC scales of the calibrated 30 ns pi and of the 600 ns-swap pump
     (from the archived beta_program.json; a faster swap is that pump scale x 600 ns / swap: an assumed linear gain);
  2. the words go through the RF data converter's documented 8x interpolation filters (AMD PG269, Gen-3), including
     the filter's start-up and ring-down tail (+283 fine samples);
  3. the result is integrated sample by sample (at 6.4 GS/s) by an independent C++ Taylor propagator
     (native_waveform_check.cpp), not by the optimiser's cell matrices.
The qubit frame correction (Rz) is FIXED to the optimiser's value; nothing is re-fitted.
No static dispersive shift. Both target and output columns must converge in norm.
Not included: the analog line response after the DAC, and the converter's internal rounding.
"""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import ctypes
import hashlib
import json
import subprocess
import math
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from scipy.signal import upfirdn
from acadia.sample_arithmetic import complex_to_sample, sample_to_complex
from acadia.firmware_configurations import CONFIG_200
from acadia_qmsmt.qmsmt import InputOutput

import cd_trains as cd

HERE = Path(__file__).resolve().parent
PROGRAM = ROOT.parents[1] / ("msmt_workbench/msmt_workbench/rfsoc/post_processing/Tunmay_analysis/quantum_simulation_paper/"
                             "11052026_qsim_paper_data/.work/model_controls/ideal_two_level/beta_program.json")


def dac_scales(swap):
    program = json.loads(PROGRAM.read_text())
    return program["idle_pi"]["scale"], program["pump"]["scale"] * 0.600 / swap     # qubit pi, pump at this swap


class OfflineAcadia:
    """Only SDK memory/clock operations, with ordinary RAM instead of a board."""
    _firmware=CONFIG_200
    def channel(self,name):
        return SimpleNamespace(is_dac=True,num=lambda:int(name[3:]),interface_sample_frequency=800e6)
    def sequencer_clock_frequency(self):
        return 200e6
    def create_waveform_memory(self,channel,length):
        count=round(length*800e6)
        assert count>0 and count%4==0
        return SimpleNamespace(array=np.zeros((count,2),np.int16))


def sdk_load(dac,channel,pulse):
    """Exercise unchanged compute_pulse AND load_pulse; compare every word."""
    dac=np.asarray(dac,complex)
    assert len(dac)%4==0 and len(dac)*4<=131072
    config=dict(channel=channel,pulses={pulse:dict(data=dac,scale=1.,phase=None,
                                                detune=None,use_stretch=False)})
    io=InputOutput(channel,OfflineAcadia(),config)
    io.load_pulse(pulse)
    words=io.get_waveform_memory(pulse).array.copy()
    assert words.shape==(len(dac),2) and np.array_equal(words,complex_to_sample(dac))
    return words,sample_to_complex(words)


def to_words_and_back(samples, full_scale_value, dac_scale, qubit=False):
    """Ideal samples (rad/us) -> DAC words (what the FPGA stores) -> the values the words actually represent."""
    # Model coefficients multiply sigma_+, while physical qubit IQ multiplies
    # sigma_- in the supplied programmed_model.py. The pump multiplies sigma_+a.
    dac = (np.conj(samples) if qubit else samples) / full_scale_value * dac_scale
    assert max(np.max(abs(dac.real)), np.max(abs(dac.imag))) < 1 - 2 ** -16      # no int16 overflow
    program=json.loads(PROGRAM.read_text())
    pulse=program['idle_pi' if qubit else 'pump']['name']
    words,restored=sdk_load(dac,'DAC7' if qubit else 'DAC10',pulse)
    model=restored.conj() if qubit else restored
    return words,model / dac_scale * full_scale_value


def rfdc_interpolate(samples):
    """The three half-band interpolation stages of the RFDC (x2 each), causal, full start-up and tail."""
    halves = [[5, 0, -17, 0, 44, 0, -96, 0, 187, 0, -335, 0, 565, 0, -906, 0, 1401, 0, -2112, 0, 3145, 0, -4723, 0, 7415,
               0, -13331, 0, 41526],
              [-12, 0, 84, 0, -337, 0, 1008, 0, -2693, 0, 10142],
              [-6, 0, 54, 0, -254, 0, 1230]]
    out = np.asarray(samples, complex)
    for half, center in zip(halves, [65536, 16384, 2048]):
        h = np.asarray(half + [center] + half[::-1], float) / center                # sums to 2 (zero insertion)
        out = upfirdn(h, out, up=2)
    return out


def propagate(qubit, sideband, initial, dt, order=12, fast=False):
    """Independent sample-by-sample propagation (C++), 12th-order Taylor per sample."""
    cpp = HERE / ("fast_waveform_replay.cpp" if fast else "native_waveform_check.cpp")
    library = Path("/tmp") / f"cd_waveform_{hashlib.sha256(cpp.read_bytes()).hexdigest()[:16]}.so"
    if not library.exists():
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(cpp), "-o", str(library)], check=True)
    function = getattr(ctypes.CDLL(str(library)),"fast_waveform_replay" if fast else "propagate_native_samples")
    function.argtypes = [ctypes.c_int] * 3 + [ctypes.c_double] * 2 + [ctypes.c_int] + [ctypes.c_void_p] * 4
    q, s = np.ascontiguousarray(qubit, complex), np.ascontiguousarray(sideband, complex)
    state = np.ascontiguousarray(initial, complex)
    assert q.shape==s.shape and q.ndim==1 and len(q)>0
    levels=len(state)//2
    bound=(np.max(abs(q))+np.max(abs(s))*np.sqrt(levels))*dt
    remainder=len(q)*math.exp(bound)*bound**(order+1)/math.factorial(order+1)
    assert remainder < 1e-10,(order,bound,remainder)
    out = np.empty_like(state)
    function(len(state) // 2, state.shape[1], len(q), dt, 0.0, order,               # 0.0: no static dispersive shift
             q.ctypes.data, s.ctypes.data, state.ctypes.data, out.ctypes.data)
    assert np.max(abs(np.sum(abs(out) ** 2, axis=0) - 1)) < 1e-9
    return out


def played_arrays(train,x,interpolate=True):
    """Physical int16 payloads and their Hamiltonian coefficients after RFDC."""
    _,q,s=cd.waveform(train,x)
    qs,ss=dac_scales(train.swap)
    qw,q=to_words_and_back(q,np.pi/cd.PI_TIME,qs,qubit=True)
    sw,s=to_words_and_back(s,train.g,ss)
    if interpolate:
        q,s=rfdc_interpolate(q),rfdc_interpolate(s)
    return q,s,dict(qubit=qw,sideband=sw)


def check(result, start_levels=128, step=64, tolerance=1e-9, interpolate=True,outputs=False):
    """result = an optimiser result dict (job, qubit_phases, pump_phases, qubit_frame_Rz). Returns the fidelity."""
    job = result["job"]
    strength = job.get("strength", {"squeeze": cd.R_8DB, "rotation": np.pi / 2}.get(job["gate"]))
    n_input = job.get("n_input", 9)
    train = cd.Train(job["gate"], strength, job["swap"], job["cells"], job["gap"], levels=16)   # waveform and timing
    x = np.r_[result["qubit_phases"], result["pump_phases"]]
    q_played,s_played,words=played_arrays(train,x,interpolate)
    dt=cd.DT/8 if interpolate else cd.DT
    sim_time = len(q_played) * dt
    history, previous = [], None
    for levels in range(start_levels, 1025, step):
        space = cd.Space(levels, n_input)
        target = cd.target_columns(space, job["gate"], train.strength)
        initial = np.eye(2 * levels, dtype=complex)[:, space.inputs]
        out = propagate(q_played, s_played, initial, dt,order=8,fast=True)
        frames = np.exp(-1j * result["qubit_frame_Rz"] * space.z / 2)
        out = frames[:, None] * out
        overlaps = np.sum(np.conj(target) * out, axis=0)
        fidelity = float(abs(overlaps.sum() / len(overlaps)) ** 2)
        item=dict(levels=levels,fidelity=fidelity,columns=2*n_input)
        if previous is not None:
            old_out,old_target=previous
            padded=np.zeros_like(out);padded[:len(old_out)]=old_out
            target_padded=np.zeros_like(target);target_padded[:len(old_target)]=old_target
            item['output_column_change']=float(np.max(np.linalg.norm(out-padded,axis=0)))
            item['target_column_change']=float(np.max(np.linalg.norm(target-target_padded,axis=0)))
        history.append(item)
        if previous is not None and max(item['output_column_change'],item['target_column_change']) < tolerance:
            report=dict(fidelity=fidelity, worst_input_fidelity=float(np.min(abs(overlaps) ** 2)), levels=levels,
                        cutoff_history=history, command_time_us=len(words['qubit']) * cd.DT, simulated_time_us=sim_time,
                        columns=2*n_input,complex_amplitudes=out.size,RFDC_samples=len(q_played),
                        words=words)
            report['command_time_us']=len(words['qubit'])*cd.DT
            return (report,out,target) if outputs else report
        previous = out,target
    raise RuntimeError("cutoff did not converge")
