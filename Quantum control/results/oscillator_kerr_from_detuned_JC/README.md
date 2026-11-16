# Oscillator Kerr: frequency, fidelity, pulse sequence

[Executed notebook](../../oscillator_kerr_from_detuned_JC.ipynb) ·
[Paper, Appendix B](https://arxiv.org/pdf/2611.05460) ·
[Counted validation](execution.log)

$$
\boxed{\frac{H_K}{h}=\frac{k}{2}n(n-1),\qquad k\equiv\frac{K}{2\pi}}
$$

Report $k$ in **kHz**. Fitted rates include all ramp time; plateau spectral
curvature is shown separately. At fixed duration $T$, equivalent fitted rates
differ by $1/T$; the displayed fit uses the representative closest to zero.

Model: 250 ns complete swap, $g/(2\pi)=1\,\mathrm{MHz}$,
$\alpha/(2\pi)=-150\,\mathrm{MHz}$.
Inputs $n=0,\ldots,8$; cavity dimension 10; transmon dimension 5; no decoherence.

```text
Hann up → flat → Hann down → πx → Hann up → flat → Hann down → πx
 0.5 us  12.003 us  0.5 us         0.5 us  12.003 us  0.5 us
```

Both $\pi$ pulses have coupling off. The frame change uses the **bare** qubit:

$$
R(T)=\exp\!\left(+i2\pi\delta n_qT\right),\qquad n_q=q^\dagger q.
$$

Pulse axes are transformed consistently with $R(t)$.

**Required operation:** preserve an arbitrary unknown qubit superposition. The qualifying result is the joint g/e echo gate; ground-only curves are comparisons.

**140 MHz detuning, whole-subspace fit:**

$$
\boxed{\frac{K_{\mathrm{eff}}}{2\pi}=-2.426\,\mathrm{kHz},
\qquad F_{\mathrm{avg}}=99.475\%}
$$

Total time $T=26.006\,\mu\mathrm{s}$. Using the $n=0,1,2$ curvature instead gives
$K_{\mathrm{eff}}/(2\pi)=-3.191\,\mathrm{kHz}$ and $F_{\mathrm{avg}}=55.64\%$.
The plateau low-photon Kerr is $K_{\mathrm{echo}}(0)/(2\pi)=-3.379\,\mathrm{kHz}$.

- [Played sequence](sequence_diagram.png)
- [Kerr frequency and fidelity together](Kerr_frequency_and_fidelity.png)
- [Paper-style spectrum and signed echo](paper_comparison_and_echo.png)
- [Full-H Kerr versus photon number and return error](full_time_domain_phase_and_return.png)
- [Ramp duration: error, frequency, fidelity](ramp_duration_study.png)
- [Resonances: spectrum, fitted frequency, fidelity](near_resonance_spectra_and_fidelity.png)
- [Longer ramps near resonances](near_resonance_ramp_effect.png)
- [Full time traces](full_time_domain_traces.png)

Validation covers 1,163 sweep detunings, 32 ramp cases, 93 near-resonance pulse
cases, and both g/e inputs. Independent rotating-frame propagation compared 30
complex elements with maximum difference $2.10\times10^{-8}$. Timestep checks
compared 900 main-protocol and 3,600 near-pole/short-ramp elements. CSV files retain
numerical values with explicit kHz columns; the notebook presents plots instead
of tables.

**Detuning reference:** all sweep axes use the g/e exchange detuning, not the e/f detuning:

$$
\delta_{ge}=\Delta,\qquad \delta_{ef}=\Delta-K,\qquad K=-\alpha>0.
$$

Thus $\Delta/(2\pi)=140\,\mathrm{MHz}$ means $\delta_{ef}/(2\pi)=-10\,\mathrm{MHz}$.

## Conclusions supported by the simulation data

**Required gate:** $U_{\mathrm{target}}=U_K\otimes I_{ge}$, preserving an arbitrary qubit state.

**Main echo: g/e detuning = 140 MHz; e/f detuning = -10 MHz**, 0.5 us Hann edges, 12.003 us per flat top:

$$
\boxed{\frac{K_{\mathrm{eff}}}{2\pi}=-2.426\,\mathrm{kHz},\qquad F_{\mathrm{avg}}=99.475\%,\qquad T=26.006\,\mu\mathrm{s}}
$$

- **The 55.64% result is a phase/target-strength mismatch.** Using only n=0,1,2 sets the target to -3.191 kHz; fitting all n=0..8 gives the result above. Worst product-state return error is only 9.85e-06. [Same-pulse data](main_Kerr_gate_fidelity.csv).
- **Near the g/e exchange resonance, echo cancels most Kerr:** 46.0 Hz at 10 MHz and 2.57 Hz at 20 MHz, with 0.5 us edges. These are g/e detunings, far from e/f exchange. [Data](ramp_duration_study.csv).
- **Near e/f exchange, chi prime does dominate the echo.** At g/e detuning 140 MHz, e/f detuning is -10 MHz: $K_g/(2\pi)=$ -0.841 Hz, $\chi'_e/(2\pi)=$ -6.755 kHz, giving plateau echoed Kerr -3.379 kHz (curvature from n=0,1,2). [Data](exact_rate_table.csv).
- **Moving still closer to e/f exchange reduces Kerr fidelity for these pulses:** at 145 MHz the fit gives -7.23 kHz but only 67.23% fidelity. At exact 75/150/225 MHz resonances, fidelities are 43.3/37.0/39.5%. [Same-duration resonance data](near_resonance_time_domain.csv).
- **Slower ramps do not by themselves produce a purer Kerr.** At 140 MHz, changing each edge from 0.5 to 1 us reduces return error from 9.85e-06 to 1.01e-06; joint Kerr fidelity changes from 99.475% to 99.454%. [Ramp data](ramp_duration_study.csv).

**Scope:** joint average fidelity over n=0..8 and both g/e, with dimensions 10 x 5, ideal pi pulses and no decoherence. These data demonstrate the stated finite pulse; they do not establish long-time Kerr fidelity, a worst-case fidelity bound, or an optimal operating point.
