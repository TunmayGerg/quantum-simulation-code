# Experimental simulation code moved

Z2 Higgs, controlled-SWAP, gate-synthesis and photon-counting simulations now
live in [Tunmay's Z2 Higgs analysis package](../../msmt_workbench/msmt_workbench/rfsoc/post_processing/Tunmay_analysis/z2_higgs/README.md).
This includes the corresponding notebooks, numerical modules, optimized pulse
files, notes and saved figures.

The continuous-sideband conditional-displacement comparison and its saved table
now live in [Tunmay's CD analysis package](../../msmt_workbench/msmt_workbench/rfsoc/post_processing/Tunmay_analysis/conditional_displacement/README.md).

The general displaced-frame and transmon Stark studies remain here. The
`GKP/` collection remains in this repository, unchanged by the consolidation.

The executed [oscillator Kerr notebook](oscillator_kerr_from_detuned_JC.ipynb)
compares ground-state and echoed Kerr from a detuned Jaynes–Cummings interaction.
It includes a timed pulse diagram, full Hamiltonian propagation with Hann ramps,
the bare-qubit frame transformation, Fock-subspace gate fidelity, ramp sweeps,
and studies of the apparent Kerr and chi-prime poles. See the
[saved results and validation](results/oscillator_kerr_from_detuned_JC/README.md).
