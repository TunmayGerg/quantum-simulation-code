# Codex -> Claude research handoff (2026-11-15 17:14:36 EST)

I found code/claude_replay.py and runs/claude_replay_squeeze.log. Your fully
independent SDK/FIR/sparse-expm replay agrees with our historical squeeze_8db
F to3e-15 at256/288levels. Thank you; this is useful independent evidence.
I will not edit your file or interrupt your processes.

IMPORTANT latestuserconstraints: plateau sideband BOTH magnitude and phase
fixed preferred. Previous300nsCD and3.61usSQ are modulated and now historical,
notrecommendations. Minswap200ns stillhard. Qubitpi/N envelopes30ns, Ninteger,
phase/angle/timingcanvary. Mainpriority strategicMagnuscancellation notrawOCT.
Optionalpump-phase musthave explicitMagnusbenefit, constant-radiusIQarc10ns
centeredqubitamplitudezero; straightCartesianIQblendchangesmagnitude and is
notallowed. Needactualfilterenveloperipplemeasuredifusingarc.

UserrequestedallPythonin code/, notebooksin notebooks/. Mainmovedexisting
filesandpatchedpaths;projectrootisproject_paths.ROOT. PYTHONPATH=code
../../acadia_env/bin/python fromprojectroot. Historicalnotebooksarchive/.

Liveboard: WORK_STATUS.md. Pleasewrite runs/claude_research/STATUS.md and
challengeourhypotheses, ratherthanduplicatingrunningfits. Codexsubagents own:
- strategic_cd: code/strategic_cd.py,runs/strategic_cd/
- strategic_squeezing: code/strategic_squeezing.py,runs/strategic_squeezing/
- independent_review: code/review_hypotheses.py,runs/independent_review/
MainownsnewMagnuscatalogue/layout/notebooks. Existingindependent codeis yours.

Bestcurrentfullyconstantreferences:CDbeta1,370ns,swap283.2079,F.99227794;
SQ8dB,160pi4.81us,swap353.9474,F.99018309. Targetall18coherentinputs,n0..8.

Reviewtargets:
1)CD B4inversequbitloop B=[R_phi(alpha),R_psi(beta),R_(psi+pi)(beta),
R_(phi+pi)(alpha)] closesI. B,Xpi,B,Xpi XtwirlsfirstJCaverageonto complexsigma_x.
Outer5nsrampsgive5nseffectivebareJC;add5nsX-frameJCidletofirst-ordercompensate.
Oneconstantpumpphaseorientssigma_xdesiredcavityquadrature. Withphi0,beta pi,
approxMx=.095+.06cos²psi -i*.06cospsi sinpsi us,so315nsblockcanreachbeta1
atswap200ifpsi~pi/4. Outeralpha=pi/Nchangeshigherterms. VerifyexactMagnus,
bridgecontributionsandfullgatebeforebelievingit. Timinggrowforweakerg, fixedbeta.
2)SQcode/magnus_blocks.py exactpulseCauchyFFT+BCH:g0throughg4,65536eightpulse
+/-X,+/-Ywords ->680closedzeroO1 ->16puredesiredO2squeezewords. c2.00174495.
Quibitpi-phasecyclingcancelscubicoddjointparity .000122butleaves .000607.
DoNOTclaimcubiccancellation. New24pulseEulerianZbridgecatalogue1024propercases
inruns/fixed_sideband/dcg_block_catalogue.json:bestc2.00348990,cubic.00068498
nonzero. Squeezingagenttestsnativefiniteg/R/scalarstrength;youcouldderivea
betterbridgeorprovewhichcubicoperatorremainsandhowtoreverseitwithoutpumpmods.
3)CheckCauchypulseseries/Magnussigns/lowphotonprojectionboundaries. Rawdata
runs/fixed_sideband/proper_block_terms.npz,pulse_series.npz. Needfalsification
andcomparisoncounts,notonlyagreement. Full18statefidelityindependentofgeneratorfit.

Pleasekeepalltargetstrengthcomparisonsfixedbetaor8dB,notfixed-durationweaker
achievedgate. Mainplotsmustshowbestverifieddurationandfidelityvsstrength.
