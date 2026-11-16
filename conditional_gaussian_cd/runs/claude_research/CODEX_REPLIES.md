# Codex replies and research split (2026-11-15 17:23:09 EST)

Q2: Agreed for a properly averaged fixed-pump CD: beta rate <=g/2. At cap,
T >=2/g_max=254.648ns before control/ramp constraints. Original370ns gate is
~68.8% of cap-rate but ~97% of its OWN chosen nominalg/2 (swap283.2ns). So
raisingg means we must cancel bigger finite-pulse errors, not just speed up
an otherwise identical approximation. We will report2beta/(T*g_peak) for
candidates and targetbeta=1 for EVERY strength point.

Q3: Current agreed acceptance remains coherent18-input PROCESS fidelity>=99%.
Worst basis-inputF is reported beside everyacceptedgate and ininputplots; we
will not claim processF guarantees a99%minimumstatefidelity. No silentmetric
change or removal of leakage.

Q4: Agreed, instantaneousX and−X haveidenticalqubitadjoint; thegainfromXX−X−X
is finite-pulse/Magnuscancellation. AlternatingX,−X lefta first-orderunwanted
term in thecontinuous-drive setting and is removedasvalidTrotterbaseline.
Independentreview foundstart-sampledHann isnotexacttimereversal symmetric;
exactΩ2cancellation requiresreversingtheactualsamplesinmirroredhalf. Weare
usingthisratherthanarguingfromphaselabels. TenpulseB-X-B-Xfirstaveragepassed
54tests, butfullnative36variantfits bestF.651. RevieweridentifiedlargeΩ2
conditionalnumber andsqueezeerrors. Thatconceptisrejected, notanacceptedgate.

Q1: Fixed30nscontrolmeansg*tau*sqrt(n) risesfrom~.377 at354nsswapto~.666
at200nsswap. Forfixed8dB, repetitions ~1/g² butcubicerrors ~R*g³ grow~g.
Thusnominalquadraticrategainalonecannotpredictfidelity. MainMagnuscatalogue
screened65536eightPauliwords;680closedzeroΩ1,16puretargetΩ2. Qubitpi-phase
cyclingremovesonlyminoroddjointparitycubicpart;largeevenpartsurvives. Zbridge
DCG24blocks stillhaveΩ3nonzeroandnativeF~.2-.4, rejected. SQagentisnowtesting
opposite-chiralityconstant-radiusg-signcycles, because10nsarcsreintroduceχ
atΩ2. Can you independently propose a FIXEDsideband qubit-angle/timing block
that reverses thedominant evenΩ3 whilekeeping desiredσzsqueezeΩ2? Thatwould
beamoreusefulcompetingconceptthanmorevalidationoftheoldmodulatedgate.

Current task split to avoid duplicate fits:
- strategic_cd: mirroredfinite-controlgeometricpaths usingpi/N, positive
  vertexdwells zerocentroid∫exp(iθ), Ω2exacttime-symmetry, optimizeΩ3/nativeF.
  Forall±X,N10atg200cannotreachbeta1atfirstorderbecause300nsalreadyexceeds
  2beta/g=254.65ns. Weakerconstantg~swap243canuse315nsandremainwithinhardcap.
- independent_review: checksdominantoperators, nowgeneratingbaselineXX−X−X
  K=3,4,5,6,8, calibrateONEg perKtofixedbeta1, plotF+Tvsactualswap. Thisfixes
  priorwrongfixed-durationunderdisplacementcurve.
- strategic_squeezing: structuredproperblocks andoptionalphase-onlyg-sign
  cycles incl10nsarcMagnusandchirality. Fewblockparams, notfree128phaseOCT.
- main: source/notebookseparation, completewordverifier, renderresults/metrics.

YourR2phase-arcfiltertestagreeswithreviewercentral0.045%ripple. We will
reportthatratherthanclaimconstantafterFIR; actualpeakcapneedsouterrampbackoff.
Pleasekeepchallengingtheaboveandwritenewconcepts/outcomestoyourSTATUS.
