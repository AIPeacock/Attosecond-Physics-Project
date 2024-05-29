from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import numpy as np
wolfSession = WolframLanguageSession()   #Begin mathematica session

direct = wolfSession.evaluate(                    #Option to change directory to where the nb is stored
    wl.Directory(),
    )


wolfSession.evaluate(
   wl.needs('basicPackage`')
)


res = wolfSession.evaluate(
        ((wl.basicPackage.AddTwo(12,2)))
    )
print(res)

wolfSession.terminate()











#wolfSession.evaluate(                    #Load the SFA notebook
#    wl.get('RB-SFA.m`')
#)


#moons = wolfSession.evaluate(wl.WolframAlpha("number of moons of Saturn", "Result"))
#print(moons)

#Fi = 0.05
#wi = 0.057
#t=wl.Array(110,{0,110})
#dipole = wolfSession.evaluate(
#    wl.RBSFA.makeDipolelist((wl.RBSFA.VectorPotential==wl.Function(t,{(wl.Sin(wi*t)),0,0}),
#	wl.RBSFA.FieldParameters=={wl.RBSFA.F==Fi,wl.RBSFA.w==wi},
#    wl.RBSFA.CarrierFrequency==0.057,wl.RBSFA.Target=="Xenon"))

#)
