from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

# Start the Wolfram Language session
with WolframLanguageSession() as wolfSession:
    direct = wolfSession.evaluate(
    wl.Directory()
    )
    print("Current Directory:",direct)
    # Load the package
    wolfSession.evaluate(wl.Get("basicPackage`"))
    
    add_func =  wl.Function(wl.basicPackage.AddTwo)
    # Call the AddTwo function
    result = (add_func(12, 2))

print("Result:",result)
    
