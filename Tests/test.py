import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy.fft import fft,ifftshift,fftshift,fftfreq
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft

def Waveeqft(A,z,w,c):           # A = (B_1, B_2)
    return(A[1],-w**2/c**2 * A[0])             # Return (B_1',B_2')
c=1
L = 10
N=100
dz = L/N
Z = np.arange(-L/2,L/2,dz)

sampling = 1                                              #defining sampling rate
t = np.arange(-10,10 + sampling,sampling)           #defining time axis
w = 2*np.pi*fftshift(fftfreq(t.shape[0],d = sampling))
print(w)
Init = 1/np.cosh(Z)
for w in 2*np.pi*fftshift(fftfreq(t.shape[0],d = sampling)):
    Asolft = odeint(Waveeqft,y0 =[Init[0],0],t=Z, args=(w,c))

print(Asolft)
Asol= fftshift(fft(ifftshift(Asolft)))
print(Asol)