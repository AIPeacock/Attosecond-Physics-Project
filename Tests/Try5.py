import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy.fft import fft,ifftshift,fftshift,fftfreq
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm 

c = 1 
L = 40           #Length of propagation (Width)
N = 1000         #Discritisation points 
dz = L/N         #Spatial size
z = np.arange(-10,10,dz)

def waveEQ(u,w,c,z):                                      #Wave Equation Fourier Transformed 
    dadz = (-w**2/c**2) * u
    return(dadz)

Init = 1/np.cosh(z)                                       #Inital value (y0)V

sampling = 1                                              #defining sampling rate
for t in np.arange(-10,10 + sampling,sampling):           #defining time axis
    
    w = 2*np.pi*fftshift(fftfreq(t.shape[0],d = sampling))   #defining freq axis
    
    AsolFT = odeint(waveEQ, Init,z, args=(w,c))
    
    Asol= fftshift(fft(ifftshift(Asol)))
