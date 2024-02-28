import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
from numpy.fft import fft,ifftshift,fftshift,fftfreq

amplitude = 1
sampling = 0.01                       #Sampling rate or step size
T =10                                 # period of function


FWHM = T*np.arccos(amplitude/2)/np.pi # define a FWHM to determine frequency of pulse                                      # 
f0=np.arccos(amplitude/2)/(FWHM*np.pi)
f0 = 0.1

f=f0*10
                                        #Time array defined in steps of "sampling"
N = 10
t=np.arange(0,N*(2*np.pi)/f0,sampling)
Carrier = np.sin(2*np.pi*f0*(t-N/f0))              
Field=(Carrier*((np.sin(np.pi*f0*t/N))**2))      #Define function to Fourier Transform
#print(y)

plt.figure(1)
#plt.xlim(-2,2)
plt.plot(t*f0,Field)                         #Plot (r,t) of Function y
#plt.plot(fwhmx,halfmax)
#plt.plot(-1.6666,0.5,'o')
#plt.plot(1.6666,0.5,'o')
#Fourier Transform
#print(t.shape[-1])
yft = fftshift(fft(ifftshift(Field)))                           
freq = fftshift(fftfreq(t.shape[-1],d = sampling ))         #Returns frequency bins using using window length(n) and 
                                                            #sample spacing (d).

spectrum = np.abs(yft)**2

plt.figure(2)
plt.xlim(-1,1)
plt.plot(freq,spectrum)