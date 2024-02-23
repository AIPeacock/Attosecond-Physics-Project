import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
from numpy.fft import fft,ifftshift,fftshift,fftfreq

amplitude = 1
sampling = 0.01                       #Sampling rate or step size
T =10                                 # period of function
FWHM = T*np.arccos(amplitude/2)/np.pi # define a FWHM to determine frequency of pulse                                      # 
f0=np.arccos(amplitude/2)/(FWHM*np.pi)
f=f0*10
t=np.arange(-10,10+sampling,sampling)   #Time array defined in steps of "sampling"
print(t)        
ft = np.cos(2*np.pi*f0*t)
y=(ft*((np.sin(2*np.pi*f*t))**2))      #Define function to Fourier Transform
print(y)

Fwhm = np.arccos(amplitude/2)/(np.pi*f0)
print(Fwhm)
plt.figure(1)
#plt.xlim(-2,2)
plt.plot(t,y)                         #Plot (r,t) of Function y

#Fourier Transform
print(t.shape[-1])
yft = fftshift(fft(ifftshift(y)))                           
freq = fftshift(fftfreq(t.shape[-1],d = sampling ))         #Returns frequency bins using using window length(n) and 
                                                            #sample spacing (d).

spectrum = np.abs(yft)**2

plt.figure(2)
plt.xlim(-4,4)
plt.plot(freq,spectrum)