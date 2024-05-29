import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
from numpy.fft import fft,ifftshift,fftshift,fftfreq

amplitude = 10
sampling = 0.01                       #Sampling rate or step size
f0=1
f=f0
t=np.arange(-10,10+sampling,sampling)   #Time array defined in steps of "sampling"
print(t)        
y=amplitude*(np.sin(2*np.pi*f*t))**2              #Define function to Fourier Transform
print(y)

plt.figure(1)
plt.plot(t,y)                         #Plot (r,t) of Function y

#Fourier Transform
print(t.shape[-1])
yft = fftshift(fft(ifftshift(y)))                           
freq = fftshift(fftfreq(t.shape[-1],d = sampling ))         #Returns frequency bins using using window length(n) and 
                                                            #sample spacing (d).

spectrum = np.abs(yft)**2

plt.figure(2)
plt.xlim(-5,5)
plt.plot(freq,spectrum)