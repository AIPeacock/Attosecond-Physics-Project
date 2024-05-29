import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
from numpy.fft import fft,ifftshift,fftshift,fftfreq
sampling = 0.01
t = np.arange(-10,10 + sampling,sampling)
3
w  = 1
# x = np.sin(w*t)
x = np.sin(2*np.pi*w*t)#np.exp(1j*2*np.pi*w*t)                 #Sin function of wt
print(x)        
#x[t<-2] = 0
#x[t>2] = 0


#Fourier transform 
y = fftshift(fft(ifftshift(x)))
#generate frequency axis
freq = fftshift(fftfreq(t.shape[-1],d = sampling ))

#find spec and phase
spectrum = np.abs(y)**2
phase = np.log(y).imag

print(y)

plt.figure(1)
#plot wave
plt.plot(t,x)
plt.xlim(-5,5)

plt.figure(2)
#plot spectrum
plt.plot(freq,np.abs(y)**2)
plt.xlim(-5,5)

plt.figure(3)
#plot phase
plt.plot(freq,phase)
plt.xlim(-5,5)

plt.show()