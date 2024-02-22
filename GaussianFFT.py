import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
from numpy.fft import fft,ifftshift,fftshift,fftfreq
s=0.1
t=np.arange(-10,10+s,s)
c = 0.5
x =0
A=1
y0=A                       #1/(sigma*np.sqrt(2*np.pi))
y0=y0*np.exp(-0.5*(((t-x)/c)**2))

plt.figure(1)
plt.plot(t,y0)

yft= fftshift(fft(ifftshift(y0)))
freq = fftshift(fftfreq(t.shape[-1],d = s ))

plt.figure(2)
plt.plot(freq,yft)