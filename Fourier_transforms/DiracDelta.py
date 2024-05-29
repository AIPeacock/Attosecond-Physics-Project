import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
from numpy.fft import fft,ifftshift,fftshift,fftfreq
from scipy import signal

def delta(n,w):
    if n == w:
        return 1
    else:
        return 0
    

w=5
sampling = 0.05
print(w)

t=np.arange(-10,10+sampling,sampling)
print(t.shape[-1])

Dirac = signal.unit_impulse(t.shape[-1],w)

y = np.sqrt(2 *np.pi)*((np.sin(w*t))**2)

Dirac_2=[]
for i in np.arange(-10,10+sampling,sampling):
    d_2=(delta(i,w))
    print(i)
    Dirac_2.append(d_2)
print(Dirac_2)


plt.figure(1)
plt.plot(t,y)

plt.figure(2)
plt.plot(t,Dirac)
plt.title('First Dirac attempt')

plt.figure(3)
plt.plot(t,Dirac_2)
plt.title('Second Dirac attempt,uses definition of w')

plt.figure(4)
plt.plot(t,y*Dirac_2)
plt.title('Second Dirac attempt times y(t)')
