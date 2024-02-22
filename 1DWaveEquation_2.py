import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fft import fft, ifft, fftshift
from scipy.integrate import odeint


def wave(y, x, omega, c):              #solving fourier transformed SODE 
    u, A = y 
    dydt = [A, -u*(omega**2/c**2)]
    return dydt

omega = 1
c = 1

y0 = [0, 200]     #Intial conditions, initally at x= 0 with some velocity

x = np.linspace(0, 10, 101)

Sol = odeint(wave, y0, x, args=(omega,c))


uinv = ifft(Sol[:,0]) 
L = fftshift(uinv)
#print(uinv)
plt.plot(x,Sol[:,0], label = 'Amplitude for frequency, w')
plt.plot(x,L, label ='frequency')
plt.legend(loc = 'best')
plt.xlabel('x')
plt.ylabel('Amplitude')


