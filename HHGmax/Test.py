import pylewenstein
import numpy as np

# set parameters (everything is SI units)
wavelength = 800e-9 # 800 nm
fwhm = 30e-15 # 30 fs
ionization_potential = 15.7*pylewenstein.e # 15.7 eV (Ar)
peakintensity = 1e14 * 1e4 # 1e14 W/cm^2

# define time axis
T = wavelength/pylewenstein.c # one period of carrier 
t = np.linspace(-20.*T,20.*T,200*40+1)

# define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)

# use module to calculate dipole response (returns dipole moment in SI units)
d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)

# plot result
import pylab
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t,np.real(d))
plt.title('time domain')

plt.figure(2)
plt.plot(t,Et)
plt.title('E field')
plt.figure(figsize=(5,4))
q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
plt.semilogy(q[q>=0], abs(np.fft.fft(d)[q>=0])**2)     #Plt.plot w/ log scaling
plt.xlim((0,50)) 
plt.show()