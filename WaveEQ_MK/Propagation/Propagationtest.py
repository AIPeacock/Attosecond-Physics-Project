import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifftshift, fftfreq, ifft
import time
from scipy import integrate
import importlib
import Lewenstein
import importlib
importlib.reload(Lewenstein)

def pulse_Field(t, F, w, Nc, duration):
    return (np.sin(w * (t - (Nc * 2 * np.pi / w) / 2)) * ((np.sin(np.pi * t / (Nc * 2 * np.pi / w))) ** 2)) * F

# Define constants
w = 0.057
dt = 0.1
Nc = 4
duration = 0#6 * np.pi / w  # Duration of the envelope window
t0 = 0
tf = duration + (Nc * (2 * np.pi / w)) + duration  # Total time including ramp-up and transition
N = int((tf - t0) / dt)
c = 299792458 / 2.187e6 

# Calculate field parameters
I0 = (2.5* (10 ** 14)) / (3.51 * (10 ** 16))
E0 = np.sqrt(I0)
t_full = np.linspace(t0, tf, N)
freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
Field_list = pulse_Field(t_full,E0,w,Nc, duration)
Ip = 15.7 / 27.2

#Is this correct way to propagate the Field's? 
Nmax= 200
dx = 0.01
N_at = 4.4455413e-7   #2.025e42
#dn = (N_at*dx*1j*2*np.pi*w/c) #???
dn = 0.002

#Initalise arrays to store fields in
E_t= np.zeros((len(t_full),Nmax), dtype = complex)
E_w = np.zeros((len(t_full),Nmax), dtype = complex)

#Initalise arrays to store dipole moments in 
Dipole_temp= np.zeros((len(t_full),Nmax), dtype = complex)
d_t = np.zeros((len(t_full),Nmax), dtype = complex)
d_w = np.zeros((len(t_full),Nmax), dtype = complex)

#Store driving field in first coloumn of array 
E_t[:,0]= pulse_Field(t_full,E0,w,Nc,duration)

for n in range(1,Nmax):

    Output = Lewenstein.Lewenstein(w=0.057,I0=(2.5* (10 ** 14)) / (3.51 * (10 ** 16)),Ip=Ip,dt=dt,N=int((tf - t0) / dt),Nc=Nc,t_full=t_full,Field_list=E_t[:,n-1]) # Calculation of single atom response

    Dipole_temp[:,n-1] = Output[0]  # Storing dipole moment in time, Should this be stored in n or n-1, which step is this the response of? 

    d_t[:,n-1]= Dipole_temp[:,n-1] + E_t[:,n-1] # Adding linear response by adding field from previous step to dipole moment 

    d_w[:,n-1] = fftshift(fft(ifftshift(d_t[:,n-1]))) # Fourier transform

    E_w[:,n-1] =Output[1] # Store fourier transfrom of field used to generate response 

    E_w[:,n] = E_w[:,n-1] -  1j*dn*d_w[:,n-1] # Propagation calculation

    E_t[:,n] = fftshift(ifft(ifftshift(E_w[:,n])))  # IFT back to time domain 
    print('Step number',n)

plt.figure(0)
plt.plot(t_full,d_t[:,Nmax-1])
plt.title('Time domain response(d_t +E_t)')
plt.figure(1)
plt.semilogy(freq_axis/w,np.abs(d_w[:,Nmax-1]**2))
plt.xlim(-20,70)
plt.title('Spectrum of response FFT(d_w)')
plt.figure(3)
plt.plot(t_full,E_t[:,Nmax-1])
plt.title('Electric Field at last Slice')
plt.figure(4)
plt.semilogy(freq_axis/w,np.abs(E_w[:,Nmax-1])**2)
plt.xlim(-20,70)
plt.title('Spectrum of electric field at last slice')

# NL_pol = N_at * Output[0]    
# Field_dx = Output[1] - (dx*(1j*(2*np.pi*w/c)*NL_pol))
# Field_list = fftshift(ifft(ifftshift(Field_dx)))

