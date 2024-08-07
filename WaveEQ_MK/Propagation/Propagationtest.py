import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifftshift, fftfreq, ifft
import time
from scipy import integrate
import importlib
import Lewenstein
import importlib
import pandas as pd
importlib.reload(Lewenstein)

def pulse_Field(t, F, w,Nc, duration):
    return (np.sin(w * (t - (Nc * 2 * np.pi / w) / 2)) * ((np.sin(np.pi * t / (Nc * 2 * np.pi / w))) ** 2)) * F 

def pulse_Field_HFM(t, F, w, F2, w2, Nc, duration):
    return (np.sin(w * (t - (Nc * 2 * np.pi / w) / 2)) * ((np.sin(np.pi * t / (Nc * 2 * np.pi / w))) ** 2)) * F - ((np.sin(w2 * (t - (Nc * 2 * np.pi / w2) / 2)) * ((np.sin(np.pi * t / (Nc * 2 * np.pi / w2))) ** 2)) * F2)

# Define constants
w = 0.057
dt = 0.1
Nc = 6
duration = 0#6 * np.pi / w  # Duration of the envelope window
t0 = 0
tf = duration + (Nc * (2 * np.pi / w)) + duration  # Total time including ramp-up and transition
N = int((tf - t0) / dt)
c = 299792458 / 2.187e6 

# Calculate field parameters
I0 = (2.5* (10 ** 14)) / (3.51 * (10 ** 16))
E0 = np.sqrt(I0)

I1 = I0/100
E1 = np.sqrt(I1)
w1= w/6

t_full = np.linspace(t0, tf, N)
freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
Ip = 15.7 / 27.2

#Is this correct way to propagate the Field's? 
Nmax= 500
#dx = 0.01
#N_at = 4.4455413e-7   #2.025e42
#dn = (N_at*dx*1j*2*np.pi*w/c) #???
dn = 0.001

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

plt.figure(figsize=(12,12),dpi=120.0)
plt.imshow(np.abs(E_w[(np.arange(3306,4000,1)),:])**2,norm="log",origin='lower')
plt.colorbar()

plt.figure(figsize=(15,15),dpi=120.0)
plt.imshow(np.abs(E_w[(np.arange(3306,3806,1)),:])**2,norm="log",origin='lower',extent=[0,Nmax,0,(np.abs(3306-3806))/6,])
plt.xlabel('Slice num (n)')
plt.ylabel('Harmonic Order')
plt.colorbar(shrink = 0.35)

slice_length = range(0,Nmax,1)
plt.plot(slice_length,np.abs(E_w[(3306+3*6),:])**2)

plt.figure(figsize=(15,15),dpi=120.0)
plt.imshow(np.abs(E_w[(np.arange(3306+1*6,3306+6*6,1)),:])**2,norm="log",origin='lower',extent=[0,Nmax,1,6],aspect=10)
plt.xlabel('Slice num (n)')
plt.ylabel('Harmonic Order')
plt.colorbar(shrink = 0.35)

# plt.figure(figsize = (12,12))
# slice_length = np.arange(0,Nmax,1)
# plt.semilogy(slice_length,np.abs(E_w[(3306+17*6),:])**2,label = 'Harmonic: 17')
# plt.semilogy(slice_length,np.abs(E_w[(3306+15*6),:])**2,label = 'Harmonic: 15')
# plt.semilogy(slice_length,np.abs(E_w[(3306+13*6),:])**2,label = 'Harmonic: 13')
# plt.semilogy(slice_length,np.abs(E_w[(3306+11*6),:])**2,label = 'Harmonic: 11')
# plt.semilogy(slice_length,np.abs(E_w[(3306+9*6),:])**2,label = 'Harmonic: 9')
# plt.semilogy(slice_length,np.abs(E_w[(3306+7*6),:])**2,label = 'Harmonic: 7')
# plt.semilogy(slice_length,np.abs(E_w[(3306+5*6),:])**2,label = 'Harmonic: 5')
# plt.semilogy(slice_length,np.abs(E_w[(3306+3*6),:])**2,label = 'Harmonic: 3')
# plt.semilogy(slice_length,np.abs(E_w[(3306+1*6),:])**2,label = 'Harmonic: 1')
# plt.semilogy(slice_length,1e-5*slice_length**2,label = 'sqaure')
# plt.xscale('log')
# plt.ylabel('Intensity')
# plt.xlabel('slice')
# plt.legend()

# import numpy as np
# import pandas as pd

# df = pd.DataFrame(E_w)
# df.to_csv('E_w_N100_dn001',index=False,header=False)

# ddf = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N100_dn001',header = None)
# df  = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N10_dn01',header = None)

# ddf = ddf.to_numpy(dtype ='complex')
# df = df.to_numpy(dtype ='complex')

# ddf_slice_length = np.arange(0,100,1)
# df_slice_length = np.arange(0,10,1)


# plt.semilogy(ddf_slice_length,np.abs(ddf[(3306+7*6),:])**2,label = 'Harmonic: 17, dn = 0.001')
# plt.semilogy(df_slice_length,np.abs(df[(3306+7*6),:])**2,label = 'Harmonic: 17, dn = 0.01 ')
# plt.semilogy(slice_length,1e-5*slice_length**2,label = 'sqaure')
# plt.xscale('log')
# plt.ylabel('Intensity')
# plt.xlabel('slice')
# plt.legend()

# ddf = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N100_dn001',header = None)
# df  = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N10_dn01',header = None)

# ddf = ddf.to_numpy(dtype ='complex')
# df = df.to_numpy(dtype ='complex')

# ddf_slice_length = np.arange(0,0.1,0.001)
# df_slice_length = np.arange(0,0.1,0.01)

# ddf = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N100_dn001',header = None)
# df  = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N10_dn01',header = None)
# dddf  = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N20_dn005',header = None)
# ddddf  = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N200_dn0005',header = None)
# N500_dn001  = pd.read_csv(r'C:\Users\Aaron\Kings Code\Attosecond Physics Project\WaveEQ_MK\Propagation\E_w_N500_dn001',header = None)

# ddf = ddf.to_numpy(dtype ='complex')
# df = df.to_numpy(dtype ='complex')
# dddf = dddf.to_numpy(dtype ='complex')
# ddddf = ddddf.to_numpy(dtype ='complex')
# N500_dn001 = N500_dn001.to_numpy(dtype='complex')

# ddf_slice_length = np.arange(0,0.1,0.001)
# df_slice_length = np.arange(0,0.1,0.01)
# dddf_slice_length = np.arange(0,0.1,0.005)
# ddddf_slice_length = np.arange(0,0.1,0.0005)
# N500_dn001_slice_length = np.arange(0,0.5,0.001)

# plt.semilogy(df_slice_length,np.abs(df[(3306+9*6),:])**2,label = 'Harmonic: 9, dn = 0.01')
# plt.semilogy(dddf_slice_length,np.abs(dddf[(3306+9*6),:])**2,label = 'Harmonic: 9, dn = 0.005')
# plt.semilogy(ddf_slice_length,np.abs(ddf[(3306+9*6),:])**2,label = 'Harmonic: 9, dn = 0.001 ')
# plt.semilogy(ddddf_slice_length,np.abs(ddddf[(3306+9*6),:])**2,label = 'Harmonic: 9, dn = 0.0005 ')
# plt.semilogy(N500_dn001_slice_length,np.abs(N500_dn001[(3306+9*6),:])**2,label = 'Harmonic: 9, dn = 0.001 , N = 500')
# plt.semilogy(ddddf_slice_length,1e-1*ddddf_slice_length**2,label = 'sqaure')
# plt.xscale('log')
# #plt.xlim(6e-2,1e-1)
# plt.ylabel('Intensity')
# plt.xlabel('slice')
# plt.legend()

