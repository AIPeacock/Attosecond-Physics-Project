import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
#w = 1
#Nc = 10
#t = np.linspace(0,Nc*2*np.pi/w,1000)
#sp = np.fft.fft(np.sin(w*t)*np.sin(w*t)**2)
#Carrier = np.sin(w*(t-(Nc*2*np.pi/w)/2))             
#st=(Carrier*((np.sin(np.pi*t/(Nc*2*np.pi/w)))**2))
#sp = np.fft.fft(st)
#freq = (2*np.pi)*(np.fft.fftfreq(t.shape[-1],d =Nc*2*np.pi/w/1000))
#plt.figure(1)
#plt.plot(t,st)
#plt.figure(2)
#plt.plot(freq,abs(sp),marker = 'o')
#plt.xlim(-5,5)
#plt.show()

def Field(t, F, w):
    return np.sin(w * t) * F

def pulse_field(t,F,w,Nc):
     Carrier = np.sin(w*(t-(Nc*2*np.pi/w)/2))             #Defines the carrier, sin, with centering(t - tmax/2),   Nc*2*np.pi/w gives us the tmax by multiplying the number of cycles by f0 
     st=(Carrier*((np.sin(np.pi*t/(Nc*2*np.pi/w)))**2))   #Defines the envelope function sinsquared(pi* t/tmax)
     return st*F

# Calculate field parameters
I0 = (1 * (10**14)) / (3.51 * (10**16))  # Intensity
E0 = np.sqrt(I0)  # Electric field amplitude


# Define constants
w = 0.057  # Frequency of field
dt = 0.1  # Time steps
t0 = 0  # Initial time
tf =4*(2 * np.pi / w)  # Final time
Nc = 4
N = int((tf-t0) / dt)  # Number of time steps/samples


# Generate a time array for plotting
t_full = np.linspace(t0, tf, N)

Field_list = pulse_field(t_full,E0,w,Nc)
plt.figure(2)
plt.plot(t_full/((2 * np.pi / w)),Field_list)       
plt.xlabel('Time (Cycles)')

plt.figure(3)
field_freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1],d=dt))
Freq_Field = fftshift(fft((Field_list)))
plt.plot(field_freq_axis/w,abs(Freq_Field)**2)
plt.xlabel('Freq (Harmonic Order)')
plt.xlim(-2,2)
