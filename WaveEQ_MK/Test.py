import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
# Define constants
w = 0.057  # Frequency of field
dt = 1  # Time steps
t0 = 0  # Initial time
tf = (2 * np.pi / w)  # Final time
N = int((tf-t0)/dt)  # Number of time steps/samples


# Calculate field parameters
I0 = (1 * (10**14)) / (3.51 * (10**16))  # Intensity
E0 = np.sqrt(I0)  # Electric field amplitude
Ip = 15.7 / 27.2  # Ionisation potential

# Generate a time array for plotting
t_full = np.linspace(t0, tf, N)
f0 = w/(2*np.pi)

def pulse_field(t,F,w):
     Carrier = np.sin(w*(t-N/(w/(2*np.pi))))
     return F * ((np.sin(w*t/N)**2) * Carrier)

plt.figure(1)
plt.plot(t_full,pulse_field(t_full,E0,w))


        