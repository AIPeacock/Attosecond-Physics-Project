import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
import time


# Define constants
w = 0.057  # Frequency of field
dt = 1  # Time steps
t0 = 0  # Initial time
tf = (2 * np.pi / w)  # Final time
N = int(tf / dt)  # Number of time steps

# Calculate field parameters
I0 = (1 * (10**14)) / (3.51 * (10**16))  # Intensity
E0 = np.sqrt(I0)  # Electric field amplitude
Ip = 15.7 / 27.2  # Ionisation potential

# Generate a time array for plotting
t_full = np.linspace(t0, tf - dt, N)

# Initialize empty list to store dipole moments
Dipole = []

# Calculate Up, gamma, and kappa
Up = (E0**2) / (4 * (w**2))  # Ponderomotive force/potential
gamma = np.sqrt(Ip / (2 * Up))  # Keldysh parameter
kappa = np.sqrt(2 * Ip)  # Kappa

# Record start time for performance measurement
time_a = time.time()

# Generate all combinations of ti and tr using vectorized operations
ti_grid, tr_grid = np.meshgrid(np.linspace(t0, tf, N), np.linspace(t0, tf, N))
valid_indices = np.triu_indices_from(ti_grid, k=-1)  # Get indices for lower triangle of the matrix
tr_list = ti_grid[valid_indices]  # List of valid ionization times
ti_list = tr_grid[valid_indices]  # List of valid recombination times

# Record time after generating time combinations
time_b = time.time()
print('Time nested for loop')
print(time_b - time_a)

plt.figure(1)
plt.plot(ti_list,tr_list,marker ='o',color='k',linestyle='none')
plt.xlabel('ti_list')
plt.ylabel('tr_list')


x=[]
y=[]
for ti, tr in zip(ti_list, tr_list):
        if (tr-ti) <= 2*dt:
                print(ti, tr)
                continue
        x = np.append(x,ti)
        y = np.append(y,tr)

plt.figure(2)
plt.plot(x,y,marker = 'o',color = 'k',linestyle ='none')
plt.show()
