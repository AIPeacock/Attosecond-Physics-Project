import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
import time

def Field(t, F, w):
    return np.sin(w * t) * F


# Define constants
w = 0.057  # Frequency of field
dt =0.1  # Time steps
t0 = 0  # Initial time
tf =(2 * np.pi / w)  # Final time
N = int((tf-t0) / dt)  # Number of time steps/samples
period = 2

# Calculate field parameters
I0 = (1 * (10**14)) / (3.51 * (10**16))  # Intensity
E0 = np.sqrt(I0)  # Electric field amplitude
Ip = 15.7 / 27.2  # Ionisation potential

# Generate a time array for plotting
t_full = np.linspace(t0, tf, N)

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
valid_indices = np.triu_indices_from(ti_grid, k=-1)  # Get indices for upper triangle of the matrix
tr_list = ti_grid[valid_indices]  # List of valid ionization times
ti_list = tr_grid[valid_indices]  # List of valid recombination times

# Record time after generating time combinations
time_b = time.time()
print('Time nested for loop')
print(time_b - time_a)

# Record start time for integration
time_c = time.time()
#lenat= []

# Loop over all valid combinations of ionization and recombination times
for ti, tr in zip(ti_list, tr_list):
    if tr <= ti:
            continue
    #print('Tr',tr,'Ti',ti)
    n =int((tr-ti)/dt)                         # Generates an array of times with each element including the increments between ti,tr
    #print(n)
    t_eva = np.linspace(ti, tr, n)
    #print(t_eva)
    # Calculate the vector potential A(t)
    At = integrate.cumulative_trapezoid(Field(t_eva, E0, w), dx=dt, initial=0)
    #lenat = np.append(lenat,len(At))
time_d = time.time()
print('Time to calculate AT')
print(time_d- time_c)

time_e = time.time()
At=[]
for i in range(0,N,1):                                  # Pre-calculating the vector potential for all possible times. 
    #print('First Field',Field(i,E0,w))
    Atint = (dt/2)*(Field(i,E0,w)+Field(i-1,E0,w))   
    At = np.append(At,-1*Atint)

At2= np.square(At)

time_f = time.time()
print('Time to calculate At2')
print(time_f-time_e)