import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
import time
import cProfile
import pstats
#%reload_ext line_profiler
# Function to calculate the electric field at time t
def Field(t, F, w):
    return np.sin(w * t) * F

def pulse_field(t,F,w,Nc):
     #Carrier = np.sin(w*(t-(Nc*2*np.pi/w)/2))           #Defines the carrier, sin, with centering(t - tmax/2),   Nc*2*np.pi/w gives us the tmax by multiplying the number of cycles by f0 
     #envelope =((np.sin(np.pi*t/(Nc*2*np.pi/w)))**2))   #Defines the envelope function sinsquared(pi* t/tmax)
     return (np.sin(w*(t-(Nc*2*np.pi/w)/2))*((np.sin(np.pi*t/(Nc*2*np.pi/w)))**2))*F

# Function to calculate the Dipole Transition Matrix Element (DTME)
def hydroDTME(p, k):
    return (1j * 8 * np.sqrt(2 * (k**5)) * p) / (np.pi * (((p**2) + (k**2))**3))

def time_to_index(time_value, dt):
    return int(np.round(time_value/dt))

# Define constants
w = 0.057  # Frequency of field
dt = 0.1  # Time steps
t0 = 0  # Initial time
Nc = 10
tf =Nc*(2 * np.pi / w)  # Final time
N = int((tf-t0) / dt)  # Number of time steps/samples



# Calculate field parameters
I0 = (2 * (10**14)) / (3.51 * (10**16))  # Intensity
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
epsilon = 1e-4 # 1/((10*kappa)**2)  #Comupational prefactor 

#Calculate Cut-off frequency

w_c= Ip + 3.17*Up
cutoff = w_c/w
print('Cutoff (Harmonic Order)',cutoff)
# Record start time for performance measurement
time_a = time.time()

# Generate all combinations of ti and tr using vectorized operations
ti_grid, tr_grid = np.meshgrid(np.linspace(t0, tf, N), np.linspace(t0, tf, N))
valid_indices = np.triu_indices_from(ti_grid, k=-1)  # Get indices for upper triangle of the matrix
tr_list = ti_grid[valid_indices]  # List of valid ionization times
ti_list = tr_grid[valid_indices]  # List of valid recombination times

# Record time after generating time combinations
time_b = time.time()
print('Time to generate time values')
print(time_b - time_a)

w = 0.057  # Frequency of field
dt = 0.1  # Time steps
t0 = 0  # Initial time
Nc = 10
tf =Nc*(2 * np.pi / w)  # Final time
N = int((tf-t0) / dt)  # Number of time steps/samples

At=np.zeros(N)
time_e = time.time()
for i in np.arange(t0,tf+dt,dt):                               # Pre-calculating the vector potential for all possible times.
     Atint = (dt/2)*(pulse_field(i,E0,w,Nc)+pulse_field(i-1,E0,w,Nc))  
     index = time_to_index(i,dt)
     #print(index)
     if 0<=index <N:
        At[index] = -1*Atint
     else:
        print(index,'Index')
At2 = np.square(At)


time_1 = time.time()
sol1 = np.zeros(N)
solsquared2 = np.zeros(N)

for i in np.arange(t0,tf+dt,dt):
     index = time_to_index(i,dt)
     if 0<= index <N:
        solint = (dt/2)*(At[index]+At[index-1])
        sol1[index]=solint
        sol2int = (dt/2)*(At2[index]+At2[index-1])
        solsquared2[index]=sol2int
     else:
        print(index,'Index')

plt.figure(1)
plt.plot(t_full,pulse_field(t_full,E0,w,Nc))
#plt.xlim(973.5,974)
plt.title('Field')
plt.figure(2)
plt.plot(t_full,At)
plt.title('Vector Potential')
#plt.xlim(973.5,974)
plt.figure(3)
plt.plot(t_full,sol1)
plt.title('Integral of A(t)')
#plt.xlim(973.5,974)
plt.figure(4)
plt.plot(t_full,solsquared2)
plt.title('Integral of A(t)^2 ')
#plt.xlim(973.5,974)