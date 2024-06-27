import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
import time

# Function to calculate the electric field at time t
def Field(t, F, w):
    return np.sin(w * t) * F

def pulse_field(t,F,w,period):
     Carrier = np.sin(w*(t-period/(w/(2*np.pi))))
     return F * ((np.sin(w*t/period)**2) * Carrier)

# Function to calculate the Dipole Transition Matrix Element (DTME)
def hydroDTME(p, k):
    DTME = (1j * 8 * np.sqrt(2 * (k**5)) * p) / (np.pi * (((p**2) + (k**2))**3))
    return DTME

# Define constants
w = 0.057  # Frequency of field
dt = 0.1  # Time steps
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
    A_t = -1 * At  # Invert the sign for the calculation
    A_t2 = A_t**2  # Square of A(t)
    
    # Integrate A(t) and A(t)^2 over time to get sol and solsquared
    sol = integrate.trapezoid(A_t, t_eva)
    solsquared = integrate.trapezoid(A_t2, t_eva)
    
    # Calculate stationary momentum Ps
    Ps = -sol / (tr - ti)
    
    # Calculate the classical action Sv
    Sv = Ip * (tr - ti) + 0.5 * (Ps**2) * (tr - ti) + Ps * sol + 0.5 * solsquared

    # Calculate ionization and recollision dipole matrix elements
    d_matrix_ion = hydroDTME(Ps + A_t[0], kappa)
    d_matrix_recol = np.conj(hydroDTME(Ps + A_t[-1], kappa))
    
    # Define prefactors for the calculation
    prefactor = 1#((2*np.pi)/(1j*(tr-ti)))**(3/2)  # Prefactor from momentum integration using saddle point

    # Calculate and store the dipole moment
    Dipole.append((1j * prefactor * Field(ti, E0, w) * d_matrix_ion*np.exp(-1j * Sv) )* d_matrix_recol )

# Record end time for integration
time_d = time.time()
print('Time integration')
print(time_d - time_c)

# Convert Dipole list to a NumPy array for efficient computation
Dipole = np.array(Dipole)

# Summation of all dipole moments
Dipole_total = np.zeros(N, dtype=complex)
i1 = 0
i2 = 0 
for i in range(N, 0, -1):
    print(i)
    i2 = i1 + i
    print('i2',i2,'i1',i1)
    Dipole_total[N-i] = np.sum(Dipole[i1:i2])
    # print('Splice i1',i1,'i2',i2,'Dipole',Dipole[i1:i2])
    i1 = i2
    
#Plot the field 
#pulse_field_list = pulse_field(t_full,E0,w,period=2)
#plt.figure(1)
#plt.plot(t_full,pulse_field_list)


Field_list = Field(t_full,E0,w)
plt.figure(2)
plt.plot(t_full,Field_list)


# Plot the real part of the dipole moment over time
plt.figure(3)
plt.plot(t_full, np.real(Dipole_total))
plt.xlabel('Time')
plt.ylabel('Dipole Moment')

# Calculate the Fourier transform of the dipole moment
Dipole_freq = fftshift(fft(ifftshift(Dipole_total)))
freq_axis = fftshift(fftfreq(t_full.shape[-1], d=dt))
D_spectrum = np.abs(Dipole_freq)**2

# Plot the spectrum
plt.figure(4)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0])
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.show()