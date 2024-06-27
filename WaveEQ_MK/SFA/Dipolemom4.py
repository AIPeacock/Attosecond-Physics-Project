import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
import time

# Function to calculate the electric field at time t
def Field(t, F, w):
    return np.cos(w * t) * F

# Function to calculate the Dipole Transition Matrix Element (DTME)
def hydroDTME(p, k):
    DTME = (1j * 8 * np.sqrt(2 * (k**5)) * p) / (np.pi * (((p**2) + (k**2))**3))
    return DTME

# Define constants
w = 0.057  # Frequency of field
dt = 0.2  # Time steps
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
valid_indices = np.tril_indices_from(ti_grid, k=-1)  # Get indices for lower triangle of the matrix
ti_list = ti_grid[valid_indices]  # List of valid ionization times
tr_list = tr_grid[valid_indices]  # List of valid recombination times

# Record time after generating time combinations
time_b = time.time()
print('Time nested for loop')
print(time_b - time_a)

# Record start time for integration
time_c = time.time()

# Loop over all valid combinations of ionization and recombination times
for ti, tr in zip(ti_list, tr_list):
    # Ensure positive number of samples
    num_samples = max(int((tr - ti) / dt), 2)  # Ensure at least 2 samples
    t_eva = np.linspace(ti, tr, num_samples)

    if tr - ti <= dt:
        continue
    print(ti,tr)
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
    prefactor1 = -1  # Prefactor from derivative of action wrt ti
    prefactor2 = 1  # Prefactor from momentum integration using saddle point

    # Calculate and store the dipole moment
    Dipole.append(1j * prefactor1 * prefactor2 * d_matrix_recol * d_matrix_ion * Field(ti, E0, w) * np.exp(-1j * Sv))

# Record end time for integration
time_d = time.time()
print('Time integration')
print(time_d - time_c)

# Convert Dipole list to a NumPy array for efficient computation
Dipole = np.array(Dipole)

# Summation of all dipole moments
Dipole_total = np.zeros(N, dtype=complex)
i1 = 0
for i in range(N):
    Dipole_total[i] = np.sum(Dipole[i1:i1 + N - i])
    i1 += N - i

# Plot the real part of the dipole moment over time
plt.figure(1)
plt.plot(t_full, np.real(Dipole_total))
plt.xlabel('Time')
plt.ylabel('Dipole Moment')

# Calculate the Fourier transform of the dipole moment
Dipole_freq = fftshift(fft(ifftshift(Dipole_total)))
freq_axis = fftshift(fftfreq(t_full.shape[-1], d=dt))
D_spectrum = np.abs(Dipole_freq)**2

# Plot the spectrum
plt.figure(2)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0])
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.show()