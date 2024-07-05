import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import time

# Function to calculate the electric field at time t
def Field(t, F, w):
    return np.sin(w * t) * F

def pulse_field(t, F, w, Nc):
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
Nc = 8
tf = Nc*(2 * np.pi / w)  # Final time
N = int((tf-t0) / dt) + 1  # Number of time steps/samples

# Calculate field parameters
I0 = (2 * (10**14)) / (3.51 * (10**16))  # Intensity
E0 = np.sqrt(I0)  # Electric field amplitude
Ip = 15.7 / 27.2  # Ionisation potential

# Generate a time array for plotting
t_full = np.linspace(t0, tf, N)

# Calculate Up, gamma, and kappa
Up = (E0**2) / (4 * (w**2))  # Ponderomotive force/potential
gamma = np.sqrt(Ip / (2 * Up))  # Keldysh parameter
kappa = np.sqrt(2 * Ip)  # Kappa
epsilon = 1e-4  # 1/((10*kappa)**2)  # Computational prefactor

# Calculate Cut-off frequency
w_c = Ip + 3.17*Up
cutoff = w_c/w
print('Cutoff (Harmonic Order)', cutoff)

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

At = np.zeros(N)
time_e = time.time()
for i in np.arange(t0, tf+dt, dt):  # Pre-calculating the vector potential for all possible times.
    Atint = (dt/2)*(pulse_field(i, E0, w, Nc)+pulse_field(i-1, E0, w, Nc))
    index = time_to_index(i, dt)
    if 0 <= index < N:
        At[index] = -1*Atint
At2 = np.square(At)

time_f = time.time()
print('Time to calculate At2')
print(time_f-time_e)

time_1 = time.time()
sol1 = np.zeros(N)
solsquared2 = np.zeros(N)

for i in np.arange(t0, tf+dt, dt):
    index = time_to_index(i, dt)
    if 0 <= index < N:
        solint = (dt/2)*(At[index]+At[index-1])
        sol1[index] = solint
        sol2int = (dt/2)*(At2[index]+At2[index-1])
        solsquared2[index] = sol2int

time_2 = time.time()
print('Time to calculate sol and solsquared')
print(time_2 - time_1)

def init_worker(at, sol1_shared, solsquared2_shared):
    global At, sol1, solsquared2
    At = at
    sol1 = sol1_shared
    solsquared2 = solsquared2_shared

def calculate_dipole(ti_tr_pairs, dt, E0, w, Nc, Ip, kappa, At, sol1, solsquared2):
    dipole_list = []
    for ti, tr in ti_tr_pairs:
        if (tr-ti) <= dt:
            continue
        x = time_to_index(ti, dt)  # Generate index corresponding to ionisation time
        y = time_to_index(tr, dt)  # Generate index corresponding to recombination time

        if x >= N or y >= N:
            continue
        solsum = np.sum(sol1[x:y+1])  # Access integrals
        solsquaredsum = np.sum(solsquared2[x:y+1])
        delta_t = tr - ti

        # Calculate stationary momentum Ps
        Ps = -solsum / delta_t

        # Calculate the classical action Sv
        Sv = Ip * delta_t + 0.5 * (Ps**2) * delta_t + Ps * solsum + 0.5 * solsquaredsum

        # Calculate ionization and recollision dipole matrix elements
        d_matrix_ion = hydroDTME(Ps + At[x], kappa)
        d_matrix_recol = np.conj(hydroDTME(Ps + At[y], kappa))

        # Define prefactors for the calculation
        prefactor = 1  # ((np.pi)/(epsilon+(1j*delta_t/2)))**(3/2)  # Prefactor from momentum integration using saddle point

        # Calculate and store the dipole moment
        Dipolemom = ((1j * prefactor * d_matrix_recol * (d_matrix_ion * pulse_field(ti, E0, w, Nc)) * np.exp(-1j * Sv)))
        dipole_list.append(Dipolemom)
    return dipole_list

# Record start time for integration
time_c = time.time()

# Convert the arrays to shared memory arrays
At_shared = mp.Array('d', At)
sol1_shared = mp.Array('d', sol1)
solsquared2_shared = mp.Array('d', solsquared2)

# Split the time pairs into chunks for multiprocessing
num_cores = 4
chunks = np.array_split(list(zip(ti_list, tr_list)), num_cores)

# Use multiprocessing to calculate the dipole moments in parallel
with mp.Pool(num_cores, initializer=init_worker, initargs=(At_shared, sol1_shared, solsquared2_shared)) as pool:
    results = pool.map(partial(calculate_dipole, dt=dt, E0=E0, w=w, Nc=Nc, Ip=Ip, kappa=kappa, At=At_shared, sol1=sol1_shared, solsquared2=solsquared2_shared), chunks)

# Flatten the results list
Dipole = [item for sublist in results for item in sublist]

# Record end time for integration
time_d = time.time()
print('Time integration (mins)')
print((time_d - time_c)/60)

# Convert Dipole list to a NumPy array for efficient computation
Dipole = np.array(Dipole)

# Summation of all dipole moments
Dipole_total = np.zeros(N, dtype=complex)
i1 = 0
i2 = 0
for i in range(N, 0, -1):
    i2 = i1 + i
    Dipole_total[N-i] = np.sum(Dipole[i1:i2]) * dt / 2
    i1 = i2

# Plot the field
Field_list = pulse_field(t_full, E0, w, Nc)
plt.figure(2)
plt.plot(t_full/(2 * np.pi / w), Field_list/max(Field_list))
plt.xlabel('Time (Cycles of carrier)')

plt.figure(3)
field_freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
Freq_Field = fftshift(fft(Field_list))
plt.plot(field_freq_axis/w, abs(Freq_Field)**2/max(abs(Freq_Field)**2))
plt.xlabel('Frequency (Harmonic order)')
plt.xlim(-2, 2)

# Plot the real part of the dipole moment over time
plt.figure(4)
plt.plot(t_full/(2 * np.pi / w), np.real(Dipole_total)/max(np.real(Dipole_total)))
plt.xlabel('Time (Cycles of carrier)')
plt.ylabel('Dipole Moment')

# Calculate the Fourier transform of the dipole moment
Dipole_freq = fftshift(fft((Dipole_total)))
freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
D_spectrum = np.abs(Dipole_freq)**2

# Plot the spectrum
plt.figure(figsize=(12, 12), dpi=120.0)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0])
plt.xlabel('Frequency (Harmonic Order)')
plt.ylabel('Intensity')
plt.xlim(0, 50)
plt.show()

# Plot the spectrum
plt.figure(figsize=(12, 12), dpi=120.0)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0], marker='o')
plt.xlabel('Frequency (Harmonic Order)')
plt.ylabel('Intensity')
plt.xticks(np.arange(0, 50, 1))
plt.xlim(0, 50)
plt.vlines(x=[1, 3, 5, 7, 9, 11, cutoff/2], ymin=0, ymax=1e-4, colors=['red', 'red', 'red', 'red', 'red', 'red', 'green'])
plt.axvline(x=cutoff, ymin=0, ymax=0.9, color='blue')
plt.show()