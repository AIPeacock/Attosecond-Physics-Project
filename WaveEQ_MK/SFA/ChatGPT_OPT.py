import numpy as np
import time
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt

def Field(t, F, w):
    return np.sin(w * t) * F

def pulse_field(t, F, w, Nc):
    Carrier = np.sin(w * (t - (Nc * 2 * np.pi / w) / 2))
    st = Carrier * (np.sin(np.pi * t / (Nc * 2 * np.pi / w)) ** 2)
    return st * F

def hydroDTME(p, k):
    return (1j * 8 * np.sqrt(2 * (k**5)) * p) / (np.pi * ((p**2 + k**2) ** 3))

def time_to_index(time_value, dt):
    return int(time_value / dt)

# Define constants
w = 0.057
dt = 0.1
t0 = 0
Nc = 2
tf = Nc * (2 * np.pi / w)
N = int((tf - t0) / dt)

# Calculate field parameters
I0 = 1e14 / 3.51e16
E0 = np.sqrt(I0)
Ip = 15.7 / 27.2

# Generate time array
t_full = np.linspace(t0, tf, N)

# Calculate Up, gamma, and kappa
Up = E0**2 / (4 * w**2)
gamma = np.sqrt(Ip / (2 * Up))
kappa = np.sqrt(2 * Ip)
epsilon = 1 / (10 * kappa)**2

# Record start time for performance measurement
time_a = time.time()

# Generate all combinations of ti and tr using vectorized operations
ti_grid, tr_grid = np.meshgrid(t_full, t_full)
valid_indices = np.triu_indices_from(ti_grid, k=1)
tr_list = ti_grid[valid_indices]
ti_list = tr_grid[valid_indices]

# Record time after generating time combinations
time_b = time.time()
print('Time to generate time values:', time_b - time_a)

At = np.zeros(N)
for i in range(1, N):
    Atint = (dt / 2) * (Field(t_full[i], E0, w) + Field(t_full[i-1], E0, w))
    At[i] = At[i-1] - Atint
At2 = np.square(At)

time_e = time.time()
print('Time to calculate At2:', time_e - time_b)

sol1 = np.zeros(N)
solsquared2 = np.zeros(N)
for i in range(1, N):
    solint = (dt / 2) * (At[i] + At[i-1])
    sol1[i] = sol1[i-1] + solint
    sol2int = (dt / 2) * (At2[i] + At2[i-1])
    solsquared2[i] = solsquared2[i-1] + sol2int

time_f = time.time()
print('Time to calculate sol and solsquared:', time_f - time_e)

# Record start time for integration
time_c = time.time()

# Calculate Dipole moments
Dipole = []
for ti, tr in zip(ti_list, tr_list):
    x = time_to_index(ti, dt)
    y = time_to_index(tr, dt)

    solsum = sol1[y] - sol1[x]
    solsquaredsum = solsquared2[y] - solsquared2[x]

    Ps = -solsum / (tr - ti)
    Sv = Ip * (tr - ti) + 0.5 * Ps**2 * (tr - ti) + Ps * solsum + 0.5 * solsquaredsum

    d_matrix_ion = hydroDTME(Ps + At[x], kappa)
    d_matrix_recol = np.conj(hydroDTME(Ps + At[y], kappa))

    prefactor = 1  # Adjust if necessary

    Dipole.append(1j * prefactor * pulse_field(ti, E0, w, Nc) * d_matrix_ion * np.exp(-1j * Sv) * d_matrix_recol)

# Record end time for integration
time_d = time.time()
print('Time integration:', time_d - time_c)

# Summation of all dipole moments
Dipole = np.array(Dipole)
Dipole_total = np.zeros(N, dtype=complex)
i1 = 0
for i in range(N, 0, -1):
    i2 = i1 + i
    Dipole_total[N - i] = np.sum(Dipole[i1:i2])
    i1 = i2



Field_list = pulse_field(t_full,E0,w,Nc)
plt.figure(2)
plt.plot(t_full/(2 * np.pi / w),Field_list)
plt.xlabel('Time (Cycles of carrier)')

plt.figure(3)
field_freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1],d=dt))
Freq_Field = fftshift(fft(Field_list))
plt.plot(field_freq_axis/w,abs(Freq_Field)**2)
plt.xlabel('Frequency (Harmonic order)')
plt.xlim(-2,2)

# Plot the real part of the dipole moment over time
plt.figure(4)
plt.plot(t_full/(2 * np.pi / w), np.real(Dipole_total))
plt.xlabel('Time (Cycles of carrier) ')
plt.ylabel('Dipole Moment')

# Calculate the Fourier transform of the dipole moment
Dipole_freq = fftshift(fft((Dipole_total)))
freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
D_spectrum = np.abs(Dipole_freq)**2

# Plot the spectrum
plt.figure(5)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0])
plt.xlabel('Frequency (Harmonic Order)')
plt.ylabel('Intensity')
plt.show()