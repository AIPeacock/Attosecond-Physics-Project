import numpy as np 
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp
from numpy.fft import fft,ifftshift,fftshift,fftfreq

# Constants
E0 = 0.1  # Electric field amplitude (in atomic units)
omega = 0.057  # Laser frequency (in atomic units, corresponds to 800 nm)
Ip = 0.5  # Ionization potential (in atomic units)
tau = 2 * np.pi / omega  # Optical cycle duration

# Time grid
t = np.linspace(-2 * tau, 2 * tau, 1000)  # Time array

# Electric field as a function of time
def electric_field(t, E0, omega):
    return E0 * np.cos(omega * t)

# Vector potential as a function of time
def vector_potential(t, E0, omega):
    return -E0 / omega * np.sin(omega * t)

# Saddle-point equation for the ionization time
def saddle_point_eq(t_prime, t, E0, omega, Ip):
    A = vector_potential(t_prime, E0, omega)
    return Ip + 0.5 * (A**2) + A * vector_potential(t, E0, omega)

# Dipole acceleration calculation using SFA
def dipole_acceleration(t, E0, omega, Ip):
    A_t = vector_potential(t, E0, omega)
    acc = np.zeros_like(t, dtype=np.complex128)
    
    for i, t_prime in enumerate(t):
        A_tp = vector_potential(t_prime, E0, omega)
        integral = np.exp(-1j * (Ip * (t - t_prime) + (A_tp**2 - A_t**2) / (2 * omega)))
        integral *= electric_field(t_prime, E0, omega)
        acc[i] = np.sum(integral) * (t[1] - t[0])
    
    return np.real(acc)

# Calculate the dipole acceleration
dipole_acc = dipole_acceleration(t, E0, omega, Ip)

# Fourier transform to get the HHG spectrum
hhg_spectrum = np.abs(np.fft.fft(dipole_acc))**2
freqs = np.fft.fftfreq(t.size, d=t[1] - t[0])

# Plot the dipole acceleration and HHG spectrum
plt.figure(figsize=(12, 5))

# Plot dipole acceleration
plt.subplot(1, 2, 1)
plt.plot(t, dipole_acc)
plt.title('Dipole Acceleration')
plt.xlabel('Time (a.u.)')
plt.ylabel('Acceleration (a.u.)')

# Plot HHG spectrum
plt.subplot(1, 2, 2)
plt.plot(freqs[:t.size // 2], hhg_spectrum[:t.size // 2])
plt.title('HHG Spectrum')
plt.xlabel('Frequency (a.u.)')
plt.ylabel('Intensity (a.u.)')
plt.yscale('log')

plt.tight_layout()
plt.show()