import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifftshift, fftfreq
import time
from scipy import integrate
def Field(t, F, w, Nc):
    return np.cos(w * t ) * F #+ F*np.cos(2*w*t + np.pi/2)

def pulse_Field(t, F, w, Nc):
    return (np.sin(w * (t - (Nc * 2 * np.pi / w) / 2)) * ((np.sin(np.pi * t / (Nc * 2 * np.pi / w))) ** 2)) * F

def hydroDTME(p, k):
    return (1j * 8 * np.sqrt(2 * (k ** 5)) * p) / (np.pi * (((p ** 2) + (k ** 2)) ** 3))

def time_to_index(time_value, dt):
    return np.round(time_value / dt).astype(int)

# Define constants
w = 0.057
dt = 0.1
Nc =5
t0 = 0
tf = (Nc * (2 * np.pi / w))
N = int((tf - t0) / dt) 

# Calculate field parameters
I0 = (2 * (10 ** 14)) / (3.51 * (10 ** 16))
E0 = np.sqrt(I0)
Ip = 15.7 / 27.2

# Generate a time array for plotting
t_full = np.linspace(t0, tf, N)

# Initialize empty list to store dipole moments
Dipole = []

# Calculate Up, gamma, and kappa
Up = (E0 ** 2) / (4 * (w ** 2))
gamma = np.sqrt(Ip / (2 * Up))
kappa = np.sqrt(2 * Ip)
epsilon = 1e-4

# Calculate Cut-off frequency
w_c = Ip + 3.17 * Up
cutoff = w_c / w
print('Cutoff (Harmonic Order)', cutoff)

time_0 = time.time()
#Pre-computation of vector potential 
At = -1*integrate.cumulative_trapezoid(Field(t_full,E0,w,Nc),x=t_full,dx=dt,initial=0)
plt.figure(2)
plt.plot(t_full,Field(t_full,E0,w,Nc))
plt.title('Field')
plt.figure(3)
plt.plot(t_full,At)
plt.title('Vector potential')


At2 = np.square(At)
time_1 =time.time()
print('Time to precompute A(t)',time_1-time_0)

time_0 =time.time()
#Pre-computation of Secondary Integrals 
Sol1 = integrate.cumulative_trapezoid(At,x=t_full,dx=dt,initial=0)
#Sol1 = Sol1 + max(abs(Sol1))/2
Sol2 = integrate.cumulative_trapezoid(At2,x=t_full,dx=dt,initial=0)

sol1analytical = E0/(w**2) * np.cos(t_full*w)


plt.figure(4)
plt.plot(t_full,Sol1)
plt.title('Integral of A(t)')
plt.figure(1)
plt.plot(t_full,sol1analytical)
plt.title('Analytical Vector potential')
plt.figure(5)
plt.plot(t_full,Sol2)
plt.title('Integral of A^2(t)')

time_1=time.time()
print('Time to precompute Sol1 and Sol2',time_1-time_0)


# Record start time for performance measurement
time_a = time.time()

# Generate all combinations of ti and tr using vectorized operations
tr_grid, ti_grid = np.meshgrid(np.linspace(t0, tf, N), np.linspace(t0, tf, N))
valid_indices = np.triu_indices_from(ti_grid, k=0)
tr_list = tr_grid[valid_indices]
ti_list = ti_grid[valid_indices]

delta_t_matrix = np.subtract(tr_list,ti_list)
Field_matrix_ion = np.array(Field(ti_list,E0,w,Nc))

ti_indices = valid_indices[0]
tr_indices = valid_indices[1]

Sol1_matrix = np.subtract(Sol1[tr_indices],Sol1[ti_indices])    # Currently does Sol[tr] - Sol[ti], So this contains all possible values of Int A(t) for every combination of ti,tr  
                                                                # As Int A(t) has already been pre computed for all dt'' this contains the contribution for all times between every ti,tr 
Sol2_matrix = np.subtract(Sol2[tr_indices],Sol2[ti_indices])

Ps_matrix = -1 * (Sol1_matrix/(delta_t_matrix))

delta_t_zeros = delta_t_matrix == 0

Ps_matrix[delta_t_zeros] = At[ti_indices[delta_t_zeros]]

Sv_matrix = (Ip *delta_t_matrix
             + 0.5 * np.square(Ps_matrix)*delta_t_matrix 
             + Ps_matrix * Sol1_matrix
             + 0.5 * Sol2_matrix
             )

d_ion_matrix = hydroDTME(Ps_matrix + At[ti_indices],kappa)
d_recombination_matrix = np.conj(hydroDTME(Ps_matrix + At[tr_indices],kappa))

prefactor_matrix = ((np.pi) / (epsilon + (1j * delta_t_matrix / 2)))**(3 / 2)

Dipolemom_matrix = np.array(1j * prefactor_matrix * d_recombination_matrix *
                    (d_ion_matrix * Field_matrix_ion) * np.exp(-1j * Sv_matrix))

time_b = time.time()
print('Time to compute Dipolemom', time_b-time_a)

time_0=time.time()

# Dipole_total = np.zeros(N, dtype=complex)
# i1 = 0
# i2 = 0 
# for i in range(N, 0, -1):
#     #print(i)
#     i2 = i1 + i 
#     #print('i2',i2,'i1',i1)
#     Dipole_total[N-i] = np.sum(Dipolemom_matrix[i1:i2])*dt
#     #print('Splice i1',i1,'i2',i2,'Dipole',Dipole[i1:i2])
#     i1 = i2

# Sum over ti to get the total dipole
Dipole_total = np.zeros(N, dtype=complex)
cumsum = np.cumsum(np.arange(N, 0, -1))+1
indices = np.hstack(([0], cumsum))

for i in range(N):
    Dipole_total[i] = np.sum(Dipolemom_matrix[indices[i]:indices[i + 1]]) * dt

time_1=time.time()
print('Time to complete summation of Dipolemom',time_1-time_0)
    
#Plot the field 
Field_list = Field(t_full,E0,w,Nc)
plt.figure(6)
plt.plot(t_full/(2 * np.pi / w),Field_list)
plt.xlabel('Time (Cycles of carrier)')

plt.figure(7)
field_freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1],d=dt))
Freq_Field = fftshift(fft(ifftshift(Field_list)))
plt.plot(field_freq_axis/w,abs(Freq_Field)**2/max(abs(Freq_Field)**2))
plt.xlabel('Frequency (Harmonic order)')
plt.xlim(-2,2)

# Plot the real part of the dipole moment over time
plt.figure(8)
plt.plot(t_full/(2 * np.pi / w), np.real(Dipole_total))
plt.xlabel('Time (Cycles of carrier) ')
plt.ylabel('Dipole Moment')

# Calculate the Fourier transform of the dipole moment
Dipole_freq = fftshift(fft(ifftshift(Dipole_total)))
freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
D_spectrum = np.abs(Dipole_freq)**2

# Plot the spectrum
plt.figure(figsize=(12,12),dpi=120.0)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0])
plt.xlabel('Frequency (Harmonic Order)')
plt.ylabel('Intensity')
plt.xlim(0,50)
plt.show()

#Plot the spectrum
plt.figure(figsize=(12,12),dpi=120.0)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0],marker = 'o')
plt.xlabel('Frequency (Harmonic Order)')
plt.ylabel('Intensity')
plt.xticks(np.arange(0,50,1))
plt.xlim(0,50)
plt.vlines(x=[1,3,5,7,9,11,cutoff/2],ymin=0,ymax=1e14,colors =['red','red','red','red','red','red','green'])
plt.axvline(x=cutoff,ymin=0,ymax=0.9,color='blue')
plt.show()


