import numpy as np
from numpy.fft import fft, ifftshift, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
import time
#%reload_ext line_profiler
# Function to calculate the electric field at time t
def Field(t, F, w):
    return np.sin(w * t) * F

def pulse_field(t,F,w,Nc):
     Carrier = np.sin(w*(t-(Nc*2*np.pi/w)/2))          #Defines the carrier, sin, with centering(t - tmax/2),   Nc*2*np.pi/w gives us the tmax by multiplying the number of cycles by f0 
     st=(Carrier*((np.sin(np.pi*t/(Nc*2*np.pi/w)))**2))   #Defines the envelope function sinsquared(pi* t/tmax)
     return st*F

# Function to calculate the Dipole Transition Matrix Element (DTME)
def hydroDTME(p, k):
    DTME = (1j * 8 * np.sqrt(2 * (k**5)) * p) / (np.pi * (((p**2) + (k**2))**3))
    return DTME

def time_to_index(time_value, dt):
    return int(time_value * (1/dt))

# Define constants
w = 0.057  # Frequency of field
dt = 0.1  # Time steps
t0 = 0  # Initial time
Nc = 4
tf =Nc*(2 * np.pi / w)  # Final time
N = int((tf-t0) / dt)  # Number of time steps/samples


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
epsilon = 1/((10*kappa)**2)

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

At=np.zeros(N+1)
time_e = time.time()
for i in np.linspace(t0,tf,N):                               # Pre-calculating the vector potential for all possible times.
     Atint = (dt/2)*(Field(i,E0,w)+Field(i-1,E0,w))  
     index = time_to_index(i,dt)
     At[index] = -1*Atint
At2 = np.square(At)

time_f = time.time()
print('Time to calculate At2')
print(time_f-time_e)

time_1 = time.time()
sol1 = np.zeros(N+1)
solsquared2 = np.zeros(N+1)

for i in np.linspace(t0,tf,N):
     index = time_to_index(i,dt)
     solint = (dt/2)*(At[index]+At[index-1])
     sol1[index]=solint
     sol2int = (dt/2)*(At2[index]+At2[index-1])
     solsquared2[index]=sol2int

time_2 = time.time()
print('Time to calculate sol and solsquared')
print(time_2 - time_1)

# Record start time for integration
time_c = time.time()
# Loop over all valid combinations of ionization and recombination times
 
for ti, tr in zip(ti_list, tr_list):
    if tr <= ti:
            continue
    x = time_to_index(ti,dt)
    y = time_to_index(tr,dt)
    #print('Tr',tr,'Ti',ti)
    #n = int((tr-ti)/dt)                         # Generates an array of times with each element including the increments between ti,tr
    #print(n)
    #t_eva = np.linspace(ti, tr, n)
    #print(t_eva)
    #A_t = At[x:y]          # x:y is the indexs of time that ti and tr correspond too.  
     
    #print('A_t',len(A_t),'A_t2',len(A_t2),'T_eva',len(t_eva))
    # Integrate A(t) and A(t)^2 over time to get sol and solsquared
    #sol = integrate.trapezoid(A_t, t_eva)
    solsum = np.sum(sol1[x:y])
    #print(sol - solsum)
    #solsquared = integrate.trapezoid(A_t2, t_eva)
    solsquaredsum = np.sum(solsquared2[x:y])
    
    # Calculate stationary momentum Ps
    Ps = -solsum / (tr - ti)
    
    # Calculate the classical action Sv
    Sv = Ip * (tr - ti) + 0.5 * (Ps**2) * (tr - ti) + Ps * solsum + 0.5 * solsquaredsum

    # Calculate ionization and recollision dipole matrix elements
    d_matrix_ion = hydroDTME(Ps + At[x], kappa)
    d_matrix_recol = np.conj(hydroDTME(Ps + At[y], kappa))
    
    # Define prefactors for the calculation
    prefactor = ((2*np.pi)/(epsilon+1j*(tr-ti)))**(3/2)  # Prefactor from momentum integration using saddle point

    # Calculate and store the dipole moment
    Dipole.append((1j * prefactor * pulse_field(ti, E0, w,Nc) * d_matrix_ion*np.exp(-1j * Sv) )* d_matrix_recol )
    ti_check = ti

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
    #print(i)
    i2 = i1 + i
    #print('i2',i2,'i1',i1)
    Dipole_total[N-i] = np.sum(Dipole[i1:i2])
    # print('Splice i1',i1,'i2',i2,'Dipole',Dipole[i1:i2])
    i1 = i2
    
#Plot the field 
#pulse_field_list = pulse_field(t_full,E0,w,period=2)
#plt.figure(1)
#plt.plot(t_full,pulse_field_list)


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
freq_axis = fftshift(fftfreq(t_full.shape[-1], d=dt))
D_spectrum = np.abs(Dipole_freq)**2

# Plot the spectrum
plt.figure(5)
plt.semilogy((freq_axis[freq_axis >= 0]) / w, D_spectrum[freq_axis >= 0])
plt.xlabel('Frequency (Harmonic Order)')
plt.ylabel('Intensity')
plt.show()