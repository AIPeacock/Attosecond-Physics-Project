import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifftshift, fftfreq
import time
from scipy import integrate
# def Field(t, F, w, Nc):
#     return  np.cos(w * t ) * F #+  F*np.sin(3*w*t) + F*np.sin(5*w*t)

def Field_window(t, F, w, Nc, duration):
    # Length of the sine signal
    tf_sin = Nc * (2 * np.pi / w)
    
    # Initialize the field with zeros
    field = np.zeros_like(t)
    
    # Ramp-up window (cosine-squared envelope)
    ramp_up_end = duration
    ramp_up_region = (t >= 0) & (t <= ramp_up_end)
    ramp_up_window = np.sin((t[ramp_up_region] / duration) * np.pi / 2) ** 2
    

    # Set the field in the ramp-up region
    field[ramp_up_region] = np.cos(w * t[ramp_up_region]) * F * ramp_up_window
    
    # Main sine wave region
    main_region = (t > duration) & (t <= (duration + tf_sin))
    field[main_region] = np.cos(w * (t[main_region] - duration)) * F
    
    # Transition window (cosine-squared envelope)
    transition_start = duration + tf_sin
    transition_end = transition_start + duration
    transition_region = (t > transition_start) & (t <= transition_end)
    transition_window = np.cos((t[transition_region] - transition_start) / duration * np.pi / 2) ** 2
    
    # Apply the transition window to the field
    field[transition_region] = np.cos(w * (t[transition_region] - duration)) * F * transition_window
    
    return field


def Field(t, Fx, w,Nc,duration):
    return np.cos(w * t) * Fx 

def pulse_Field(t, F, w, Nc, duration):
    return (np.sin(w * (t - (Nc * 2 * np.pi / w) / 2)) * ((np.sin(np.pi * t / (Nc * 2 * np.pi / w))) ** 2)) * F

def hydroDTME(p, k):
    return (1j * 8 * np.sqrt(2 * (k ** 5)) * p) / (np.pi * (((p ** 2) + (k ** 2)) ** 3))


# Define constants
w = 0.057
dt = 0.1
Nc = 6
duration = 3*(2 * np.pi / w)  # Duration of the envelope window
t0 = 0
tf = duration + (Nc * (2 * np.pi / w)) + duration  # Total time including ramp-up and transition
N = int((tf - t0) / dt)
c = 299792458 / 2.187e6  # Atomic Units ??

# Calculate field parameters
I0 = (2* (10 ** 14)) / (3.51 * (10 ** 16))
E0 = np.sqrt(I0)
Ip = 15.7 / 27.2

# Generate a time array for plotting
t_full = np.linspace(t0, tf, N)

# Calculate Up, gamma, and kappa
Up = (E0 ** 2) / (4 * (w ** 2))
gamma = np.sqrt(Ip / (2 * Up))
kappa = np.sqrt(2 * Ip)
epsilon = 1e-8

# Calculate Cut-off frequency
w_c = Ip + 3.17 * Up
cutoff = w_c / w
print('Cutoff (Harmonic Order)', cutoff)

#Generate matrix of 2D array for all time values 
tr_grid, ti_grid = np.meshgrid(np.linspace(t0, tf, N), np.linspace(t0, tf, N))     # Ti_grid goes across horizontally, Tr_grid goes up vertically So both are NxN arrays 
valid_indices = np.triu_indices_from(ti_grid, k=0)                            # Ti_grid is all the horizontal values so first element is N zeros then the second is N 1's etc until the last 
                                                                              # Element which is N N's, The valid 
#tr_list = tr_grid[valid_indices]                                              # 
#ti_list = ti_grid[valid_indices]                                              # 
mask = np.triu(np.ones_like(tr_grid), k=0)
delta_t_matrix = (tr_grid - ti_grid) * mask

plt.figure(100)
plt.imshow(delta_t_matrix, extent=(t0, tf, t0, tf), origin='lower', aspect='auto')
plt.colorbar()
plt.title('delta_t_matrix')
plt.xlabel('tr')
plt.ylabel('ti')

#Find vector potential of given field
Field_list = Field_window(t_full,E0,w,Nc, duration)
At = -1*integrate.cumulative_trapezoid(Field_list,x=t_full,dx=dt,initial=0)
#At = At + max(abs(At))/2   #Only needed to correct for sin field
#Square of vector potential 
At2 = np.square(At)

At_ti = np.tile(At, (N, 1)).T
At_ti[np.tril_indices(N)] = 0
At_tr = np.tile(At,(N,1))
At_tr[np.tril_indices(N)] = 0

plt.figure(0)
plt.imshow(At_ti, extent=(t0, tf, t0, tf), origin='lower', aspect='auto')
plt.colorbar()
plt.title('At_ti Matrix')
plt.xlabel('tr')
plt.ylabel('ti')

plt.figure(10)
plt.imshow(At_tr, extent=(t0, tf, t0, tf), origin='lower', aspect='auto')
plt.colorbar()
plt.title('At_tr Matrix')
plt.xlabel('tr')
plt.ylabel('ti')


#Let's Generate a 2D matrix for sol (integral of A(t)) and for sol2 (Integral of A(t)^2)

Sol1 = integrate.cumulative_trapezoid(At,x=t_full,dx=dt,initial= 0)                      #First pre-compute S A(t)
Sol2 = integrate.cumulative_trapezoid(At2,x=t_full,dx=dt,initial= 0)
#Sol1 = Sol1 + max(abs(Sol1))/2
Sol1_expanded = np.expand_dims(Sol1, axis=0)
Sol2_expanded = np.expand_dims(Sol2,axis=0)
Sol1_matrix_square = Sol1_expanded[:,valid_indices[1]]-Sol1_expanded[:,valid_indices[0]]     #Generate a 2D matrix of values by subtracting all ti_indicies of Sol1 from tr_indices of Sol1 
Sol2_matrix_square = Sol2_expanded[:,valid_indices[1]]-Sol2_expanded[:,valid_indices[0]]
# Reshape back to the upper triangular form
Sol1_matrix = np.zeros((N, N))
Sol1_matrix[valid_indices] = Sol1_matrix_square
Sol2_matrix = np.zeros((N, N))
Sol2_matrix[valid_indices] = Sol2_matrix_square

plt.figure(1)
plt.plot(t_full,Field_list)
plt.figure(2)
plt.plot(t_full,At)
plt.figure(3)
plt.plot(t_full,Sol1)
plt.figure(4)
plt.plot(t_full,Sol2)

#Plot to verify the results if needed
plt.figure(5)
plt.imshow(Sol1_matrix, extent=(t0, tf, t0, tf), origin='lower')
plt.colorbar()
plt.title('Integral of vector potential for all combinations of ti,tr')
plt.xlabel('tr')
plt.ylabel('ti')
plt.figure(6)
plt.imshow(Sol2_matrix, extent=(t0, tf, t0, tf), origin='lower')
plt.colorbar()
plt.title('Integral of At^2 for all combinations of ti,tr')
plt.xlabel('tr')
plt.ylabel('ti')

# Generate a 2D matrix for the field where all ti values have the same field value
Field_matrix_ion = np.tile(Field_list[:, np.newaxis], (1, N))

# Ensure the top-left triangle half is zero
Field_matrix_ion[np.tril_indices(N)] = 0

plt.figure()
plt.imshow(Field_matrix_ion, extent=(t0, tf, t0, tf), origin='lower')
plt.colorbar()
plt.title('Field Matrix')
plt.xlabel('tr')
plt.ylabel('ti')

plt.show()

#Generate 2D matrix of stationary momentum 
#want to do Sol1 * -1/(tr-ti) or -1 * Sol1/delta_t_matrix

Ps_matrix = -1 * np.divide(Sol1_matrix, delta_t_matrix, out=np.zeros_like(Sol1_matrix), where=np.abs(delta_t_matrix) > epsilon)

np.fill_diagonal(Ps_matrix, At)   

plt.figure(7)
plt.imshow(Ps_matrix, extent=(t0, tf, t0, tf), origin='lower')
plt.colorbar()
plt.title('Stationary momentum for all combinations of ti,tr')
plt.xlabel('tr')
plt.ylabel('ti')
plt.show()

#Generate action matrix
Sv_matrix = (Ip *delta_t_matrix
             + 0.5 * np.square(Ps_matrix)*delta_t_matrix 
             + Ps_matrix * Sol1_matrix
             + 0.5 * Sol2_matrix
             )

plt.figure(8)
plt.imshow(Sv_matrix, extent=(t0, tf, t0, tf), origin='lower')
plt.colorbar()
plt.title('Action for all combinations of ti,tr')
plt.xlabel('tr')
plt.ylabel('ti')
plt.show()

#Generate Dipole matrix element matrices

d_ion_matrix = hydroDTME(Ps_matrix + At_ti, kappa)
d_recomb_matrix = np.conj(hydroDTME(Ps_matrix  + At_tr , kappa))

#Prefactor Matrix

prefactor_matrix2 = ((np.pi) / (epsilon + (1j * delta_t_matrix / 2)))**(3 / 2)

Dipolemom_matrix = 1j * (prefactor_matrix2 * d_recomb_matrix * (d_ion_matrix * Field_matrix_ion) * np.exp(-1j * Sv_matrix))
 
Dipole_total2 = np.sum(Dipolemom_matrix, axis =1)*dt


plt.figure(6)
plt.plot(t_full/(2 * np.pi / w),Field_list)
plt.xlabel('Time (Cycles of carrier)')

plt.figure(7)
field_freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1],d=dt))
Freq_Field = fftshift(fft(ifftshift(Field_list)))
plt.plot(field_freq_axis/w,abs(Freq_Field)**2/max(abs(Freq_Field)**2))
plt.xlabel('Frequency (Harmonic order)')
plt.xlim(-2,2)

# Plot the real part of the dipole moment over time      #Dipole_total2[: int((((Nc-1) * (2 * np.pi / w)) - t0) / dt)],to be used when plotting CW signals to remove weird end of dipole mom
plt.figure(8)
plt.plot(t_full/(2 * np.pi / w), np.real(Dipole_total2))
plt.xlabel('Time (Cycles of carrier) ')
plt.ylabel('Dipole Moment')

# Calculate the Fourier transform of the dipole moment
#Harmonic_scale = (w**4)/(2*np.pi*(c**3))
Dipole_freq = fftshift(fft(ifftshift(Dipole_total2)))
freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
scaled_harmonic_intensity = ((freq_axis)**2)#/(2*np.pi*(c**3))
D_spectrum = (np.abs(Dipole_freq)**2)*scaled_harmonic_intensity              #Need to Scale to harmonic intensity ??

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
plt.vlines(x=[1,3,5,7,9,11,cutoff/2,Ip/w],ymin=0,ymax=1e-1,colors =['red','red','red','red','red','red','green','orange'])
plt.axvline(x=cutoff,ymin=0,ymax=0.9,color='blue')
plt.show()
