import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifftshift, fftfreq
import time
from scipy import integrate

def hydroDTME(p, k):
    return (1j * 8 * np.sqrt(2 * (k ** 5)) * p) / (np.pi * (((p ** 2) + (k ** 2)) ** 3))

def Lewenstein(w,I0,Ip,dt,N,Nc,t_full,Field_list):     #To make this function work it needs to be passed, w (omega0),dt (timestep), t_full,intensity(I0 or E0?), and the Ionisation pot
    #Calc Field strength based on intensity
    E0 = np.sqrt(I0)

    # Calculate Up, gamma, and kappa
    Up = (E0 ** 2) / (4 * (w ** 2))
    gamma = np.sqrt(Ip / (2 * Up))
    kappa = np.sqrt(2 * Ip)
    epsilon = 1e-4

    #Generate matrix of 2D array for all time values 
    tr_grid, ti_grid = np.meshgrid(t_full, t_full)     # Ti_grid goes across horizontally, Tr_grid goes up vertically So both are NxN arrays 
    valid_indices = np.triu_indices_from(ti_grid, k=0)                            # Ti_grid is all the horizontal values so first element is N zeros then the second is N 1's etc until the last 
                                                                                # Element which is N N's, The valid 
    mask = np.triu(np.ones_like(tr_grid), k=0)
    delta_t_matrix = (tr_grid - ti_grid) * mask

    #Find vector potential of given field
    
    At = -1*integrate.cumulative_trapezoid(Field_list,x=t_full,dx=dt,initial=0)
    #At = At + max(abs(At))/2   #Only needed to correct for sin field
    #Square of vector potential 
    At2 = np.square(At)

    At_ti = np.tile(At, (N, 1)).T
    At_ti[np.tril_indices(N)] = 0
    At_tr = np.tile(At,(N,1))
    At_tr[np.tril_indices(N)] = 0

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

    # Generate a 2D matrix for the field where all ti values have the same field value
    Field_matrix_ion = np.tile(Field_list[:, np.newaxis], (1, N))

    # Ensure the top-left triangle half is zero
    Field_matrix_ion[np.tril_indices(N)] = 0

    #Generate 2D matrix of stationary momentum 

    Ps_matrix = -1 * np.divide(Sol1_matrix, delta_t_matrix, out=np.zeros_like(Sol1_matrix), where=np.abs(delta_t_matrix) > epsilon)

    np.fill_diagonal(Ps_matrix, At)   

    #Generate action matrix
    Sv_matrix = (Ip *delta_t_matrix
                + 0.5 * np.square(Ps_matrix)*delta_t_matrix 
                + Ps_matrix * Sol1_matrix
                + 0.5 * Sol2_matrix
                )

    #Generate Dipole matrix element matrices

    d_ion_matrix = hydroDTME(Ps_matrix + At_ti, kappa)
    d_recomb_matrix = np.conj(hydroDTME(Ps_matrix  + At_tr , kappa))

    #Prefactor Matrix

    prefactor_matrix2 = ((np.pi) / (epsilon + (1j * delta_t_matrix / 2)))**(3 / 2)

    Dipolemom_matrix = 1j * (prefactor_matrix2 * d_recomb_matrix * (d_ion_matrix * Field_matrix_ion) * np.exp(-1j * Sv_matrix))
    
    Dipole_total2 = np.sum(Dipolemom_matrix, axis =1)*dt

    #Fourier transforms
    Freq_Field = fftshift(fft(ifftshift(Field_list)))

    Dipole_freq = fftshift(fft(ifftshift(Dipole_total2)))
    freq_axis = (2*np.pi)*fftshift(fftfreq(t_full.shape[-1], d=dt))
    scaled_harmonic_intensity = ((freq_axis)**2)#/(2*np.pi*(c**3))
    D_spectrum = (np.abs(Dipole_freq)**2)*scaled_harmonic_intensity  

    return(Dipole_total2, Freq_Field)