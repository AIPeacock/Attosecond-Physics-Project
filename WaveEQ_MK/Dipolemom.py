import numpy as np 
from numpy.fft import fft,ifftshift,fftshift,fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp

w = 0.057              #Define frequency of vector potential
t0 = 0                 #Define intial time (t_ref)
tf = (2*np.pi/w)       #Define final time 
dt = 0.01              #Time steps
t_eval = np.arange(t0,tf,dt)

Ip=-13.6 

def hydroDTME(p,k):              # K is ionisation potential, p is momentum
     DTME = (1j*8*((2*(k**5))**(1/2))*p)/(np.pi*(((p**2)+(k**2))**3))
     return (DTME)


def A_t(t,A):
    return np.sin(w*t)

def A_t2(t,A):
    return (np.sin(w*t))**2


sol = solve_ivp(A_t,[t0,tf],y0=[0],t_eval=t_eval)                        # Solving integral of S a(t'') dt''

solsquared = solve_ivp(A_t2,[t0,tf],y0=[0],t_eval=t_eval)                # Solving integral of S a(t'')^2 dt''

Ps = -(1/(tf-t0))*sol.y[0]                                               # Calculating stationary momentum

Sv = Ip*(tf-t0)+0.5*(Ps**2)*(tf-t0)+Ps*sol.y[0]+0.5*solsquared.y[0]      #This is for one t_ion, Does this need to be a number for each time t_ion so does generate an array but for steps through t_ion

d_matrix_ion = hydroDTME(Ps+A_t(t0,1),Ip)                                # Ionisation Dipole matrix elements contains (ps+A(t_ion)) 

d_matrix_recol = np.conj(hydroDTME(Ps+A_t(tf,1),Ip))                     # Recollision Dipole Matrix Element contaion (ps+A(t_recol))


Field =-np.gradient(A_t(t_eval,0))/np.gradient(t_eval)
print(Field)

plt.figure(1)
plt.plot(t_eval,A_t(t_eval,0))

plt.figure(2)
plt.plot(t_eval,Field)
plt.show