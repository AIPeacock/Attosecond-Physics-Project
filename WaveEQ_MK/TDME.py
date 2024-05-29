import numpy as np 
from numpy.fft import fft,ifftshift,fftshift,fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp

w = 0.057              #Define frequency of vector potential
t0 = 0                 #Define intial time (t_ref)
tf = (2*np.pi/w)       #Define final time 
dt = 0.01              #Time steps
t_eval = np.arange(0,tf,dt)     #Generate an array of times to evaluate the function at
def A_t(t,A):                   #Define func to be solved (This is for the ODE situation)
    return np.sin(w*t)


sol =  solve_ivp(A_t,[t0,tf],y0=[0],t_eval=t_eval)


ps = (1/(tf-t0))*sol.y[0]           #Need to find a way to loop through all the t_eval times


def hydroDTME(p,k):              # K is ionisation potential, p is momentum
     DTME = (1j*8*((2*(k**5))**(1/2))*p)/(np.pi*(((p**2)+(k**2))**3))
     return (DTME, np.conj(DTME))

y = hydroDTME(ps,46)
b=hydroDTME(2,46)
print('b=',b) 
#print(y)
print((529*1j*(23)**(1/2))/(74438500*np.pi)) 