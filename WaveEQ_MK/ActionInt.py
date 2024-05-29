import numpy as np 
from numpy.fft import fft,ifftshift,fftshift,fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp

w = 0.057              #Define frequency of vector potential
t0 = 0                 #Define intial time (t_ref)
tf = (2*np.pi/w)       #Define final time 
dt = 0.01              #Time steps
t_eval = np.arange(0,tf,dt)
Ip=4

def A_t(t,A):
    return np.sin(w*t)

def A_t2(t,A):
    return (np.sin(w*t))**2

sol = solve_ivp(A_t,[t0,tf],y0=[0],t_eval=t_eval)

solsquared = solve_ivp(A_t2,[t0,tf],y0=[0],t_eval=t_eval)

#v = Ip*(tf-t0)+0.5*(Ps**2)*(tf-t0)+Ps*sol.y[0]+0.5*solsquared.y[0] #This is for one t_ion 


print("t_eval",t_eval)

alpha = -(1/w)*(np.cos(w*t_eval)-np.cos(w*t0))
print("alpha = ",alpha)

plt.figure(1)
plt.plot(t_eval,alpha)
plt.title("analytical")

plt.figure(2)
plt.plot(t_eval,sol.y[0])
plt.title("numerical")

plt.figure(3)
at = A_t(t_eval,1000)
plt.plot(t_eval,at)
plt.show()















