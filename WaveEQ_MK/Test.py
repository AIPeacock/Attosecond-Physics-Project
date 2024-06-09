import numpy as np 
from numpy.fft import fft,ifftshift,fftshift,fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp

w = 0.057              #Define frequency of field
t0 = 0                 #Define intial time (t_ref)
tf = (2*np.pi/w)       #Define final time 
dt = 1             #Time steps
I0 = (1*(10**14))/(3.51*(10**16))
E0 = I0**(1/2)

def Field(t,F): 
    return np.sin(w*t)*(F)

t_full=np.arange(t0,tf-dt,dt)
t_list = []
t_array = np.zeros((int(tf),int(tf)))

for ti in np.arange(t0,tf,dt):    # Maybe this dt should be bigger and 

    for tr in np.arange(t0,tf,dt):
        if tr <= ti:              # Should this just be less than? 
            continue
        t_eva = np.arange(ti,tr,dt).tolist()   # This dt should be smaller ? 
        #print(t_eva)
        t_list.append(t_eva)
        #np.array(t_list)
        #t_list = np.append(t_list,t_eva)

Field_list = Field(t_full,E0)

A_t = -integrate.cumulative_trapezoid(Field_list,t_full,dx=dt,initial=0)






        