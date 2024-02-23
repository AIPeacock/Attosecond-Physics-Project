import numpy as np
import matplotlib.pyplot as plt



    
    

tf = 12.4
dt = 0.1

t = np.arange(0,tf,dt)

e = -1.6*(10**-19)
E0 = 1
w0 = 1 
E1 = 0.1
w1 = 0.1 
plt.figure(1)
for t0 in np.arange(0,0.6,0.1):
    xt = (E0/(w0**2))*(np.cos(w0*t)-np.cos(w0*t0))
    xt = xt + (E0/w0) * np.sin(w0*t0)*(t-t0)
    xt = xt + (E1/(w1**2))*(np.cos(w1*t)-np.cos(w1*t0))
    xt = xt + (E1/w1) * np.sin(w1*t0)*(t-t0)
    xt = e*xt
    plt.plot(t,xt)
    plt.xlabel(r't (hbar/$E_h$)')
    plt.ylabel(r'x[t] ($a_{0}$)')
    plt.title(r'Electron Trajectroies (2 colour field, E0 = 1 + E1 = 0.1 )')

plt.figure(2)
for t1 in np.arange(0,0.6,0.1):
    xt1 = (E0/(w0**2))*(np.cos(w0*t)-np.cos(w0*t1))
    xt1 = xt1 + (E0/w0) * np.sin(w0*t1)*(t-t1)
    xt1 = e*xt1
    plt.plot(t,xt1)
    plt.xlabel(r't (hbar/$E_h$)')
    plt.ylabel(r'x[t] ($a_{0}$)')
    plt.title(r'Electron Trajectroies (1 colour field, E0)')
