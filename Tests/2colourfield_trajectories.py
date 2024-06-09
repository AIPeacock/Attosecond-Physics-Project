import numpy as np
import matplotlib.pyplot as plt

#Atomic units, h = e = me = 1 
tf = 6.4                    # s 
dt = 0.1                    # s 

t = np.arange(0,tf,dt)
Iontimes = []

e = 1          # C
E0 = 1                    # V/m
w0 = 1                      # 2pif, units of 1/s (Hz) Time period = 
Period0 = 1/(w0/(2*np.pi))
#print(Period0)
E1 = E0*0.1
w1 = 0.1 
Period1 = 1/(w1/(2*np.pi))
#print(Period1)

def Traj_2(E0,w0,E1,w1,t,t0):
        xt = (E0/(w0**2))*(np.cos(w0*t)-np.cos(w0*t0))
        xt = xt + (E0/w0) * np.sin(w0*t0)*(t-t0)
        xt = xt + (E1/(w1**2))*(np.cos(w1*t)-np.cos(w1*t0))
        xt = xt + (E1/w1) * np.sin(w1*t0)*(t-t0)
        xt = e*xt
        return(xt)

#plt.figure(1)
#for t0 in np.arange(0,0.6,0.1):
#    xt = (E0/(w0**2))*(np.cos(w0*t)-np.cos(w0*t0))
#    xt = xt + (E0/w0) * np.sin(w0*t0)*(t-t0)
#    xt = xt + (E1/(w1**2))*(np.cos(w1*t)-np.cos(w1*t0))
#    xt = xt + (E1/w1) * np.sin(w1*t0)*(t-t0)
#    xt = e*xt
#    plt.plot(t/Period0,xt)
#    plt.xlabel(r't (Cycles of Field E0)')
#    plt.ylabel(r'x[t] ($a_{0}$)')
#    plt.title(rf'Electron Trajectories (2 colour field, E0 = {E0} + E1 = {E0} )')

# plt.figure(2)
# w = 0 
# for t1 in np.arange(0,0.6,0.1):
#    xt1 = (E0/(w0**2))*(np.cos(w0*t)-np.cos(w0*t1))
#    xt1 = xt1 + (E0/w0) * np.sin(w0*t1)*(t-t1)
#    xt1 = e*xt1
#    Iontimes = np.append(Iontimes,round(t1/Period0,2))
#    plt.plot(t/Period0,xt1, label = f"t0 ={Iontimes[w]}")
#    w = w + 1
#    plt.xlabel(r't (Cycles)')
#    plt.ylabel(r'x[t] ($a_{0}$)')
#    plt.title(r'Electron Trajectories (1 colour field, E0)')
# print(Iontimes)
# plt.legend(loc = 'best')

scaling = 3
t0 = 0
tf = 200
phase = 0
phase0 = 2*np.pi/3
t = np.arange(0,tf,dt)
plt.figure(3)
for E1 in np.linspace(E1,E1*scaling,5):
    xt2 = (E0/(w0**2))*(np.cos(w0*t+phase0)-np.cos(w0*t0+phase0))
    xt2 = xt2 + (E0/w0) * np.sin(w0*t0+phase0)*(t-t0)
    xt2 = xt2 + (E1/(w1**2))*(np.cos(w1*t+phase)-np.cos(w1*t0+phase))
    xt2 = xt2 + (E1/w1) * np.sin(w1*t0+phase)*(t-t0)
    xt2 = e*xt2
    plt.plot(t/Period0,xt2,label = f"E1 ={E1}")
    plt.xlabel(r't (Cycles of Field E0)')
    plt.ylabel(r'x[t] ($a_{0}$)')
    plt.title(rf'Electron Trajectories (2 colour field, E0 = {E0} + E1)')
    #plt.axis([0,2,-0.25,0.25])
    plt.legend(loc= 'best')