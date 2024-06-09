import numpy as np
import matplotlib.pyplot as plt

E0= 1
w0= 1
E1= 0.1
w1= 0.1
v0 = 0
Period0 = 1/(w0/(2*np.pi))
Period1 = 1/(w1/(2*np.pi))


dt = 0.1
tf = 120.6
t = np.arange(0,tf,dt)
v=[]
plt.figure(1)
for t0 in np.arange(0,0.1,0.1):
    v = v0+ (E0/(w0))*(np.sin(w0*t)-np.sin(w0*t0))
    v = v + E1/w1 *(np.sin(w1*t)-np.sin(w1*t0))
    np.append(v,v)
    Ek = 1/2*v**2
    plt.plot(t/Period0,Ek, label = f"t0 ={t0}")
    plt.xlabel(r't (Cycles of Field E0)')
    plt.ylabel(r'Kinetic Energy')

ip = 1
Cutoff_w = Ek + ip


