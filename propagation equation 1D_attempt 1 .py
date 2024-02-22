import numpy as np 
import matplotlib.pyplot as plt

nz= 10 # Discretising spatial propagation 
nt= 10 # Discretising temporal propagation 
t0 = 0.0 #inital time
tf = 2.0 #final time
z0 = 0 # Inital position 
zf = 10 # final position 

n = (nz+1,nt+1)    # Defining size of grid of points 
E = np.zeros(n)

def ddz(E, dz):
     i
     ((E[i+1] - 2*E[i] + E[i-1])/( dz ** 2))    # finite difference method where E[i-1] is the previous step

def ddt(E, dt):
     ((E[j+1] - 2*E[j] + E[j-1])/(dt**2)) 
     
for i in np.arange(0,nz+1,1):
    dE2dt = ddz(E[0][0],4) 
