import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fft import fft, ifft, fftshift
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm 


c = 10            #Wave speed
Lx = 40           #Length of propagation 
Nx = 1000         #Discritisation points
dx = Lx/Nx        #Spatial size
Ly= 40
Ny= 1000
dy=Ly/Ny
x = np.arange(-Lx/2,Lx/2,dx)
y= np.arange(-Ly/2,Ly/2,dy)

kx = 2*np.pi*np.fft.fftfreq(Nx,d=dx) 

ky = 2*np.pi*np.fft.fftfreq(Ny,d=dy)   # Define the discritised wavevectors

#Inital condition

u0x= 1/np.cosh(x)                    #In real space  can be any shape
u0y= 1/np.cosh(y)
u0ftx= fft(u0x)         
u0fty= fft(u0y)                            #In Fourier space

u0ft_rix = np.concatenate((u0ftx.real,u0ftx.imag))
u0ft_riy = np.concatenate((u0fty.real,u0fty.imag))  #Solve issues in odeint 

#Fourier domain simulation 

dt = 0.0025
t = np.arange(0,100*dt,dt)


def waveeq(u0ft_rix, u0ft_riy, t, kx, ky, c):                                 #Defining the fourier transformed function we want to solve
    u0ft_rix = u0ft_rix[:Nx] + (1J)*u0ft_rix[Nx :]   #Define u(k,t) 
    u0ft_riy = u0ft_rix[:Ny] + (1J)*u0ft_rix[Ny :] 
    dAdt = -kx**2 * c**2 * u0ftx + -ky**2 * c**2 * u0fty 
    dAdt_ri = np.concatenate((dAdt.real,dAdt.imag)).astype('float64')
    return dAdt_ri

uft_ri = odeint(waveeq, u0ft_rixy, t, args=(k,c))
uft = uft_ri[:,:N] + (1j)* uft_ri[:,N:]

u = np.zeros_like(uft)
for i in range(len(t)):
    u[i,:]=ifft(uft[i,:])

u=u.real