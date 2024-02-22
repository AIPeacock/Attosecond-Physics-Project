import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fft import fft, ifft, fftshift
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm 

#Set up initally in real space

c = 2            #Wave speed
L = 40           #Length of propagation 
N = 1000         #Discritisation points
dz = L/N         #Spatial size
z = np.arange(-L/2,L/2,dz)

k = 2*np.pi*np.fft.fftfreq(N,d=dz)   # Define the discritised wavevectors

#Inital condition
u0= 1/np.cosh(z)            #In real space
u0ft= fft(u0)            #In Fourier space

u0ft_ri = np.concatenate((u0ft.real,u0ft.imag)) 


#Fourier domain simulation

dt = 0.0025
t = np.arange(0,100*dt,dt)

def waveeq(u0ft_ri, t, k, c):                                 #Defining the fourier transformed function we want to solve
    u0ft = u0ft_ri[:N] + (1J)*u0ft_ri[N :]   #Define u(k,t) 
    dAdt = -k**2 * c**2 * u0ft
    dAdt_ri = np.concatenate((dAdt.real,dAdt.imag)).astype('float64')
    return dAdt_ri

uft_ri = odeint(waveeq, u0ft_ri, t, args=(k,c))
uft = uft_ri[:,:N] + (1j)* uft_ri[:,N:]

u = np.zeros_like(uft)
for i in range(len(t)):
    u[i,:]=ifft(uft[i,:])

u=u.real

fig =plt.figure()
ax= fig.add_subplot(111,projection='3d',xlabel = 'Width (N)', ylabel = 'Length(z)', zlabel ='Amplitude',autoscalez_on = True)

ax.zaxis.label.set_position((-0.5, 0.5))

u_plot = u[0:-1:10,:]
for j in range(u_plot.shape[0]):
    ys = j*np.ones(u_plot.shape[1])
    ax.plot(z,ys,u_plot[j,:],color= cm.jet(j*20))

#plt.savefig('u_plot', bbox_inches='tight')
plt.figure()
plt.imshow(np.flipud(u), aspect=8)
plt.set_cmap('jet_r')
plt.show()