import numpy as np 
from numpy.fft import fft,ifftshift,fftshift,fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp

def Field(t,E0): 
    return np.cos(w*t)*(E0)

def hydroDTME(p,k):              # K is ionisation potential(Sure ? ), p is momentum
     DTME = (1j*8*((2*(k**5))**(1/2))*p)/(np.pi*(((p**2)+(k**2))**3))
     return (DTME)

w = 0.057              #Define frequency of field
t0 = 0                 #Define intial time (t_ref)
tf = 4*(2*np.pi/w)       #Define final time 
dt = 1                 #Time steps
I0 = (1*(10**14))/(3.51*(10**16))
E0 = I0**(1/2)
Ip=15.7/27.2

t_full=np.arange(t0,tf -dt,dt)              # Time for plotting, has -dt as needs to match length of Dipole
Dipole=[]
t_list=[]

                                                         # Are these in atomic units
Up = (E0**2)/(4*(w**2))                                    #Ponderomotive force/potential
gamma = (Ip/2*Up)**(1/2)                                 # Keldysh parameter
kappa = (2*Ip)**(1/2)                                    # Kappa ? 

for ti in np.arange(t0,tf,dt):                           #Loop for ionisation times
    
    for tr in np.arange(t0,tf,dt):                       #Loop for Recombination times
        if tr <= ti:
            continue                                     #Generates an array of times with each element including the increments between ti,tr
        
        t_eva = np.arange(ti,tr,dt).tolist()
        t_eval =np.round(t_eva,6)
        t_list.append(t_eval)

for i in range(0,len(t_list)):                           # Main dipole calc
    #print(i)
    if t_list[i][0] == t_list[i][-1]:
        continue
    At = integrate.cumulative_trapezoid(Field(t_list[i],E0),dx=dt,initial=0)  # Integrate the field to find Potential, Is it okay to always have inital = 0 even for stepping through fields ? 
    A_t = -1*At
    A_t2 = A_t * A_t
    sol = integrate.trapezoid(A_t,t_list[i],dx=dt)                                                       #Integral for A_t for all times

    solsquared = integrate.trapezoid(A_t2,t_list[i],dx=dt)                                               # Integral for A_t^2 for all times

    Ps = -(1/(t_list[i][-1]-t_list[i][0]))*sol                                                      # Calculating stationary momentum

    Sv = Ip*(t_list[i][-1]-t_list[i][0])+0.5*(Ps**2)*(t_list[i][-1]-t_list[i][0])+Ps*sol+0.5*solsquared     # Classical action 

    d_matrix_ion = hydroDTME(Ps+A_t[0],kappa)                                # Ionisation Dipole matrix elements contains (ps+A(t_ion)) 

    d_matrix_recol = np.conj(hydroDTME(Ps+A_t[-1],kappa))                     # Recollision Dipole Matrix Element contaion (ps+A(t_recol))
    
    prefactor1 =-1# -((2*np.pi*w*gamma)/((kappa*kappa)*(1+gamma*gamma)**(1/2)))       #Prefactor from derivative of action wrt ti

    prefactor2 = 1#((2*np.pi)/(1j*(t_list[i][-1]-t_list[i][0])))**(3/2)              #Prefactor from Momentum integration using saddle point

    Dipole = np.append(Dipole,1j*prefactor1*prefactor2*d_matrix_recol*d_matrix_ion*Field(t_list[i][0],E0)*np.exp(-1j*Sv))   #Calculation of dipole mom for all times(incl increments)

# np.savetxt('Dipolelist.txt',Dipole)

tmax = ((tf-t0)/dt) -1                                                       
i1=0
i2=0
i_sum = 0
Dipole_total = []
for i in range(int(tmax),-1,-1):                                                    #Summation of all dipole moms at individual ti times to generate array of Dipole Mom
    print(i)
    i2 = i2 + i 
    Dipole_total = np.append(Dipole_total,sum(Dipole[i1:i2]))
    # print('Splice i1',i1,'i2',i2,'Dipole',Dipole[i1:i2])
    i1 = i1 + i 

plt.figure(1)
plt.plot(t_full,np.real(Dipole_total))

Dipole_freq = fftshift(fft((Dipole_total)))                                #Fourier transform for spectrum, but not right1 ?
freq_axis = fftshift(fftfreq(t_full.shape[-1],d = dt))
D_spectrum = np.abs(Dipole_freq)**2

plt.figure(2)
plt.semilogy(freq_axis/w,D_spectrum)
plt.xlabel('Frequency')
#plt.xlim(-1,20)
plt.ylabel('Intensity')
plt.show()