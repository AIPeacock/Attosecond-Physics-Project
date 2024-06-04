import numpy as np 
from numpy.fft import fft,ifftshift,fftshift,fftfreq
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
from scipy.integrate import solve_ivp

def Field(t,F): 
    return np.cos(w*t)

def hydroDTME(p,k):              # K is ionisation potential, p is momentum
     DTME = (1j*8*((2*(k**5))**(1/2))*p)/(np.pi*(((p**2)+(k**2))**3))
     return (DTME)

w = 0.057              #Define frequency of field
t0 = 0                 #Define intial time (t_ref)
tf = (2*np.pi/w)       #Define final time 
dt = 1               #Time steps
Ip=-15.7/13.6/2
Dipole=[]
t_list=[]


for ti in np.arange(t0,tf,dt):                           #Loop for ionisation times
    
    for tr in np.arange(t0,tf,dt):                       #Loop for Recombination times
        if tr <= ti:
            continue
        
        t_eva = np.arange(ti,tr,dt).tolist()
        t_eval =np.round(t_eva,6)
        t_list.append(t_eval)


for i in range(0,len(t_list)):
    print(i)
    if t_list[i][0] == t_list[i][-1]:
        continue
    At = solve_ivp(Field,[round(t_list[i][0],6),round(t_list[i][-1],6)],y0=[0],t_eval=t_list[i])
    A_t = -1*At.y[0]
    A_t2 = A_t * A_t

    sol = integrate.trapezoid(A_t,t_list[i])

    solsquared = integrate.trapezoid(A_t2,t_list[i])

    Ps = -(1/(t_list[i][-1]-t_list[i][0]))*sol                                        # Calculating stationary momentum

    Sv = Ip*(t_list[i][-1]-t_list[i][0])+0.5*(Ps**2)*(t_list[i][-1]-t_list[i][0])+Ps*sol+0.5*solsquared                   # This is for one t_ion, Does this need to be a number for each time t_ion so does generate an array but for steps through t_ion

    d_matrix_ion = hydroDTME(Ps+A_t[0],Ip)                                # Ionisation Dipole matrix elements contains (ps+A(t_ion)) 

    d_matrix_recol = np.conj(hydroDTME(Ps+A_t[-1],Ip))                     # Recollision Dipole Matrix Element contaion (ps+A(t_recol))

    Dipole = np.append(Dipole,d_matrix_recol*d_matrix_ion*Field(t_list[i][0],0)*np.exp(-1j*Sv))











# print('ti=',ti,'tr',tr,'t_eval',t_eval)
        # if np.any(t_eval< min(ti,tr)):
        #      print('its the minimum,ti',ti,'t_eval',t_eval)
        # elif np.any(t_eval> max(ti,tr)):
        #      print('its the maximum,tr',tr,'t_eval',t_eval)
        # else:
        #      ()

#print('t0',t0,'tf',tf,'ti',ti,'tr',tr)

# plt.figure(1)
# plt.plot(t_eval,A_t.y[0])
# plt.plot(t_eval,A_t2)
# plt.title('Vector potential')

# A_tt= interpolate.BarycentricInterpolator(xi=t_eval,yi=A_t.y[0])
# plt.figure(4)
# plt.plot(t_eval,A_tt.__call__(t_eval))