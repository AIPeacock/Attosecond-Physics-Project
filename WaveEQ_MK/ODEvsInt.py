import numpy as np 
from numpy.fft import fft,ifftshift,fftshift,fftfreq
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp
import time 


w = 0.057              #Define frequency of vector potential
t0 = 0                 #Define intial time (t_ref)
tf = (2*np.pi/w)              #Define final time 
dt = 0.01              #Time steps
t_eval = np.arange(0,tf,dt)     #Generate an array of times to evaluate the function at
start = time.time()
def A_t(t,A):                   #Define func to be solved (This is for the ODE situation)
    return np.sin(w*t)


sol = solve_ivp(A_t,[t0,tf],y0=[0],t_eval=t_eval)   #Solve the ode da/dt = A[a,t] using an inital value, a(t0)= 0 
total = 0
for i in range(0,len(sol.y[0])):
    total = total + sol.y[0][i]


plt.figure(1)
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Solution of ODE, A_t')

end = time.time()
print ("the length to run this script is:", end-start)

def A_t2(t):                #Define func to be solved (This is for the Integral situation)
    return np.sin(w*t)

y_values = A_t2(t_eval)   #generate an array of y values for the func defined previously

sol2= integrate.cumtrapz(y_values,dx=dt,initial = 0)            #Solve the integral a = int_t0^tf (A(t)) dt 
total2 = 0 
for i in range (0, len(sol2)):
    total2 = total2 + sol2[i]
print("total =",total2, "total2 = ",total2)


plt.figure(2)
plt.plot(t_eval,sol2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Solution of integral,A_t')


def A_tsquared(t,A):
    return ((np.sin(w*t))**2)

solsquared = solve_ivp(A_tsquared,[t0,tf],y0=[0],t_eval=t_eval)

plt.figure(3)
plt.plot(solsquared.t, solsquared.y[0])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Solution of ODE,A_t^2')

def A_t2squared(t):
    return ((np.sin(w*t))**2)

y_valuessquared = A_t2squared(t_eval)

sol2= integrate.cumtrapz(y_valuessquared,dx=dt,initial = 0)

plt.figure(4)
plt.plot(t_eval,sol2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Solution of integral,A_t^2')
plt.show()