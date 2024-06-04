import numpy as np 

w = 0.057              #Define frequency of field
t0 = 0                 #Define intial time (t_ref)
tf = (2*np.pi/w)       #Define final time 
dt = 0.1             #Time steps

t_list = []

for ti in np.arange(t0,tf,dt):    # Maybe this dt should be bigger and 

    for tr in np.arange(t0,tf,dt):
        if tr <= ti:              # Should this just be less than? 
            continue
        t_eva = np.arange(ti,tr,dt).tolist()   # This dt should be smaller ? 
        #print(t_eva)
        t_list.append(t_eva)
        #np.array(t_list)
        #t_list = np.append(t_list,t_eva)

print((t_list[5][4]))