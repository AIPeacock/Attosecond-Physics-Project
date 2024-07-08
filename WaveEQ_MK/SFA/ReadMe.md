#Strong Field Approximation

This code calculates the Time-dependent dipole moment as outlined in Lewenstein 1993 [https://journals.aps.org/pra/pdf/10.1103/PhysRevA.49.2117].

There are two functions to define the field being used to generate the dipole response:

`Field(t, F, w)` is used to generates a plane wave and `pulse_field(t,F,w,Nc)` generates a sin^2 envelope pulse similar to the one used in [https://www.mdpi.com/2304-6732/10/10/1122].

Next a function to describe the Dipole Transition Matrix Element (DTME) for the 1s orbital in hydrogen-like atomic as outlined in [https://journals.aps.org/pr/pdf/10.1103/PhysRev.34.109?casa_token=7fBH4TTYD5cAAAAA%3A5jQmXIZAM52w0bL_NilNRsCbu8CBBLeV2yP3ONms65hp0sYosq26DZWZLW30nHGWr6vWmqxgbUMVtg] is defined. 

`def hydroDTME(p, k):`
    `return (1j * 8 * np.sqrt(2 * (k**5)) * p) / (np.pi * (((p**2) + (k**2))**3))`

Lastly a function is defined which converts individual time steps into intger numbers to be used for indexing, 
`def time_to_index(time_value, dt):`
    `return int(np.round(time_value/dt))`
