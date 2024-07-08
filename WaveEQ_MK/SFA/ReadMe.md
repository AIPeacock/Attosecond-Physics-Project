# Strong Field Approximation (SFA)

This code calculates the Time-dependent dipole moment as outlined in Lewenstein 1993 [https://journals.aps.org/pra/pdf/10.1103/PhysRevA.49.2117].

There are two functions to define the field being used to generate the dipole response:

`Field(t, F, w)` is used to generate a plane wave and `pulse_field(t,F,w,Nc)` generates a  $sin^2$ envelope pulse similar to the one used in [https://www.mdpi.com/2304-6732/10/10/1122].

Next a function to describe the Dipole Transition Matrix Element (DTME), `def hydroDTME(p, k):`, for the 1s orbital in hydrogen-like atomic as outlined in [https://journals.aps.org/pr/pdf/10.1103/PhysRev.34.109] is defined. 


Lastly a function is defined which converts individual time steps into integer numbers to be used for indexing, `def time_to_index(time_value, dt):`.

## Constants 
Next a series of constants which are related to the length and strength of pulse are defined in atomic units, $\hslash = m = e = 1$.

**Define constants**\
w =  _Frequency of field_ \
dt =  _Time steps_\
t0 =  _Initial time_\
Nc =  _Number of carrier cycles_\
tf = $Nc*(2 * np.pi / w)$  _Final time_\
N = int $((tf-t0)/dt)+1$ _Number of time steps/samples_ 
 
**Calculate field parameters**\
I0 = _Intensity_\
E0 = np.sqrt(I0)  _Electric field amplitude_\
Ip = 15.7 / 27.2  _Ionisation potential_ 

**Calculate Up, gamma, and kappa**\
Up = (E0^2)/(4*(w^2))  _Ponderomotive force/potential_\
gamma = np.sqrt(Ip / (2 * Up))  _Keldysh parameter_\
kappa = np.sqrt(2 * Ip)  _Kappa_\
epsilon = 1e-4 or 1/((10*kappa)^2)  _Comupational prefactor_ 

Lastly the harmonic cutoff is calculated using the experssion found in [https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.71.1994] and converted to harmonic order. 

w_c= Ip + 3.17*Up _cutoff_\
cutoff = w_c/w    _Harmonic order_

## Discretisation and generation of times 

An array of all possible time values for the ionisation and recombination of an electron is generated using the `np.meshgrid` function where each of the axes are defined using `np.linspace(t0, tf, N)` which generates a list of evenly spaced times from t0 to tf. N is used to select the number of elements to generate such that the spacing between adjacent values is dt steps. 

As a square mesh is generated, unphysical times are included such that an ionisation time (ti) may be greater(later) than a recombination time (tr) so to remove these the `np.triu_indices_from(ti_grid, k=-1)` is used which selects the indices of the values that are above the diagonal that is to the upper left of the central diagonal going from the bottom left to the top right. 

_In fact this isn't quite true but is corrected for by reassigning the lists to each other_ /
`tr_list = ti_grid[valid_indices]`  
`ti_list = tr_grid[valid_indices]`

## Pre-computation of Integrals 

In the SFA theory used in the Lewenstein model the intgeral for the time dependent dipole moment is given by;

$$
\textbf{D}(t) = i \int_{t}^{t0} \ dt'\int\ d\textbf{p} \textbf{d}*(\textbf{p}+\textbf{A}(t)) \exp{-iS(\textbf(p),t,t')} \textbf{F}(t') \textbf{d}(\textbf{p}+\textbf{A}(t'))+ c.c.
$$
