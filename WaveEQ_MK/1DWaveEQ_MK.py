import numpy as np
import subprocess

dn = 0.001      # Define 
nmax = 200

param = np.loadtxt('paramet.prn')
dt = param[4, 0]
b = int(param[1, 0])
omega_las = param[10, 0]

E1 = np.loadtxt('E.prn')
a = len(E1)
E = np.zeros((b, 1))
E[:b, 0] = E1[:b]

N_T = int(round(b * dt * omega_las / (2 * np.pi)))
d_omega = 2 * np.pi / (dt * b)

w1 = np.arange(0, d_omega * (np.round(b / 2) - 1), d_omega)
w2 = np.arange(-d_omega * (np.round(b / 2)), -d_omega, d_omega)
w11 = -w1[::-1]
w22 = -w2[::-1]
w = np.zeros((b, 1))
w[:int(round(b / 2)), 0] = w11
w[int(round(b / 2)):b, 0] = w22

E0_w = np.fft.fft(E[:, 0])
E_w = np.zeros((b, 1), dtype=complex)
E_w[:, 0] = E0_w
sp_field = np.abs(E0_w) ** 2

ww = -w11
q = ww / omega_las

tt = np.arange(1, b + 1)
time = tt * dt
Tt = time / (2 * np.pi) * omega_las

sm = 0.0015
wm = 15
mask = 1 / (1 + np.exp(-(ww - d_omega * wm) / sm))
mask_ = 1 / (1 + np.exp((ww - (ww[-1] - d_omega * (wm - 1))) / sm))
mask_s = np.concatenate((mask, mask_))

Emasked = np.zeros((b, nmax))
d_t = np.zeros((b, nmax))
d_w = np.zeros((b, nmax))

for n in range(nmax):
    n += 1
    print(n)
    subprocess.run(['Ar1.exe'])  # assuming Ar1.exe is an external program to be executed

    dtmp = np.loadtxt('d.prn')
    dtmp = np.pad(dtmp, (0, b - len(dtmp)), 'constant')
    d_t[:, n - 1] = dtmp[:b] + E[:b, n - 1]

    d_w[:, n - 1] = np.fft.fft(d_t[:, n - 1])
    d_w[:, n - 1] *= mask_s[:, 0]

    E_w[:, n] = E_w[:, n - 1]
    E_w[1:, n] -= 1j * dn * d_w[1:, n - 1] / w[1:, 0]
    E[:, n] = np.real(np.fft.ifft(E_w[:, n]))

    Er = E[:, n]
    Er = np.pad(Er, (0, a - b), 'constant')

    np.savetxt('E.prn', Er)

s_f0 = np.abs(E_w) ** 2
sp_n = s_f0[:, :201:25]