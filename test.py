import numpy as np
from grape_core import *
dt = 0.002 #in μs
n_t, n_c = 2, 24
A, B = make_ops(n_t, n_c)
H0, Hc = make_hamiltonian(n_t, n_c)

N = 250
psi_i = basis_state(n_t, n_c, 0, 0) # initial state |g,0⟩
psi_f = basis_state(n_t, n_c, 0, 6) # target state |g,6⟩

print("Hermitian H0:", np.allclose(H0, H0.conj().T)) # Check if H0 is Hermitian

offdiag = H0 - np.diag(np.diag(H0)) # To confind H0 has nothing off the diagonal 
print("max off-diagonal element:",np.max(np.abs(offdiag)))

d=np.diag(H0) #Check transmon transition frequency E|e,c) − E|g,c) for #c photons in cavity
print("Kerr spot-check d[2] vs Kerr:", d[2], Kerr) #Check if the Kerr term is correct for 2 photons in cavity

transition = (d[n_c+6]-d[6])/(2*np.pi) #transition frequency for 6 photons in cavity
print("Transition frequency for 6 photons in cavity (MHz):", transition)

for i, H in enumerate(Hc, start=1):
    print(f"H{i} Hermitian:", np.allclose(H, H.conj().T)) # Check if H1, H2, H3, H4 are Hermitian

#Check zero control can't move eigen state
#u0 = np.zeros((N, 4)) # initial controls (all zeros)
#print("F(zero -> |g,0>):", fidelity(u0, H0, Hc, psi_i, psi_i,  dt))  # expect 1.0
#print("F(zero -> |g,6>):", fidelity(u0, H0, Hc, psi_i, psi_f, dt))  # expect 0.0

#Check norm preservation of the hamiltonian in random drive 
#rng = np.random.default_rng(0) # random number generator
#u_rand = 0.3 * rng.standard_normal((N, 4))
#F, psi_end = fidelity(u_rand, H0, Hc, psi_i, psi_f, dt)
#print ("norm after random drive:", np.linalg.norm(psi_end)) # expect 1.0

rng = np.random.default_rng(0)
u = 3.0 * rng.standard_normal((N, 4))            

print(optimize_controls(H0, Hc, psi_i, psi_f, dt, N, u))