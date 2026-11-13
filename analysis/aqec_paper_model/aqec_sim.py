"""Numpy replication of Roy et al.'s gate model + idealised AQEC variants.

Gate model = their notebook cells 152-155 / utilities.build_ham:
  H(t) = H0 + Re e(t) (a+a') + Im e(t) (-i(a-a')) + Re W(t) (q+q') + Im W(t) (-i(q-q'))
  e, W = flattop_gaussian_pulse(best_amps): clip each quadrature to max_amp, 12-point Gaussian rise/fall.
  c_ops = sqrt(1/T1q) q, sqrt(1/Tphiq) q'q, sqrt(1/T1c) a, sqrt(1/Tphic) a'a  (their exact forms).
Ordering: cavity (x) transmon, index n*nt + q.
"""
import numpy as np
from scipy.linalg import expm

T1q, T2q, T1c, T2c = 70.0, 30.0, 180.0, 290.0
Tphiq = 1 / (1 / T2q - 1 / (2 * T1q))
Tphic = 1 / (1 / T2c - 1 / (2 * T1c))


def flattop(x, amp):
    n = len(x)
    rp = int(250 * 0.05)                       # = 12, as in their code
    s = rp / 5.0
    rise = np.exp(-np.linspace(-5 * s, 0, rp) ** 2 / (2 * s * s))
    fall = np.exp(-np.linspace(0, 5 * s, rp) ** 2 / (2 * s * s))
    env = np.concatenate([rise, np.ones(n - 2 * rp), fall])
    return np.clip(x, -amp, amp) * env


class Gate:
    def __init__(self, path, amps_key="best_amps"):
        z = np.load(path)
        self.nc, self.nt = int(z["ncav"]), int(z["ntr"])
        self.d = self.nc * self.nt
        self.H0 = z["H0"].astype(complex)
        A = z[amps_key].astype(complex)
        amp = float(z["max_amp"])
        self.re0, self.im0 = flattop(A[0].real, amp), flattop(A[0].imag, amp)
        self.re1, self.im1 = flattop(A[1].real, amp), flattop(A[1].imag, amp)
        self.dt = float(z["tpulse"][1] - z["tpulse"][0])
        self.psi0 = z["psi0"][:, :, 0]
        self.exp_ops = z["exp_ops"]
        nc, nt = self.nc, self.nt
        a = np.diag(np.sqrt(np.arange(1, nc)), 1)
        q = np.diag(np.sqrt(np.arange(1, nt)), 1)
        Ic, It = np.eye(nc), np.eye(nt)
        self.a, self.q = np.kron(a, It), np.kron(Ic, q)
        self.Xa = self.a + self.a.conj().T
        self.Ya = -1j * (self.a - self.a.conj().T)
        self.Xq = self.q + self.q.conj().T
        self.Yq = -1j * (self.q - self.q.conj().T)
        nq = self.q.conj().T @ self.q
        na = self.a.conj().T @ self.a
        self.cops = [np.sqrt(1 / T1q) * self.q, np.sqrt(1 / Tphiq) * nq,
                     np.sqrt(1 / T1c) * self.a, np.sqrt(1 / Tphic) * na]
        self.cdc = sum(c.conj().T @ c for c in self.cops)

    def H(self, k):
        return (self.H0 + self.re0[k] * self.Xa + self.im0[k] * self.Ya
                + self.re1[k] * self.Xq + self.im1[k] * self.Yq)

    def closed_fidelity(self):
        psi = self.psi0.T.copy()
        for k in range(len(self.re0)):
            psi = expm(-1j * self.H(k) * self.dt) @ psi
        return np.mean([np.real(psi[:, i].conj() @ self.exp_ops[i] @ psi[:, i])
                        for i in range(psi.shape[1])])

    def lind(self, H, R, cops=None, on=True):
        Heff = H - 0.5j * self.cdc if on else H
        out = -1j * (Heff @ R - R @ Heff.conj().T)
        if on:
            for c in self.cops:
                out = out + c @ R @ c.conj().T
        return out

    def run_gate(self, R, nsub=20, on=True):
        """Propagate a stack R (m,d,d) through the gate (RK4, nsub substeps per 1 ns)."""
        h = self.dt / nsub
        for k in range(len(self.re0)):
            H = self.H(k)
            for _ in range(nsub):
                k1 = self.lind(H, R, on=on)
                k2 = self.lind(H, R + 0.5 * h * k1, on=on)
                k3 = self.lind(H, R + 0.5 * h * k2, on=on)
                k4 = self.lind(H, R + h * k3, on=on)
                R = R + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return R

    # ---- idle superoperator (constant H0), column-stacked vec ----
    def idle_super(self, tau, on=True):
        d = self.d
        I = np.eye(d)
        Heff = self.H0 - 0.5j * self.cdc if on else self.H0
        L = -1j * (np.kron(I, Heff) - np.kron(Heff.conj(), I))
        if on:
            for c in self.cops:
                L = L + np.kron(c.conj(), c)
        return expm(L * tau)

    def apply_super(self, S, R):
        m, d, _ = R.shape
        v = R.transpose(0, 2, 1).reshape(m, d * d)          # column-stack
        return (v @ S.T).reshape(m, d, d).transpose(0, 2, 1)

    # ---- code words ----
    def cav(self, n):
        v = np.zeros(self.nc, complex); v[n] = 1; return v

    def kets(self):
        L0 = (self.cav(0) + self.cav(4)) / np.sqrt(2); L1 = self.cav(2)
        E0, E1 = self.cav(3), self.cav(1)
        return L0, L1, E0, E1

    def joint(self, vc, qlev=0):
        vq = np.zeros(self.nt, complex); vq[qlev] = 1
        return np.kron(vc, vq)


def logical_inputs(G):
    L0, L1, _, _ = G.kets()
    basis = [G.joint(L0), G.joint(L1)]
    return np.array([np.outer(basis[i], basis[j].conj()) for i in range(2) for j in range(2)])


def process_fidelity(G, R, n_gates):
    """F_pro of the stack R (inputs |i><j|, i,j in {0_L,1_L}) against X^n, transmon must be in g."""
    L0, L1, _, _ = G.kets()
    t = [G.joint(L0), G.joint(L1)]
    if n_gates % 2:
        t = t[::-1]
    F = 0
    for idx, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        F += np.real(t[i].conj() @ R[idx] @ t[j])
    return F / 4


np.seterr(divide="ignore", over="ignore", invalid="ignore")   # Accelerate matmul noise (benign)


def _chunk_super(path, k0, k1, on=True):
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    G = Gate(path)
    d = G.d; I = np.eye(d)
    S = np.eye(d * d, dtype=complex)
    jump = sum(np.kron(c.conj(), c) for c in G.cops) if on else 0
    for k in range(k0, k1):
        Heff = G.H(k) - (0.5j * G.cdc if on else 0)
        L = -1j * (np.kron(I, Heff) - np.kron(Heff.conj(), I)) + jump
        S = expm(L * G.dt) @ S
    return S


def gate_super(path, n_jobs=8, on=True):
    from joblib import Parallel, delayed
    G = Gate(path); N = len(G.re0)
    edges = np.linspace(0, N, n_jobs + 1).astype(int)
    parts = Parallel(n_jobs=n_jobs)(delayed(_chunk_super)(path, edges[i], edges[i + 1], on)
                                    for i in range(n_jobs))
    S = np.eye(G.d ** 2, dtype=complex)
    for P in parts:
        S = P @ S
    return S


# ---- fast exact gate action: sparse Liouvillian, expm_multiply per 1 ns step ----
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


def _parts(G):
    d = G.d; I = sp.identity(d, format="csr", dtype=complex)
    comm = lambda H: -1j * (sp.kron(I, sp.csr_matrix(H)) - sp.kron(sp.csr_matrix(H.T), I))
    C = G.cdc
    L0 = comm(G.H0) - 0.5 * (sp.kron(I, sp.csr_matrix(C)) + sp.kron(sp.csr_matrix(C.T), I))
    for c in G.cops:
        L0 = L0 + sp.kron(sp.csr_matrix(c.conj()), sp.csr_matrix(c))
    return [L0.tocsr()] + [comm(X).tocsr() for X in (G.Xa, G.Ya, G.Xq, G.Yq)]


def apply_gate(G, R):
    if not hasattr(G, "_parts"):
        G._parts = _parts(G)
    L0, La, Lb, Lc, Ld = G._parts
    m, d, _ = R.shape
    V = R.transpose(0, 2, 1).reshape(m, d * d).T.copy()
    for k in range(len(G.re0)):
        L = L0 + G.re0[k] * La + G.im0[k] * Lb + G.re1[k] * Lc + G.im1[k] * Ld
        V = expm_multiply(L * G.dt, V)
    return V.T.reshape(m, d, d).transpose(0, 2, 1)
