"""Idealised AQEC variants on the paper's model, B1 and B2 protocols.

AQEC step (cavity = c, transmon = q, 2 levels as in the paper):
  0. trace out q (any transmon excitation left by the gate is forgiven: optimistic)
  [B only] 1 us lossy idle BEFORE the map (pulse losses get corrected: optimistic bracket)
  1. ideal map: K_g = 1 - |E0><E0| - |E1><E1| (q stays g),  K_e = |L0><E0| + |L1><E1| (q -> e)
  2. pre-twist each branch by exp(+i H0_q tau): cancels Kerr (g branch) and Kerr + chi (e branch)
     over the lossy window that follows, exactly as the paper says its pulse does
  3. lossy idle for tau (their c_ops, transmon held in its branch state)
  4. ideal reset: trace q, put it in g
Variants: A tau = 2.2 us (your outline: 1 us pulse + 1.2 us readout, all losses after the map)
          B 1 us idle, map, tau = 1.2 us
          C tau = 1.0 us (no readout window simulated)
          D tau = 0 (instant perfect AQEC)
Baselines: none = gates only;  idle = gates + 2.2 us lossy idle with Kerr cancelled for free.
"""
import sys
import numpy as np
from aqec_sim import Gate, logical_inputs, process_fidelity, apply_gate

D = sys.argv[1] if len(sys.argv) > 1 else "."


class AQEC:
    def __init__(self, G):
        self.G = G
        nc, nt = G.nc, G.nt
        L0, L1, E0, E1 = G.kets()
        self.Kg = np.eye(nc) - np.outer(E0, E0.conj()) - np.outer(E1, E1.conj())
        self.Ke = np.outer(L0, E0.conj()) + np.outer(L1, E1.conj())
        self.h = [np.real(np.diag(G.H0))[q::nt] for q in range(nt)]    # cavity diag per transmon level
        self._S = {}

    def S(self, tau):
        if tau not in self._S:
            self._S[tau] = self.G.idle_super(tau)
        return self._S[tau]

    def trq(self, R):
        m = R.shape[0]; nc, nt = self.G.nc, self.G.nt
        return np.einsum("mnqkq->mnk", R.reshape(m, nc, nt, nc, nt))

    def embed(self, Rc, q):
        P = np.zeros((self.G.nt, self.G.nt)); P[q, q] = 1
        return np.array([np.kron(r, P) for r in Rc])

    def twist(self, Rc, q, tau):
        u = np.exp(1j * self.h[q] * tau)
        return u[None, :, None] * Rc * u.conj()[None, None, :]

    def lossy_idle_g(self, Rc, tau):
        """Lossy idle of a cavity state with q in g, Kerr phase cancelled up front."""
        R = self.G.apply_super(self.S(tau), self.embed(self.twist(Rc, 0, tau), 0))
        return self.trq(R)

    def step(self, R, variant):
        Rc = self.trq(R)
        if variant == "B":
            Rc = self.lossy_idle_g(Rc, 1.0)
        tau = {"A": 2.2, "B": 1.2, "C": 1.0, "D": 0.0}[variant]
        sg = np.einsum("ij,mjk,lk->mil", self.Kg, Rc, self.Kg.conj())
        se = np.einsum("ij,mjk,lk->mil", self.Ke, Rc, self.Ke.conj())
        if tau > 0:
            J = self.embed(self.twist(sg, 0, tau), 0) + self.embed(self.twist(se, 1, tau), 1)
            Rc = self.trq(self.G.apply_super(self.S(tau), J))
        else:
            Rc = sg + se
        return self.embed(Rc, 0)

    def idle(self, R, tau=2.2):
        return self.embed(self.lossy_idle_g(self.trq(R), tau), 0)


def run(g, nmax=20):
    import time
    G = Gate(f"{D}/{g}_X_gate.npz")
    A = AQEC(G)
    R0 = logical_inputs(G)
    rows = []
    # lockstep stack: block 0 = B1 chain (gates only), blocks 1..4 = B2 chains for A, B, C, D
    R = np.concatenate([R0] * 5)
    t0 = time.time()
    for N in range(1, nmax + 1):
        R = apply_gate(G, R)
        B1 = R[0:4]
        row = dict(gate=g, protocol="B1", N=N, none=process_fidelity(G, B1, N),
                   none_traced=process_fidelity(G, A.embed(A.trq(B1), 0), N),
                   idle=process_fidelity(G, A.idle(B1), N))
        for v in "ABCD":
            row[v] = process_fidelity(G, A.step(B1, v), N)
        rows.append(row)
        for i, v in enumerate("ABCD"):
            blk = slice(4 * (i + 1), 4 * (i + 2))
            R[blk] = A.step(R[blk], v)
            rows.append(dict(gate=g, protocol="B2", N=N, variant=v, F=process_fidelity(G, R[blk], N)))
        print(g, N, f"{time.time()-t0:.0f}s", {k: round(row[k], 4) for k in ("none", "idle", "A", "B", "C", "D")},
              flush=True)
    return rows


if __name__ == "__main__":
    import csv
    allrows = run("ET") + run("Ord")
    keys = sorted({k for r in allrows for k in r})
    with open("aqec_ideal_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    print("wrote aqec_ideal_results.csv", len(allrows), "rows")
