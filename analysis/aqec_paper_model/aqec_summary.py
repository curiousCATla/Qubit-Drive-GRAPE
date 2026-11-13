import sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = sys.argv[1]
rows = list(csv.DictReader(open("aqec_ideal_results.csv")))
b1 = np.load(glob.glob(P + "/Nxgates_AQEC_sims_X_*")[0])
b2 = np.load(glob.glob(P + "/Nx_gatesAQEC_sims_X_*")[0])
paper = {("B1", "ET"): b1["fidelity_ET"][1:], ("B1", "Ord"): b1["fidelity_L"][1:],
         ("B2", "ET"): b2["fidelity_ET"][1:], ("B2", "Ord"): b2["fidelity_L"][1:]}


def series(g, proto, key):
    if proto == "B1":
        r = [x for x in rows if x["gate"] == g and x["protocol"] == "B1"]
        return np.array([int(x["N"]) for x in r]), np.array([float(x[key]) for x in r])
    r = [x for x in rows if x["gate"] == g and x["protocol"] == "B2" and x["variant"] == key]
    return np.array([int(x["N"]) for x in r]), np.array([float(x["F"]) for x in r])


for g in ("ET", "Ord"):
    print(f"\n=== {g} gate ===")
    N, none = series(g, "B1", "none")
    _, idle = series(g, "B1", "idle")
    print(" N | none   idle2.2 |  B1: A      B      C      D     paper |  B2: A      B      C      D     paper")
    for n in (1, 2, 3, 5, 10, 15, 20):
        i = n - 1
        s = f"{n:2d} | {none[i]:.4f} {idle[i]:.4f} | "
        s += " ".join(f"{series(g, 'B1', v)[1][i]:.4f}" for v in "ABCD")
        pb1 = paper[("B1", g)]; s += f" {pb1[i]:.4f} |" if i < len(pb1) else "   -    |"
        s += " ".join(f"{series(g, 'B2', v)[1][i]:.4f}" for v in "ABCD")
        pb2 = paper[("B2", g)]; s += f" {pb2[i]:.4f}" if i < len(pb2) else "   -"
        print(s)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
col = {"ET": "#4c72b0", "Ord": "#c44e52"}
for ax, proto in zip(axes, ("B1", "B2")):
    for g in ("ET", "Ord"):
        N, none = series(g, "B1", "none")
        ax.plot(N, none, ":", color=col[g], label=f"{g}: gates only")
        for v, ls in (("A", "-"), ("D", "--")):
            n, F = series(g, proto, v)
            ax.plot(n, F, ls, color=col[g], label=f"{g}: ideal AQEC {'2.2 us lossy' if v == 'A' else 'instant'}")
        pf = paper[(proto, g)]
        ax.plot(np.arange(1, len(pf) + 1), pf, "o", ms=3.5, color=col[g], label=f"{g}: paper Fig. A6")
    ax.set_title({"B1": "N gates + one AQEC (A6b)", "B2": "N x (gate + AQEC) (A6d)"}[proto])
    ax.set_xlabel("number of gates N"); ax.grid(alpha=.3)
axes[0].set_ylabel("process fidelity"); axes[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("aqec_ideal_vs_paper.png", dpi=150)
print("saved aqec_ideal_vs_paper.png")
