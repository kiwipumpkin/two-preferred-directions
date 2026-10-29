import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
J = json.load(open("../13_combined/combined.json"))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for k, col in (("A", "0.6"), ("B", "#3b6fb0")):
    d = np.load(f"../13_combined/null_{k}.npz"); lab = "null A: whole frame random" if k == "A" else "null B: CoE random, AoE fixed"
    axs[0].hist(d["t1"], bins=np.arange(-0.5, 13), density=True, histtype="step", lw=2, color=col, label=lab)
    axs[1].hist(d["t2"], bins=np.linspace(0, 90, 31), density=True, histtype="step", lw=2, color=col, label=lab)
axs[0].axvline(J["observed"]["T1"], color="#c0392b", lw=2.5, label="observed (11/12)"); axs[0].set_xlabel("T1: features matching in sign (of 12)")
axs[1].axvline(J["observed"]["T2"], color="#c0392b", lw=2.5, label="observed (14.8°)"); axs[1].set_xlabel("T2: ℓ = 4 axis to plane-of-evil normal (deg)")
for a in axs: a.set_ylabel("fraction of draws"); a.legend(fontsize=7); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/fig6_nulls.png", dpi=200); print("ok")
