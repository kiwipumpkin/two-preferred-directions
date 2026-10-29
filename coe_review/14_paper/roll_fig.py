import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
j = json.load(open(PKG + "/coe_review/09_values_blend/compare/compare_2d.json"))
r = np.array(sorted((int(k), v) for k, v in j["roll_scan"].items())); rb = json.load(open(PKG + "/coe_review/09_values_blend/compare/ring_blend.json"))
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(r[:, 0], r[:, 1], "k-", lw=2, label="ring model vs Migkas 2-D map")
ax.axvline(60.18, color="#1b7f3b", ls="--", lw=2, label="roll fixed by the Axis of Evil (60.18°)")
ax.axvline(r[np.argmax(r[:, 1]), 0], color="#c0392b", ls=":", lw=2, label=f"best roll from the map ({int(r[np.argmax(r[:, 1]), 0])}°)")
ax.set_xlabel("roll about the CoE axis (deg)"); ax.set_ylabel("weighted correlation r"); ax.set_xlim(0, 355); ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="lower left"); ax.set_title("Roll preferred by the cluster map (best roll is 50° at every blend width 0–90°)", fontsize=9)
fig.tight_layout(); fig.savefig("figs/fig2_roll.png", dpi=200); print("ok")
