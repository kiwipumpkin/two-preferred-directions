"""One chain: Migkas 2-D map -> our ring model -> data - model -> l = 4 part of the difference."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import numpy as np, healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open(PKG + "/coe_review/09_values_blend/compare/two_scale.py").read()
G = {"__file__": PKG + "/coe_review/09_values_blend/compare/two_scale.py", "__name__": "chain"}
exec(src.split("out = {}")[0], G); exec(src[src.index("# ---------- figures"):src.index("def scaled(")], G)
Amap, n, ng, blend, linfit, wr, data_g, mpanel, LO, LA, ell, bb = (G[k] for k in ("Amap", "n", "ng", "blend", "linfit", "wr", "data_g", "mpanel", "LO", "LA", "ell", "bb"))
L = blend(50, n); S = blend(20, n)-L; beta, fitA = linfit([L, S]); r = wr(fitA)
Lg = blend(50, ng); model_g = beta[0]+beta[1]*Lg+beta[2]*(blend(20, ng)-Lg)
NS, LMAX = 16, 6
res = 100*(Amap-fitA); alm = hp.map2alm(res-res.mean(), lmax=LMAX); a4 = np.zeros_like(alm)
for m in range(5): a4[hp.Alm.getidx(LMAX, 4, m)] = alm[hp.Alm.getidx(LMAX, 4, m)]
l4_g = hp.get_interp_val(hp.alm2map(a4, NS, lmax=LMAX), ell, bb, lonlat=True)
fig = plt.figure(figsize=(18, 12)); fig.subplots_adjust(hspace=0.32, wspace=0.08)
def cbar(ax, pc, lab):
    c = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, extend="both"); c.set_label(lab, fontsize=9)
ax0, pc = mpanel(fig, (2, 2, 1), data_g, "① Migkas 2-D map (θ = 75° cones)"); cbar(ax0, pc, "A/A_all")
ax, pc = mpanel(fig, (2, 2, 2), model_g, f"② Our ring model: blend σ = 50° + finer rings 20°, AoE roll 60.18°   (r = {r:.2f} with ①)"); cbar(ax, pc, "A/A_all")
ax, pc = mpanel(fig, (2, 2, 3), 100*(data_g-model_g), "③ Difference ① − ② (%)", norm=matplotlib.colors.Normalize(-8, 8)); cbar(ax, pc, "%")
ax, pc = mpanel(fig, (2, 2, 4), l4_g, "④ ℓ = 4 part of ③ (%) — 2.5× noise power, p = 0.016", norm=matplotlib.colors.Normalize(-3, 3)); cbar(ax, pc, "%")
for x0, y0, x1, y1 in ((0.49, 0.75, 0.53, 0.75), (0.53, 0.535, 0.47, 0.475), (0.49, 0.28, 0.53, 0.28)):
    fig.add_artist(matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=25, color="0.3", lw=2))
h, l = ax0.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig("flow_chain_migkas_model_diff_l4.png", dpi=130, bbox_inches="tight"); print("ok", r)
