"""Flow fig 3: multipole content of the residual (Migkas - ring model (a)), no patch circles, no repeated maps."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np, healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = PKG + "/coe_review/10_residual_P2/"
J = json.load(open(D+"resid_p2.json")); AX = json.load(open(D+"axes_check.json")); alm = np.load(D+"resid_maps.npz")["alm"]
NS, LMAX = 16, 6
def comp(l):
    a = np.zeros_like(alm)
    for m in range(l+1): a[hp.Alm.getidx(LMAX, l, m)] = alm[hp.Alm.getidx(LMAX, l, m)]
    return hp.alm2map(a, NS, lmax=LMAX)
G = J["geometry"]
pts = [(G["P1 (AoE)"]["antipode"], "P", "k", "AoE / P1 (260,+60)"), (G["P1 (AoE)"]["lb"], "X", "k", "P1 antipode (80,−60)"),
       (G["P2"]["lb"], "D", "yellow", "P2 (234,+52.5)"), (G["P2"]["antipode"], "d", "yellow", "P2 antipode (54,−52.5)"),
       (G["CoE axis"]["lb"], "*", "k", "CoE (180,−15)"), ((0, 15), "o", "k", "anti-CoE (0,+15)")]
fig = plt.figure(figsize=(16, 10.6)); fig.subplots_adjust(hspace=0.35)
# panel 1: power per multipole vs noise
ax = fig.add_subplot(2, 2, 1)
ls = np.arange(1, 7); ratio = [J["Cl"][str(l)]["ratio"] for l in ls]; p = [J["Cl"][str(l)]["p_noise"] for l in ls]
bars = ax.bar(ls, ratio, color=["#d7301f" if l == 4 else "0.6" for l in ls], edgecolor="k")
ax.axhline(1, color="k", ls="--", lw=1, label="noise level (Migkas map covariance, 1000 sims)")
for l, r, pp in zip(ls, ratio, p): ax.text(l, r+0.05, f"p={pp:.3f}" if pp < 0.05 else f"p={pp:.2f}", ha="center", fontsize=9)
ax.set_xlabel("multipole ℓ"); ax.set_ylabel("power in residual / mean noise power"); ax.set_xticks(ls)
ax.set_title("Which multipoles carry the residual? (Migkas − ring model (a))", fontsize=11); ax.legend(fontsize=9, loc="upper right"); ax.set_ylim(0, 2.9)
panels = [(100*comp(2), f"ℓ = 2 part (power {J['Cl']['2']['ratio']}× noise)   axis {tuple(J['axes']['2']['axis'])}", 2, J["axes"]["2"]["axis"]),
          (100*comp(4), f"ℓ = 4 part (power {J['Cl']['4']['ratio']}× noise, p = {J['Cl']['4']['p_noise']})   axis {tuple(AX['l4_axis'])}", 3, AX["l4_axis"]),
          (100*comp(3), f"ℓ = 3 part (power {J['Cl']['3']['ratio']}× noise)   axis {tuple(J['axes']['3']['axis'])}", 4, J["axes"]["3"]["axis"])]
for m, ttl, k, key in panels:
    lim = 6 if k == 2 else 3
    hp.mollview(m, fig=fig.number, sub=(2, 2, k), title=ttl, min=-lim, max=lim, cmap="coolwarm", flip="astro", unit="%", notext=True)
    hp.graticule(dpar=30, dmer=60, alpha=.3)
    for (l, b), mk, fc, lab in pts: hp.projplot(l, b, mk, lonlat=True, color="k", mfc=fc, ms=10, label=lab)
    for l0, b0 in ((key[0], key[1]), ((key[0]+180) % 360, -key[1])):
        hp.projplot(l0, b0, "s", lonlat=True, mfc="none", mec="lime", mew=2.5, ms=14)
h, l = plt.gca().get_legend_handles_labels()
fig.legend(h[:6], l[:6], loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.text(0.5, -0.01, "green squares = best axis of that multipole (and antipode).  Galactic, ℓ increases left.", ha="center", fontsize=9)
fig.savefig("flow_3_residual_multipoles.png", dpi=130, bbox_inches="tight"); print("ok")
