#!/usr/bin/env python3
"""Fig 4 — VALUES, NO BANDS, BLEND version (Chris: rings blend smoothly).
Panels B/C: value(n) = w*A(theta) + (1-w)*A(360-theta), theta=arccos(n.D),
w=(1+cos psi)/2, cos psi = (n.y)/sqrt(1-(n.D)^2), y = row ring_basis +y.
Panel A rim unchanged.

(Verbatim copy of Chris's script; run copy: base module taken from
../../06_consistent/makers, outputs here. FIG_DIR as given.)
"""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))  # package root
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG_DIR = Path(PKG + "/coe_review/06_consistent/makers")
sys.path.insert(0, str(HERE))
sys.path.insert(1, PKG + "/coe_review/06_consistent/makers")
import types; sys.modules.setdefault('paths', types.SimpleNamespace(OUT=HERE))
import make_fig05_sky_orientation as base  # reuse compose_four_row unchanged

d = pd.read_csv(PKG + "/workspace/cosmology/migkas/fig6_digitized.csv")
L, AV = d.l_deg.values.astype(float), d.A_over_Aall.values.astype(float)
def A_of(theta):
    return np.interp(np.mod(theta, 360.0), L, AV, period=360)
NORM = matplotlib.colors.Normalize(vmin=AV.min(), vmax=AV.max())
CMAP = plt.get_cmap("coolwarm")

def basis_y(mod, D):
    a = np.asarray(D, float); a /= np.linalg.norm(a)
    if hasattr(mod, "ring_basis"):
        return mod.ring_basis(a)[0]
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(a @ tmp) > 0.9: tmp = np.array([0.0, 1.0, 0.0])
    y = tmp - (tmp @ a) * a
    return y / np.linalg.norm(y)

def A_on_dirs(mod, D, n):  # n (...,3)
    a = np.asarray(D, float); a /= np.linalg.norm(a)
    y = basis_y(mod, a)
    x = np.clip(n @ a, -1, 1)
    th = np.rad2deg(np.arccos(x))                      # 0..180
    cpsi = np.clip((n @ y) / np.sqrt(np.clip(1 - x*x, 1e-12, None)), -1, 1)  # cos psi, psi=0 at +y
    w = 0.5 * (1.0 + cpsi)
    return w * A_of(th) + (1.0 - w) * A_of(360.0 - th)

def wrap(mod, tag):
    mod.draw_panelA_ribbon = lambda *a, **k: None
    mod.fill_mollweide_ribbon = lambda *a, **k: None
    mod.fill_sphere_ribbon = lambda *a, **k: None
    oA, oB, oC = mod.panel_A_model_frame, mod.panel_B_mollweide, mod.panel_C_sphere
    def pA(ax, geo):
        oA(ax, geo)
        t = np.linspace(0, 360, 721)
        x, y = np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))
        seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], 1)
        ax.add_collection(LineCollection(seg, colors=CMAP(NORM(A_of(0.5*(t[:-1]+t[1:])))),
                                         linewidths=6, zorder=1.5, capstyle="butt"))
    def pB(ax, geo):
        oB(ax, geo)
        lon = np.linspace(-np.pi, np.pi, 361); lat = np.linspace(-np.pi/2, np.pi/2, 181)
        LO, LA = np.meshgrid(lon, lat)
        ell = np.mod(-np.rad2deg(LO), 360.0)  # module: lon_disp = -l  (l increases LEFT)
        # verify against module convention
        n = np.stack([np.cos(LA)*np.cos(np.deg2rad(ell)), np.cos(LA)*np.sin(np.deg2rad(ell)), np.sin(LA)], -1)
        ax.pcolormesh(LO, LA, A_on_dirs(mod, geo["D"], n), cmap=CMAP, norm=NORM,
                      shading="gouraud", alpha=0.85, zorder=0.5, rasterized=True)
    def pC(ax, geo):
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 121), np.linspace(0, np.pi, 61))
        n = np.stack([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)], -1)
        fc = CMAP(NORM(A_on_dirs(mod, geo["D"], n))); fc[..., 3] = 0.6
        ax.plot_surface(n[..., 0]*0.995, n[..., 1]*0.995, n[..., 2]*0.995, facecolors=fc,
                        rstride=1, cstride=1, linewidth=0, shade=False, zorder=1)
        oC(ax, geo)
    mod.panel_A_model_frame, mod.panel_B_mollweide, mod.panel_C_sphere = pA, pB, pC

def load(script, i):
    if str(FIG_DIR) not in sys.path: sys.path.insert(0, str(FIG_DIR))
    spec = importlib.util.spec_from_file_location(f"vals_{i}", FIG_DIR / script)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def main():
    pngs = []
    for i, s in enumerate(base.RING_SCRIPTS):
        m = load(s, i)
        # display-lon convention check
        print(s, "lon_disp(90)=", m.lon_disp_from_ell(np.array([90.0])))
        m.PNG_PATH = HERE / f"fig8_row{i}_values_nobands_blend.png"
        m.JSON_PATH = HERE / f"fig8_row{i}_values_nobands_blend.json"
        wrap(m, i); m.main(); pngs.append(m.PNG_PATH)
    comp = base.compose_four_row(pngs)
    fig = plt.figure(figsize=(comp.width/160, 0.75), dpi=160)
    cax = fig.add_axes([0.2, 0.62, 0.6, 0.22])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(NORM, CMAP), cax=cax, orientation="horizontal")
    cb.set_label(r"Migkas $A/A_{\rm all}$ at scan angle $\theta$ (fig6 digitized, linear interp.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    tmp = HERE / "_cbar_values_nobands_blend_tmp.png"; fig.savefig(tmp, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    cbi = Image.open(tmp).convert("RGB"); cbi.thumbnail((int(comp.width*0.7), 10**4))
    out = Image.new("RGB", (comp.width, comp.height + cbi.height), "white")
    out.paste(comp, (0, 0)); out.paste(cbi, ((comp.width-cbi.width)//2, comp.height)); tmp.unlink()
    out.save(HERE / "fig8_orientation_seq_values_nobands_blend.png"); print("wrote", HERE / "fig8_orientation_seq_values_nobands_blend.png", out.size)

if __name__ == "__main__":
    main()
