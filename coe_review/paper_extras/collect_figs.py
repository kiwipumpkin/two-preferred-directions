"""Copy each script's output PNG to the name used in the paper (14_paper/figs). Run after the figure scripts."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import os, shutil
R = PKG + "/coe_review"; FIGS = R + "/14_paper/figs"; os.makedirs(FIGS, exist_ok=True)
MAP = {  # paper file name : produced by
    "fig_knobs.png": "15_knobs/knobs.png",
    "fig3_ringblends.png": "09_values_blend/compare/ring_blend_30_40_50_60.png",
    "fig4_chain.png": "12_flow/flow_chain_migkas_model_diff_l4.png",
    "fig5_multipoles.png": "12_flow/flow_3_residual_multipoles.png",
    "fig_spots.png": "06_consistent/figs/chris_AB_residual_spots.png",
    "fig_dipoles.png": "17_paperfigs/fig_dipoles_connected.png",
    "fig_sky.png": "17_paperfigs/fig_where_to_look.png",
}   # fig_AB, fig_dipole_cmp, fig2_roll, fig6_nulls are written straight to 14_paper/figs; Fig. 2 and Fig. 13 by make_frame_views.py
for dst, src in MAP.items():
    if os.path.exists(os.path.join(R, src)): shutil.copy(os.path.join(R, src), os.path.join(FIGS, dst)); print("ok ", dst)
    else: print("MISSING", src, "(run its script first)")
