"""Table 2 and Sec. 5 text: summary of the knob sweep from the saved sweep results (15_knobs/sweep_*.json).
Rows: valid setups; maxima at both poles + one minimum ring; minimum-ring offset from the equal-distance (critical) circle;
global maximum on the CoE side; the R >= D runs (sweep_inside)."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import json, numpy as np
K = PKG + "/coe_review/15_knobs/"
N = [r for r in json.load(open(K + "sweep_newton.json")) if r.get("ok")]
L = [r for r in json.load(open(K + "sweep_ltb.json")) if r.get("ok")]
crit = lambda DR: 180 - np.degrees(np.arccos(1/(2*DR)))   # critical circle, measured from the anti-CoE direction
def shape_ok(r): return r["A0_is_local_max"] and r["A180_is_local_max"] and r["n_interior_extrema"] == 1
for name, S in (("Newtonian", N), ("LTB", L)):
    print(f"{name}: {len(S)} valid; maxima at both poles + one minimum ring: {sum(map(shape_ok, S))}/{len(S)}")
    tm = np.array([r["theta_min"] for r in S if shape_ok(r)], float); dr = np.array([r["DR"] for r in S if shape_ok(r)], float)
    off = tm - crit(dr)
    print(f"  minimum ring at {tm.min():.1f}-{tm.max():.1f} deg from anti-CoE; offset from critical circle {off.min():.1f}-{off.max():.1f} deg (mean {off.mean():.1f})")
    gmax = [r for r in S if r.get("theta_max") is not None and float(r["theta_max"]) > 90]
    print(f"  global maximum on the CoE side (theta_max > 90): {len(gmax)}/{len(S)}")
I = [r for r in json.load(open(K + "sweep_inside.json")) if r.get("ok")]
print(f"R >= D runs: {len(I)} valid, D/R {min(r['DR'] for r in I)}-{max(r['DR'] for r in I)}")
for dr in sorted({r["DR"] for r in I}):
    S = [r for r in I if r["DR"] == dr]
    print(f"  D/R {dr:5.2f}: ring (one interior minimum, maxima at both poles) {sum(map(shape_ok, S))}/{len(S)};  "
          f"no interior extremum (simple dipole) {sum(r['n_interior_extrema'] == 0 for r in S)}/{len(S)}")
