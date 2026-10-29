"""Turn the knobs: which CoE-geometry observables survive across setups?
Newtonian: point mass + shell (ptshell) and 8 distributed profiles (prof_core); exact GR: LTB (relfree).
Per setup: A(theta) with theta from the anti-CoE direction (0) to the CoE (180), A = directional expansion af, H ~ sqrt(A).
Records: angle of global max / min, number of interior extrema on 0<theta<180 (= rings about the CoE axis on the sky),
density slope across the observer (denser toward CoE?), R^2 vs the Migkas curve at best l0 (no extra harmonic)."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, os, json, time, itertools, numpy as np
B = PKG + '/workspace/cosmology/papers/spreadsheet_model/bubble'; sys.path.insert(0, B); os.chdir(B)
import newt_omb as N, prof_core as P, ptshell  # ptshell patches P.run for kind "ptshell"
from bubble_core import shifted
def feats(a):
    a = np.asarray(a, float); n = len(a); th = np.arange(n)*360/n
    half = (a[: n//2+1] + a[::-1][: n//2+1][::-1]) / 2 if False else a[: n//2+1]; t = th[: n//2+1]
    rng_ = a.max()-a.min(); k = max(1, n//72)
    sm = np.convolve(np.r_[half[k:0:-1], half, half[-2:-k-2:-1]], np.ones(2*k+1)/(2*k+1), 'valid')
    ext = [i for i in range(1, len(sm)-1) if (sm[i]-sm[i-1])*(sm[i+1]-sm[i]) < 0 and abs(sm[i]-sm[i-1])+abs(sm[i+1]-sm[i]) >= 0]
    # keep only prominent extrema (> 2% of range from neighbours' mean over +-15 deg)
    w = max(1, int(15*n/360)); keep = []
    for i in ext:
        lo, hi = max(0, i-w), min(len(sm), i+w+1)
        if abs(sm[i]-np.median(sm[lo:hi])) > 0.02*rng_: keep.append(float(t[i]))
    g = np.sign(np.diff(sm)); g = g[g != 0]; nint = int(np.sum(g[1:] != g[:-1]))
    return dict(n_interior_extrema=nint, A0_is_local_max=bool(half[0] > half[min(len(half)-1, max(1, n//36))]), A180_is_local_max=bool(half[-1] > half[-1-max(1, n//36)]), theta_max=float(t[np.argmax(half)]), theta_min=float(t[np.argmin(half)]), rings=len(keep), ring_angles=keep,
                amp=float(rng_/a.mean()))
def r2mig(af):
    v, l0, _, _ = N.fit_l0(af/af.mean(), False); return float(v), float(l0)
rows = []; t0 = time.time()
DRS = [2.0, 3.0, 3.5, 4.5, 6.0, 8.0]
# --- Newtonian point mass + shell
for DR, mcc, (kin, kout) in itertools.product(DRS, [1e-3, 1.7e-3, 1e-2, 0.1], [(1.04, 1.66), (1.5, 3.0)]):
    D = DR*N.RV
    if kin*N.RV >= 0.99*D: continue
    r = N.af_of(N.geo(D, mcc, kin, kout, N.RV))
    if r is None: rows.append(dict(model='Newton point+shell', DR=DR, knobs=f'mcc={mcc},kin={kin},kout={kout}', ok=False)); continue
    af, rr = r; f = feats(af); v, l0 = r2mig(af)
    rows.append(dict(model='Newton point+shell', DR=DR, knobs=f'mcc={mcc},kin={kin},kout={kout}', ok=True, contrast=rr['contrast'], R2=v, l0=l0, **f))
print('ptshell done', len(rows), '%.0fs' % (time.time()-t0), flush=True)
# --- Newtonian distributed profiles
PROF = [('exp', dict(LD=0.01)), ('exp', dict(LD=0.1)), ('exp', dict(LD=0.3)), ('pl', dict(LD=0.05, n=3.0)), ('gauss', dict(LD=0.18)),
        ('cored', dict(LD=0.05)), ('shell', dict(rs=0.25, w=0.1)), ('st53', dict(rst=0.24))]
for (kind, pars), DR, mc, Ob in itertools.product(PROF, DRS, [0.003, 0.05], [0.05, 0.3]):
    a = dict(pars, DR=DR)
    try:
        rr = P.run(kind, a, mc, Ob); af = rr['af']
        ok = bool(np.all(np.isfinite(af)) and rr['vrmin'] > 0)
    except Exception as e: ok = False
    kn = f"{kind} {','.join(f'{k}={v}' for k, v in pars.items())}, mc={mc}, Ob={Ob}"
    if not ok: rows.append(dict(model='Newton '+kind, DR=DR, knobs=kn, ok=False)); continue
    f = feats(af); v, l0 = r2mig(af)
    rows.append(dict(model='Newton '+kind, DR=DR, knobs=kn, ok=True, contrast=rr['contrast'], R2=v, l0=l0, **f))
print('profiles done', len(rows), '%.0fs' % (time.time()-t0), flush=True)
json.dump(rows, open(PKG + '/coe_review/15_knobs/sweep_newton.json', 'w'), indent=1, default=float)
np.save(PKG + '/coe_review/15_knobs/dummy.npy', 0)
