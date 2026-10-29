"""Combined significance of the two-direction frame.
Statistics (fixed before running, as claimed in the draft):
  T1 feature-sign score (12 features, Migkas residual curves, 23_sky_mapA_al recipe)      high = good
  T2 angle between the Migkas residual l=4 axis (115.3, 31.4) and the plane-of-evil normal low  = good
Nulls (N draws each):
  A: whole frame rotated rigidly at random (CoE and AoE together)
  B: AoE held at its observed axis; CoE direction uniform on the sky; frame rebuilt from (CoE, AoE)
Joint p: Fisher statistic -2 sum ln p_i with each p_i taken as the empirical tail rank inside the same null draws.
T3 (separate, analytic): Migkas 2-D best roll 50 deg vs AoE-fixed roll 60.18 deg; p = 2*10.18/360 for a uniform roll."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import sys, json, numpy as np
sys.argv = ['x']; exec(open(PKG + '/coe_review/joint/spotmc.py').read().split("F0=np.array")[0])
F0 = np.array([f[2] for f in feat]); C0 = vec(180, -15); A0 = vec(80, -60)       # P1 end of the AoE axis (81.94 deg from CoE)
L4 = vec(115.3, 31.4)
def frame2(C, A):
    """model->Galactic rotation taking model CoE (-1,0,0) to C and model P1 (in x-y plane, y>0) to A."""
    if C @ A < 0: A = -A
    a = A - (A @ C)*C; a /= np.linalg.norm(a)
    G = np.column_stack([C, a, np.cross(C, a)])
    M = np.column_stack([[-1, 0, 0], [0, 1, 0], np.cross([-1, 0, 0], [0, 1, 0])])
    return G @ M.T
Rm0 = frame2(C0, A0); assert np.allclose(Rm0, frame(-15, 60.18), atol=2e-3), (Rm0, frame(-15, 60.18))
def T1(Rm, F=F0):
    s = 0
    for (K, nm, v, c), vv in zip(feat, F):
        m = vv @ Rm; lm = np.degrees(np.arctan2(m[1], m[0])) % 360; s += (pi(MM['l'], MM[K], lm) > 0) == (c == 'hot')
    return s
def T2(C, A):
    N = np.cross(C, A); N /= np.linalg.norm(N); return np.degrees(np.arccos(min(1, abs(N @ L4))))
obs = dict(T1=T1(Rm0), T2=T2(C0, A0)); print("observed", obs)
rng = np.random.default_rng(11)
def randrot():
    q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)], [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
def runnull(kind, N):
    t1, t2 = np.zeros(N), np.zeros(N)
    for i in range(N):
        if kind == 'A':
            Q = randrot(); C, A = Q @ C0, Q @ A0
        else:
            C = rng.normal(size=3); C /= np.linalg.norm(C); A = A0
        t1[i] = T1(frame2(C, A)); t2[i] = T2(C, A)
    return t1, t2
out = dict(observed=obs)
for kind in ('A', 'B'):
    N = 20000; t1, t2 = runnull(kind, N)
    np.savez(f'null_{kind}.npz', t1=t1, t2=t2)
    p1 = (t1 >= obs['T1']).mean(); p2 = (t2 <= obs['T2']).mean()
    # empirical tail p for every draw (ranks), Fisher combine, compare
    r1 = np.array([(t1 >= x).mean() for x in np.unique(t1)]); P1 = {x: (t1 >= x).mean() for x in np.unique(t1)}
    q1 = np.array([P1[x] for x in t1]); o2 = np.sort(t2); q2 = np.searchsorted(o2, t2, side='right')/N
    F = -2*(np.log(q1)+np.log(q2)); Fobs = -2*(np.log(max(p1, 1/N))+np.log(max(p2, 1/N)))
    pj = (F >= Fobs).mean()
    out['null_'+kind] = dict(N=N, p_T1=p1, p_T2=p2, p_joint_T1T2=pj, T1_mean=float(t1.mean()), T2_median=float(np.median(t2)))
    print(kind, out['null_'+kind])
pT3 = 2*10.18/360; out['T3_roll'] = dict(best_roll_2D=50, aoe_roll=60.18, p=pT3)
from scipy.stats import chi2, norm
for kind in ('A', 'B'):
    pj = out['null_'+kind]['p_joint_T1T2']; Fall = -2*(np.log(pj)+np.log(pT3)); pall = chi2.sf(Fall, 4)
    out['null_'+kind]['p_all_three'] = float(pall); out['null_'+kind]['sigma_all_three'] = float(norm.isf(pall/2))
    out['null_'+kind]['sigma_T1T2'] = float(norm.isf(pj/2))
json.dump(out, open('combined.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
