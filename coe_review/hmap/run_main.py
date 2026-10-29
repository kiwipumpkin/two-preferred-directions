"""Data maps, bootstrap covariance, model templates and full-sky axis scans.
usage: python run_main.py   (single core; ~ few minutes)"""
from common import *
import json, sys, time
t0 = time.time()
rng = np.random.default_rng(20261025)
NB = 2000
cat = load_catalogue(); Ncl = len(cat["z"])
pl, pb, pv = pix_centres()
ang = cone_angles(cat, pv)
print("clusters", Ncl)

res = {}
store = {}
for theta in (75, 90):
    Mem = (ang < theta).astype(float)
    print("theta", theta, "clusters/cone min/med/max", Mem.sum(1).min(), np.median(Mem.sum(1)), Mem.sum(1).max())
    mw = mig_weights(cat, ang, theta)
    ests = {"OLS": (lambda W: ols_maps(cat, Mem, W), ols_kernel(cat, Mem)),
            "MIG": (lambda W: mig_maps(cat, mw, W), mig_kernel(mw))}
    for en, (fn, K) in ests.items():
        tag = "%s_t%d" % (en, theta)
        y0 = fn(np.ones((Ncl, 1)))[:, 0]
        bs = np.empty((NPIX, NB))
        for s in range(0, NB, 250):
            W = np.stack([np.bincount(rng.integers(0, Ncl, Ncl), minlength=Ncl) for _ in range(250)], 1).astype(float)
            bs[:, s:s + 250] = fn(W)
        mu = bs.mean(1); C = np.cov(bs); sig = np.sqrt(np.diag(C))
        lam, V = np.linalg.eigh(C); lam, V = lam[::-1], V[:, ::-1]
        cf = np.cumsum(lam) / lam.sum()
        kk = {f: int(np.searchsorted(cf, f) + 1) for f in (0.9, 0.95, 0.99, 0.999)}
        # sanity: does the linear forward kernel reproduce the estimator? (isotropic data + known delta)
        dlt = 0.02 * cat["n"][:, 0]
        cat2 = dict(cat); cat2["y"] = cat["y"] + dlt
        fn2 = (lambda W: ols_maps(cat2, Mem, W)) if en == "OLS" else (lambda W: mig_maps(cat2, mw, W))
        kerr = np.abs((fn2(np.ones((Ncl, 1)))[:, 0] - y0) - 0.5 * K @ dlt).max()
        H = 10 ** y0; imax, imin = H.argmax(), H.argmin()
        info = dict(Hmax=float(H[imax]), lb_max=[float(pl[imax]), float(pb[imax])], Hmin=float(H[imin]), lb_min=[float(pl[imin]), float(pb[imin])],
                    ratio_max_min=float(H[imax] / H[imin]), sig_med=float(np.median(sig)), sig_range=[float(sig.min()), float(sig.max())],
                    boot_bias_med=float(np.median(np.abs(mu - y0) / sig)), modes=kk, lam_top=(lam[:12] / lam.sum()).tolist(),
                    kernel_check_maxabs=float(kerr), N_eff=float(lam.sum() ** 2 / np.sum(lam ** 2)))
        if en == "MIG": info.update(B_all=float(mw["B"]), sig_int=float(mw["sig_int"]), A_all=float(10 ** mw["a_all"]))
        print(tag, json.dumps(info)[:900]); sys.stdout.flush()
        res[tag] = dict(data=info)
        store[tag] = dict(y0=y0, sig=sig, lam=lam[:200], V=V[:, :200], K=K, C=C if (en == "OLS" and theta == 75) else None)
        del bs
print("bootstrap done %.0fs" % (time.time() - t0))
np.savez_compressed(OUT + "/maps_cov.npz", **{"%s__%s" % (t, k): v for t, d in store.items() for k, v in d.items() if v is not None},
                    pl=pl, pb=pb)
json.dump(res, open(OUT + "/data_summary.json", "w"), indent=1)
