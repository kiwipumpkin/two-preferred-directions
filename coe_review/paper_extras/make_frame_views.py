"""Fig. 2 crop and Fig. 13 (Appendix D) composition.
1. Runs plane_of_evil_coe_obs_ceq_p1p2.py three times: default view (Fig. 2), along the CoE axis (ELEV, AZIM = 0, 180),
   and from above the plane of evil (90, -90).
2. Fig. 2 = default view with long white bands removed.  Fig. 13 = the two edge-on views stacked vertically."""
import os as _os; PKG = globals().get("PKG") or _os.environ.get("COE_PKG") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))  # package root
import os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
D = PKG + "/coe_review/11_plane_of_evil"; FIGS = PKG + "/coe_review/14_paper/figs"; os.makedirs(FIGS, exist_ok=True)
src = open(D + "/plane_of_evil_coe_obs_ceq_p1p2.py").read()
for elev, azim, tag in [(28, -58, "default"), (0, 180, "alongaxis"), (90, -90, "top")]:
    s = re.sub(r"^ELEV, AZIM = .*$", f"ELEV, AZIM = {elev}, {azim}", src, flags=re.M)
    s = s.replace("plane_of_evil_coe_obs_ceq_p1p2.png", f"view_{tag}.png")
    p = D + f"/_view_{tag}.py"; open(p, "w").write(s)
    subprocess.run([sys.executable, p], cwd=D, check=True, env=dict(os.environ, COE_PKG=PKG)); os.remove(p)
def drop_white_bands(im, run_max=30):
    a = np.asarray(im.convert("RGB")); white = a.min(axis=(1, 2)) >= 250; keep = []; run = 0
    for i, w in enumerate(white):
        run = run + 1 if w else 0
        if not (w and run > run_max): keep.append(i)
    return Image.fromarray(a[keep])
drop_white_bands(Image.open(D + "/view_default.png")).save(FIGS + "/fig1_frame3d_coe_p1p2.png")
def trim(im, pad=20):
    a = np.asarray(im.convert("L")); c = np.where(a.min(0) < 250)[0]; r = np.where(a.min(1) < 250)[0]
    return im.crop((max(c[0]-pad, 0), max(r[0]-pad, 0), min(c[-1]+pad, im.width), min(r[-1]+pad, im.height)))
A = Image.open(D + "/view_alongaxis.png").convert("RGB"); T = Image.open(D + "/view_top.png").convert("RGB")
a = trim(A.crop((0, 600, A.width, 1380))); t = trim(T.crop((0, 190, T.width, 1748)), 0)
w, h = t.size; t = t.crop((0, int(0.32*h), int(0.70*w), int(0.68*h)))
leg = trim(A.crop((0, 1748, A.width, A.height)))
W = 1800; a = a.resize((W, int(a.height*W/a.width))); t = t.resize((W, int(t.height*W/t.width))); lab = 60
out = Image.new("RGB", (W, lab + a.height + 30 + lab + t.height + 20 + leg.height), "white"); y = lab
out.paste(a, (0, y)); y += a.height + 30; yb = y; y += lab; out.paste(t, (0, y)); y += t.height + 20; out.paste(leg, ((W - leg.width)//2, y))
d = ImageDraw.Draw(out)
try: f = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
except OSError: f = ImageFont.load_default()
d.text((10, 10), "(a) Looking along the CoE axis", fill="black", font=f); d.text((10, yb + 10), "(b) Looking down onto the plane of evil", fill="black", font=f)
out.save(FIGS + "/figA_frame_views_vertical.png"); print("wrote Fig. 2 and Fig. 13 to", FIGS)
