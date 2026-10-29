# Code package: "Two preferred directions" (working draft, version 4)

This package holds the code, and the small inputs it needs, to reproduce the calculations and figures in
`coe_review/14_paper/two_directions_working_draft_v4.tex`.

**Quick start:** the paper's figures are included in `coe_review/14_paper/figs/`, so the paper builds directly:
`cd coe_review/14_paper && pdflatex two_directions_working_draft_v4.tex` (run twice). The compiled paper is
`Alameddin_2026_two_preferred_directions_v4.pdf` at the package root. `run_all.sh` regenerates the figures and numbers.

Contact: C. Alameddin, calameddin@gmail.com


The folder layout mirrors the working folders the analysis was done in:
- `coe_review/` — the analysis scripts and their saved results;
- `workspace/cosmology/` — the model code and the digitized Migkas data;
- `kiwipumpkin_work/` — files from the author's repository `kiwipumpkin/coe-cosmology` (`work/2026-10-23_25`).

Every absolute path in the original scripts has been replaced by `PKG + "..."`. `PKG` is the package root: each script works it out from its own location, or takes it from the `COE_PKG` environment variable. The package therefore runs from wherever it is unpacked.

## Requirements

Tested with Python 3.13 and the following packages:

| Package | Tested version |
|---|---|
| numpy | 2.5 |
| scipy | 1.18 |
| pandas | 3.0 |
| matplotlib | 3.11 |
| healpy | 1.20 |
| astropy | 8.0 |
| Pillow | 12.3 |
| numba | 0.68 |

`numba` is used by the model code. Building the paper needs LaTeX (pdfTeX). Install the Python packages with:

```
pip install numpy scipy pandas matplotlib healpy astropy pillow numba
```

## Quick start

```
bash run_all.sh                # fast and medium steps, about 20-25 min on one core
bash run_all.sh --with-resid   # also re-run the 2-D residual analysis (adds about 10 min)
cd coe_review/14_paper && pdflatex two_directions_working_draft_v4.tex && pdflatex two_directions_working_draft_v4.tex
```

`run_all.sh` writes the figures to `coe_review/14_paper/figs/` under the names the paper uses. The steps in the "Heavy steps" section are not run; their saved outputs are included.

## What reproduces what

| Paper item (section) | Script (under `coe_review/`) | Inputs | Output / value |
|---|---|---|---|
| Table 1: frame, D/R = 3.564, θ0 = 81.94°, roll 60.18°/240.18° (Sec. 4) | `06_consistent/geometry_from_aoe.py` | CoE (180°, −15°), AoE (260°, +60°) | `geometry_from_aoe.json` |
| See row note (1) | `paper_extras/frame_angles.py` | `geometry_from_aoe.json` | printed |
| Fig. 1; B 4.2σ strict; joint Δχ² 70.9 → 7.1σ (Sec. 3, 8) | `17_paperfigs/sheet_joint356.py` (uses `sheet_AB.py`) | Migkas `fig6_digitized.csv` | `14_paper/figs/fig_AB.png`, `sheet_joint356.json`. Fit gives χ²A 4.47, χ²B 7.97, L/D 0.257, a 0.180, l0 11.8° |
| Spreadsheet's own inputs: χ²A 10.3, χ²B 7.3 (App. B) | `17_paperfigs/sheet_AB.py` | same | `sheet_AB.json` |
| Free D/R: 2.83, Δχ² 71.1; relaxed bound: flat down to D/R ≈ 1.7 (App. B) | `17_paperfigs/sheet_joint.py` | same | `AMPMAX=0.2` gives D/R 2.83; the default 0.6 gives 1.65. Not re-run here |
| A vs isotropy: Δχ² 40.5, 6 extra params, 5.1σ (Sec. 8) | `17_paperfigs/chi2_test.py` | `06_consistent/fig4/newt_fixedD.json`, model code | `chi2_test.json`. Its B and A+B entries come from the superseded free-sine B model; the paper's B is from `sheet_joint356.py` |
| Fig. 8; A vs dipole: χ² 26.2 vs 4.2, Δχ² 22.0, 4 params, 3.7σ (Sec. 8) | `17_paperfigs/fig_dipole_cmp.py` | same | `14_paper/figs/fig_dipole_cmp.png` |
| Fourier comparison: 5 params χ² 7.56 (R² 0.81); 7 params χ² 3.15 (Sec. 8, Fig. 1 caption) | `paper_extras/fourier_compare.py` | `fig6_digitized.csv` | printed |
| Rotation-term direction l = 73° ± 14° (Sec. 3) | `17_paperfigs/phase_profile.py` | `sheet_joint356.json` | printed phase profile (Δχ² ≤ 1: 58–87°) |
| Table 6: full covariance Δχ² 9.2 → 2.6σ; 36 effective bins 3.1σ / 2.2σ (App. C) | `07_coe_direction/dir_fit.py`, then `paper_extras/robustness.py` | `af_D51.json`, `rerun/teamA/mfit.py` + `migkas_data.npz`, `hmap/common.py` | `dir_fit.npz`; printed |
| Tier B: 1.1% patch (3.2% / 3.6% for wider D/R) (Sec. 8) | `paper_extras/tierB_patch.py` | `geometry_from_aoe.py` | printed (2×10⁶ random axes) |
| Table 3 T1, T2, T1+T2 (Sec. 8) | `13_combined/combined.py` (uses `joint/spotmc.py`) | `kiwipumpkin_work/.../rnx/fr/residuals.csv`, `l50/curves.json`, Migkas CSV | `combined.json`, `null_A.npz`, `null_B.npz` |
| Table 3 T3 (either branch) 0.113; all three 2.9σ / 2.4σ | `paper_extras/tierC_combine.py` | `combined.json` | printed |
| Fig. 9 (null distributions) | `14_paper/null_fig.py` | `combined.json`, `null_*.npz` | `figs/fig6_nulls.png` |
| Table 4, Fig. 10 (residual features) | `06_consistent/features_at_fixed.py`; `06_consistent/figs/chris_AB_residual_spots.py` | `spotmc.py` inputs; `figs/M25.json` | `features_at_fixed.json`; `fig_spots.png` |
| Table 2 and Sec. 5 knob numbers: 208/208, 0.4–8.5° (mean 3.8°), LTB 97.5–100.5°, 180/208, 18/21, R ≥ D thresholds | `paper_extras/knobs_summary.py` | `15_knobs/sweep_newton.json`, `sweep_ltb.json`, `sweep_inside.json` | printed |
| Fig. 3 (knobs) | `15_knobs/fig_knobs.py` | sweep JSONs, model code, `kiwipumpkin_work/cosmology/papers/cmb_redshift` | `15_knobs/knobs.png` → `fig_knobs.png` |
| Fig. 4 (ring blends 30/40/50/60°); r = 0.59, 0.70, 0.77, 0.79 (Sec. 7) | `09_values_blend/compare/compare_2d.py`, `ring_blend.py`, `two_scale.py` | `hmap/maps_cov.npz` (trimmed), ring makers in `06_consistent/makers`, `09_values_blend/rot60/...blend.py` | `ring_blend_30_40_50_60.png` → `fig3_ringblends.png`; `*_2d.json`, `ring_blend.json`, `two_scale.json` |
| Fig. 5 (roll scan) | `14_paper/roll_fig.py` | `compare_2d.json`, `ring_blend.json` | `figs/fig2_roll.png` |
| Fig. 6 (chain: map → model → difference → ℓ = 4) | `12_flow/flow_chain.py` (uses `two_scale.py`) | same as Fig. 4 | `fig4_chain.png` |
| ℓ = 4 at 2.5× noise, p = 0.016; T2 axis 14.8° from plane normal (Sec. 7, 8) | `10_residual_P2/resid_p2.py`; `10_residual_P2/axes_check.py` (heavy) | as Fig. 4 | `resid_p2.json` (`ratio` 2.52, `p_noise` 0.016), `resid_maps.npz`, `axes_check.json` |
| Fig. 7 (multipoles) | `12_flow/flow_3.py` | `resid_p2.json`, `axes_check.json`, `resid_maps.npz` | `fig5_multipoles.png` |
| Fig. 2 (3-D frame) and Fig. 13 (edge-on views, App. D) | `paper_extras/make_frame_views.py` (runs `11_plane_of_evil/plane_of_evil_coe_obs_ceq_p1p2.py` three times) | `geometry_from_aoe.json`, `resid_maps.npz` | `fig1_frame3d_coe_p1p2.png`, `figA_frame_views_vertical.png` |
| Figs. 11, 12 (dipoles; sky chart) | `17_paperfigs/fig_dipoles_and_chart.py` | built-in coordinates | `fig_dipoles.png`, `fig_sky.png` |
| Limitations: rebuilt frame on (198°, −4°): D/R 2.88, T1 7/12, T2 6.7°, null-B 0.075 | `18_rebuild/frame198.py`, `18_rebuild/chi2_198.py` | `spotmc.py` inputs; `06_consistent/fig4/prof_DR2.879.json` | `frame198.json`, `chi2_198.json` |
| Supernova test: dipole 1.05 ± 0.46 (`sn_split.py`), even part 0.04 ± 0.05 (`sn_test.py`) (Sec. 9; summary in `joint/sn/P3_result.md`) | `joint/sn/sn_test.py`, `joint/sn/sn_split.py` | Pantheon+ (see "External data"); `rerun/fixed/res/*.json` | printed |
| Table 7 and Sec. 11: SMICA axes; CoE 0.2–4.9° from plane; 99% / 93% sectoral | `20_smica/lowl.py`, `20_smica/masked.py` | `data/smica_I_nside64_fwhm3deg_uK.json` | `axes.json`, `masked.json` |
| Appendix B model code | `workspace/cosmology/papers/spreadsheet_model/bubble/{prof_core,ptshell,newt_omb}.py`; LTB: `.../gr_ltb/refit_mass/relfree.py` (+ `rm_core.py`, `ltb_fill.py`) | — | used by the scripts above |

Note (1): `frame_angles.py` computes:
- P2;
- the plane-of-evil normal;
- the P1–P2 axis and the 163.9° end separation;
- the 8.1° offset of the plane normal to the published axis;
- the "Other axes" angles (CMB dipole, Copi normal, hemispherical asymmetry, Bianchi, Bao);
- the RA/Dec of the CoE (Orion, 5° from Betelgeuse; frame CoE near Aldebaran).

## Heavy steps (not run by `run_all.sh`; outputs included)

| Script | What it makes | Rough time |
|---|---|---|
| `15_knobs/sweep.py`, `sweep_ltb.py`, `sweep_inside.py` | `sweep_newton.json` (240 setups), `sweep_ltb.json` (36), `sweep_inside.json` (160) | tens of minutes (Newtonian) to hours (LTB) |
| `06_consistent/fig4/refit_newt_fixedD.py`, `refit_ltb_fixedD.py`, `prof_one.py` | `newt_fixedD.json`, `ltb_fixedD.json`, `prof_DR*.json` (model fits at fixed D) | minutes to an hour each |
| `07_coe_direction/af_curve.py` | `af_D51.json` | minutes |
| `10_residual_P2/axes_check.py` | `axes_check.json` (random-axis nulls for the ℓ = 4 axis) | over 25 min |
| `hmap/run_main.py` | the full `maps_cov.npz` (99 MB) from the 313 Migkas clusters | long. Only the four arrays the paper uses (`MIG_t75__y0/sig/lam/V`) are included, compressed (5 MB) |
| `rerun/fixed/run.py` (+ `coe_ltb.py`) | `rerun/fixed/res/*.json` (exact-GR builds used by the supernova test) | long. Also needs the precomputed tables under `phase1C_v11/` (58 MB, mostly `h_M*.npy`); only `base.py` is included, so this generator cannot run from the package alone |
| `rerun/teamA/mkdata.py` | `migkas_data.npz` | needs `phase1B` inputs not included; the output is included |

## External data

- **Pantheon+ covariance:** needed only for the supernova test. Download `Pantheon+SH0ES_STAT+SYS.cov` (33 MB) from the Pantheon+ data release (github.com/PantheonPlusSH0ES/DataRelease, `Pantheon+_Data/4_DISTANCES_AND_COVAR/`) into `coe_review/data/pantheon/`. The data table `Pantheon+SH0ES.dat` is included.
- **Planck SMICA map:** `data/smica_I_nside64_fwhm3deg_uK.json` (1.3 MB) is the author's own downgraded copy of Planck PR3 `COM_CMB_IQU-smica_2048_R3.00_full.fits` (smoothed to 3°, N_side 64, monopole and dipole removed).
- **Migkas data:**
  - `workspace/cosmology/migkas/fig6_digitized.csv` is the author's digitization of Migkas et al. (2020) Fig. 6 (see its README).
  - `migkas2020_tablec1_vizier.tsv` is the public VizieR table of the 313 clusters.
- **`kiwipumpkin_work/`:** files copied from the author's repository `kiwipumpkin/coe-cosmology`.
- **`06_consistent/fig4/ltb_curve.py`:** keeps one absolute path to a stored validation curve in that repository. It is used only for a self-check, not for any number in the paper.

## Verification (29 Oct 2026)

`run_all.sh` was run from a fresh copy of this package; it took 4 min 20 s on one core. The results:

- **Figures:** all 13 figures came out pixel-identical to the paper's figures.
- **Saved results:** these JSON files were identical to the originals:
  - `chi2_test`, `sheet_joint356`, `geometry_from_aoe`;
  - `compare_2d`, `ring_blend`, `two_scale`;
  - `features_at_fixed`, `combined`;
  - `frame198`, `chi2_198`, `masked`.
- **Paper:** the paper built to 19 pages.
- **Also run and identical:** `10_residual_P2/resid_p2.py` (about 9 min), and the supernova scripts with the Pantheon+ covariance supplied (`sn_test.py`: even part 0.04 ± 0.05; `sn_split.py`: dipole 1.05 ± 0.46).
- **Not re-run:** the steps listed under "Heavy steps".


License: MIT, see [LICENSE](LICENSE).

Citation: see [CITATION.cff](CITATION.cff).
