#!/usr/bin/env bash
# Reproduce the paper's numbers and figures (fast and medium steps, ~25 min on one core).
# Heavy sweeps are NOT run here; their saved outputs are included (see README, "Heavy steps").
# Usage:  bash run_all.sh            (from the package root)
#         bash run_all.sh --with-resid   also re-run 10_residual_P2/resid_p2.py (~10 min)
set -e
export COE_PKG="$(cd "$(dirname "$0")" && pwd)"
export MPLBACKEND=Agg OMP_NUM_THREADS=1
R="$COE_PKG/coe_review"
run() { echo; echo "=== $1"; ( cd "$R/$(dirname "$1")" && python3 "$(basename "$1")" ) 2>&1 | grep -v -i "warning" | tail -n "${2:-12}"; }

# Frame geometry (Table 1, Sec. 4)
run 06_consistent/geometry_from_aoe.py
run paper_extras/frame_angles.py 30
# Cluster curves: Tier A, Fig. 1, Fig. 8, Fourier comparison, rotation-term direction
run 17_paperfigs/chi2_test.py 30
run 17_paperfigs/fig_dipole_cmp.py
run paper_extras/fourier_compare.py
run 17_paperfigs/sheet_AB.py
run 17_paperfigs/sheet_joint356.py          # ~3 min -> 14_paper/figs/fig_AB.png
run 17_paperfigs/phase_profile.py 30        # ~1 min
# Robustness (Appendix C)
run 07_coe_direction/dir_fit.py
run paper_extras/robustness.py
# Tier B and Tier C (Sec. 8, Table 3, Fig. 9)
run paper_extras/tierB_patch.py
run 13_combined/combined.py                 # ~1 min, 20,000 draws per null
run paper_extras/tierC_combine.py
( cd "$R/14_paper" && python3 null_fig.py )
# Residual features (Table 4, Fig. 10)
run 06_consistent/features_at_fixed.py
run 06_consistent/figs/chris_AB_residual_spots.py
# 2-D map: ring blends, roll, chain, multipoles (Figs. 4-7)
run 09_values_blend/compare/compare_2d.py
run 09_values_blend/compare/ring_blend.py
run 09_values_blend/compare/two_scale.py
( cd "$R/14_paper" && python3 roll_fig.py )
run 12_flow/flow_chain.py
if [ "$1" = "--with-resid" ]; then run 10_residual_P2/resid_p2.py 30; fi
run 12_flow/flow_3.py
# Knob sweep figure (Fig. 3, from saved sweep results)
run 15_knobs/fig_knobs.py
run paper_extras/knobs_summary.py 20
# 3-D frame (Fig. 2) and edge-on views (Fig. 13)
run paper_extras/make_frame_views.py
# Dipoles and sky chart (Figs. 11, 12)
run 17_paperfigs/fig_dipoles_and_chart.py
# Rebuilt frame on (198, -4) (Limitations)
run 18_rebuild/frame198.py
run 18_rebuild/chi2_198.py
# Planck SMICA low-l axes (Sec. 11, Appendix E)
run 20_smica/lowl.py 20
run 20_smica/masked.py 20                   # ~4 min
# Supernova test (Sec. 9): needs the Pantheon+ covariance (see README)
if [ -f "$R/data/pantheon/Pantheon+SH0ES_STAT+SYS.cov" ]; then run joint/sn/sn_test.py 30; run joint/sn/sn_split.py 30; else echo; echo "=== skipping joint/sn/sn_test.py (Pantheon+ covariance not present)"; fi
# Copy outputs to the paper's figure names
run paper_extras/collect_figs.py 20
echo; echo "Done. Figures are in coe_review/14_paper/figs. Build the paper with: cd coe_review/14_paper && pdflatex two_directions_working_draft_v4.tex (twice)"
