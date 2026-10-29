# P3 outcome: supernova test of the pre-registered Hubble-rate anisotropy

The prediction is P3 in `../blind_predictions.md` (SHA-256 ed0490f2…7a86).

**Data and method.**
- Data: Pantheon+ (GitHub DataRelease), 490 Hubble-flow supernovae at 0.0233 < z < 0.15, calibrators excluded.
- Covariance: full STAT+SYS.
- Absolute magnitude: marginalised.
- Scripts: `sn_test.py`, `sn_split.py`.

**Results.**

| Test | Result |
|---|---|
| Full C* template at the predicted amplitude (β = 1), frame axis | Δχ² = **+335** relative to isotropy (strongly excluded) |
| Best-fit amplitude of the full template | β = 0.03 ± 0.05 |
| Same, using CMB-frame redshifts instead of flow-corrected ones | Δχ² = +328; β = 0.04 ± 0.05 |
| Dipole part only (1.0% along the frame axis) | amplitude **1.05 ± 0.46** (prediction 1) — consistent |
| Even part only (quadrupole-dominated; P₂ coefficient 15%) | amplitude **0.04 ± 0.05** (prediction 1) — excluded at about 19σ |
| Context: free dipole + quadrupole fit (8 parameters) | Δχ² = −14; dipole 1.9% toward (339°, +6°), about 23° from the frame axis |

**Reading.**
- **The dipole passes.** Pantheon+ shows the predicted direction and size of the dipole.
- **The quadrupole fails.** The large tidal-shear quadrupole (H_r/H_t ≈ 1.27) is absent. That quadrupole is what fits the Migkas cluster anisotropy. Supernovae limit the even part to less than about 10% of the predicted size (2σ), i.e. a peak-to-peak H variation of about 3% or less, against about 22% predicted.
- **Implication.** Either the Migkas cluster anisotropy is not mainly an anisotropy in the cosmic Hubble rate (it could be cluster-specific, for example scaling-relation systematics or absorption), or the CoE shear that reproduces it is too large.
