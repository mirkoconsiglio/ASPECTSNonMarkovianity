# Bath-state (in)dependence of non-Markovianity measures — code

Code and data behind the figures and table of

> M. Consiglio, I. Khomchenko and T. J. G. Apollaro,
> *Bath-state (In)dependence of Non-Markovianity Measures in Quadratic Open Systems*.

The model (Sec. III of the paper) is a single fermionic site d̂ with on-site energy E, Ĥ_S = E d̂†d̂,
coupled with strength √(ΓΩ)/2 to the terminal site ĉ₁ of a semi-infinite chain of free fermions ĉ_k with
uniform hopping Ω/2, so that the band is [−Ω, Ω]. The chain is held in a Gibbs state at temperature
T = 1/β and chemical potential μ, with k_B = ħ = 1. In the code these are `E`, `Gamma`, `Omega`, `T` and
`mu`; energies are in units of Ω (`Omega = 1` by default) and times in units of 1/Ω.

## Requirements

- Python ≥ 3.9, `numpy`, `scipy`, `matplotlib`
- `qutip` ≥ 4.6 (tested with 5.3). It is needed only for the master-equation cross-checks (`mesolve`).
  The default ("direct") estimators do not call it.
- Optional: `tqdm` (progress bars), `pytest` (tests)
- The figure scripts typeset with LaTeX by default. Set `NM_USETEX=0` to draw without a LaTeX install.

```bash
pip install numpy scipy matplotlib qutip tqdm pytest
```

## Files

| File                     | Role                                                                                                                                                                                                          |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `non_markovianity.py`    | Core model: kernels, G(t), Z_β(t), the rates Λ^±(t), and the five measures (direct and master-equation routes). Also a command-line tool.                                                                     |
| `combined_panels.py`     | Computes one parameter point and draws the two-panel figure (time series + measures vs T). Writes `data/E_<E>_Gamma_<Gamma>/series.{csv,npz}`, `data/sweep_measures.csv` and the figure. Figs. 3–5.           |
| `replot.py`              | Redraws a two-panel figure from a saved `series.npz`, without recomputing. Optionally overlays the master-equation check, cached in `mesolve_check.npz`.                                                      |
| `blind_map.py`           | 𝒩_BLP over the (E, Γ) plane (Fig. 6), and an optional cut across the threshold.                                                                                                                               |
| `mu_sweep.py`            | Measures against the chemical potential μ at fixed T (Fig. 7).                                                                                                                                                |
| `msv_families.py`        | MSV lower bounds per family (J_pop, J_coh, J_anti), plus a free 5-parameter optimum (Table I, Appendix B 3).                                                                                                  |
| `convergence.py`         | Window dependence at the Fig. 3 point: writes `data/conv_tmax*.csv` and `data/dip_tmax*.csv` and prints the truncation estimate (≈6% at Ω t_max = 320) and the MSV dip depth and crossover quoted in Sec. IV. |
| `mesolve_gaps.py`        | Prints the direct vs master-equation differences for BLP and LPP (figure captions).                                                                                                                           |
| `quadratic_schematic.py` | Schematic of the general quadratic model (Fig. 1).                                                                                                                                                            |
| `test_nm.py`             | Tests: the filled-band sum rule Z_β = 1 − \|G\|², its fourth-order convergence, ∂_t Z_β, and the fast vs slow Z_β routes.                                                                                     |
| `data/`                  | Saved runs (see below).                                                                                                                                                                                       |
| `figures/`               | Generated figures. Every plot is written as both `.pdf` and `.png`.                                                                                                                                           |

`figures/` also holds exploratory `E_<E>_Gamma_<Gamma>/` folders from parameter sweeps. Only the three
points listed below appear in the paper.

## How the code maps onto the paper

The paper's equations are implemented in `non_markovianity.py` as follows (`F = 0` throughout, since the
chain has no pairing terms). Equation names are the `\label`s in the manuscript's LaTeX source.

| Paper                                                                  | Code                                                  | Notes                                                                                                                                                                                             |
|------------------------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Newns density 𝒥_N(ω), Eq. (newns)                                      | `NonMarkovianModel._J`                                |                                                                                                                                                                                                   |
| φ(ω), Eq. (varphi)                                                     | `NonMarkovianModel._phi`                              | Uses ω − E − Re Σ(ω) = ω(1 − Γ/2Ω) − E and π𝒥_N(ω) = (Γ/2Ω)√(Ω² − ω²).                                                                                                                            |
| Localized levels ω_b with residues R_b, Eq. (G-bound)                  | `bound_states(E, Gamma, Omega)`                       | Returns `(ω_b, R_b)`; saved as `R_b` in `series.npz` and the CSVs.                                                                                                                                |
| G(t), Eqs. (G-fourier), (G-bound)                                      | `G_t`, built in `compute_kernels`                     | Band integral on the grid ω = Ω sin θ, plus the localized-level terms R_b e^(−iω_b t).                                                                                                            |
| γ(t) and H(t), from iH + γ/2 = −∂_t G/G, Eq. (gammaH)                  | `gamma_t` and `H_t`                                   | `gamma_t` is the total rate γ(t); `Gamma` (capital) is always the coupling Γ.                                                                                                                     |
| Z_β(t), Eq. (noise_app)                                                | `Z_Tt`, built in `compute_rates`                      | Single-integral form ∫ 𝒥_N(ω) f(ω) \|g(ω,t)\|² dω; the double integral, Eq. (Zdouble), is kept as `fast=False`.                                                                                   |
| Λ^+ = ∂_t Z_β + γ Z_β, Λ^- = γ − Λ^+, Eqs. (lambdaplus), (lambdaminus) | `Lambda_plus_t`, `Lambda_minus_t`                     | Checked in `_validate`: G(0) = 1, γ(0) = 0, 0 ≤ Z_β ≤ 1 − \|G\|².                                                                                                                                 |
| 𝒩_BLP, Eq. (blpbound)                                                  | `blp_direct`                                          | Maximum over P ∈ {\|G\|, \|G\|²} of ∫ max{0, ∂_t P} dt: the equatorial and polar endpoints of Appendix B 1. `theta_scan=True` scans the whole antipodal family.                                   |
| 𝒩_LPP, det **A** = \|G\|⁴                                              | `lpp_direct`                                          |                                                                                                                                                                                                   |
| 𝒩_MSV, Eq. (msvbound)                                                  | `msv_direct` (figures), `msv_families.py` (table)     | Best of J_pop, J_coh and J_anti (Appendix B 3); a lower bound.                                                                                                                                    |
| 𝒩_RHPE, concurrence 𝒞, Eq. (concurrence)                               | `rhpe_direct`                                         | ∫ \|∂_t 𝒞\| dt − [𝒞(0) − 𝒞(∞)], i.e. twice the summed increases of 𝒞.                                                                                                                             |
| 𝒩_RHP, Eq. (RHPresult)                                                 | `rhp_direct`                                          | ∫ (max{0, −Λ^+} + max{0, −Λ^-}) dt. Diverges logarithmically at zeros of G (a warning is issued).                                                                                                 |
| RHP floor 𝒩^(0)_RHP, Eq. (rhpfloor)                                    | ∫ max{0, −γ} dt, from `gamma_t`                       | Computed from `series.npz` or the model; not a separate method.                                                                                                                                   |
| Master-equation check, Eq. (master_equation)                           | `run(..., direct=False)` → `blp`, `lpp`, `msv`, `rhp` | Evolves the reconstructed generator with `qutip.mesolve`. `msv` sees only the polar and equatorial pairs (a lower bound on `msv_direct`); RHPE has no such route and falls back to `rhpe_direct`. |

**Names.** Variables follow the paper: `gamma_t` = γ(t), `H_t` = H(t), `R_b` = residue, `Z_Tt` = Z_β(t) on the (T, t) grid, `Lambda_plus_t`/`Lambda_minus_t` = Λ^±(t), `A_mat` = the LPP matrix **A**, `s` = the longitudinal shift |G|² + 2Z_β − 1 (Appendix B 3). Archives written before this renaming used `Gamma_t`, `omega_eff_t` and `Z_b`; `replot.py` still reads them.

**Index convention.** Inside the code, index 0 is the **occupied** state and index 1 the vacuum, because
QuTiP's `sigmap() = |0><1|`. This mirrors the paper, where |0⟩ is the vacuum and |1⟩ the occupied state. When comparing code with
the paper, read `rho_00` in the code as ρ₁₁ in the paper. None of the measures depends on this.

**Discretization.** The time integrals use a uniform grid: `t = linspace(0, tmax, nt)` with `nt` odd, for
the cumulative Simpson rule. The direct BLP, LPP and MSV estimators sum positive increments on that grid.

## Reproducing the paper

Run all commands from this folder. Settings are those used for the paper. The three two-panel figures are
expensive at Ω t_max = 320, but every array is saved, so appearance changes only need `replot.py`.

### Fig. 1 — quadratic model schematic
```bash
python quadratic_schematic.py              # -> figures/quadratic_model.{pdf,png}
```
Fig. 2 (`figures/fermionic_model.pdf`) is a drawing and is not generated here.

### Figs. 3–5 — dynamics and measures at three parameter points
Compute (writes `data/E_*/series.{csv,npz}` and appends to `data/sweep_measures.csv`):
```bash
# Fig. 3: above the threshold (localized level), E = 2Ω, Γ = Ω/2
python combined_panels.py --E 2.0 --Gamma 0.5 --tmax 320 --nt 32001 \
    --T-log 0.01 10 25 --T-panel-a 1.0 --no-figs
# Fig. 4: on the threshold, E = Ω/2, Γ = Ω
python combined_panels.py --E 0.5 --Gamma 1.0 --tmax 40 --nt 4001 \
    --T-log 0.005 10 25 --T-panel-a 0.02 --no-figs
# Fig. 5: below the threshold, E = Γ = Ω/2
python combined_panels.py --E 0.5 --Gamma 0.5 --tmax 40 --nt 4001 \
    --T-log 0.005 10 25 --T-panel-a 0.02 --no-figs
```
Draw (with the master-equation open circles where the paper shows them):
```bash
python replot.py data/E_2.0_Gamma_0.5/series.npz --t-window 40 --mesolve-check 0.01 0.1 1 10 \
    --legend-loc "lower center" -o figures/E_2.0_Gamma_0.5/figure
python replot.py data/E_0.5_Gamma_1.0/series.npz --t-window 25 --logy --no-inset \
    --annot-loc "upper right" -o figures/E_0.5_Gamma_1.0/figure
python replot.py data/E_0.5_Gamma_0.5/series.npz --logy --no-inset --annot-loc "upper right" \
    --legend-loc "center right" --zero-loc "center left" --mesolve-check 0.01 0.1 1 10 \
    -o figures/E_0.5_Gamma_0.5/figure
```
The first `--mesolve-check` run solves the master equation and caches the result in
`data/E_*/mesolve_check.npz`. Add `--refresh-mesolve` to recompute it. `mesolve_gaps.py` prints the same
comparison as text, for example:
```bash
python mesolve_gaps.py --E 2.0 --Gamma 0.5 --tmax 320 --nt 32001 --T 0.01 0.1 1 10
```

### Fig. 6 — where BLP and LPP vanish
```bash
python blind_map.py --tmax 1280 --dt 0.05 --plot figures/blind_map.pdf    # full scan
python blind_map.py --from data/blind_map.npz --plot figures/blind_map.pdf  # redraw only
```
`data/blind_map.npz` also stores `t_first`, the first revival time quoted in Sec. IV (Ωt ≃ 61, 152 and 633 along
Γ = Ω/2, at E − E_c = −0.15, −0.10 and −0.05Ω).

### Fig. 7 — chemical-potential scan
```bash
python mu_sweep.py --E 2.0 --Gamma 0.5 --T 0.05 0.5 --out data/mu_sweep_E2.0_G0.5
python mu_sweep.py --E 0.5 --Gamma 1.0 --T 0.05 0.5 --out data/mu_sweep_E0.5_G1.0
python mu_sweep.py --E 0.5 --Gamma 0.5 --T 0.05 0.5 --out data/mu_sweep_E0.5_G0.5
python mu_sweep.py --figure data/mu_sweep_E2.0_G0.5.npz data/mu_sweep_E0.5_G1.0.npz \
    data/mu_sweep_E0.5_G0.5.npz --logy-cols 1 2 --plot figures/fig_mu_sweep.pdf
```
`--T-index 1` draws the T = Ω/2 panels instead. `mu_sweep_E-2.0_G0.5` is the particle–hole mirror of
E = 2Ω, 𝒩(E, μ) = 𝒩(−E, −μ), an end-to-end check of the implementation.

### Table I — MSV families
```bash
python msv_families.py --free          # -> data/tab.csv, data/tab.tex  (Ω t_max = 100, nt = 2001, 40 restarts)
```

### Truncation and MSV-dip checks (Sec. IV, Fig. 3 point)
```bash
python convergence.py        # Ω t_max = 160 and 320; ~5 GB peak memory, ~25 min
```
Prints the change of every measure between the two windows, the implied residual for a t_max^(−1/2) tail, and the depth and crossover of the MSV minimum.

### Tests
```bash
pytest test_nm.py
```

## Command-line use of the core module

```bash
python non_markovianity.py --E 2.0 --Gamma 0.5 --tmax 40 --nt 4001 --T 0 0.1 1 10 \
    --measure all --save out/run --plot-file out/run
python non_markovianity.py --help
```
From Python:
```python
import numpy as np
from non_markovianity import ModelConfig, NonMarkovianModel
m = NonMarkovianModel(ModelConfig(E=2.0, Gamma=0.5, tmax=40, nt=4001, T=np.array([0.02, 1.0])))
res = m.run(["BLP", "LPP", "MSV", "RHPE", "RHP"])   # dict of arrays over T
print(res["BLP"], m._bound)                         # bound states [(omega_b, R_b)]
```

## Data files

| File                                        | Contents                                                                                                                                                                                          |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data/E_<E>_Gamma_<Gamma>/series.npz`       | `t`, `G`, `gamma_t` (= γ), `H_t` (= H), `T`, `Z_Tt`, `Lambda_plus_Tt`, `Lambda_minus_Tt`, the five measures over `T`, `sqrtJ_pol_panel_a`, and run metadata (`tmax`, `nt`, `min_absG`, `R_b`, …). |
| `data/E_*/series.csv`                       | The panel-(a) time series as text.                                                                                                                                                                |
| `data/E_*/mesolve_check.npz`                | Master-equation BLP and LPP at the check temperatures.                                                                                                                                            |
| `data/sweep_measures.csv`                   | One row per (E, Γ, T) from every `combined_panels.py` run.                                                                                                                                        |
| `data/blind_map.{npz,csv}`                  | The (E, Γ) scan of Fig. 6.                                                                                                                                                                        |
| `data/mu_sweep_*.{npz,csv}`                 | The chemical-potential scans of Fig. 7.                                                                                                                                                           |
| `data/conv_tmax*.csv`, `data/dip_tmax*.csv` | Fig. 3 point at Ω t_max = 160 and 320: six temperatures, and the full Fig. 3 grid.                                                                                                                |
| `data/tab.{csv,tex}`                        | Table I.                                                                                                                                                                                          |
