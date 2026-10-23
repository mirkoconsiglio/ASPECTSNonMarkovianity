# Bath-state (in)dependence of non-Markovianity measures — code

Code and data behind the figures and table of

> M. Consiglio, I. Khomchenko and T. J. G. Apollaro,
> *Bath-state (In)dependence of Non-Markovianity Measures in Quadratic Open Systems*.

The model is a single fermionic level (a spin, after the Jordan–Wigner mapping) with on-site energy
`E`, coupled with strength `sqrt(Gamma*Omega)/2` to the end of a semi-infinite free-fermion chain with
hopping `Omega/2` (band `[-Omega, Omega]`). The chain is held in a Gibbs state at temperature `T` and
chemical potential `mu`. All energies are in units of `Omega` (`Omega = 1` throughout the paper), with
`k_B = hbar = 1`.

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

| File | Role |
|---|---|
| `non_markovianity.py` | Core model: kernels, `G(t)`, `Z_beta(t)`, rates `Lambda^±(t)`, and the five measures (direct and master-equation routes). Also a command-line tool. |
| `combined_panels.py` | Computes one parameter point and draws the two-panel figure (time series + measures vs `T`). Writes `data/E_<E>_Gamma_<Gamma>/series.{csv,npz}`, `data/sweep_measures.csv` and the figure. Figs. 3–5. |
| `replot.py` | Redraws a two-panel figure from a saved `series.npz`, without recomputing. Optionally overlays the master-equation check, cached in `mesolve_check.npz`. |
| `blind_map.py` | `N_BLP` over the `(E, Gamma)` plane (Fig. 6), and an optional cut across the threshold. |
| `mu_sweep.py` | Measures against the chemical potential at fixed `T` (Fig. 7). |
| `msv_families.py` | MSV lower bounds per family (populations, coherences, antipodal), plus a free 5-parameter optimum (Table I, Appendix B 3). |
| `convergence.py` | Window dependence at the Fig. 3 point: writes `data/conv_tmax*.csv` and `data/dip_tmax*.csv` and prints the truncation estimate (≈6% at Ωt = 320) and the MSV dip depth and crossover quoted in Sec. IV. |
| `mesolve_gaps.py` | Prints the direct vs master-equation differences for BLP and LPP (figure captions). |
| `quadratic_schematic.py` | Schematic of the general quadratic model (Fig. 1). |
| `test_nm.py` | Tests: sum rule `Z_beta = 1 - |G|^2` for a filled band, its fourth-order convergence, `dZ/dt`, fast vs slow `Z`. |
| `data/` | Saved runs (see below). |
| `figures/` | Generated figures. Every plot is written as both `.pdf` and `.png`. |

`figures/` also holds exploratory `E_<E>_Gamma_<Gamma>/` folders from parameter sweeps. Only the three
points listed below appear in the paper.

## How the code maps onto the paper

The paper's equations are implemented in `non_markovianity.py` as follows (`F = 0` throughout, since the
chain has no pairing terms). Equation names are the `\label`s in the manuscript's LaTeX source.

| Paper | Code | Notes |
|---|---|---|
| Newns density `J_N(ω)`, Eq. (newns) | `NonMarkovianModel._J` | |
| `φ(ω)`, Eq. (varphi) | `NonMarkovianModel._phi` | Uses `ω − E − Re Σ = ω(1 − Γ/2Ω) − E` and `π J_N = (Γ/2Ω)√(Ω² − ω²)`. |
| Localized levels `ω_b`, residue `R_b`, Eq. (G-bound) | `bound_states(E, Gamma, Omega)` | Returns `(ω_b, R_b)`; saved as `R_b` in `series.npz` and the CSVs. |
| `G(t)`, Eqs. (G-fourier), (G-bound) | `G_t`, built in `compute_kernels` | Band integral on the grid `ω = Ω sin θ`, plus bound-state poles. |
| `γ(t) = −∂_t ln|G|²` and `H(t)`, Eq. (gammaH) | `gamma_t` and `H_t` | `gamma_t` is the total rate γ(t); `Gamma` (capital) is always the coupling Γ. |
| `Z_β(t)`, Eq. (noise_app) | `Z_Tt`, built in `compute_rates` | Single-integral form `∫ J_N f |g(ω,t)|² dω`; the double integral (Eq. Zdouble) is kept as `fast=False`. |
| `Λ^+ = Ż + γ Z`, `Λ^- = γ − Λ^+`, Eqs. (lambdaplus), (lambdaminus) | `Lambda_plus_t`, `Lambda_minus_t` | Checked in `_validate`: `G(0) = 1`, `γ(0) = 0`, `0 ≤ Z ≤ 1 − |G|²`. |
| `N_BLP`, Eq. (blpbound) | `blp_direct` | `max(Σ₊Δ|G|, Σ₊Δ|G|²)`: the polar and equatorial endpoints of Appendix B 1. `theta_scan=True` scans the whole antipodal family. |
| `N_LPP`, `det A = |G|⁴` | `lpp_direct` | |
| `N_MSV`, Eq. (msvbound) | `msv_direct` (figures), `msv_families.py` (table) | Best of the three families of Appendix B 3; a lower bound. |
| `N_RHPE`, concurrence Eq. (concurrence) | `rhpe_direct` | `∫|∂_t C| − [C(0) − C(∞)]`, i.e. twice the summed increases of `C`. |
| `N_RHP`, Eq. (RHPresult) | `rhp_direct` | `∫ max(0, −Λ^+) + max(0, −Λ^-) dt`. Diverges logarithmically at zeros of `G` (a warning is issued). |
| RHP floor `N^(0)_RHP`, Eq. (rhpfloor) | `∫ max(0, −gamma_t) dt` | Computed from `series.npz` or the model; not a separate method. |
| Master-equation check, Eq. (master_equation) | `run(..., direct=False)` → `blp`, `lpp`, `msv`, `rhp` | Evolves the reconstructed generator with `qutip.mesolve`. `msv` sees only the polar and equatorial pairs (a lower bound on `msv_direct`); RHPE has no such route and falls back to `rhpe_direct`. |

**Names.** Variables follow the paper: `gamma_t` = γ(t), `H_t` = H(t), `R_b` = residue, `Z_Tt` = Z_β(t) on the (T, t) grid, `Lambda_plus_t`/`Lambda_minus_t` = Λ^±(t), `A_mat` = the LPP matrix **A**, `s` = the longitudinal shift |G|² + 2Z_β − 1 (Appendix B 3). Archives written before this renaming used `Gamma_t`, `omega_eff_t` and `Z_b`; `replot.py` still reads them.

**Index convention.** Inside the code, index 0 is the **occupied** state and index 1 the vacuum, because
QuTiP's `sigmap() = |0><1|`. This mirrors the paper, where `|0>` is the vacuum. When comparing code with
the paper, read `rho_00` in the code as `rho_11` in the paper. None of the measures depends on this.

**Discretization.** The time integrals use a uniform grid: `t = linspace(0, tmax, nt)` with `nt` odd, for
the cumulative Simpson rule. The direct BLP, LPP and MSV estimators sum positive increments on that grid.

## Reproducing the paper

Run all commands from this folder. Settings are those used for the paper. The three two-panel figures are
expensive at `tmax = 320`, but every array is saved, so appearance changes only need `replot.py`.

### Fig. 1 — quadratic model schematic
```bash
python quadratic_schematic.py              # -> figures/quadratic_model.{pdf,png}
```
Fig. 2 (`Schemes/Spin_chain_illustration_V4.pdf`) is a drawing and is not generated here.

### Figs. 3–5 — dynamics and measures at three parameter points
Compute (writes `data/E_*/series.{csv,npz}` and appends to `data/sweep_measures.csv`):
```bash
# Fig. 3: above the threshold (bound state), E = 2, Gamma = 0.5
python combined_panels.py --E 2.0 --Gamma 0.5 --tmax 320 --nt 32001 \
    --T-log 0.01 10 25 --T-panel-a 1.0 --no-figs
# Fig. 4: on the threshold, E = 0.5, Gamma = 1
python combined_panels.py --E 0.5 --Gamma 1.0 --tmax 40 --nt 4001 \
    --T-log 0.005 10 25 --T-panel-a 0.02 --no-figs
# Fig. 5: below the threshold, E = Gamma = 0.5
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
`data/blind_map.npz` also stores `t_first`, the first revival time quoted in Sec. IV (61, 152 and 633 at
`Gamma = 0.5`, `E − E_c = −0.15, −0.10, −0.05`).

### Fig. 7 — chemical-potential scan
```bash
python mu_sweep.py --E 2.0 --Gamma 0.5 --T 0.05 0.5 --out data/mu_sweep_E2.0_G0.5
python mu_sweep.py --E 0.5 --Gamma 1.0 --T 0.05 0.5 --out data/mu_sweep_E0.5_G1.0
python mu_sweep.py --E 0.5 --Gamma 0.5 --T 0.05 0.5 --out data/mu_sweep_E0.5_G0.5
python mu_sweep.py --figure data/mu_sweep_E2.0_G0.5.npz data/mu_sweep_E0.5_G1.0.npz \
    data/mu_sweep_E0.5_G0.5.npz --logy-cols 1 2 --plot figures/fig_mu_sweep.pdf
```
`--T-index 1` draws the `T = 0.5` panels instead. `mu_sweep_E-2.0_G0.5` is the particle–hole mirror of
`E = 2`: `N(E, mu) = N(−E, −mu)`, an end-to-end check of the implementation.

### Table I — MSV families
```bash
python msv_families.py --free          # -> data/tab.csv, data/tab.tex  (tmax = 100, nt = 2001, 40 restarts)
```

### Truncation and MSV-dip checks (Sec. IV, Fig. 3 point)
```bash
python convergence.py        # t_max = 160 and 320; ~5 GB peak memory, ~25 min
```
Prints the change of every measure between the two windows, the implied residual for a `t_max^{-1/2}` tail, and the depth and crossover of the MSV minimum.

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

| File | Contents |
|---|---|
| `data/E_<E>_Gamma_<Gamma>/series.npz` | `t`, `G`, `gamma_t` (= γ), `H_t` (= H), `T`, `Z_Tt`, `Lambda_plus_Tt`, `Lambda_minus_Tt`, the five measures over `T`, `sqrtJ_pol_panel_a`, and run metadata (`tmax`, `nt`, `min_absG`, `R_b`, …). |
| `data/E_*/series.csv` | The panel-(a) time series as text. |
| `data/E_*/mesolve_check.npz` | Master-equation BLP and LPP at the check temperatures. |
| `data/sweep_measures.csv` | One row per `(E, Gamma, T)` from every `combined_panels.py` run. |
| `data/blind_map.{npz,csv}` | The `(E, Gamma)` scan of Fig. 6. |
| `data/mu_sweep_*.{npz,csv}` | The chemical-potential scans of Fig. 7. |
| `data/conv_tmax*.csv`, `data/dip_tmax*.csv` | Fig. 3 point at `t_max = 160` and `320`: six temperatures, and the full Fig. 3 grid. |
| `data/tab.{csv,tex}` | Table I. |
