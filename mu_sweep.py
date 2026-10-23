#!/usr/bin/env python
r"""Sweep the chemical potential of the reservoir at fixed temperature(s).

Theorem 1 asserts that N_BLP and N_LPP are independent of the bath state,
"in particular of beta and mu".  Temperature moves the noise term Z_beta(t)
over part of its allowed range; mu moves it over all of it.  At low T, taking
mu from below the band bottom to above the band top turns f(w) from 0 into 1
across the whole spectral support, so Z_beta runs from ~0 up to its ceiling
1 - |G(t)|^2 -- the full width of the affine shift that Eq. (affine) says
cancels.  This is the most severe numerical test of the theorem available.

Implementation: spin_chain_1d accepts mu as one value per temperature row, so
every (T_i, mu_j) pair becomes one row of a single model.  G(t), the Fourier
table and the rates are built once, and all measures come out of one run.

Particle-hole symmetry maps (E, mu) -> (-E, -mu), so N(E, mu) = N(-E, -mu);
running the mirrored point is a free end-to-end check of the implementation.

Outputs
-------
<out>.npz  arrays shaped (n_T, n_mu) for every measure, plus the noise
           filling fraction  zfrac = Z(t_max) / (1 - |G(t_max)|^2)  in [0, 1],
           which shows how much of the allowed shift each bath state explores
<out>.csv  one tidy row per (T, mu)

Usage
-----
    python mu_sweep.py --E 2.0 --Gamma 0.5 --T 0.05 0.5 --mu -1.5 1.5 61 \
        --out data/mu_sweep_E2.0_G0.5 --plot figures/mu_sweep_E2.0_G0.5.pdf
    python mu_sweep.py --E 0.5 --Gamma 1.0 --T 0.05 0.5 --logy \
        --out data/mu_sweep_E0.5_G1.0 --plot figures/mu_sweep_E0.5_G1.0.pdf
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from non_markovianity import ModelConfig, NonMarkovianModel, VALID_MEASURES


def save_figure(fig, path, dpi=600):
	"""Write the figure as vector PDF and raster PNG side by side.

	``path`` may end in .pdf, .png or have no extension; both
	``<stem>.pdf`` and ``<stem>.png`` are written either way, so the paper
	can always include the PDF.
	"""
	stem, ext = os.path.splitext(path)
	if ext.lower() not in (".pdf", ".png"):
		stem = path
	os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
	fig.savefig(stem + ".pdf")
	fig.savefig(stem + ".png", dpi=dpi)
	print("wrote", stem + ".pdf", "and", stem + ".png")


_ORDER = ("BLP", "LPP", "MSV", "RHPE", "RHP")


def sweep(E, Gamma, T_list, mu_grid, Omega=1.0, tmax=40.0, nt=4001,
          n_omega=2049, verbose=True):
	"""All measures on the (T, mu) grid from a single model run."""
	T_list = np.asarray(T_list, dtype=float)
	mu_grid = np.asarray(mu_grid, dtype=float)
	nT, nmu = T_list.size, mu_grid.size
	T_rows = np.repeat(T_list, nmu)
	mu_rows = np.tile(mu_grid, nT)
	
	cfg = ModelConfig(E=E, Gamma=Gamma, Omega=Omega, tmax=tmax, nt=nt,
	                  T=T_rows, mu=mu_rows, n_omega=n_omega)
	model = NonMarkovianModel(cfg, verbose=False)
	t0 = time.time()
	res = model.run(list(VALID_MEASURES))
	if verbose:
		print(f"[mu] E={E:g}, Gamma={Gamma:g}: {nT} x {nmu} bath states in "
		      f"{time.time() - t0:.1f}s", flush=True)
	
	# Bath-independent floor of the divisibility measure (see rhp_floor):
	# N_RHP >= int max(0, -Gamma) dt, saturated when f is flat over the band.
	wS = np.ones(nt)
	wS[1:-1:2], wS[2:-1:2] = 4.0, 2.0
	rhp_floor = float(np.sum(wS * np.clip(-model.gamma_t, 0.0, None)) * cfg.dt / 3.0)
	g2_end = abs(model.G_t[-1]) ** 2
	zfrac = model.Z_Tt[:, -1] / max(1.0 - g2_end, 1e-300)
	bound = getattr(model, "_bound", []) or []
	out = dict(E=E, Gamma=Gamma, Omega=Omega, T=T_list, mu=mu_grid,
	           tmax=tmax, nt=nt, min_absG=model.G_abs_min,
	           bound_state=bool(bound),
	           R_b=float(bound[0][1]) if bound else np.nan,
	           omega_b=float(bound[0][0]) if bound else np.nan,
	           rhp_floor=rhp_floor,
	           zfrac=zfrac.reshape(nT, nmu))
	for k in _ORDER:
		out[k] = np.asarray(res[k], dtype=float).reshape(nT, nmu)
	return out


def report(res):
	"""The theorem check, stated as numbers."""
	print(f"  min_t|G| = {res['min_absG']:.4f}, bound state: {res['bound_state']}")
	for i, T in enumerate(res["T"]):
		z = res["zfrac"][i]
		line = (f"  T = {T:g}:  noise filling spans {z.min():.3f} .. {z.max():.3f} "
		        f"of its allowed range;")
		for k in ("BLP", "LPP"):
			v = res[k][i]
			line += f"  {k} spread over mu = {np.max(v) - np.min(v):.1e}"
		print(line)
		for k in ("MSV", "RHPE", "RHP"):
			v = res[k][i]
			j0, j1 = int(np.argmin(v)), int(np.argmax(v))
			print(f"      {k:4s}: min {v[j0]:.4g} at mu={res['mu'][j0]:+.2f}, "
			      f"max {v[j1]:.4g} at mu={res['mu'][j1]:+.2f}")


def save(res, stem):
	os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
	np.savez_compressed(stem + ".npz", **res)
	with open(stem + ".csv", "w", newline="") as fh:
		w = csv.writer(fh)
		w.writerow(["E", "Gamma", "Omega", "T", "mu", *_ORDER, "zfrac",
		            "tmax", "nt", "min_absG", "bound_state"])
		for i, T in enumerate(res["T"]):
			for j, mu in enumerate(res["mu"]):
				w.writerow([res["E"], res["Gamma"], res["Omega"], T, mu,
				            *[res[k][i, j] for k in _ORDER], res["zfrac"][i, j],
				            res["tmax"], res["nt"], res["min_absG"],
				            res["bound_state"]])
	print("wrote", stem + ".npz", "and", stem + ".csv")


def plot(res, path, logy=False):
	import matplotlib.pyplot as plt
	try:
		from combined_panels import _STYLE, use_paper_style
		use_paper_style()
	except Exception:  # pragma: no cover
		_STYLE = {k: dict(label=k) for k in _ORDER}
	mu, O, E = res["mu"], res["Omega"], res["E"]
	nT = len(res["T"])
	fig = plt.figure(figsize=(3.4 * nT, 3.3), layout="constrained")
	gs = fig.add_gridspec(2, nT, height_ratios=[1, 3])
	for i, T in enumerate(res["T"]):
		top = fig.add_subplot(gs[0, i])
		ax = fig.add_subplot(gs[1, i], sharex=top)
		top.plot(mu, res["zfrac"][i], color="0.3", lw=1.0)
		top.set_ylim(-0.05, 1.05)
		top.set_ylabel(r"$Z/(1-|G|^2)$", fontsize=7)
		top.tick_params(labelbottom=False, labelsize=7)
		top.set_title(rf"$T = {T:g}\,\Omega$", fontsize=9)
		vanish = []
		for k in _ORDER:
			y = res[k][i].astype(float)
			style = dict(_STYLE[k])
			if logy:
				# values at round-off level (~1e-16) are zero, not revivals
				y = np.where(np.abs(y) < 1e-12, 0.0, y)
				if np.all(y == 0.0):
					vanish.append(k)
					style["label"] = "_nolegend_"
				y = np.where(y > 0.0, y, np.nan)
			ax.plot(mu, y, **style)
		for a in (top, ax):
			for x in (-O, O):
				a.axvline(x, color="0.6", lw=0.6, ls=":")
		if logy:
			ax.set_yscale("log")
			if vanish:
				ax.annotate(" = ".join(_STYLE[k]["label"] for k in vanish)
				            + r"$\,\equiv 0$", xy=(0.03, 0.03),
				            xycoords="axes fraction", ha="left", va="bottom",
				            fontsize=8)
		else:
			ax.set_ylim(0.0, None)
		ax.set_xlabel(r"$\mu/\Omega$")
		if i == 0:
			ax.set_ylabel(r"$\mathcal{N}$")
			ax.legend(fontsize=7, handlelength=1.5, labelspacing=0.25)
		ax.set_xlim(mu.min(), mu.max())
	save_figure(fig, path)
	return fig


def paper_figure(results, path, iT=0, logy=(False, True), labels=None):
	"""Side-by-side mu sweeps at one temperature, one column per parameter point.

	Each column: the noise filling Z/(1-|G|^2) on top, the five measures below.
	The bath-independent floor of N_RHP is drawn where it is nonzero; measures
	that vanish identically are named in a corner rather than listed in the
	legend.
	"""
	import matplotlib.pyplot as plt
	try:
		from combined_panels import _STYLE, use_paper_style
		use_paper_style()
	except Exception:  # pragma: no cover
		_STYLE = {k: dict(label=k) for k in _ORDER}
	n = len(results)
	fig = plt.figure(figsize=(7.0, 3.1), layout="constrained")
	gs = fig.add_gridspec(2, n, height_ratios=[1, 3.2])
	for c, res in enumerate(results):
		mu, O, E = res["mu"], res["Omega"], res["E"]
		lg = logy[c] if c < len(logy) else False
		top = fig.add_subplot(gs[0, c])
		ax = fig.add_subplot(gs[1, c], sharex=top)
		top.plot(mu, res["zfrac"][iT], color="0.3", lw=1.0)
		top.set_ylim(-0.08, 1.08)
		top.set_yticks([0, 1])
		top.set_ylabel(r"$Z_\beta/(1-|G|^2)$", fontsize=7)
		top.tick_params(labelbottom=False, labelsize=7)
		lab = (labels[c] if labels else
		       rf"({chr(97 + c)}) $E={E:g}\,\Omega$, $\Gamma={res['Gamma']:g}\,\Omega$")
		top.set_title(lab, fontsize=9, loc="left")
		vanish = []
		for k in _ORDER:
			y = np.asarray(res[k][iT], dtype=float)
			st = dict(_STYLE[k])
			if lg:
				y = np.where(np.abs(y) < 1e-12, 0.0, y)
				if np.all(y == 0.0):
					vanish.append(k)
					st["label"] = "_nolegend_"
				y = np.where(y > 0.0, y, np.nan)
			ax.plot(mu, y, **st)
		fl = float(res.get("rhp_floor", 0.0))
		if fl > 0.0:
			ax.axhline(fl, color=_STYLE["RHP"]["color"], lw=0.7, ls=(0, (1, 1.5)),
			           alpha=0.8, label=r"$\mathcal{N}^{(0)}_{\rm RHP}$")
		for a in (top, ax):
			for x in (-O, O):
				a.axvline(x, color="0.6", lw=0.6, ls=":")
		if lg:
			ax.set_yscale("log")
			if vanish:
				ax.annotate(" = ".join(_STYLE[k]["label"] for k in vanish)
				            + r"$\,\equiv 0$", xy=(0.03, 0.04),
				            xycoords="axes fraction", ha="left", va="bottom",
				            fontsize=8)
		else:
			ax.set_ylim(0.0, None)
		ax.set_xlim(mu.min(), mu.max())
		ax.set_xlabel(r"$\mu/\Omega$")
		if c == 0:
			ax.set_ylabel(r"$\mathcal{N}$")
		# linear panels: the band below the flat BLP line is empty; log panels:
		# the middle decades are empty (curves sit high, tails run down the
		# edges), and the bottom left holds the "= 0" label
		ax.legend(fontsize=6.5, handlelength=1.5, labelspacing=0.2,
		          columnspacing=0.9, loc="center" if lg else "lower center",
		          bbox_to_anchor=(0.5, 0.36) if lg else None,
		          ncol=1 if lg else 3)
	save_figure(fig, path)
	return fig


def _load(npz):
	d = dict(np.load(npz, allow_pickle=True))
	return {k: (v.item() if np.ndim(v) == 0 else v) for k, v in d.items()}


def main(argv=None):
	p = argparse.ArgumentParser(description=__doc__,
	                            formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--E", type=float, default=None)
	p.add_argument("--Gamma", type=float, default=None)
	p.add_argument("--Omega", type=float, default=1.0)
	p.add_argument("--T", type=float, nargs="+", default=[0.05, 0.5],
	               help="Temperatures, one panel each.")
	p.add_argument("--mu", type=float, nargs=3, default=[-1.5, 1.5, 61],
	               metavar=("LO", "HI", "N"))
	p.add_argument("--tmax", type=float, default=40.0)
	p.add_argument("--nt", type=int, default=4001)
	p.add_argument("--n-omega", type=int, default=2049)
	p.add_argument("--logy", action="store_true")
	p.add_argument("--out", default=None)
	p.add_argument("--plot", default=None)
	p.add_argument("--figure", nargs="+", default=None, metavar="NPZ",
	               help="Skip the sweep; draw the side-by-side paper figure "
	                    "from saved sweeps (one column per file).")
	p.add_argument("--T-index", type=int, default=0,
	               help="Which temperature of the saved sweeps to draw.")
	p.add_argument("--logy-cols", type=int, nargs="*", default=None,
	               help="Columns (0-based) with a log vertical axis; default: "
	                    "any column in which BLP vanishes identically.")
	a, rest = p.parse_known_args(argv)
	if a.figure:
		res = [_load(f) for f in a.figure]
		logy = tuple((c in a.logy_cols) if a.logy_cols is not None
		             else bool(np.all(r["BLP"] == 0.0)) for c, r in enumerate(res))
		paper_figure(res, a.plot or "figures/mu_sweep.pdf", iT=a.T_index, logy=logy)
		return 0
	if rest:
		p.error(f"unrecognised arguments: {rest}")
	if a.E is None or a.Gamma is None:
		p.error("--E and --Gamma are required unless --figure is given")
	mu = np.linspace(a.mu[0], a.mu[1], int(a.mu[2]))
	res = sweep(a.E, a.Gamma, a.T, mu, Omega=a.Omega, tmax=a.tmax, nt=a.nt,
	            n_omega=a.n_omega)
	report(res)
	stem = a.out or f"data/mu_sweep_E{a.E:.1f}_G{a.Gamma:.1f}"
	save(res, stem)
	if a.plot:
		plot(res, a.plot, logy=a.logy)
	return 0


if __name__ == "__main__":
	main()