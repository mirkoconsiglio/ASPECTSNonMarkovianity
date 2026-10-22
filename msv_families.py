#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
msv_families.py
===============

Lower bounds on the MSV (quantum Jensen--Shannon) non-Markovianity measure for
the F = 0 model of ``non_markovianity.py``, evaluated family by family, plus the
appendix table comparing them.

Background
----------
For F = 0 the map acts differently on the two Bloch directions:

    transverse (coherences)  ->  scaled by |G(t)|, rotated by arg G(t)
    longitudinal (populations) -> scaled by |G(t)|^2, then shifted by

        sigma(t) = |G(t)|^2 + 2 Z(T,t) - 1 .

All the bath dependence sits in sigma. Since J is unitarily invariant and the
rotation acts equally on both inputs, arg G drops out and a pair of inputs is
fixed by five numbers (a1, w1, a2, w2, phi): transverse radii a_i, heights w_i
(with a_i^2 + w_i^2 <= 1) and the relative azimuth phi. With
eta(r) = h((1+r)/2) the entropy of a qubit of Bloch radius r,

    J = eta(m) - eta(r1)/2 - eta(r2)/2 ,
    r_i^2 = |G|^2 a_i^2 + (|G|^2 w_i + sigma)^2 ,
    m^2   = |G|^2 (a1^2 + a2^2 + 2 a1 a2 cos phi)/4
            + (|G|^2 (w1 + w2) + 2 sigma)^2 / 4 .

Because N_MSV is a maximum over pairs, ANY family of pairs gives a lower bound.
Three two-parameter families have closed forms:

  (a) populations only   a1 = a2 = 0            -> parametrised by (c0, Delta0),
                                                   the initial mean occupation
                                                   and occupation gap;
                                                   contains the polar pair.
  (b) coherences only    w1 = w2 = w, a1 = a2 = a, phi = pi
                                                   -> contains the equatorial pair.
  (c) antipodal          v2 = -v1                 -> the family used for the
                                                   trace distance in the BLP
                                                   appendix; contains both the
                                                   polar and equatorial pairs.

Unlike the trace-distance case there is no proof that the optimum lies in any of
them, so ``--free`` additionally runs a multistart Nelder-Mead over all five
parameters as a check.

Usage
-----
    python msv_families.py                             # default table
    python msv_families.py --free                      # + 5-parameter optimum
    python msv_families.py --params 2.0,0.5 1.0,0.5 --T 0.02 0.3 1.0
    python msv_families.py --latex table_msv.tex --csv table_msv.csv

Or import it:

    from msv_families import msv_lower_bound, family_values
    lb = msv_lower_bound(model, iT)          # max over the three families

Requires: numpy, scipy, and ``non_markovianity.py`` (or ``non_markovianity.py``)
on the import path.
"""

from __future__ import annotations

import argparse
from typing import Dict, Sequence, Tuple

import numpy as np

from non_markovianity import ModelConfig, NonMarkovianModel, _h2, _eta, _revivals

FAMILIES = ("polar", "populations", "coherences", "antipodal")


# --------------------------------------------------------------------------- #
# The three closed-form families                                              #
# --------------------------------------------------------------------------- #
def J_populations(absG: np.ndarray, Z: np.ndarray, c0: float, D0: float) -> np.ndarray:
	"""Family (a): both inputs diagonal, initial mean occupation c0 and gap D0.

	p_i(t) = |G|^2 p_i(0) + Z, hence c(t) = |G|^2 c0 + Z and Delta(t) = |G|^2 D0.
	The polar pair is (c0, D0) = (1/2, 1).
	"""
	g2 = absG ** 2
	c = g2 * c0 + Z
	D = g2 * D0
	return _h2(c) - 0.5 * _h2(c + D / 2.0) - 0.5 * _h2(c - D / 2.0)


def J_coherences(absG: np.ndarray, sigma: np.ndarray, a: float, w: float) -> np.ndarray:
	"""Family (b): equal heights w, opposite coherences +-a (phi = pi).

	The equatorial pair is (a, w) = (1, 0).
	"""
	z = absG ** 2 * w + sigma
	r = np.sqrt((absG * a) ** 2 + z ** 2)
	return _eta(np.abs(z)) - _eta(r)


def J_antipodal(absG: np.ndarray, sigma: np.ndarray, a: float, w: float) -> np.ndarray:
	"""Family (c): v2 = -v1 with v1 = (a, 0, w); the BLP family.

	The mean Bloch vector is pair independent, m = |sigma|. Polar pair at
	(a, w) = (0, 1), equatorial pair at (1, 0).
	"""
	g2 = absG ** 2
	r1 = np.sqrt((absG * a) ** 2 + (g2 * w + sigma) ** 2)
	r2 = np.sqrt((absG * a) ** 2 + (-g2 * w + sigma) ** 2)
	return _eta(np.abs(sigma)) - 0.5 * _eta(r1) - 0.5 * _eta(r2)


def J_general(absG, sigma, a1, w1, a2, w2, phi):
	"""The general five-parameter pair (used by the free optimisation)."""
	g2 = absG ** 2
	z1, z2 = g2 * w1 + sigma, g2 * w2 + sigma
	r1 = np.sqrt((absG * a1) ** 2 + z1 ** 2)
	r2 = np.sqrt((absG * a2) ** 2 + z2 ** 2)
	m = 0.5 * np.sqrt(g2 * (a1 ** 2 + a2 ** 2 + 2.0 * a1 * a2 * np.cos(phi))
	                  + (z1 + z2) ** 2)
	return _eta(m) - 0.5 * _eta(r1) - 0.5 * _eta(r2)


# --------------------------------------------------------------------------- #
# Maximisation within each family                                             #
# --------------------------------------------------------------------------- #
def _grid_populations(absG, Z, n: int) -> Tuple[float, Tuple[float, float]]:
	best, arg = 0.0, (0.5, 1.0)
	for c0 in np.linspace(0.0, 1.0, n):
		for D0 in np.linspace(0.0, 2.0 * min(c0, 1.0 - c0), n):
			v = _revivals(J_populations(absG, Z, c0, D0))
			if v > best:
				best, arg = v, (float(c0), float(D0))
	return best, arg


def _grid_disc(fun, absG, sigma, n: int) -> Tuple[float, Tuple[float, float]]:
	"""Maximise over the unit half-disc a >= 0, a^2 + w^2 <= 1."""
	best, arg = 0.0, (0.0, 1.0)
	for a in np.linspace(0.0, 1.0, n):
		wmax = np.sqrt(max(1.0 - a * a, 0.0))
		for w in np.linspace(-wmax, wmax, 2 * n - 1):
			v = _revivals(fun(absG, sigma, a, w))
			if v > best:
				best, arg = v, (float(a), float(w))
	return best, arg


def _free_optimum(absG, sigma, nstart: int, seed: int) -> Tuple[float, np.ndarray]:
	"""Multistart Nelder-Mead over (a1, w1, a2, w2, phi). Diagnostic only."""
	from scipy.optimize import minimize
	
	def ok(p):
		a1, w1, a2, w2, _ = p
		return (a1 >= 0.0 and a2 >= 0.0
		        and a1 * a1 + w1 * w1 <= 1.0 + 1e-12
		        and a2 * a2 + w2 * w2 <= 1.0 + 1e-12)
	
	def neg(p):
		return -_revivals(J_general(absG, sigma, *p)) if ok(p) else 1e3
	
	rng = np.random.default_rng(seed)
	best, arg = 0.0, None
	for p0 in ((0.0, 1.0, 0.0, -1.0, 0.0), (1.0, 0.0, 1.0, 0.0, np.pi)):
		v = -neg(np.array(p0, dtype=float))
		if v > best:
			best, arg = v, np.array(p0, dtype=float)
	for _ in range(nstart):
		p0 = rng.uniform([0, -1, 0, -1, 0], [1, 1, 1, 1, np.pi])
		if not ok(p0):
			continue
		r = minimize(neg, p0, method="Nelder-Mead",
		             options=dict(maxiter=4000, xatol=1e-7, fatol=1e-12))
		if ok(r.x) and -r.fun > best:
			best, arg = -r.fun, r.x
	return best, arg


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def family_values(model: NonMarkovianModel, iT: int, n_grid: int = 121,
                  free: bool = False, nstart: int = 40, seed: int = 0) -> Dict:
	"""All family lower bounds on N_MSV at temperature index ``iT``.

	Returns a dict with the value of each family, the maximising parameters, and
	(with ``free=True``) the unconstrained five-parameter optimum.
	"""
	model.compute_rates()
	absG = np.abs(model.G_t)
	Z = np.asarray(model.Z_Tt[iT], dtype=float)
	sigma = absG ** 2 + 2.0 * Z - 1.0
	
	out: Dict = {"T": float(model.cfg.T[iT])}
	out["polar"] = _revivals(J_populations(absG, Z, 0.5, 1.0))
	out["populations"], out["populations_arg"] = _grid_populations(absG, Z, n_grid)
	out["coherences"], out["coherences_arg"] = _grid_disc(J_coherences, absG, sigma, n_grid)
	out["antipodal"], out["antipodal_arg"] = _grid_disc(J_antipodal, absG, sigma, n_grid)
	out["bound"] = max(out[k] for k in FAMILIES)
	if free:
		out["free"], out["free_arg"] = _free_optimum(absG, sigma, nstart, seed)
	return out


def msv_lower_bound(model: NonMarkovianModel, iT: int = 0, n_grid: int = 121) -> float:
	"""Lower bound on N_MSV: the best of the three closed-form families.

	Drop-in companion to the ``*_direct`` estimators of ``non_markovianity.py``:

		model.run(["BLP", "LPP"]) ...
		msv = [msv_lower_bound(model, i) for i in range(len(model.cfg.T))]
	"""
	return family_values(model, iT, n_grid=n_grid)["bound"]


def build_table(param_list: Sequence[Tuple[float, float]],
                T_list: Sequence[float],
                Omega: float = 1.0, mu: float = 0.0,
                tmax: float = 100.0, nt: int = 2001,
                n_grid: int = 121, free: bool = False,
                nstart: int = 40, verbose: bool = True) -> list:
	"""Run every (E, Gamma) x T combination and return a list of row dicts."""
	rows = []
	for E, Gamma in param_list:
		cfg = ModelConfig(E=E, Gamma=Gamma, Omega=Omega, mu=mu,
		                  tmax=tmax, nt=nt, T=np.asarray(T_list, dtype=float))
		model = NonMarkovianModel(cfg, verbose=False)
		model.compute_rates()
		for iT in range(len(cfg.T)):
			row = family_values(model, iT, n_grid=n_grid, free=free, nstart=nstart)
			row.update(E=E, Gamma=Gamma, Omega=Omega,
			           min_absG=model.G_abs_min)
			rows.append(row)
			if verbose:
				extra = f"  free {row['free']:.6f}" if free else ""
				arg = tuple(round(v, 3) for v in row["populations_arg"])
				print(f"E={E:4} G={Gamma:4} T={row['T']:6.3f} | "
				      f"polar {row['polar']:.6f}  pop {row['populations']:.6f} "
				      f"(c0,D0)={str(arg):16s} coh {row['coherences']:.6f}  "
				      f"anti {row['antipodal']:.6f}{extra}", flush=True)
	return rows


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def write_csv(rows: Sequence[Dict], path: str):
	import csv
	keys = ["E", "Gamma", "Omega", "T", "min_absG", "polar", "populations",
	        "coherences", "antipodal", "bound"] + (["free"] if "free" in rows[0] else [])
	with open(path, "w", newline="") as fh:
		w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
		w.writeheader()
		w.writerows(rows)
	print(f"saved -> {path}")


def write_latex(rows: Sequence[Dict], path: str):
	"""REVTeX table matching the MSV appendix."""
	free = "free" in rows[0]
	cols = "ccc" + "d" * (5 if free else 4)
	head = (r"   $E/\Omega$ & $\Gamma/\Omega$ & $T/\Omega$ & "
	        r"\multicolumn{1}{c}{polar} & \multicolumn{1}{c}{(a) populations} & "
	        r"\multicolumn{1}{c}{(b) coherences} & \multicolumn{1}{c}{(c) antipodal}")
	if free:
		head += r" & \multicolumn{1}{c}{free}"
	lines = [r"\begin{table}[t]",
	         r"  \caption{Revivals of $\sqrt{J}$ maximised within each family of "
	         r"Appendix~\ref{app:MSV}, compared with the polar pair. Every entry is a "
	         r"lower bound on $\cN_\text{MSV}$.}",
	         r"  \label{tab:msvfamilies}",
	         r"  \begin{ruledtabular}",
	         rf"  \begin{{tabular}}{{{cols}}}",
	         head + r" \\ \colrule"]
	for r in rows:
		cells = [f"{r['E']:.1f}", f"{r['Gamma']:.1f}", f"{r['T']:.2f}",
		         f"{r['polar']:.5f}", f"{r['populations']:.5f}",
		         f"{r['coherences']:.5f}", f"{r['antipodal']:.5f}"]
		if free:
			cells.append(f"{r['free']:.5f}")
		lines.append("   " + " & ".join(cells) + r" \\")
	lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table}"]
	with open(path, "w") as fh:
		fh.write("\n".join(lines) + "\n")
	print(f"saved -> {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _pair(s: str) -> Tuple[float, float]:
	E, G = s.split(",")
	return float(E), float(G)


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Family-by-family lower bounds on the MSV measure.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--params", type=_pair, nargs="+",
	               default=[(0.5, 1.0), (1.0, 0.5), (2.0, 0.5)],
	               help="E,Gamma pairs, e.g. --params 2.0,0.5 1.0,0.5")
	p.add_argument("--T", dest="T_list", type=float, nargs="+",
	               default=[0.02, 0.3, 1.0], help="Temperatures.")
	p.add_argument("--Omega", type=float, default=1.0)
	p.add_argument("--mu", type=float, default=0.0)
	p.add_argument("--tmax", type=float, default=100.0)
	p.add_argument("--nt", type=int, default=2001, help="Time points (odd).")
	p.add_argument("--n-grid", type=int, default=121,
	               help="Grid resolution per family parameter.")
	p.add_argument("--free", action="store_true",
	               help="Also run the five-parameter multistart optimisation.")
	p.add_argument("--nstart", type=int, default=40)
	p.add_argument("--csv", default="data/tab.csv")
	p.add_argument("--latex", default="data/tab.tex")
	p.add_argument("--quiet", "-q", action="store_true")
	return p


def main(argv: Sequence[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	rows = build_table(args.params, args.T_list, Omega=args.Omega, mu=args.mu,
	                   tmax=args.tmax, nt=args.nt, n_grid=args.n_grid,
	                   free=args.free, nstart=args.nstart, verbose=not args.quiet)
	if args.csv:
		write_csv(rows, args.csv)
	if args.latex:
		write_latex(rows, args.latex)
	return 0


if __name__ == "__main__":
	main()
