#!/usr/bin/env python3
r"""Window (t_max) dependence of the measures at the Fig. 3 point.

Backs two statements in Sec. IV of the paper, for E = 2, Gamma = 0.5 (above the
threshold, where the bound state keeps |G| ~ 1 and the rates decay only as
t^{-3/2}):

  * Truncation. The measures change by a few percent between t_max = 160 and
    320. For a t_max^{-1/2} tail, N(t_max) = N_inf - c t_max^{-1/2}, the
    residual at the larger window W2 follows from the change between W1 and W2
    as  Delta / (sqrt(W2/W1) - 1)  (2.4 x Delta for W2 = 2 W1).
  * The MSV minimum near T = 0.1. Its depth below the T = 10 plateau, and the
    crossover temperature at which the optimal pair switches from mixed
    diagonal states (family (a)) to the polar pair, do not move with t_max.

Outputs, in the CSV format of combined_panels.py (one row per temperature):

  data/conv_tmax<W>.csv   T = 0, 1, 2.5, 5, 7.5, 10
  data/dip_tmax<W>.csv    T = 0 and 25 log-spaced values from 0.01 to 10
                          (the temperature grid of Fig. 3)

Usage:
    python convergence.py                     # t_max = 160 and 320
    python convergence.py --tmax 160 320 640  # nt = 100 t_max + 1 by default

Memory: at t_max = 320 (nt = 32001) the rates need ~5 GB at peak. N_MSV is
evaluated one family at a time (the maximum over families is unchanged), so
the arrays of only one family are held at once.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings

import numpy as np

from combined_panels import CSV_FIELDS, rows_from_run
from non_markovianity import ModelConfig, NonMarkovianModel

T_CONV = np.array([0.0, 1.0, 2.5, 5.0, 7.5, 10.0])
T_DIP = np.concatenate(([0.0], np.geomspace(0.01, 10.0, 25)))
FAMILIES = ("pop", "coh", "anti")


def run_window(E, Gamma, Omega, mu, tmax, nt, verbose=True):
    """All five measures on the union of both temperature grids, plus, per T,
    which MSV family wins and whether family (a) beats the polar pair."""
    T = np.union1d(T_CONV, T_DIP)
    cfg = ModelConfig(E=E, Gamma=Gamma, Omega=Omega, mu=mu, tmax=tmax, nt=nt, T=T)
    m = NonMarkovianModel(cfg, verbose=False)
    t0 = time.time()
    res = m.run(["BLP", "LPP", "RHPE", "RHP"])
    msv, winner, pop_gain = [], [], []
    for iT in range(len(T)):
        vals = {}
        for f in FAMILIES:  # one family at a time keeps the peak memory low
            _, v = m.msv_direct(iT, families=(f,), return_all=True)
            vals[f] = v[f]
            vals["polar"] = v["polar"]
        best = max(vals[f] for f in FAMILIES)
        msv.append(best)
        winner.append(max(FAMILIES, key=lambda f: vals[f]))
        pop_gain.append(vals["pop"] / vals["polar"] - 1.0 if vals["polar"] > 0 else 0.0)
        if verbose:
            print(f"  t_max={tmax:g}  T={T[iT]:8.4f}  MSV={best:.6f} ({winner[-1]})"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    res["MSV"] = np.array(msv)
    bound = m._bound or []
    info = dict(E=E, Gamma=Gamma, Omega=Omega, mu=mu, tmax=tmax, nt=nt,
                min_absG=m.G_abs_min, min_absG_t=m.G_abs_min_t,
                bound_state=bool(bound),
                R_b=float(bound[0][1]) if bound else float("nan"))
    return rows_from_run(res, info), np.array(pop_gain), T


def select(rows, T_sel):
    return [r for r in rows if np.any(np.isclose(r["T"], T_sel, rtol=1e-12, atol=0.0))]


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {path}")


def summary(results):
    """Print the numbers quoted in the paper."""
    W = sorted(results)
    print("\nMSV minimum (T > 0) and plateau at T = 10:")
    for w in W:
        rows = select(results[w]["rows"], T_DIP)
        T = np.array([r["T"] for r in rows])
        msv = np.array([r["MSV"] for r in rows])
        pos = T > 0
        i = np.argmin(msv[pos])
        plateau = msv[np.isclose(T, 10.0)][0]
        gain = results[w]["pop_gain"]
        Tall = results[w]["T"]
        mixed = Tall[(Tall > 0) & (gain > 1e-9)]
        polar = Tall[(Tall > 0) & (gain <= 1e-9)]
        lo = mixed.max() if mixed.size else float("nan")
        hi = polar[polar > lo].min() if polar.size and np.any(polar > lo) else float("nan")
        print(f"  t_max={w:g}: min {msv[pos][i]:.5f} at T={T[pos][i]:.4f}, plateau {plateau:.5f}, "
              f"depth {100 * (plateau - msv[pos][i]) / plateau:.2f}%, "
              f"crossover {lo:.4g} < T* < {hi:.4g}")
    for w1, w2 in zip(W[:-1], W[1:]):
        r1 = select(results[w1]["rows"], T_CONV)
        r2 = select(results[w2]["rows"], T_CONV)
        print(f"\nChange from t_max={w1:g} to {w2:g} (T > 0), and the implied residual at "
              f"{w2:g} for a t_max^-1/2 tail:")
        fac = 1.0 / (np.sqrt(w2 / w1) - 1.0)
        for k in ("BLP", "LPP", "MSV", "RHPE", "RHP"):
            a = np.array([r[k] for r in r1 if r["T"] > 0])
            b = np.array([r[k] for r in r2 if r["T"] > 0])
            rel = (b - a) / b
            print(f"  {k:5s}: change {100 * rel.min():.2f}..{100 * rel.max():.2f}%   "
                  f"residual ~{100 * fac * rel.min():.1f}..{100 * fac * rel.max():.1f}%")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--E", type=float, default=2.0)
    p.add_argument("--Gamma", type=float, default=0.5)
    p.add_argument("--Omega", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--tmax", type=float, nargs="+", default=[160.0, 320.0])
    p.add_argument("--dt", type=float, default=0.01, help="Time step; nt = tmax/dt + 1.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--quiet", "-q", action="store_true")
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    results = {}
    for w in a.tmax:
        nt = int(round(w / a.dt)) + 1
        if nt % 2 == 0:
            nt += 1
        print(f"t_max = {w:g}, nt = {nt}", flush=True)
        rows, gain, T = run_window(a.E, a.Gamma, a.Omega, a.mu, w, nt, verbose=not a.quiet)
        tag = f"{w:g}"
        write_csv(select(rows, T_CONV), os.path.join(a.data_dir, f"conv_tmax{tag}.csv"))
        write_csv(select(rows, T_DIP), os.path.join(a.data_dir, f"dip_tmax{tag}.csv"))
        results[w] = dict(rows=rows, pop_gain=gain, T=T)
    summary(results)
    return 0


if __name__ == "__main__":
    main()
