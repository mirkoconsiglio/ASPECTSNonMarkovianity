#!/usr/bin/env python
"""Redraw a two-panel figure from a saved ``series.npz`` -- no recomputation.

``combined_panels.py`` writes every array it uses into
``data/E_<E>_Gamma_<Gamma>/series.npz``, so the figure can be rebuilt from disk
in a second whenever only its appearance needs to change (inset corner, log
axes, panel-(a) temperature).  Nothing here touches the model, so qutip is not
needed.

Examples
--------
    python replot.py data/E_0.5_Gamma_1.0/series.npz --inset-loc "lower left" --logy
    python replot.py data/E_2.0_Gamma_0.5/series.npz -o figures/alt.pdf
    python replot.py data/E_2.0_Gamma_0.5/series.npz --mesolve-check 0.01 0.1 1 10
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from combined_panels import (_ANNOT_LOC, _INSET_LOC, _LEGEND_LOC, _ORDER,
                             _ZERO_LOC, draw_panel_a, draw_panel_b,
                             mesolve_check, mesolve_gaps, use_paper_style)


def load_or_run_mesolve(npz_path, z, check_T, refresh=False):
    """Master-equation BLP/LPP at the temperatures ``check_T`` (open circles).

    series.npz does not store the cross-check, so it is computed once with the
    point's own (E, Gamma, Omega, mu, tmax, nt) and cached beside the archive as
    mesolve_check.npz; later replots with the same temperatures reuse it.
    """
    T_req = np.asarray(check_T, dtype=float)
    cache = os.path.join(os.path.dirname(os.path.abspath(npz_path)),
                         "mesolve_check.npz")
    if os.path.exists(cache) and not refresh:
        c = np.load(cache)
        if (c["T"].shape == T_req.shape and np.allclose(c["T"], T_req)
                and int(c["nt"]) == int(z["nt"])
                and float(c["tmax"]) == float(z["tmax"])):
            print("using cached master-equation check", cache)
            return {"T": c["T"], "BLP": c["BLP"], "LPP": c["LPP"]}
    print("running master-equation check at T =", T_req.tolist(), flush=True)
    r = mesolve_check(T_req, E=float(z["E"]), Gamma=float(z["Gamma"]),
                      Omega=float(z["Omega"]), mu=float(z["mu"]),
                      tmax=float(z["tmax"]), nt=int(z["nt"]))
    check = {"T": T_req, "BLP": np.asarray(r["BLP"], dtype=float),
             "LPP": np.asarray(r["LPP"], dtype=float)}
    np.savez(cache, tmax=float(z["tmax"]), nt=int(z["nt"]), **check)
    print("cached in", cache)
    return check


def default_out(npz_path):
    """Where combined_panels.py would have put this point's figure.

    The archives live in data/<point>/series.npz and the figures in
    figures/<point>/figure.*, so writing beside the npz would quietly produce a
    second copy in the data tree while the one the paper includes stays stale.
    Swap the last path component named "data" for "figures"; if there is none,
    fall back to the archive's own directory.
    """
    d = os.path.dirname(os.path.abspath(npz_path))
    parts = d.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data":
            parts[i] = "figures"
            return os.path.join(os.sep.join(parts), "figure")
    return os.path.join(d, "figure")


def model_from_npz(z, iT):
    """A stand-in carrying exactly the attributes the drawing code reads."""
    bound = ([(float("nan"), float(z["Z_b"]))]
             if bool(z["bound_state"]) and np.isfinite(z["Z_b"]) else [])
    cfg = SimpleNamespace(t=z["t"], T=z["T"], E=float(z["E"]),
                          Gamma=float(z["Gamma"]), Omega=float(z["Omega"]),
                          mu=float(z["mu"]), tmax=float(z["tmax"]),
                          nt=int(z["nt"]))
    return SimpleNamespace(
        cfg=cfg, G_t=z["G"], Gamma_t=z["Gamma_t"],
        Lambda_plus_t=np.asarray(z["Lambda_plus_Tt"], dtype=float),
        Lambda_minus_t=np.asarray(z["Lambda_minus_Tt"], dtype=float),
        Z_Tt=np.asarray(z["Z_Tt"], dtype=float),
        G_abs_min=float(z["min_absG"]), G_abs_min_t=float(z["min_absG_t"]),
        _bound=bound,
        sqrtJ_pol=(z["sqrtJ_pol_panel_a"] if "sqrtJ_pol_panel_a" in z.files
                   else None),
        sqrtJ_pol_iT=int(z["iT_a"]))


def replot(npz_path, inset_loc="upper right", inset=True, logy=False,
           logx=True, T_panel_a=None, t_window=None,
           annot_loc="lower right", legend_loc="upper right",
           zero_loc="lower left", out=None, check_T=None,
           refresh_mesolve=False):
    z = np.load(npz_path)
    T = z["T"]
    iT = (int(np.argmin(np.abs(T - T_panel_a))) if T_panel_a is not None
          else int(z["iT_a"]))
    res = {"T": T, **{k: z[k] for k in _ORDER if k in z.files}}

    use_paper_style()
    fig = plt.figure(figsize=(7.0, 4.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.1],
                          height_ratios=[1.0, 1.2, 1.0])
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    a2 = fig.add_subplot(gs[2, 0], sharex=a0)
    b = fig.add_subplot(gs[:, 1])

    draw_panel_a(model_from_npz(z, iT), iT, (a0, a1, a2),
                 t_window=t_window, annot_loc=annot_loc)
    check = None
    if check_T:
        check = load_or_run_mesolve(npz_path, z, check_T, refresh_mesolve)
        mesolve_gaps(res, check, T)
    draw_panel_b(res, b, check=check, inset=inset, inset_loc=inset_loc,
                 logy=logy, logx=logx, legend_loc=legend_loc,
                 zero_loc=zero_loc)
    for ax, lab in ((a0, "(a)"), (b, "(b)")):
        ax.annotate(lab, xy=(0.02, 0.96), xycoords="axes fraction",
                    ha="left", va="top", fontsize=9)

    if out is None:
        out = default_out(npz_path)
    stem = os.path.splitext(out)[0] if out.endswith((".pdf", ".png")) else out
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=600 if ext == "png" else None)
        print("wrote", f"{stem}.{ext}")
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", help="Path to a series.npz written by combined_panels.py")
    p.add_argument("--inset-loc", choices=tuple(_INSET_LOC), default="upper right")
    p.add_argument("--inset-rect", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "W", "H"),
                   help="Explicit inset rectangle in axes fractions, "
                        "overriding --inset-loc.")
    p.add_argument("--no-inset", action="store_true")
    p.add_argument("--logy", action="store_true")
    p.add_argument("--no-logx", action="store_true")
    p.add_argument("--annot-loc", choices=tuple(_ANNOT_LOC),
                   default="lower right",
                   help="Corner of panel (a) holding the parameter annotation.")
    p.add_argument("--legend-loc", choices=_LEGEND_LOC, default="upper right",
                   help="Position of the legend in panel (b).")
    p.add_argument("--zero-loc", choices=tuple(_ZERO_LOC), default="lower left",
                   help="Where panel (b) names the measures that vanish "
                        "identically.")
    p.add_argument("--t-window", type=float, default=None, metavar="OMEGA_T",
                   help="Truncate the panel (a) time axis at this Omega*t.")
    p.add_argument("--T-panel-a", type=float, default=None,
                   help="Override the panel-(a) temperature (nearest on the grid).")
    p.add_argument("-o", "--out", default=None,
                   help="Output stem; defaults to the matching figures/<point>/"
                        "figure, i.e. where combined_panels.py writes it.")
    p.add_argument("--mesolve-check", type=float, nargs="+", default=None,
                   metavar="T",
                   help="Overlay master-equation BLP/LPP (open circles) at these "
                        "temperatures; computed once, then cached in "
                        "mesolve_check.npz beside the series.npz.")
    p.add_argument("--refresh-mesolve", action="store_true",
                   help="Recompute the master-equation check even if cached.")
    a = p.parse_args(argv)
    replot(a.npz, inset_loc=(a.inset_rect or a.inset_loc),
           inset=not a.no_inset, logy=a.logy,
           logx=not a.no_logx, T_panel_a=a.T_panel_a, t_window=a.t_window,
           annot_loc=a.annot_loc, legend_loc=a.legend_loc,
           zero_loc=a.zero_loc, out=a.out, check_T=a.mesolve_check,
           refresh_mesolve=a.refresh_mesolve)
    return 0


if __name__ == "__main__":
    main()
