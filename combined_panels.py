#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_panels.py
==================

The measures figure of the paper, assembled as a single two-column figure, plus
a sweep driver that builds one such figure for every (E, Gamma) in a grid. This
file replaces the former panel_a.py / panel_b.py pair: the shared style, the
span helpers and both panel drawings now live here.

  (a) three stacked time-series axes at one temperature --- the propagator
      |G(t)|, the canonical rates Gamma, Lambda^+, Lambda^-, and the MSV
      quantifier sqrt(J) of the polar pair. Grey bands mark Gamma < 0, i.e. the
      windows that feed N_BLP and N_LPP; purple bands mark the windows on which
      sqrt(J_pol) rises.

  (b) the five measures as functions of reservoir temperature at the same
      parameters. N_BLP and N_LPP are flat (the Theorem), while N_MSV, N_RHPE
      and N_RHP vary. Optional open markers overlay an independent evaluation of
      BLP and LPP through the time-local master equation.

Both panels are built from a SINGLE model per parameter point: the temperature
used by panel (a) is inserted into the panel (b) grid, so the kernels and the
noise term Z(T,t) are computed once.

Every run is also written out in full, so any plot can be redrawn later without
recomputing:

  * a tidy CSV with one row per (E, Gamma, T) holding the five measures and the
    run diagnostics (data/sweep_measures.csv by default);
  * per parameter point, the underlying time series --- t, |G|, arg G, Gamma(t),
    Lambda^+-(t), Z(t) and sqrt(J_pol)(t) at the panel-(a) temperature --- as a
    CSV, and every array including the full Z(T,t) grid as a compressed .npz.

Usage:
    python combined_panels.py                                   # default point
    python combined_panels.py --E 0.5 1.0 2.0 --Gamma 0.5 1.0   # explicit lists
    python combined_panels.py --E-range 0.2 3.0 15 --Gamma 0.5  # linspace in E
    python combined_panels.py --E-range 1.0 3.0 9 --Gamma 0.2 0.5 1.0 1.5 \
                              --nT 41 --mesolve-check 0 2 4 6 8
    python combined_panels.py --E 2.0 --Gamma 0.5 --T-log 0.01 10 25
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from spin_chain_1d import ModelConfig, NonMarkovianModel, VALID_MEASURES, _h2
except ImportError as _exc:  # drawing-only use (replot.py) needs none of these
    ModelConfig = NonMarkovianModel = None
    VALID_MEASURES = ("BLP", "LPP", "MSV", "RHPE", "RHP")
    _IMPORT_ERROR = _exc

    def _h2(x):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        out = np.zeros_like(x)
        m = (x > 0.0) & (x < 1.0)
        out[m] = -x[m] * np.log2(x[m]) - (1.0 - x[m]) * np.log2(1.0 - x[m])
        return out

# Cool = bath-state independent, warm = bath-state dependent, purple = the new one.
_STYLE = {
    "BLP":  dict(color="C0", ls="-",  label=r"$\mathcal{N}_{\rm BLP}$"),
    "LPP":  dict(color="C9", ls="-",  label=r"$\mathcal{N}_{\rm LPP}$"),
    "MSV":  dict(color="C4", ls="-.", label=r"$\mathcal{N}_{\rm MSV}$"),
    "RHPE": dict(color="C1", ls="--", label=r"$\mathcal{N}_{\rm RHPE}$"),
    "RHP":  dict(color="C3", ls="--", label=r"$\mathcal{N}_{\rm RHP}$"),
}

_ORDER = ("BLP", "LPP", "MSV", "RHPE", "RHP")

CSV_FIELDS = ["E", "Gamma", "Omega", "mu", "T", *_ORDER,
              "min_absG", "min_absG_t", "bound_state", "Z_b", "tmax", "nt"]


# --------------------------------------------------------------------------- #
# Style and small helpers (formerly panel_a.py)                               #
# --------------------------------------------------------------------------- #
def use_paper_style(usetex=None):
    """Single-column PRA width is 3.4 in; a figure spanning both columns ~7 in.

    ``usetex`` defaults to True; set NM_USETEX=0 in the environment (or pass
    False) on a machine without a working LaTeX installation.
    """
    if usetex is None:
        usetex = os.environ.get("NM_USETEX", "1") not in ("0", "false", "False")
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def negative_spans(t, y):
    """Contiguous [t_start, t_end] intervals on which y < 0."""
    neg = np.asarray(y) < 0.0
    spans, k, n = [], 0, len(neg)
    while k < n:
        if not neg[k]:
            k += 1
            continue
        j = k
        while j < n and neg[j]:
            j += 1
        spans.append((float(t[k]), float(t[min(j, n - 1)])))
        k = j
    return spans


def rising_spans(t, y, min_rise=1e-9):
    """Contiguous intervals on which y increases."""
    up = np.diff(np.asarray(y)) > 0
    out, k, n = [], 0, len(up)
    while k < n:
        if not up[k]:
            k += 1
            continue
        j = k
        while j < n and up[j]:
            j += 1
        if y[j] - y[k] > min_rise:
            out.append((float(t[k]), float(t[j])))
        k = j
    return out


def polar_msv_curve(model, iT):
    """sqrt(J) for the polar pair: occupations |G|^2 + Z and Z.

    A model may carry a precomputed double-precision curve (``sqrtJ_pol`` for
    temperature index ``sqrtJ_pol_iT``); replot.py uses that, because Z(T, t) is
    archived in single precision and its round-off would otherwise show up as
    spurious hairline revivals in the shading.
    """
    stored = getattr(model, "sqrtJ_pol", None)
    if stored is not None and iT == getattr(model, "sqrtJ_pol_iT", None):
        return np.asarray(stored, dtype=float)
    g2 = np.abs(model.G_t) ** 2
    Z = np.asarray(model.Z_Tt[iT], dtype=float)
    p1, p2 = g2 + Z, Z
    J = _h2((p1 + p2) / 2.0) - 0.5 * (_h2(p1) + _h2(p2))
    return np.sqrt(np.clip(J, 0.0, None))


def mesolve_gaps(res, check, T_grid):
    """Largest |direct - master equation| over the cross-check temperatures.

    BLP and LPP are bath-state independent, so the direct curve is constant in
    T and the comparison needs no interpolation; interpolating anyway costs
    nothing and guards against that ever ceasing to hold. These are the numbers
    quoted in the figure caption.
    """
    out = {}
    for k in ("BLP", "LPP"):
        if k not in check or k not in res:
            continue
        direct = np.interp(np.asarray(check["T"], dtype=float),
                           np.asarray(T_grid, dtype=float),
                           np.asarray(res[k], dtype=float))
        gap = float(np.max(np.abs(direct - np.asarray(check[k], dtype=float))))
        out["mesolve_gap_" + k] = gap
        print("    master-equation cross-check, %-3s: max |direct - ODE| "
              "= %.3e" % (k, gap), flush=True)
    return out


def mesolve_check(T_check, **cfg_kwargs):
    """BLP and LPP through the ODE solver, as an independent cross-check.

    This route builds the time-local generator from the T-dependent rates and
    integrates it, so agreement with the direct route across temperatures is
    evidence for bath-state independence rather than a restatement of it.
    """
    cfg = ModelConfig(T=np.asarray(T_check, dtype=float), **cfg_kwargs)
    m = NonMarkovianModel(cfg, verbose=False)
    return m.run(["BLP", "LPP"], direct=False)


# --------------------------------------------------------------------------- #
# Panel drawing                                                               #
# --------------------------------------------------------------------------- #
# Where the parameter annotation sits in the top panel of column (a), as
# (x, y, ha, va) in axes fractions. Which corner is free depends on the
# propagator: |G| stays near 1 across the frame in the bound-state regime, so
# only the bottom is clear there, while a decaying |G| leaves the top right
# open. "upper left" collides with the (a) panel label.
_ANNOT_LOC = {
    "lower right": (0.98, 0.05, "right", "bottom"),
    "upper right": (0.98, 0.95, "right", "top"),
    "lower left":  (0.02, 0.05, "left", "bottom"),
    "upper left":  (0.02, 0.95, "left", "top"),
}


def draw_panel_a(model, iT, axs, shade_color="0.88", annotate=True,
                 t_window=None, annot_loc="lower right"):
    """Draw the time-series column into the axes (ax0, ax1[, ax2]).

    ``t_window`` truncates the horizontal axis without touching the data: the
    measures in panel (b) are still integrated over the full run, only the
    time series are shown over a shorter stretch.  A long integration window
    is needed for the integrals to converge, but compresses the rate
    oscillations into an unreadable smear if it is also used for the plot.
    """
    t = model.cfg.t
    g = np.abs(model.G_t)
    ax0, ax1 = axs[0], axs[1]
    ax2 = axs[2] if len(axs) > 2 else None

    for ax in axs:
        for (ta, tb) in negative_spans(t, model.Gamma_t):
            ax.axvspan(ta, tb, color=shade_color, lw=0, zorder=0)

    ax0.plot(t, g, color="C0")
    bound = getattr(model, "_bound", []) or []
    if bound:
        Zb = bound[0][1]
        ax0.axhline(Zb, color="C0", ls=":", lw=0.8)
        ax0.annotate(r"$Z_b$", xy=(t[-1], Zb), xytext=(-2, 2),
                     textcoords="offset points", ha="right", va="bottom",
                     color="C0", fontsize=8)
    ax0.set_ylabel(r"$|G(t)|$")
    ax0.set_ylim(model.G_abs_min - 0.02, 1.02)

    ax1.axhline(0.0, color="0.6", lw=0.6, zorder=1)
    ax1.plot(t, model.Gamma_t, color="0.25", lw=0.9, label=r"$\gamma$")
    ax1.plot(t, model.Lambda_plus_t[iT], color="C3", label=r"$\Lambda^{+}$")
    ax1.plot(t, model.Lambda_minus_t[iT], color="C2", label=r"$\Lambda^{-}$")
    ax1.set_ylabel(r"rates$\,/\,\Omega$")
    ax1.legend(ncol=3, handlelength=1.3, columnspacing=0.9, loc="upper right")

    if ax2 is not None:
        sJ = polar_msv_curve(model, iT)
        for (ta, tb) in rising_spans(t, sJ):
            ax2.axvspan(ta, tb, facecolor="C4", alpha=0.18, lw=0, zorder=0)
        ax2.plot(t, sJ, color="C4")
        ax2.set_ylabel(r"$\sqrt{J_{\rm pol}}$")

    axs[-1].set_xlabel(r"$\Omega t$")
    ax0.set_xlim(t[0], float(t_window) if t_window else t[-1])

    if annotate:
        cfg = model.cfg
        x, y, ha, va = _ANNOT_LOC[annot_loc]
        ax0.annotate(rf"$E = {cfg.E:g}\,\Omega$,  $\Gamma = {cfg.Gamma:g}\,\Omega$"
                     "\n"
                     rf"$\min_t|G| = {model.G_abs_min:.3f}$,  "
                     rf"$T = {cfg.T[iT]:g}\,\Omega$",
                     xy=(x, y), xycoords="axes fraction",
                     ha=ha, va=va, fontsize=7.5)
    return axs


# Inset placement in panel (b), as [x0, y0, w, h] in axes fraction. The corner
# to use depends on where the curves happen to run; "lower left" is the empty
# corner whenever every measure decays with temperature.
_INSET_LOC = {
    "upper right": [0.52, 0.66, 0.42, 0.28],
    "upper left":  [0.16, 0.66, 0.40, 0.28],
    "lower right": [0.54, 0.13, 0.40, 0.28],
    "lower left":  [0.14, 0.13, 0.40, 0.28],
    # A short strip along the bottom, for panels too crowded for a corner:
    # tall enough to read a flat line against, short enough to sit under a
    # horizontal curve low in the frame.
    "bottom strip": [0.14, 0.10, 0.38, 0.16],
}


# Legend positions for panel (b). "upper right" suits curves that fall with
# T; when every curve is flat (e.g. below the bound-state threshold) the
# middle of a log axis is usually the empty part.
_LEGEND_LOC = ("upper right", "upper left", "lower left", "lower right",
               "center right", "center left", "lower center", "upper center",
               "best")

# Where panel (b) names the measures that vanish identically, as
# (x, y, ha, va) in axes fractions. Move it when a flat curve runs along the
# bottom of the frame.
_ZERO_LOC = {
    "lower left":   (0.03, 0.03, "left", "bottom"),
    "lower right":  (0.97, 0.03, "right", "bottom"),
    "center left":  (0.03, 0.50, "left", "center"),
    "center right": (0.97, 0.50, "right", "center"),
    "upper left":   (0.03, 0.90, "left", "top"),
}


def draw_panel_b(res, ax, check=None, inset=True, logy=False, logx=False,
                 inset_loc="upper right", legend_loc="upper right",
                 zero_loc="lower left"):
    """Draw the measures-versus-temperature panel into ``ax``.

    With ``logx`` the temperature axis is logarithmic; a T = 0 point, if present,
    stays in the data (it is the reference for the inset) but falls outside a log
    axis and is simply not drawn.
    """
    T = np.asarray(res["T"])
    pos = T[T > 0]
    xlim = (float(pos.min()), float(T.max())) if (logx and pos.size) else \
           (float(T[0]), float(T[-1]))
    present = [k for k in _ORDER if k in res]
    # A measure that is identically zero cannot be drawn on a log axis, and one
    # that reaches zero part way along would otherwise plunge off the bottom of
    # the frame rather than ending where it vanishes. Mask the non-positive
    # values, mark the last finite point with an open circle, and keep the
    # all-zero measures out of the legend -- a legend entry with no curve to
    # find is worse than no entry -- naming them in a corner instead.
    # Values at round-off level (|N| < 1e-12, e.g. +-1e-16 from a difference
    # of equal concurrences) are zero, not revivals.
    clean = {k: np.where(np.abs(np.asarray(res[k], dtype=float)) < 1e-12, 0.0,
                         np.asarray(res[k], dtype=float)) for k in present}
    vanish = [k for k in present if logy and np.all(clean[k] == 0.0)]
    for k in present:
        y = clean[k] if logy else np.asarray(res[k], dtype=float)
        style = dict(_STYLE[k])
        if logy:
            y = np.where(y > 0.0, y, np.nan)
            if k in vanish:
                style["label"] = "_nolegend_"
        ax.plot(T, y, **style)
        if logy and np.isnan(y).any() and not np.isnan(y).all():
            j = int(np.flatnonzero(~np.isnan(y))[-1])
            if j + 1 < len(y):
                ax.plot([T[j]], [y[j]], ls="none", marker="o", ms=3.5,
                        mfc="white", mec=_STYLE[k]["color"], mew=0.9, zorder=6)

    if check is not None:
        for k in ("BLP", "LPP"):
            if k in check:
                ax.plot(check["T"], check[k], ls="none", marker="o", ms=4,
                        mfc="none", mec=_STYLE[k]["color"], mew=0.9, zorder=5)
        # One legend entry for the whole overlay rather than one per measure.
        ax.plot([], [], ls="none", marker="o", ms=4, mfc="none", mec="0.35",
                mew=0.9, label=r"master eq.")

    ax.set_xlabel(r"$T/\Omega$")
    ax.set_ylabel(r"$\mathcal{N}$")
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    if logy:
        ax.set_yscale("log")
    else:
        ymax = max(float(np.max(res[k])) for k in present)
        ax.set_ylim(0.0, 1.08 * ymax if ymax > 0 else 1.0)
    # "lower/upper center" lay the legend out in rows of three, a strip that
    # fits under (or over) the curves instead of a tall column in a corner.
    wide = legend_loc in ("lower center", "upper center")
    ax.legend(loc=legend_loc, ncol=3 if wide else 1, handlelength=1.6,
              labelspacing=0.3, columnspacing=1.0 if wide else 2.0)

    if logy and vanish:
        zx, zy, zha, zva = _ZERO_LOC[zero_loc]
        ax.annotate(" = ".join(_STYLE[k]["label"] for k in vanish) + r"$\,\equiv 0$",
                    xy=(zx, zy), xycoords="axes fraction",
                    ha=zha, va=zva, fontsize=8)

    if inset and all(k in res for k in ("BLP", "LPP")):
        bounds = (_INSET_LOC[inset_loc] if isinstance(inset_loc, str)
                  else list(inset_loc))
        axi = ax.inset_axes(bounds)
        rel = all(abs(float(np.asarray(res[k])[0])) > 0 for k in ("BLP", "LPP"))
        dmax = 0.0
        for k in ("BLP", "LPP"):
            y = np.asarray(res[k], dtype=float)
            d = (y - y[0]) / y[0] if rel else (y - y[0])
            dmax = max(dmax, float(np.max(np.abs(d))))
            axi.plot(T, d, color=_STYLE[k]["color"], lw=0.9)
        axi.axhline(0.0, color="0.6", lw=0.5)
        if logx:
            axi.set_xscale("log")
        axi.set_xlim(*xlim)
        axi.set_ylim(-1.2 * max(dmax, 1e-16), 1.2 * max(dmax, 1e-16))
        axi.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axi.tick_params(labelsize=6, pad=1.5)
        axi.yaxis.get_offset_text().set_fontsize(6)
        axi.set_ylabel(r"$\Delta\mathcal{N}/\mathcal{N}(0)$" if rel
                       else r"$\Delta\mathcal{N}$", fontsize=6, labelpad=1)
        axi.set_xlabel(r"$T/\Omega$", fontsize=6, labelpad=1)
    return ax


# --------------------------------------------------------------------------- #
# One parameter point                                                         #
# --------------------------------------------------------------------------- #
def temperature_grid(Tmax=10.0, nT=41, T_log=None):
    """Temperature grid: linear by default, geometric with ``T_log``.

    ``T_log`` is (lo, hi, n): n points geometrically spaced between lo and hi,
    with T = 0 prepended. Almost all of the temperature dependence of the
    measures sits below T ~ Omega, which a linear grid samples with a handful of
    points, so the log grid is the right default for the figures.
    """
    if T_log is not None:
        lo, hi, n = T_log
        return np.concatenate(([0.0], np.geomspace(float(lo), float(hi), int(n))))
    return np.linspace(0.0, Tmax, nT)


def build(E=2.0, Gamma=0.5, Omega=1.0, mu=0.0, tmax=40.0, nt=4001,
          T_panel_a=1.0, Tmax=10.0, nT=41, T_log=None, inset=True,
          inset_loc="upper right", logy=False, logx=None, check_T=None,
          t_window=None, annot_loc="lower right", legend_loc="upper right",
          zero_loc="lower left", make_fig=True):
    """Evaluate one (E, Gamma) point and (optionally) draw its figure.

    Returns (fig_or_None, res, info, model). ``res`` is the dict returned by
    NonMarkovianModel.run, ``info`` carries the diagnostics that go into the CSV
    (including ``iT_a``, the index of the panel-(a) temperature).
    """
    # One temperature grid for both panels: insert T_panel_a if it is missing,
    # so the model (and hence the kernels and Z) is built only once.
    T_grid = temperature_grid(Tmax=Tmax, nT=nT, T_log=T_log)
    if logx is None:
        logx = T_log is not None
    if not np.any(np.isclose(T_grid, T_panel_a)):
        T_grid = np.sort(np.append(T_grid, T_panel_a))
    iT_a = int(np.argmin(np.abs(T_grid - T_panel_a)))

    cfg = ModelConfig(E=E, Gamma=Gamma, Omega=Omega, mu=mu,
                      tmax=tmax, nt=nt, T=T_grid)
    model = NonMarkovianModel(cfg, verbose=False)
    res = model.run(list(VALID_MEASURES))

    bound = getattr(model, "_bound", []) or []
    info = dict(E=E, Gamma=Gamma, Omega=Omega, mu=mu, tmax=tmax, nt=nt,
                min_absG=model.G_abs_min, min_absG_t=model.G_abs_min_t,
                bound_state=bool(bound),
                Z_b=(float(bound[0][1]) if bound else float("nan")),
                iT_a=iT_a, T_panel_a=float(T_grid[iT_a]))

    if not make_fig:
        return None, res, info, model

    check = None
    if check_T:
        check = mesolve_check(check_T, E=E, Gamma=Gamma, Omega=Omega, mu=mu,
                              tmax=tmax, nt=nt)
        info.update(mesolve_gaps(res, check, T_grid))

    use_paper_style()
    fig = plt.figure(figsize=(7.0, 4.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.1],
                          height_ratios=[1.0, 1.2, 1.0])
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    a2 = fig.add_subplot(gs[2, 0], sharex=a0)
    b = fig.add_subplot(gs[:, 1])

    draw_panel_a(model, iT_a, (a0, a1, a2), t_window=t_window,
                 annot_loc=annot_loc)
    draw_panel_b(res, b, check=check, inset=inset, inset_loc=inset_loc,
                 logy=logy, logx=logx, legend_loc=legend_loc,
                 zero_loc=zero_loc)

    for ax, lab in ((a0, "(a)"), (b, "(b)")):
        ax.annotate(lab, xy=(0.02, 0.96), xycoords="axes fraction",
                    ha="left", va="top", fontsize=9)

    return fig, res, info, model


# --------------------------------------------------------------------------- #
# Saving                                                                      #
# --------------------------------------------------------------------------- #
def save_series(model, res, info, out_dir, fmt="both"):
    """Write the time series behind one parameter point.

    ``fmt`` is "csv", "npz", "both" or "none".

      series.csv  : t, |G|, arg G, Gamma(t), Lambda^+-(t), Z(t), sqrt(J_pol)(t)
                    at the panel-(a) temperature. Human readable, one file.
      series.npz  : every array, including the full Z(T,t) and Lambda^+-(T,t)
                    grids and the measures, plus the parameters as metadata.
                    Float32 for the 2D grids to keep the files small.
    """
    if fmt == "none":
        return
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(model.cfg.t, dtype=float)
    iT = int(info["iT_a"])
    G = np.asarray(model.G_t)
    sJ = polar_msv_curve(model, iT)

    if fmt in ("csv", "both"):
        # Pure header + data: no comment lines, so pandas.read_csv works as is.
        # The parameters live in series.npz and in the folder name.
        cols = np.column_stack([t, np.abs(G), np.angle(G), model.Gamma_t,
                                model.Lambda_plus_t[iT], model.Lambda_minus_t[iT],
                                model.Z_Tt[iT], sJ])
        np.savetxt(os.path.join(out_dir, "series.csv"), cols, delimiter=",",
                   comments="",
                   header="t,absG,argG,Gamma_t,Lambda_plus,Lambda_minus,Z,sqrtJ_pol")

    if fmt in ("npz", "both"):
        payload = dict(
            t=t, G=G, Gamma_t=np.asarray(model.Gamma_t, dtype=float),
            omega_eff_t=np.asarray(getattr(model, "omega_eff_t", []), dtype=float),
            T=np.asarray(res["T"], dtype=float),
            Z_Tt=np.asarray(model.Z_Tt, dtype=np.float32),
            Lambda_plus_Tt=np.asarray(model.Lambda_plus_t, dtype=np.float32),
            Lambda_minus_Tt=np.asarray(model.Lambda_minus_t, dtype=np.float32),
            sqrtJ_pol_panel_a=sJ,
            **{k: np.asarray(res[k], dtype=float) for k in _ORDER if k in res},
            **{k: np.asarray(info[k]) for k in
               ("E", "Gamma", "Omega", "mu", "tmax", "nt", "min_absG",
                "min_absG_t", "bound_state", "Z_b", "iT_a", "T_panel_a")},
        )
        np.savez_compressed(os.path.join(out_dir, "series.npz"), **payload)


def rows_from_run(res, info):
    """One CSV row per temperature."""
    rows = []
    for j, T in enumerate(np.asarray(res["T"])):
        row = {k: info[k] for k in ("E", "Gamma", "Omega", "mu",
                                    "min_absG", "min_absG_t", "bound_state",
                                    "Z_b", "tmax", "nt")}
        row["T"] = float(T)
        for k in _ORDER:
            row[k] = float(np.asarray(res[k])[j]) if k in res else ""
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #
def sweep(E_list, Gamma_list, csv_path="data/sweep_measures.csv",
          make_figs=True, fig_dir="figures", data_dir="data",
          series="both", verbose=True, **build_kwargs):
    """Run every (E, Gamma) combination, save one figure each, collect the CSV."""
    all_rows = []
    combos = list(itertools.product(E_list, Gamma_list))
    for n, (E, Gamma) in enumerate(combos, 1):
        t0 = time.time()
        try:
            fig, res, info, model = build(E=E, Gamma=Gamma, make_fig=make_figs,
                                          **build_kwargs)
        except Exception as exc:                      # never silently skip
            print(f"[{n}/{len(combos)}] E={E:g}, Gamma={Gamma:g}: "
                  f"FAILED {type(exc).__name__}: {exc}", flush=True)
            continue

        if fig is not None:
            folder = os.path.join(fig_dir, f"E_{E:.1f}_Gamma_{Gamma:.1f}")
            os.makedirs(folder, exist_ok=True)
            fig.savefig(os.path.join(folder, "figure.pdf"), bbox_inches="tight")
            fig.savefig(os.path.join(folder, "figure.png"), dpi=600,
                        bbox_inches="tight")
            plt.close(fig)                            # keep memory flat

        if series != "none":
            save_series(model, res, info,
                        os.path.join(data_dir, f"E_{E:.1f}_Gamma_{Gamma:.1f}"),
                        fmt=series)

        all_rows.extend(rows_from_run(res, info))

        if verbose:
            summary = "  ".join(
                f"{k} {np.min(res[k]):.4g}..{np.max(res[k]):.4g}"
                for k in _ORDER if k in res)
            print(f"[{n}/{len(combos)}] E={E:g}, Gamma={Gamma:g}  "
                  f"min|G|={info['min_absG']:.3f}"
                  f"{'  (bound state)' if info['bound_state'] else ''}  "
                  f"[{time.time() - t0:.1f}s]\n    {summary}", flush=True)

    if csv_path and all_rows:
        d = os.path.dirname(csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"saved -> {csv_path}  ({len(all_rows)} rows)")

    return all_rows


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _values(explicit, rng, default):
    """Explicit list wins; otherwise a linspace from --*-range; else default."""
    if rng is not None:
        lo, hi, num = rng
        return list(np.linspace(float(lo), float(hi), int(num)))
    if explicit is not None:
        return list(explicit)
    return list(default)


def build_parser():
    p = argparse.ArgumentParser(
        description="Build the measures figure for one or many (E, Gamma) points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--E", type=float, nargs="+", default=None,
                   help="Explicit list of E values, e.g. --E 0.5 1.0 2.0")
    p.add_argument("--E-range", type=float, nargs=3, metavar=("LO", "HI", "N"),
                   default=None, help="linspace(LO, HI, N) for E; overrides --E")
    p.add_argument("--Gamma", type=float, nargs="+", default=None,
                   help="Explicit list of Gamma values.")
    p.add_argument("--Gamma-range", type=float, nargs=3,
                   metavar=("LO", "HI", "N"), default=None,
                   help="linspace(LO, HI, N) for Gamma; overrides --Gamma")
    p.add_argument("--Omega", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=40.0)
    p.add_argument("--nt", type=int, default=4001)
    p.add_argument("--T-panel-a", type=float, default=1.0)
    p.add_argument("--Tmax", type=float, default=10.0)
    p.add_argument("--nT", type=int, default=41)
    p.add_argument("--T-log", type=float, nargs=3, metavar=("LO", "HI", "N"),
                   default=None,
                   help="Geometric temperature grid: N points from LO to HI, "
                        "plus T = 0. Overrides --Tmax/--nT and switches panel "
                        "(b) to a log temperature axis.")
    p.add_argument("--no-logx", action="store_true",
                   help="Keep a linear temperature axis even with --T-log.")
    p.add_argument("--mesolve-check", type=float, nargs="+", default=None,
                   metavar="T",
                   help="Temperatures at which to overlay the ODE cross-check.")
    p.add_argument("--csv", default="data/sweep_measures.csv",
                   help="Tidy CSV with one row per (E, Gamma, T); '' to skip.")
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--data-dir", default="data",
                   help="Where the per-point time series are written.")
    p.add_argument("--series", choices=("both", "csv", "npz", "none"),
                   default="both",
                   help="Per-point time series: CSV, compressed npz, both, none.")
    p.add_argument("--no-figs", action="store_true",
                   help="Collect the CSV only, skip the figures.")
    p.add_argument("--t-window", type=float, default=None, metavar="OMEGA_T",
                   help="Truncate the panel (a) time axis at this Omega*t. "
                        "Does not affect the integrals in panel (b).")
    p.add_argument("--no-inset", action="store_true")
    p.add_argument("--inset-loc", choices=tuple(_INSET_LOC),
                   default="upper right",
                   help="Corner of panel (b) holding the BLP/LPP inset.")
    p.add_argument("--annot-loc", choices=tuple(_ANNOT_LOC),
                   default="lower right",
                   help="Corner of the top panel (a) holding the parameter "
                        "annotation.")
    p.add_argument("--legend-loc", choices=_LEGEND_LOC, default="upper right",
                   help="Position of the legend in panel (b).")
    p.add_argument("--zero-loc", choices=tuple(_ZERO_LOC), default="lower left",
                   help="Where panel (b) names the measures that vanish "
                        "identically (log axis only).")
    p.add_argument("--inset-rect", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "W", "H"),
                   help="Explicit inset rectangle in axes fractions, "
                        "overriding --inset-loc. Use when no corner is free.")
    p.add_argument("--logy", action="store_true",
                   help="Log vertical axis in panel (b); use when a measure is 0.")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    E_list = _values(a.E, a.E_range, [2.0])
    G_list = _values(a.Gamma, a.Gamma_range, [0.5])
    print(f"sweeping {len(E_list)} x {len(G_list)} = {len(E_list) * len(G_list)} "
          f"parameter points", flush=True)
    sweep(E_list, G_list,
          csv_path=(a.csv or None), make_figs=not a.no_figs, fig_dir=a.fig_dir,
          data_dir=a.data_dir, series=a.series, verbose=not a.quiet,
          Omega=a.Omega, mu=a.mu, tmax=a.tmax, nt=a.nt,
          T_panel_a=a.T_panel_a, Tmax=a.Tmax, nT=a.nT, T_log=a.T_log,
          inset=not a.no_inset, inset_loc=(a.inset_rect or a.inset_loc),
          logy=a.logy,
          t_window=a.t_window, annot_loc=a.annot_loc,
          legend_loc=a.legend_loc, zero_loc=a.zero_loc,
          logx=(False if a.no_logx else None), check_T=a.mesolve_check)
    return 0


if __name__ == "__main__":
    main()
