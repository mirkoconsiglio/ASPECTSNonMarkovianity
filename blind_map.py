#!/usr/bin/env python
r"""Where are the trace-distance measures blind?  The (E, Gamma) plane.

For the F = 0 model every pair on the BLP Pareto front has
sgn(d D_s / dt) = sgn(d|G| / dt)  (Appendix B), so

    N_BLP = 0   <=>   N_LPP = 0   <=>   |G(t)| is non-increasing.

|G(t)| depends on (E, Gamma, Omega) alone: temperature, chemical potential and
Z_beta never enter, so one map covers every bath state at once.

What the map shows (validated to Omega t = 1280)
------------------------------------------------
* N_BLP vanishes identically on the bound-state threshold |E| = Omega - Gamma/2
  for Gamma <= Omega, and nowhere else.  The segment ends sharply at
  Gamma = Omega, where the omega^2 term of phi's denominator vanishes.
* Off that segment |G| eventually rises, but ever later and ever more weakly as
  the threshold is approached (first revival time roughly ~ distance^-2).  Any
  finite window therefore shows a thin grey band around the curve that narrows
  as t_max grows; it is a window effect, not an open blind region.

Numerics
--------
G(t) = K @ phi with K depending only on (t, Omega): K is built once, in time
chunks, and every parameter point goes through one matrix product.  phi, the
band-edge limit and the bound states are the model's own, so the propagator is
bitwise the one the measures use (checked to 2e-16 against compute_kernels).
N_BLP is exactly the functional of NonMarkovianModel.blp_direct,
max( sum_+ d|G|, sum_+ d|G|^2 ), evaluated on the chosen window.

Usage
-----
    python blind_map.py --tmax 1280 --dt 0.05 --plot figures/blind_map.pdf
    python blind_map.py --from data/blind_map.npz --plot figures/blind_map.pdf
    python blind_map.py --cut 0.5 --T 0.02 --plot-cut figures/blind_cut.pdf
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from spin_chain_1d import ModelConfig, NonMarkovianModel, bound_states


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


class _PhiOnly:
    """Just enough of a NonMarkovianModel to call its spectral functions."""
    _J = NonMarkovianModel._J
    _phi = NonMarkovianModel._phi

    def __init__(self, cfg):
        self.cfg = cfg


def quadrature_grid(Omega, M):
    """theta grid, Simpson weights and omega samples, as in compute_kernels."""
    if M % 2 == 0:
        M += 1
    th = np.linspace(-np.pi / 2.0, np.pi / 2.0, M)
    dth = th[1] - th[0]
    sw = np.ones(M)
    sw[1:-1:2] = 4.0
    sw[2:-1:2] = 2.0
    sw *= dth / 3.0
    return Omega * np.sin(th), Omega * np.cos(th), sw


def phi_column(E, Gamma, Omega, w, dwdth):
    """phi(w) dw/dtheta for one point, band-edge limit included.

    Mirrors NonMarkovianModel.compute_kernels line for line.
    """
    cfg = ModelConfig(E=E, Gamma=Gamma, Omega=Omega, tmax=1.0, nt=3, nT=1)
    phiw = np.nan_to_num(_PhiOnly(cfg)._phi(w)) * dwdth
    M = len(w)
    edge_lim = 2.0 * Omega / (np.pi * Gamma)
    for idx, A in ((M - 1, Omega - Gamma / 2.0 - E),
                   (0, -(Omega - Gamma / 2.0) - E)):
        if abs(A) < 1e-12 * max(1.0, Omega):
            phiw[idx] = edge_lim
    return phiw


def scan(E_vals, G_vals, Omega=1.0, tmax=40.0, dt=0.01, n_omega=2049,
         chunk=400, tol=1e-9, verbose=True):
    """Monotonicity diagnostics of |G(t)| on the (E, Gamma) grid."""
    nt = int(round(tmax / dt)) + 1
    t = np.linspace(0.0, tmax, nt)
    w, dwdth, sw = quadrature_grid(Omega, n_omega)
    EE, GG = np.meshgrid(np.asarray(E_vals, float), np.asarray(G_vals, float),
                         indexing="ij")
    Ef, Gf = EE.ravel(), GG.ravel()
    P = Ef.size

    t0 = time.time()
    Phi = np.empty((len(w), P))
    bstates = []
    for p in range(P):
        Phi[:, p] = phi_column(Ef[p], Gf[p], Omega, w, dwdth)
        bstates.append(bound_states(Ef[p], Gf[p], Omega))
    Phi *= sw[:, None]
    has_b = np.array([len(b) > 0 for b in bstates])
    Zb = np.array([b[0][1] if b else np.nan for b in bstates])

    rise_eq = np.zeros(P)
    rise_pol = np.zeros(P)
    max_step = np.full(P, -np.inf)
    t_first = np.full(P, np.nan)
    gmin = np.full(P, np.inf)
    g0 = None
    last = None
    for a in range(0, nt, chunk):
        tc = t[a:a + chunk]
        ph = np.outer(tc, w)                       # real/imag split: Phi stays real
        G = (np.cos(ph) @ Phi) - 1j * (np.sin(ph) @ Phi)
        for p in np.flatnonzero(has_b):
            for wb, zb in bstates[p]:
                G[:, p] += zb * np.exp(-1j * wb * tc)
        g = np.abs(G)
        if g0 is None:
            g0 = g[0].copy()
        gmin = np.minimum(gmin, g.min(axis=0))
        seg = g if last is None else np.vstack([last[None, :], g])
        tseg = tc if last is None else np.concatenate([[t[a - 1]], tc])
        d = np.diff(seg, axis=0)
        rise_eq += np.clip(d, 0.0, None).sum(axis=0)
        rise_pol += np.clip(np.diff(seg ** 2, axis=0), 0.0, None).sum(axis=0)
        max_step = np.maximum(max_step, d.max(axis=0))
        hit = d > tol
        new = np.isnan(t_first) & hit.any(axis=0)
        if new.any():
            t_first[new] = tseg[1:][hit[:, new].argmax(axis=0)]
        last = g[-1]
    if verbose:
        print(f"[map] {P} points over Omega t <= {tmax:g} (dt={dt:g}, "
              f"M={len(w)}) in {time.time() - t0:.1f}s; "
              f"max |G(0)|-1 = {np.max(np.abs(g0 - 1)):.1e}", flush=True)

    shape = EE.shape
    out = dict(E=np.asarray(E_vals, float), Gamma=np.asarray(G_vals, float),
               Omega=Omega, tmax=tmax, dt=dt, n_omega=len(w), tol=tol,
               rise_eq=rise_eq.reshape(shape), rise_pol=rise_pol.reshape(shape),
               max_step=max_step.reshape(shape), t_first=t_first.reshape(shape),
               min_absG=gmin.reshape(shape), G0_err=(g0 - 1).reshape(shape),
               bound_state=has_b.reshape(shape), Z_b=Zb.reshape(shape))
    out["N_BLP"] = np.maximum(out["rise_eq"], out["rise_pol"])
    out["blind"] = out["max_step"] <= tol
    return out


def save_map(res, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    np.savez_compressed(stem + ".npz", **res)
    EE, GG = np.meshgrid(res["E"], res["Gamma"], indexing="ij")
    cols = ["N_BLP", "blind", "rise_eq", "rise_pol", "max_step", "t_first",
            "min_absG", "G0_err", "bound_state", "Z_b"]
    data = np.column_stack([EE.ravel(), GG.ravel()] +
                           [np.asarray(res[k], float).ravel() for k in cols])
    np.savetxt(stem + ".csv", data, delimiter=",",
               header=",".join(["E", "Gamma"] + cols), comments="")
    print("wrote", stem + ".npz", "and", stem + ".csv")


# Parameter points of the paper's figures: above, on and below the threshold.
# Labels must match the figure numbers in the manuscript; override with --mark.
DEFAULT_MARKS = ((2.0, 0.5, "Fig. 2"), (0.5, 1.0, "Fig. 3"), (0.5, 0.5, "Fig. 4"))


def load_map(npz):
    """Read a map saved by save_map, so it can be redrawn without rescanning."""
    d = dict(np.load(npz, allow_pickle=True))
    return {k: (v.item() if np.ndim(v) == 0 else v) for k, v in d.items()}


def plot_map(res, path, marks=DEFAULT_MARKS):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch
    try:
        from combined_panels import use_paper_style
        use_paper_style()
    except Exception:
        pass
    E, G, O = res["E"], res["Gamma"], res["Omega"]
    N = np.where(res["blind"], np.nan, res["N_BLP"])
    fin = N[np.isfinite(N) & (N > 0)]
    fig, ax = plt.subplots(figsize=(3.4, 2.9), layout="constrained")
    ax.pcolormesh(G, E, np.where(res["blind"], 1.0, np.nan), shading="nearest",
                  cmap="Greys", vmin=0.0, vmax=2.5)
    # Clip at 1: near (E, Gamma) = (0, 2*Omega) both band edges approach the
    # threshold, |G| beats through near-zeros and N_BLP grows as t_max^(1/2),
    # so the values there are window-dependent and would otherwise set the scale.
    vmax = min(fin.max(), 1.0)
    pc = ax.pcolormesh(G, E, N, shading="nearest", cmap="viridis",
                       norm=LogNorm(vmin=max(fin.min(), 1e-6), vmax=vmax))
    g_in = np.linspace(max(G.min(), 0.0), min(O, G.max()), 100)
    g_out = np.linspace(O, min(G.max(), 2.0 * O), 100)
    ax.plot(g_in, O - g_in / 2.0, color="C3", lw=1.4)
    ax.plot(g_out, O - g_out / 2.0, color="C3", lw=1.0, ls="--")
    for (e, gm, lab) in marks:
        if G.min() <= gm <= G.max() and E.min() <= e <= E.max():
            ax.plot(gm, e, marker="o", ms=4.5, mfc="white", mec="k", mew=0.9)
            # labels above-right, except under the threshold line, where
            # above-right would cover the grey band: put those below-right
            below = abs(e) < O - gm / 2.0
            ax.annotate(lab, (gm, e), xytext=(4, -4) if below else (4, 3),
                        textcoords="offset points", fontsize=7,
                        va="top" if below else "baseline",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.8))
    ax.set_xlabel(r"$\Gamma/\Omega$")
    ax.set_ylabel(r"$E/\Omega$")
    ax.set_xlim(G.min(), G.max())
    ax.set_ylim(E.min(), E.max())
    cb = fig.colorbar(pc, ax=ax, extend="max" if fin.max() > vmax else "neither")
    cb.set_label(r"$\mathcal{N}_{\rm BLP}$")
    ax.legend(handles=[Patch(facecolor="0.6",
                             label=rf"no revival, $\Omega t\leq{res['tmax']:g}$")],
              loc="upper right", fontsize=6.5, frameon=True, framealpha=0.9)
    save_figure(fig, path)
    return fig


def threshold_cut(Gamma, dE, T=0.02, Omega=1.0, blp_tmax=1280.0, blp_dt=0.05,
                  tmax=80.0, nt=8001, n_omega=4097, verbose=True):
    """All measures along E = E_c + dE at fixed Gamma, across the threshold.

    N_BLP comes from the propagator on the long window (the blp_direct
    functional), so late revivals near the threshold are not missed; the
    bath-sensitive measures come from the full model on the shorter window.
    Both choices are conservative for the comparison: a longer window can only
    raise N_BLP, a shorter one can only lower N_RHP and N_MSV.
    """
    Ec = Omega - Gamma / 2.0
    Es = Ec + np.asarray(dE, dtype=float)
    lw = scan(Es, [Gamma], Omega=Omega, tmax=blp_tmax, dt=blp_dt,
              n_omega=n_omega, verbose=verbose)
    out = dict(Gamma=Gamma, Omega=Omega, T=T, E_c=Ec, dE=np.asarray(dE, float),
               E=Es, blp_tmax=blp_tmax, tmax=tmax, nt=nt,
               N_BLP_long=lw["N_BLP"][:, 0], t_first=lw["t_first"][:, 0],
               bound_state=lw["bound_state"][:, 0])
    keys = ("BLP", "LPP", "MSV", "RHPE", "RHP")
    for k in keys:
        out[k] = np.empty(Es.size)
    out["min_absG"] = np.empty(Es.size)
    t0 = time.time()
    for i, E in enumerate(Es):
        m = NonMarkovianModel(ModelConfig(E=E, Gamma=Gamma, Omega=Omega,
                                          tmax=tmax, nt=nt, T=np.array([T]),
                                          n_omega=n_omega), verbose=False)
        r = m.run(list(keys))
        for k in keys:
            out[k][i] = float(r[k][0])
        out["min_absG"][i] = m.G_abs_min
    if verbose:
        print(f"[cut] {Es.size} points at Gamma={Gamma:g}, T={T:g} in "
              f"{time.time() - t0:.1f}s", flush=True)
    return out


def report_cut(c, rhp_guard=5e-3):
    print(f"Gamma = {c['Gamma']:g}, E_c = {c['E_c']:.3f}, T = {c['T']:g}; "
          f"N_BLP on Omega t <= {c['blp_tmax']:g}, others on Omega t <= {c['tmax']:g}")
    print(f"{'E-E_c':>7} {'N_BLP':>10} {'1st rev':>8} {'N_RHP':>9} {'N_MSV':>9} "
          f"{'N_RHPE':>9} {'min|G|':>7}")
    for i, d in enumerate(c["dE"]):
        tf = c["t_first"][i]
        print(f"{d:+7.3f} {c['N_BLP_long'][i]:10.2e} "
              f"{('-' if np.isnan(tf) else f'{tf:.0f}'):>8} {c['RHP'][i]:9.2e} "
              f"{c['MSV'][i]:9.2e} {c['RHPE'][i]:9.2e} {c['min_absG'][i]:7.3f}"
              + ("   <- |G| near zero: N_RHP unreliable (App. B)"
                 if c['min_absG'][i] < rhp_guard else ""))


def save_cut(c, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    np.savez_compressed(stem + ".npz", **c)
    cols = ["dE", "E", "N_BLP_long", "t_first", "BLP", "LPP", "MSV", "RHPE",
            "RHP", "min_absG", "bound_state"]
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(c["dE"])):
            w.writerow([c[k][i] for k in cols])
    print("wrote", stem + ".npz", "and", stem + ".csv")


def plot_cut(c, path, rhp_guard=5e-3):
    import matplotlib.pyplot as plt
    try:
        from combined_panels import _STYLE, use_paper_style
        use_paper_style()
    except Exception:
        _STYLE = {k: dict(label=k) for k in ("BLP", "LPP", "MSV", "RHPE", "RHP")}
    x = c["dE"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="constrained")
    # Where |G| passes close to zero N_RHP picks up the logarithmic divergence
    # of Appendix B and stops meaning anything; do not draw it there.
    rhp = np.where(np.asarray(c["min_absG"]) >= rhp_guard, c["RHP"], np.nan)
    for k, y in (("BLP", c["N_BLP_long"]), ("MSV", c["MSV"]),
                 ("RHPE", c["RHPE"]), ("RHP", rhp)):
        y = np.asarray(y, float)
        st = dict(_STYLE[k])
        ax.plot(x, np.where(y > 0, y, np.nan), marker="o", ms=2.5, **st)
    zero = c["N_BLP_long"] <= 0
    ymin = 0.3 * np.nanmin([np.min(v[v > 0]) for v in
                            (c["N_BLP_long"], c["MSV"], rhp) if (v > 0).any()])
    ax.set_yscale("log")
    ax.set_ylim(bottom=ymin)
    if zero.any():
        ax.plot(x[zero], np.full(zero.sum(), ymin), ls="none", marker="v", ms=5,
                mfc="white", mec=_STYLE["BLP"]["color"], mew=1.0, clip_on=False,
                zorder=6, label=rf"$\mathcal{{N}}_{{\rm BLP}}=0$ to $\Omega t={c['blp_tmax']:g}$")
    ax.axvline(0.0, color="C3", lw=0.8)
    ax.set_xlabel(r"$(E-E_c)/\Omega$")
    ax.set_ylabel(r"$\mathcal{N}$")
    ax.set_title(rf"$\Gamma={c['Gamma']:g}\,\Omega$, $T={c['T']:g}\,\Omega$",
                 fontsize=8)
    ax.legend(fontsize=6.5, handlelength=1.5, labelspacing=0.25)
    save_figure(fig, path)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--E", type=float, nargs=3, default=[0.0, 3.0, 121],
                   metavar=("LO", "HI", "N"))
    p.add_argument("--Gamma", type=float, nargs=3, default=[0.05, 1.95, 77],
                   metavar=("LO", "HI", "N"))
    p.add_argument("--Omega", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=1280.0)
    p.add_argument("--dt", type=float, default=0.05,
                   help="Time step; 0.05 resolves the 2*Omega band-edge beats.")
    p.add_argument("--n-omega", type=int, default=2049)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--out", default="data/blind_map")
    p.add_argument("--plot", default=None)
    p.add_argument("--from", dest="from_npz", default=None, metavar="NPZ",
                   help="Redraw a map saved earlier (data/blind_map.npz) "
                        "instead of rescanning; use with --plot.")
    p.add_argument("--mark", nargs=3, action="append", default=None,
                   metavar=("E", "GAMMA", "LABEL"),
                   help="Mark a parameter point; repeat for several. Default: "
                        "the points of Figs. 2, 3 and 4.")
    p.add_argument("--cut", type=float, default=None, metavar="GAMMA",
                   help="Instead of the map, compute all measures on a line "
                        "crossing the threshold at this Gamma.")
    p.add_argument("--dE", type=float, nargs="+",
                   default=[-0.15, -0.1, -0.07, -0.05, -0.03, -0.02, -0.01,
                            -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--T", type=float, default=0.02)
    p.add_argument("--cut-tmax", type=float, default=80.0)
    p.add_argument("--cut-nt", type=int, default=8001)
    p.add_argument("--plot-cut", default=None)
    a = p.parse_args(argv)

    if a.cut is not None:
        c = threshold_cut(a.cut, a.dE, T=a.T, Omega=a.Omega, blp_tmax=a.tmax,
                          blp_dt=a.dt, tmax=a.cut_tmax, nt=a.cut_nt,
                          n_omega=max(a.n_omega, 4097))
        report_cut(c)
        save_cut(c, a.out if a.out != "data/blind_map"
                 else f"data/blind_cut_G{a.cut:g}")
        if a.plot_cut:
            plot_cut(c, a.plot_cut)
        return 0

    marks = (tuple((float(e), float(g), lab) for e, g, lab in a.mark)
             if a.mark else DEFAULT_MARKS)
    if a.from_npz:
        plot_map(load_map(a.from_npz), a.plot or "figures/blind_map.pdf",
                 marks=marks)
        return 0

    E = np.linspace(a.E[0], a.E[1], int(a.E[2]))
    G = np.linspace(a.Gamma[0], a.Gamma[1], int(a.Gamma[2]))
    res = scan(E, G, Omega=a.Omega, tmax=a.tmax, dt=a.dt, n_omega=a.n_omega,
               tol=a.tol)
    save_map(res, a.out)
    print(f"no revival within the window: {res['blind'].sum()}/{res['blind'].size} "
          f"points ({(res['blind'] & res['bound_state']).sum()} with a bound state)")
    if a.plot:
        plot_map(res, a.plot, marks=marks)
    return 0


if __name__ == "__main__":
    main()
