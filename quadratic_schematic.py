#!/usr/bin/env python
r"""Schematic of the generic quadratic system--bath model of Sec. II.

A single fermionic mode d coupled to a reservoir of K modes c_k through an
arbitrary quadratic Hamiltonian: normal terms U_ij f_i^dag f_j (solid) and
pairing terms V_ij f_i^dag f_j^dag (dashed). The reservoir starts in a
parity-even state, uncorrelated with the system.

    python quadratic_schematic.py                       # figures/quadratic_model.pdf/.png
    python quadratic_schematic.py --font-scale 0.9      # all text 10% smaller
    NM_USETEX=0 python quadratic_schematic.py           # without a LaTeX install
"""
from __future__ import annotations

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

SYS = "#E07B39"      # system mode
SYS_FILL = "#FBE3D3"
BATH = "#3A7CA5"     # reservoir modes
BATH_FILL = "#E6F0F7"
BOX_FILL = "#F3F8FB"
U_COL = "0.25"       # normal (hopping) terms
V_COL = "#B03A48"    # pairing terms

# Every font size below is multiplied by FS. The figure is drawn at its final
# printed width (3.4 in, one column), so FS * size is the size in the paper:
# with FS = 1 the labels print at 7-7.5 pt, just below the caption text.
FS = 1.0

# Drawing range in data units; the canvas takes this aspect ratio exactly.
XLIM = (-0.62, 9.82)
YLIM = (0.12, 5.86)


def use_style():
    usetex = os.environ.get("NM_USETEX", "1") not in ("0", "false", "False")
    mpl.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.linewidth": 0.6,
    })


def mode(ax, xy, r, label, edge, fill, fs=8.5, z=3):
    ax.add_patch(Circle(xy, r, fc=fill, ec=edge, lw=1.1, zorder=z))
    ax.text(*xy, label, ha="center", va="center", fontsize=FS * fs, zorder=z + 1)


def bond(ax, p, q, kind, r1, r2, lw=1.0, z=2):
    """Line between the rims of two modes; kind 'U' solid, 'V' dashed."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    a, b = p + r1 * u, q - r2 * u
    style = dict(color=U_COL, ls="-") if kind == "U" else dict(color=V_COL, ls=(0, (3.2, 1.8)))
    ax.plot([a[0], b[0]], [a[1], b[1]], lw=lw, zorder=z, **style)
    return (a + b) / 2, u


def panel(ax):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- reservoir box and modes ----------------
    ax.add_patch(FancyBboxPatch((4.15, 0.95), 5.55, 4.35,
                                boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=BOX_FILL, ec=BATH, lw=0.9, zorder=0))
    ax.text(6.92, 4.98, r"reservoir: $K$ modes $\hat c_k$, $\hat H_B$",
            ha="center", va="center", fontsize=FS * 7.5, color=BATH)

    rb = 0.36
    C = {1: (5.15, 3.95), 2: (5.05, 2.15), 3: (6.55, 3.15), 4: (7.15, 4.25), 5: (7.35, 1.85), "K": (8.85, 3.05)}
    labels = {1: r"$\hat c_1$", 2: r"$\hat c_2$", 3: r"$\hat c_3$",
              4: r"$\hat c_4$", 5: r"$\hat c_5$", "K": r"$\hat c_K$"}
    for k, xy in C.items():
        mode(ax, xy, rb, labels[k], BATH, BATH_FILL, fs=7.5)
    ax.text(8.1, 3.1, r"$\cdots$", ha="center", va="center", fontsize=FS * 9)

    for (i, j, kind) in [(1, 3, "U"), (2, 3, "U"), (1, 2, "V"), (3, 4, "U"), (3, 5, "V"), (4, "K", "U"), (5, "K", "U")]:
        if "K" in (i, j):
            p, q = np.array(C[i]), np.array(C[j])
            u = (q - p) / np.linalg.norm(q - p)
            a, b = p + rb * u, p + 0.62 * (q - p)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=U_COL, lw=0.8,
                    zorder=2, alpha=0.8)
            continue
        bond(ax, C[i], C[j], kind, rb, rb, lw=0.9)
    mid, _ = bond(ax, C[1], C[3], "U", rb, rb, lw=0.9)
    ax.text(mid[0] + 0.12, mid[1] + 0.2, r"$U_{kk'}$", fontsize=FS * 7,
            ha="left", va="bottom", color=U_COL)
    mid, _ = bond(ax, C[3], C[5], "V", rb, rb, lw=0.9)
    ax.text(mid[0] + 0.18, mid[1] + 0.02, r"$V_{kk'}$", fontsize=FS * 7,
            ha="left", va="center", color=V_COL)

    ax.text(6.92, 1.3, r"$\hat\rho_B$: e.g. Gibbs state at $(\beta,\mu)$",
            ha="center", va="center", fontsize=FS * 7)

    # ---------------- system mode ----------------
    D = (1.75, 3.05)
    rs = 0.55
    mode(ax, D, rs, r"$\hat d$", SYS, SYS_FILL, fs=10)
    ax.text(D[0], D[1] + 0.85, "system", ha="center", va="bottom",
            fontsize=FS * 7.5, color=SYS)
    # two-level structure of the mode
    x0, x1 = 0.25, 0.75
    for y, lab in [(3.45, r"$|1\rangle$"), (2.65, r"$|0\rangle$")]:
        ax.plot([x0, x1], [y, y], color=SYS, lw=1.0)
        ax.text(x0 - 0.06, y, lab, ha="right", va="center", fontsize=FS * 7)
    ax.text(D[0], D[1] - 0.82, r"$\hat H_S=U_{dd} \hat d^\dagger\hat d$",
            ha="center", va="top", fontsize=FS * 7)

    # ---------------- system--reservoir couplings ----------------
    mid, _ = bond(ax, D, C[1], "U", rs, rb, lw=1.2)
    ax.text(mid[0] - 0.05, mid[1] + 0.2, r"$U_{dk}$", fontsize=FS * 7.5,
            ha="center", va="bottom", color=U_COL)
    mid, _ = bond(ax, D, C[2], "V", rs, rb, lw=1.2)
    ax.text(mid[0] - 0.05, mid[1] - 0.22, r"$V_{dk}$", fontsize=FS * 7.5,
            ha="center", va="top", color=V_COL)
    ax.text(3.05, 3.05, r"$\hat H_{SB}$", ha="center", va="center", fontsize=FS * 7.5)

    # ---------------- legend and initial state ----------------
    ly = 5.62
    ax.plot([0.15, 0.75], [ly, ly], color=U_COL, lw=1.1)
    ax.text(0.85, ly, r"normal: $U_{ij} \hat f_i^\dagger\hat f_j$",
            va="center", fontsize=FS * 7)
    ax.plot([4.45, 5.05], [ly, ly], color=V_COL, lw=1.1, ls=(0, (3.2, 1.8)))
    ax.text(5.15, ly, r"pairing: $V_{ij} \hat f_i^\dagger \hat f_j^\dagger + V^*_{ij} \hat f_i \hat f_j$",
            va="center", fontsize=FS * 7)
    ax.text(5.0, 0.42,
            r"$\hat \rho(0) = \hat \rho_S(0) \otimes \hat \rho_B, [\hat \rho_B, \hat{\mathcal{P}}] = 0$",
            ha="center", va="center", fontsize=FS * 7.5)


def main(argv=None):
    global FS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default="figures/quadratic_model",
                   help="Output stem; writes <stem>.pdf and <stem>.png.")
    p.add_argument("--font-scale", type=float, default=FS,
                   help="Multiply every font size by this factor (default %(default)s).")
    a = p.parse_args(argv)
    FS = a.font_scale

    use_style()
    # The axes fill the whole canvas, and the canvas has the drawing's aspect
    # ratio (XLIM x YLIM), so the PDF is exactly one column wide and
    # \includegraphics[width=\linewidth] prints it at 1:1 -- font sizes in
    # this file are then the sizes in the paper.
    width = 3.4
    aspect = (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
    fig = plt.figure(figsize=(width, width * aspect))
    axa = fig.add_axes([0, 0, 1, 1])
    panel(axa)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=600)
    print("wrote", a.out + ".pdf", "and", a.out + ".png")


if __name__ == "__main__":
    main()
