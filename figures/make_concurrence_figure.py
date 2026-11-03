"""Summary figure for the TFG README: concurrence Delta(theta) in three regimes.

Closed forms from the thesis: eq. (44) soft breaking, eq. (50) general |+->,
eq. (57) general |++>. They agree with the amplitudes of eq. (40), with eqs. (53), (58)
and with the Mathematica plots of the thesis.
Usage: python figures/make_concurrence_figure.py  (writes the PNGs next to this script)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
th = np.linspace(0, np.pi, 20001)
c, c2, c4, s = np.cos(th), np.cos(2 * th), np.cos(4 * th), np.sin(th)


def soft(x):  # eq. (44), M_xi = 0, x = E/m_phi
    return 1 - 4 * c**2 / (x**4 * c4 + (1 - 4 * (x**4 + x**2)) * c2 + 3 * x**4 + 4 * x**2 + 3)


def pm(E, mphi, Mxi):  # eq. (50)
    E2, E4, m2, M2 = E**2, E**4, mphi**2, Mxi**2
    num = 2 * np.abs(2 * E4 + (4 * m2 - 6 * M2) * E2 + (m2 - 2 * M2) ** 2 + 2 * E2 * (M2 - E2) * c2) * s**2
    den = np.abs(3 * E4 + (4 * m2 - 7 * M2) * E2 + 3 * (m2 - 2 * M2) ** 2 + E2 * (E2 - M2) * c4
                 + (-4 * E4 + (8 * M2 - 4 * m2) * E2 + (m2 - 2 * M2) ** 2) * c2)
    return num / den


def pp(E, mphi, Mxi):  # eq. (57)
    E2, E4, E6, m2, m4, M2, M4 = E**2, E**4, E**6, mphi**2, mphi**4, Mxi**2, Mxi**4
    P1 = -2 * E4 + 2 * M4 + 2 * m4 + (E2 - 5 * M2) * m2 + (E2 - M2) * (2 * E2 + 2 * M2 - m2) * c2
    P2 = (-4 * E6 + (22 * M2 - 8 * m2) * E4 - (20 * M4 - 13 * m2 * M2 + 2 * m4) * E2 + M4 * (2 * M2 - m2)
          + (E2 - M2) * (4 * E4 - 2 * M2 * E2 + 2 * M4 - M2 * m2) * c2)
    P3 = (2 * E2 * m4 + (8 * E4 - 13 * M2 * E2 + M4) * m2 + 2 * (E2 - M2) * (2 * E4 - 9 * M2 * E2 + M4)
          - (E2 - M2) * (4 * E4 - 2 * M2 * E2 + 2 * M4 - M2 * m2) * c2)
    s22 = np.sin(2 * th) ** 2
    A = E2 * (E2 - M2) ** 2 * s22 + P1 * P2 / (m2 - 4 * E2) ** 2
    B = 4 * E2 * M2 * (E2 - M2) ** 2 * s22 + (2 * M4 * P1**2 + 2 * P3**2) / (m2 - 4 * E2) ** 2
    return 4 * M2 * np.abs(A) / B


def susy(x):  # m_phi = M_xi = 1, x = |p|/m
    return np.sqrt(1 + x**2)


panels = [
    (r"Soft SUSY breaking: $M_\xi=0,\ m_\phi\neq0$", r"$|{+}{-}\rangle$", r"$x=E/m_\phi$",
     [0.01, 1, 10, 100], lambda x: soft(x)),
    (r"Exact SUSY: $m_\phi=M_\xi=m$", r"$|{+}{-}\rangle$", r"$x=|\vec p\,|/m$",
     [0.1, 1, 2, 100], lambda x: pm(susy(x), 1.0, 1.0)),
    (r"Exact SUSY: $m_\phi=M_\xi=m$", r"$|{+}{+}\rangle$", r"$x=|\vec p\,|/m$",
     [0.1, 0.67, 2, 100], lambda x: pp(susy(x), 1.0, 1.0)),
]

themes = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", grid="#d1d9e0",
                  ramp=["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#3d444d",
                 ramp=["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"]),
}

for name, t in themes.items():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "mathtext.fontset": "dejavusans",
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["grid"], "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
    })
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for i, (ax, (title, state, xdef, xs, f)) in enumerate(zip(axs, panels)):
        ax.axvline(np.pi / 2, color=t["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        for col, x in zip(t["ramp"], xs):
            ax.plot(th, f(x), color=col, lw=2.0, solid_capstyle="round", label=f"{x:g}", zorder=3)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                      ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color=t["grid"], lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, pad=6)
        ax.set_xlabel(r"scattering angle $\theta$")
        ax.set_title(f"{title}\ninitial state {state}", fontsize=11.5, loc="left", color=t["fg"], pad=10)
        leg = ax.legend(title=xdef, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4,
                        frameon=False, handlelength=1.6, columnspacing=1.2, fontsize=10, title_fontsize=10)
        leg.get_title().set_color(t["muted"])
    axs[0].set_ylabel(r"concurrence $\Delta$")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(f"{out_dir}/concurrence_{name}.png", dpi=150, facecolor=t["bg"])
    plt.close(fig)
print("ok")
