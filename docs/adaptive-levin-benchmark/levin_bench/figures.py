"""
Redraw every report figure from the saved CSVs in ../results/.

    python -m levin_bench.figures

Self-contained: sets its own matplotlib rcParams so it does not depend on any
notebook-side styling helper.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

RESULTS = "results"
FIGS = "figs"
EPS = np.finfo(float).eps
META_GREY = "#7f7f7f"

PROB_ORDER = ["damped_sine", "sinc", "grz", "chirp", "near_singular"]
PRETTY = {
    "damped_sine": "damped sine",
    "sinc": "sinc",
    "grz": "Gradshteyn–Ryzhik",
    "chirp": "chirp (nonlinear phase)",
    "near_singular": "near-singular",
}
SPANS = {"damped_sine": 1.0, "sinc": 99.0, "grz": 1.0, "chirp": 1.0, "near_singular": 1.0}
C = {"levin": "#1f4e9c", "quad": "#c0392b", "qawo": "#2e8b57"}
LBL = {
    "levin": "adaptive Levin",
    "quad": "quad (adaptive Gauss–Kronrod)",
    "qawo": "quad QAWO (oscillatory rule)",
}
PC = {
    "damped_sine": "#1f4e9c",
    "sinc": "#e07b39",
    "grz": "#7b5aa6",
    "chirp": "#2e8b57",
    "near_singular": "#b03a5b",
}
MC = {"exact": "#1f4e9c", "naive": "#2e8b57", "fmod": "#e07b39", "reduce": "#c0392b"}
ML = {
    "exact": "exact phase (60-digit mpmath)",
    "naive": r"$\sin(\omega x)$ direct — libm reduces",
    "fmod": r"pre-reduced with fmod$(\omega x,2\pi)$",
    "reduce": "pre-reduced with range_reduce_mod_2pi",
}


def _style():
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.dpi": 110,
            "savefig.dpi": 300,
        }
    )


def _thin(ax, n=6):
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=n))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=n))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6


def fig1_ladder():
    lad = pd.read_csv(os.path.join(RESULTS, "tierA_ladder.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(13.2, 5.6), sharex=True)
    for j, prob in enumerate(PROB_ORDER):
        axtop, axbot = axes[0, j], axes[1, j]
        for meth in ("quad", "qawo", "levin"):
            s = lad[(lad.problem == prob) & (lad.method == meth) & (lad.status == "ok")]
            if meth == "levin":
                s = s[s.phase_mode == "mod2pi"]
            if s.empty:
                continue
            s = s.sort_values("omega")
            lw = 1.9 if meth == "levin" else 1.2
            al = 1.0 if meth == "levin" else 0.8
            axtop.plot(s.omega, s.rel_err.clip(lower=1e-17), "o-", color=C[meth], ms=3.2, lw=lw, alpha=al)
            axbot.plot(s.omega, s.time_s * 1e3, "o-", color=C[meth], ms=3.2, lw=lw, alpha=al)
        om = np.array(sorted(lad[lad.problem == prob].omega.unique()))
        axtop.plot(om, EPS * om * SPANS[prob], "--", color=META_GREY, lw=1.1, zorder=1)
        axtop.axhline(1.0, color="#999999", lw=0.7, ls=":", zorder=0)
        for ax in (axtop, axbot):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_minor_formatter(NullFormatter())
        axtop.set_ylim(1e-17, 1e11)
        axbot.set_ylim(3e-2, 3e3)
        axtop.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=6))
        axbot.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=5))
        axtop.set_title(PRETTY[prob], fontsize=9)
        axes[1, j].set_xlabel(r"$\omega$")
        if j:
            axtop.set_yticklabels([])
            axbot.set_yticklabels([])
    axes[0, 0].set_ylabel("relative error")
    axes[1, 0].set_ylabel("wall time (ms)")
    axes[0, 0].text(0.06, 0.93, "answer is\nworthless", transform=axes[0, 0].transAxes,
                    fontsize=7, color="#999999", va="top")
    axes[0, 3].text(0.95, 0.06, r"$\epsilon\,\theta_{\max}$", transform=axes[0, 3].transAxes,
                    fontsize=7, color=META_GREY, ha="right")
    h = [mpl.lines.Line2D([], [], color=C[m], marker="o", ms=3.2,
                          lw=1.9 if m == "levin" else 1.2, label=LBL[m])
         for m in ("levin", "quad", "qawo")]
    h.append(mpl.lines.Line2D([], [], color=META_GREY, ls="--", lw=1.1,
                              label=r"$\epsilon\,\theta_{\max}$ (attainable floor)"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Levin cost is flat in frequency and its error tracks the phase floor; "
                 "quad diverges above $\\omega\\sim10^6$", fontsize=10, y=0.985)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    out = os.path.join(FIGS, "fig1_ladder.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig2_estimator():
    est = pd.read_csv(os.path.join(RESULTS, "tierA_estimator.csv"))
    tol = pd.read_csv(os.path.join(RESULTS, "tierA_tolerance.csv"))
    fig, ax2 = plt.subplots(1, 3, figsize=(12.6, 4.4))

    a = ax2[0]
    for prob in PROB_ORDER:
        s = est[est.problem == prob].sort_values("omega")
        a.plot(s.omega, s.rel_err.clip(lower=1e-18), "o-", color=PC[prob], ms=3.4, lw=1.7)
        a.plot(s.omega, s.internal_rel_est.clip(lower=1e-18), "^--", color=PC[prob],
               ms=3.0, lw=1.0, alpha=0.55)
    a.set(xscale="log", yscale="log", xlabel=r"$\omega$", ylabel="relative error")
    a.set_title("The integrator's own estimate stops\ntracking the true error above $\\omega\\sim10^5$",
                fontsize=9)
    a.set_ylim(1e-18, 1e-3)
    a.plot([], [], "o-", color=META_GREY, ms=3.4, lw=1.7, label="true error (vs oracle)")
    a.plot([], [], "^--", color=META_GREY, ms=3.0, lw=1.0, label="reported by integrator")
    a.legend(frameon=False, fontsize=7, loc="upper left")

    b = ax2[1]
    est = est.assign(under=est.true_abs_err / est.internal_abs_est.replace(0, np.nan))
    for prob in PROB_ORDER:
        s = est[est.problem == prob].sort_values("omega")
        b.plot(s.omega, s.under, "o-", color=PC[prob], ms=3.4, lw=1.6, label=PRETTY[prob])
    b.axhline(1.0, color="#444444", lw=0.9, zorder=1)
    b.set(xscale="log", yscale="log", xlabel=r"$\omega$", ylabel="true error / reported error")
    b.set_title("Under-reporting reaches $10^{10}$:\nsilent loss of accuracy", fontsize=9)
    b.text(1.5e1, 2.0, "reported is optimistic (unsafe)", fontsize=7, color="#444444")
    b.text(1.5e1, 0.25, "reported is conservative (safe)", fontsize=7, color="#444444", va="top")
    b.legend(frameon=False, fontsize=7, loc="lower right")

    c = ax2[2]
    for prob in tol.problem.unique():
        for w, mk in zip(sorted(tol.omega.unique()), ("o", "s", "^")):
            s = tol[(tol.problem == prob) & (tol.omega == w)].sort_values("atol")
            if s.empty:
                continue
            c.plot(s.atol, s.rel_err.clip(lower=1e-18), mk + "-", color=PC[prob],
                   ms=3.2, lw=1.3, alpha=0.9)
    c.set(xscale="log", yscale="log", xlabel="requested absolute tolerance",
          ylabel="delivered relative error")
    c.set_title("Tightening the tolerance by 24 decades\nbarely changes what is delivered", fontsize=9)
    h = [mpl.lines.Line2D([], [], color=META_GREY, marker=m, ms=3.2, lw=1.3,
                          label=f"$\\omega=10^{{{int(np.log10(w))}}}$")
         for m, w in zip(("o", "s", "^"), sorted(tol.omega.unique()))]
    c.legend(handles=h, frameon=False, fontsize=7, loc="lower left")

    for ax in ax2:
        ax.margins(0.05)
        _thin(ax)
    fig.subplots_adjust(bottom=0.17, wspace=0.30, top=0.82)
    out = os.path.join(FIGS, "fig2_estimator.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_bessel_cost():
    tr = pd.read_csv(os.path.join(RESULTS, "tierB_truncation.csv"))
    rc = pd.read_csv(os.path.join(RESULTS, "tierA_reduction_cost.csv"))
    orr = pd.read_csv(os.path.join(RESULTS, "tierA_order.csv"))
    fig, ax3 = plt.subplots(1, 3, figsize=(12.6, 4.4))
    OCM = dict(zip(sorted(tr.oracle.unique()),
                   plt.cm.viridis(np.linspace(0.05, 0.9, tr.oracle.nunique()))))

    a = ax3[0]
    for orc, s in tr.groupby("oracle"):
        s = s.sort_values("max_x")
        a.plot(s.max_x, s.rel_err, "o-", color=OCM[orc], ms=3.4, lw=1.5, label=orc)
    a.axhspan(1.5e-8, 4.5e-8, color="#c0392b", alpha=0.12, zorder=0)
    a.text(2e6, 6e-8, "phase-function floor $\\approx2\\times10^{-8}$", fontsize=7, color="#a03020")
    a.set(xscale="log", yscale="log", xlabel=r"upper limit $x_{\max}$",
          ylabel="relative error vs closed form")
    a.set_title("Truncation error falls as $1/x_{\\max}$ until it\nhits the Liouville–Green phase floor",
                fontsize=9)
    a.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center",
             bbox_to_anchor=(0.5, -0.20), handlelength=1.4, columnspacing=1.0)

    b = ax3[1]
    b.plot(rc.omega, rc.range_reduce_s * 1e6, "o-", color="#c0392b", ms=3.6, lw=1.7)
    b.plot(rc.omega, rc.fmod_s * 1e6, "s-", color="#1f4e9c", ms=3.6, lw=1.7)
    b.set(xscale="log", yscale="log", xlabel=r"$\omega$",
          ylabel="time per phase evaluation ($\\mu$s)")
    b.set_title("The prime-factor reduction costs $\\sim60\\times$\nplain fmod for no accuracy gain",
                fontsize=9)
    b.text(3e3, 3.0, "range_reduce_mod_2pi", fontsize=7.5, color="#c0392b")
    b.text(3e3, 0.13, "fmod$(\\omega x,\\,2\\pi)$", fontsize=7.5, color="#1f4e9c")
    b.set_ylim(0.05, 60)

    c = ax3[2]
    sub = orr[orr.problem.isin(["sinc", "near_singular"])]
    for (prob, w), s in sub.groupby(["problem", "omega"]):
        if w > 1e9:
            continue
        s = s.sort_values("chebyshev_order")
        c.plot(s.chebyshev_order, s.levin_solves, "o-", color=PC[prob], ms=4.0, lw=1.7,
               label=PRETTY[prob])
    c.axvline(12, color=META_GREY, ls="--", lw=1.1)
    c.text(12.6, 60, "current default", fontsize=7, color=META_GREY, rotation=90, va="top")
    c.set(xlabel="Chebyshev order", ylabel="Levin solves")
    c.set_title("Raising the order cuts subdivision work;\naccuracy is unchanged from order 4 to 32",
                fontsize=9)
    c.legend(frameon=False, fontsize=7, loc="upper right")
    c.set_ylim(0, 72)

    for ax in ax3:
        ax.margins(0.05)
    _thin(ax3[0])
    _thin(ax3[1])
    ax3[2].xaxis.labelpad = 6
    ax3[2].yaxis.labelpad = 6
    fig.subplots_adjust(bottom=0.26, wspace=0.32, top=0.82)
    out = os.path.join(FIGS, "fig3_bessel_cost.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig4_mechanism():
    pf = pd.read_csv(os.path.join(RESULTS, "tierA_phase_floor.csv"))
    fig, ax4 = plt.subplots(1, 2, figsize=(9.4, 4.2))
    panels = [
        ("exact_endpoints",
         "Endpoints exactly representable ($[0,1]$):\nthe Levin core itself holds $10^{-16}$ to $\\omega=10^{12}$"),
        ("generic_endpoints",
         "Generic endpoints ($[0.07,0.93]$):\nall phase modes degrade together"),
    ]
    for i, (exp, title) in enumerate(panels):
        a = ax4[i]
        s0 = pf[pf.span == exp]
        for mode in ("exact", "naive", "fmod", "reduce"):
            s = s0[s0.phase_mode == mode].sort_values("omega")
            a.plot(s.omega, s.rel_err, "o-", color=MC[mode], ms=3.6, lw=1.6, label=ML[mode])
        om = np.array(sorted(s0.omega.unique()))
        span = 1.0 if exp == "exact_endpoints" else 0.86
        a.plot(om, EPS * om * span, "--", color=META_GREY, lw=1.2, label=r"$\epsilon\,\theta_{\max}$")
        a.set(xscale="log", yscale="log", xlabel=r"$\omega$")
        a.set_title(title, fontsize=9)
        a.set_ylim(1e-17, 1e-3)
        _thin(a)
    ax4[0].set_ylabel("relative error")
    ax4[1].set_yticklabels([])
    h4, l4 = ax4[0].get_legend_handles_labels()
    fig.legend(h4, l4, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The high-frequency error is entirely in the double-precision endpoint phase, "
                 "not in the Levin algorithm", fontsize=10, y=0.97)
    fig.subplots_adjust(bottom=0.26, top=0.80, wspace=0.06)
    out = os.path.join(FIGS, "fig4_mechanism.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    _style()
    os.makedirs(FIGS, exist_ok=True)
    for fn in (fig1_ladder, fig2_estimator, fig3_bessel_cost, fig4_mechanism):
        try:
            print("wrote", fn())
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: missing input ({exc})")


if __name__ == "__main__":
    main()
