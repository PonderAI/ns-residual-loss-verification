"""Function-space verification experiments (Section 6 of the manuscript).

Each experiment either integrates a functional gradient flow from a
prescribed initial error or evaluates a loss directly, and writes one
figure (PDF and PNG) to ./figures.  No networks are involved.

    Section  Function                      Figure
    6.2      exp1_homogenisation           fig1_homogenisation
    6.3      exp2a_tg_streamlines          fig2a_tg_streamlines
    6.3      exp2b_tg_locality             fig2b_tg_locality
    6.4      exp2c_poiseuille              fig2c_poiseuille
    6.5      exp3_balance                  fig3_balance_collapse
    6.6      exp6_transport                fig6_stacking
    6.7      exp4_uniform_rescaling        fig4_scalar_ray
    6.8      exp5_crossover                fig5_crossover

Usage:
    python experiments.py                  # all experiments
    python experiments.py exp2a exp6       # selected experiments
    python experiments.py --quick          # smoke test: run times x 0.02

Numerics: spectral derivatives with the Nyquist mode zeroed (antisymmetric
derivative); u.grad evaluated in skew-symmetric form, so every periodic flow
is the exact gradient flow of its discretised loss and no dealiasing filter
is applied; explicit RK4.  The Poiseuille experiment descends the exact
gradient of a face-based discretisation, so its natural boundary
conditions emerge from the discrete variation.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from core import (Torus, rk4, advect_skew, taylor_green, kovasznay,
                  gaussian_blob, compact_bump, TWO_PI)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

QUICK = "--quick" in sys.argv
TFAC = 0.02 if QUICK else 1.0     # scales every run time in smoke-test mode


# ======================================================================
# Figure style
# ======================================================================

# Set to the manuscript's \textwidth in inches: put \showthe\textwidth in the
# .tex file, read the value in pt from the log, and divide by 72.27.
TEXTWIDTH = 6.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "image.interpolation": "bilinear",
})

EXT = [0, TWO_PI, 0, TWO_PI]
PI_TICKS = ([0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])


def l2(f, *more):
    """RMS over the points of f (and of any further components)."""
    s = np.mean(f ** 2)
    for m in more:
        s += np.mean(m ** 2)
    return np.sqrt(s)


def panel_labels(axes):
    for i, ax in enumerate(axes):
        ax.text(-0.02, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)


def field(ax, f, cmap, vmin=None, vmax=None, norm=None):
    if norm is not None:
        return ax.imshow(f.T, origin="lower", extent=EXT, cmap=cmap, norm=norm)
    return ax.imshow(f.T, origin="lower", extent=EXT, cmap=cmap,
                     vmin=vmin, vmax=vmax)


def torus_axes(ax):
    ax.set_xticks(*PI_TICKS)
    ax.set_yticks(*PI_TICKS)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")


def cbar(fig, ax, im, label=None):
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.ax.tick_params(labelsize=7, width=0.6)
    cb.outline.set_linewidth(0.6)
    if label:
        cb.set_label(label)
    return cb


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 6.2  Additive continuity loss: homogenisation of the divergence and
#      conservation of vorticity.
#      d(gamma)/dt = 2 grad(div gamma)  =>  d(div)/dt = 2 Lap(div).
# ======================================================================

def exp1_homogenisation(N=192, T=0.2, dt=5e-5):
    T *= TFAC
    g = Torus(N)
    # divergence carried by phi, vorticity by chi, at separate locations
    phi = gaussian_blob(g, np.pi - 0.8, np.pi, 0.35)
    chi = gaussian_blob(g, np.pi + 1.2, np.pi + 0.6, 0.5)
    gx = g.dx(phi) + g.dy(chi)
    gy = g.dy(phi) - g.dx(chi)
    d0 = g.div(gx, gy)
    w0 = g.curl(gx, gy)

    def rhs(state):
        a, b = state
        d = g.div(a, b)
        return (2.0 * g.dx(d), 2.0 * g.dy(d))

    def monitor(step, t, state):
        a, b = state
        return (t, l2(g.div(a, b)))

    nsteps = max(int(round(T / dt)), 8)
    n1 = nsteps // 8                                    # snapshot at T/8
    (gxM, gyM), log1 = rk4(rhs, (gx, gy), dt, n1, monitor, every=25)
    dM = g.div(gxM, gyM)
    (gxT, gyT), log2 = rk4(rhs, (gxM, gyM), dt, nsteps - n1, monitor, every=25)
    log = log1 + [(t + n1 * dt, v) for (t, v) in log2[1:]]
    dT = g.div(gxT, gyT)
    wT = g.curl(gxT, gyT)

    t_final = nsteps * dt
    d_exact = g.heat_exact(d0, kappa=2.0, t=t_final)
    times = np.array([r[0] for r in log])
    dnorm = np.array([r[1] for r in log])
    dnorm_exact = np.array([l2(g.heat_exact(d0, 2.0, t)) for t in times])

    # both errors relative to the initial peak
    err_div = np.max(np.abs(dT - d_exact)) / np.max(np.abs(d0))
    err_vort = np.max(np.abs(wT - w0)) / np.max(np.abs(w0))

    fig, axs = plt.subplots(2, 2, figsize=(TEXTWIDTH, 0.78 * TEXTWIDTH),
                            layout="constrained")
    ax = axs.ravel()
    vm = np.max(np.abs(d0))
    im = field(ax[0], d0, "RdBu_r", -vm, vm)
    cbar(fig, ax[0], im, r"$\nabla\cdot\boldsymbol{\gamma}$")
    vm = np.max(np.abs(dM))
    im = field(ax[1], dM, "RdBu_r", -vm, vm)
    cbar(fig, ax[1], im, r"$\nabla\cdot\boldsymbol{\gamma}$")
    for a in ax[:2]:
        torus_axes(a)

    mid = N // 2                                        # slice y = pi
    ax[2].plot(g.X[:, mid], d0[:, mid], color="0.6", ls=":", label=r"$t = 0$")
    ax[2].plot(g.X[:, mid], dT[:, mid], "C0-", label="gradient flow")
    ax[2].plot(g.X[:, mid], d_exact[:, mid], "k--", lw=0.9, label="exact")
    ax[2].set_xticks(*PI_TICKS)
    ax[2].set_xlabel(r"$x$")
    ax[2].set_ylabel(r"$\nabla\cdot\boldsymbol{\gamma}$ at $y = \pi$")
    ax[2].legend()

    ax[3].semilogy(times, dnorm, "C0-", label="gradient flow")
    ax[3].semilogy(times, dnorm_exact, "k--", lw=0.9, label="exact")
    ax[3].set_xlabel(r"$t$")
    ax[3].set_ylabel(r"RMS $\nabla\cdot\boldsymbol{\gamma}$")
    ax[3].legend()

    panel_labels(ax)
    save(fig, "fig1_homogenisation")
    print(f"[exp1] divergence vs exact, max rel. err {err_div:.2e}; "
          f"vorticity change, max rel. {err_vort:.2e}; snapshot at t = {n1 * dt:g}")


# ======================================================================
# 6.3  Multiplicative continuity loss on Taylor-Green flow.
#      d(alpha)/dt = 2 w(w(alpha)),  w = u.grad in skew-symmetric form.
# ======================================================================

def _tg_flow(N, T, dt, alpha0, masks=None):
    g = Torus(N)
    u1, u2, psi = taylor_green(g)

    def w(f):
        return advect_skew(g, u1, u2, f)

    def rhs(state):
        (a,) = state
        return (2.0 * w(w(a)),)

    def monitor(step, t, state):
        # (t, overall RMS residual, RMS residual over each mask)
        (a,) = state
        r = w(a)
        out = [t, l2(r)]
        if masks is not None:
            out += [l2(r[m]) for m in masks]
        return tuple(out)

    nsteps = max(int(round(T / dt)), 1)
    (aT,), log = rk4(rhs, (alpha0,), dt, nsteps, monitor, every=100)
    return g, u1, u2, psi, aT, log


def exp2a_tg_streamlines(N=128, T=10.0, dt=1.1e-4, seed=0):
    T *= TFAC
    rng = np.random.default_rng(seed)
    g = Torus(N)

    # smooth random initial field: modes 1 <= kx, ky <= 3, random
    # amplitude and independent phases in x and y
    a0 = np.zeros((N, N))
    for kx in range(1, 4):
        for ky in range(1, 4):
            a0 += (rng.normal() * np.cos(kx * g.X + rng.uniform(0, TWO_PI))
                   * np.cos(ky * g.Y + rng.uniform(0, TWO_PI)))
    a0 = 1.0 + 0.3 * a0 / np.max(np.abs(a0))

    # regions away from and near the cell boundaries
    _, _, psi0 = taylor_green(g)
    far = np.abs(psi0) > 0.2

    g, u1, u2, psi, aT, log = _tg_flow(N, T, dt, a0, masks=(far, ~far))
    log = np.array(log)
    times = log[:, 0]

    cell = (g.X < np.pi) & (g.Y < np.pi)                # cell 1
    cell2 = (g.X > np.pi) & (g.Y < np.pi)               # neighbouring cell

    fig, axs = plt.subplots(2, 2, figsize=(TEXTWIDTH, 0.8 * TEXTWIDTH),
                            layout="constrained")
    ax = axs.ravel()

    # (a) initial field, (b) final field
    for a, f in ((ax[0], a0), (ax[1], aT)):
        im = field(a, f, "viridis")
        a.contour(g.X, g.Y, psi, 8, colors="w", linewidths=0.4,
                  linestyles="solid")
        cbar(fig, a, im, r"$\alpha$")
        torus_axes(a)

    # (c) residual: overall, away from and near the cell boundaries,
    #     each normalised by its own initial value
    for col, style, lab in ((1, "C0-", "all"),
                            (2, "k--", r"$|\psi| > 0.2$"),
                            (3, "k:", r"$|\psi| \leq 0.2$")):
        ax[2].semilogy(times, log[:, col] / log[0, col], style, label=lab)
    ax[2].set_xlabel(r"$t$")
    ax[2].set_ylabel(r"RMS $\mathbf{u}\cdot\nabla\alpha$ (normalised)")
    ax[2].legend()

    # (d) alpha against the streamline label in two neighbouring cells
    sub = slice(None, None, 7)
    ax[3].plot(psi[cell][sub], a0[cell][sub], ".", ms=1.0, color="0.7",
               label=r"$t = 0$", rasterized=True)
    ax[3].plot(psi[cell][sub], aT[cell][sub], ".", ms=1.0, color="C0",
               label=rf"$t = {T:g}$, cell 1", rasterized=True)
    ax[3].plot(psi[cell2][sub], aT[cell2][sub], ".", ms=1.0, color="C3",
               label=rf"$t = {T:g}$, cell 2", rasterized=True)
    ax[3].set_xlabel(r"$\psi$")
    ax[3].set_ylabel(r"$\alpha$")
    ax[3].legend(markerscale=6)

    panel_labels(ax)
    save(fig, "fig2a_tg_streamlines")

    # final reductions; the last points of the three curves in (c)
    # should match these
    r0 = advect_skew(g, u1, u2, a0)
    rT = advect_skew(g, u1, u2, aT)
    print(f"[exp2a] residual reduction: overall {log[0, 1] / log[-1, 1]:.1e}; "
          f"|psi| > 0.2 {l2(r0[far]) / l2(rT[far]):.1e}; "
          f"|psi| <= 0.2 {l2(r0[~far]) / l2(rT[~far]):.1e}")

# def _tg_flow(N, T, dt, alpha0, masks=None):
#     g = Torus(N)
#     u1, u2, psi = taylor_green(g)

#     def w(f):
#         return advect_skew(g, u1, u2, f)

#     def rhs(state):
#         (a,) = state
#         return (2.0 * w(w(a)),)

#     def monitor(step, t, state):
#         (a,) = state
#         r = w(a)
#         out = [t, l2(r)]
#         if masks is not None:
#             out += [l2(r[m]) for m in masks]
#         return tuple(out)

#     nsteps = max(int(round(T / dt)), 1)
#     (aT,), log = rk4(rhs, (alpha0,), dt, nsteps, monitor, every=100)
#     return g, u1, u2, psi, aT, log


# def exp2a_tg_streamlines(N=128, T=10.0, dt=1.1e-4, seed=0):
#     T *= TFAC
#     rng = np.random.default_rng(seed)
#     g = Torus(N)
#     # smooth random initial field: modes kx, ky <= 3, random amplitude/phase
#     a0 = np.zeros((N, N))
#     for kx in range(1, 4):
#         for ky in range(1, 4):
#             a0 += (rng.normal() * np.cos(kx * g.X + rng.uniform(0, TWO_PI))
#                    * np.cos(ky * g.Y + rng.uniform(0, TWO_PI)))
#     a0 = 1.0 + 0.3 * a0 / np.max(np.abs(a0))

#     g, u1, u2, psi, aT, log = _tg_flow(N, T, dt, a0)
#     times = np.array([r[0] for r in log])
#     res = np.array([r[1] for r in log])

#     cell = (g.X < np.pi) & (g.Y < np.pi)                # cell 1
#     cell2 = (g.X > np.pi) & (g.Y < np.pi)               # neighbouring cell

#     fig, axs = plt.subplots(2, 2, figsize=(TEXTWIDTH, 0.8 * TEXTWIDTH),
#                             layout="constrained")
#     ax = axs.ravel()
#     for a, f in ((ax[0], a0), (ax[1], aT)):
#         im = field(a, f, "viridis")
#         a.contour(g.X, g.Y, psi, 8, colors="w", linewidths=0.4, linestyles="solid")
#         cbar(fig, a, im, r"$\alpha$")
#         torus_axes(a)

#     ax[2].semilogy(times, res / res[0], "C0-")
#     ax[2].set_xlabel(r"$t$")
#     ax[2].set_ylabel(r"RMS $\mathbf{u}\cdot\nabla\alpha$ (normalised)")

#     sub = slice(None, None, 7)
#     ax[3].plot(psi[cell][sub], a0[cell][sub], ".", ms=1.0, color="0.7",
#                label=r"$t = 0$", rasterized=True)
#     ax[3].plot(psi[cell][sub], aT[cell][sub], ".", ms=1.0, color="C0",
#                label=rf"$t = {T:g}$, cell 1", rasterized=True)
#     ax[3].plot(psi[cell2][sub], aT[cell2][sub], ".", ms=1.0, color="C3",
#                label=rf"$t = {T:g}$, cell 2", rasterized=True)
#     ax[3].set_xlabel(r"$\psi$")
#     ax[3].set_ylabel(r"$\alpha$")
#     ax[3].legend(markerscale=6)


#     panel_labels(ax)
#     save(fig, "fig2a_tg_streamlines")

#     r0 = advect_skew(g, u1, u2, a0)
#     rT = advect_skew(g, u1, u2, aT)
#     far = np.abs(psi) > 0.2
#     print(f"[exp2a] residual reduction: overall {res[0] / res[-1]:.1e}; "
#           f"|psi| > 0.2 {l2(r0[far]) / l2(rT[far]):.1e}; "
#           f"|psi| <= 0.2 {l2(r0[~far]) / l2(rT[~far]):.1e}")


def exp2b_tg_locality(N=128, T=10.0, dt=1.1e-4,
                      x0=np.pi / 2 + 0.4, y0=np.pi / 2, R=1.0, amp=0.4):
    """Perturbation confined to one cell.  The bump is C-infinity and
    vanishes identically outside a disc inside the cell [0, pi]^2, centred
    off the cell centre so that it is not a function of psi.  The flow is
    linear in alpha, so integrating the bump alone gives the difference
    between two runs that differ only in that cell."""
    T *= TFAC
    g = Torus(N)
    cell = (g.X < np.pi) & (g.Y < np.pi)
    margin = min(x0, np.pi - x0, y0, np.pi - y0) - R
    assert margin > 0.0, f"bump support leaves the cell (margin {margin:.3f})"

    diff0 = amp * compact_bump(g, x0, y0, R)
    leak0 = l2(diff0[~cell]) / l2(diff0[cell])          # zero by construction
    _, _, _, psi, diffT, _ = _tg_flow(N, T, dt, diff0)
    leak = l2(diffT[~cell]) / l2(diffT[cell])
    evolved = l2(diffT - diff0) / l2(diff0)

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.44 * TEXTWIDTH),
                           layout="constrained")
    vm = np.max(np.abs(diff0))
    for a, f in zip(ax, (diff0, diffT)):
        im = field(a, f, "RdBu_r", -vm, vm)
        a.contour(g.X, g.Y, psi, 8, colors="k", linewidths=0.3, linestyles="solid")
        a.axvline(np.pi, color="k", lw=0.8)
        a.axhline(np.pi, color="k", lw=0.8)
        torus_axes(a)
    cbar(fig, list(ax), im, r"$\Delta\alpha$")
    panel_labels(ax)
    save(fig, "fig2b_tg_locality")
    print(f"[exp2b] leakage (RMS outside / inside cell): initial {leak0:.2e}, "
          f"final {leak:.2e}; relative change of the perturbation {evolved:.2e}")


# ======================================================================
# 6.4  Poiseuille through-flow on a bounded domain.  Exact gradient of a
#      face-based discretisation: natural (zero-slope) end conditions
#      emerge from the discrete variation, and each row sum is conserved.
# ======================================================================

def exp2c_poiseuille(Nx=240, Ny=41, Lx=4.0, T=10.0, dt=4e-5):
    T *= TFAC
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(-1.0, 1.0, Ny)
    hx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    U = 1.0 - Y ** 2                                    # u = (1 - y^2, 0)

    # a = (1.0 + 0.3 * (np.sin(2 * np.pi * X / Lx + 1.0) * np.cos(np.pi * Y)
    #                   + 0.5 * np.sin(4 * np.pi * X / Lx) * np.sin(2 * np.pi * Y))
    #      + 0.15 * np.sin(np.pi * Y))
    a = 1.0 + 0.15 * np.sin(np.pi * Y) + 0.3 * np.cos(np.pi * Y) * np.sin(2 * np.pi * X / Lx)
    a0 = a.copy()
    Uf = U[:-1, :]                                      # U on x-faces

    def grad(a):
        # residual r_{i+1/2} = U (a_{i+1} - a_i)/hx; gradient of hx*sum r^2,
        # divided by hx to give the function-space gradient
        r = Uf * (a[1:, :] - a[:-1, :]) / hx
        gL = np.zeros_like(a)
        flux = Uf * r / hx
        gL[:-1, :] -= 2.0 * flux
        gL[1:, :] += 2.0 * flux
        return gL

    nsteps = max(int(round(T / dt)), 1)
    for _ in range(nsteps):
        k1 = -grad(a)
        k2 = -grad(a + 0.5 * dt * k1)
        k3 = -grad(a + 0.5 * dt * k2)
        k4 = -grad(a + dt * k3)
        a = a + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    interior = np.abs(y) <= 0.6
    std0 = a0.std(axis=0)[interior].max()
    stdT = a.std(axis=0)[interior].max()
    slope_in = np.max(np.abs(a[1, interior] - a[0, interior]) / hx)
    slope_out = np.max(np.abs(a[-1, interior] - a[-2, interior]) / hx)
    mean0 = a0.mean(axis=0)
    meanT = a.mean(axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 0.3 * TEXTWIDTH),
                           layout="constrained")
    lo, hi = a0.min(), a0.max()
    for axi, f in ((ax[0], a0), (ax[1], a)):
        im = axi.imshow(f.T, origin="lower", aspect="equal", cmap="viridis",
                        extent=[0, Lx, -1, 1], vmin=lo, vmax=hi)
        axi.set_xlabel(r"$x$")
        axi.set_ylabel(r"$y$")
    cbar(fig, list(ax[:2]), im, r"$\alpha$")
    sc = ax[2].scatter(mean0, meanT, c=1.0 - y ** 2, s=8, cmap="viridis",
                       vmin=0.0, vmax=1.0)
    lim = [mean0.min(), mean0.max()]
    ax[2].plot(lim, lim, "k--", lw=0.8)
    ax[2].set_xlabel(r"initial along-flow mean of $\alpha$")
    ax[2].set_ylabel(r"final $\alpha$")
    cbar(fig, ax[2], sc, r"$1 - y^2$")
    panel_labels(ax)
    save(fig, "fig2c_poiseuille")
    print(f"[exp2c] interior along-flow std: initial max {std0:.2e}, "
          f"final max {stdT:.2e}; end slopes: inflow {slope_in:.1e}, "
          f"outflow {slope_out:.1e}; max |final - initial mean| "
          f"{np.max(np.abs(meanT - mean0)):.1e}")


# ======================================================================
# 6.5  lambda / ell^2 balance (additive continuity loss).
#      d(gamma)/dt = -2 gamma + 2 lambda grad(div gamma); for
#      gamma = sin(x/ell) x_hat the rate is 2(1 + lambda/ell^2).
# ======================================================================

def exp3_balance(N=128, T=0.05):
    g = Torus(N)
    kg2 = 2.0 * (N // 2 - 1) ** 2                      # largest resolved |k|^2
    ks = [2, 4, 8, 16]                                 # ell = 1/k
    lams = np.logspace(-3.5, 0.0, 8)
    pts = []
    for k in ks:
        gx0 = np.sin(k * g.X)
        gy0 = np.zeros_like(gx0)
        for lam in lams:
            rate_nom = 2.0 * (1.0 + lam * k ** 2)
            rate_max = 2.0 * (1.0 + lam * kg2)
            dt = min(2.0 / rate_max, 1e-3)
            nst = max(int(round(min(T, 2.0 / rate_nom) / dt)), 30)

            def rhs(state, lam=lam):
                a, b = state
                d = g.div(a, b)
                return (-2.0 * a + 2.0 * lam * g.dx(d),
                        -2.0 * b + 2.0 * lam * g.dy(d))

            def mon(step, t, state):
                return (t, l2(state[0], state[1]))

            _, log = rk4(rhs, (gx0, gy0), dt, nst, mon, every=1)
            t_arr = np.array([r[0] for r in log])
            n_arr = np.array([r[1] for r in log])
            rate = -np.polyfit(t_arr, np.log(n_arr), 1)[0]
            pts.append((lam * k ** 2, rate / 2.0 - 1.0, k))

    fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH, 0.42 * TEXTWIDTH),
                           layout="constrained")
    for k, m in zip(ks, "osD^"):
        xs = [p[0] for p in pts if p[2] == k]
        ys = [p[1] for p in pts if p[2] == k]
        ax.loglog(xs, ys, m, ms=4, mfc="none", mew=0.8, label=rf"$\ell = 1/{k}$")
    xr = np.logspace(-3.2, 2.7, 10)
    ax.loglog(xr, xr, "k--", lw=0.8)
    ax.set_xlabel(r"$\lambda/\ell^2$")
    ax.set_ylabel(r"Enhancement $E = r/2 - 1$")
    ax.legend()
    save(fig, "fig3_balance_collapse")
    err = max(abs(p[1] / p[0] - 1.0) for p in pts)
    print(f"[exp3] {len(pts)} points; lambda/ell^2 from "
          f"{min(p[0] for p in pts):.2e} to {max(p[0] for p in pts):.2e}; "
          f"max relative deviation {err:.2e}")


# ======================================================================
# 6.6  Combined multiplicative loss on Taylor-Green flow: transport of
#      relative error into low-velocity regions.
#      d(alpha)/dt = 2(1 - alpha)|u|^2 + 2 lambda w(w(alpha)).
# ======================================================================

def exp6_transport(N=128, T=6.0, dt=1.5e-4, lams=(0.0, 0.1, 0.4, 1.6)):
    T *= TFAC
    g = Torus(N)
    u1, u2, psi = taylor_green(g)
    speed2 = u1 ** 2 + u2 ** 2

    fast_pt = (np.pi / 2, 0.12)       # high-speed point on near-boundary orbits
    a0 = 1.0 - 0.5 * gaussian_blob(g, *fast_pt, 0.22)

    cell = (g.X < np.pi) & (g.Y < np.pi)
    m_slow = cell & (speed2 < 0.15)                     # corners and core
    m_fast = cell & (speed2 > 0.5)

    def w(f):
        return advect_skew(g, u1, u2, f)

    t_snap = 0.6 * TFAC
    results, snaps = {}, {}
    for lam in lams:
        def rhs(state, lam=lam):
            (a,) = state
            return (2.0 * (1.0 - a) * speed2 + 2.0 * lam * w(w(a)),)

        def mon(step, t, state):
            (a,) = state
            e = (1.0 - a) ** 2
            return (t, np.sqrt(np.mean(e[m_slow])), np.sqrt(np.mean(e[m_fast])))

        n1 = max(int(round(t_snap / dt)), 1)
        (aS,), log1 = rk4(rhs, (a0.copy(),), dt, n1, mon, every=100)
        snaps[lam] = aS
        n2 = max(int(round(T / dt)) - n1, 1)
        (aT,), log2 = rk4(rhs, (aS.copy(),), dt, n2, mon, every=100)
        log = log1 + [(t + n1 * dt, a, b) for (t, a, b) in log2[1:]]
        results[lam] = np.array(log)

    fig = plt.figure(figsize=(TEXTWIDTH, 0.66 * TEXTWIDTH), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    for lam in lams:
        t, es, ef = results[lam].T
        ax_s.semilogy(t, es + 1e-12, label=rf"$\lambda = {lam:g}$")
        ax_f.semilogy(t, ef + 1e-12, label=rf"$\lambda = {lam:g}$")
    for a, lab in ((ax_s, "slow set"), (ax_f, "fast set")):
        a.set_xlabel(r"$t$")
        a.set_ylabel(rf"RMS $1 - \alpha$, {lab}")
    ax_s.legend()

    xpos = np.arange(len(lams))
    peak = [results[l][1:, 1].max() for l in lams]
    final = [results[l][-1, 1] for l in lams]
    ax_b.bar(xpos - 0.18, peak, 0.36, color="C0", label="peak")
    ax_b.bar(xpos + 0.18, final, 0.36, color="0.6", label=rf"$t = {T:g}$")
    ax_b.set_xticks(xpos, [f"{l:g}" for l in lams])
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$\lambda$")
    ax_b.set_ylabel(r"RMS $1 - \alpha$, slow set")
    ax_b.legend()

    snap_axes = []
    for j, lam in enumerate((lams[0], lams[-2], lams[-1])):
        a = fig.add_subplot(gs[1, j])
        f = np.abs(1.0 - snaps[lam]) + 1e-9
        im = field(a, f, "inferno", norm=LogNorm(vmin=1e-6, vmax=0.5))
        a.contour(g.X, g.Y, psi, 6, colors="w", linewidths=0.3, linestyles="solid")
        a.contour(g.X, g.Y, speed2, [0.15], colors="cyan", linewidths=0.7)
        a.plot(*fast_pt, "w+", ms=6, mew=0.8)
        torus_axes(a)
        snap_axes.append(a)
    cbar(fig, snap_axes, im, r"$|1 - \alpha|$")

    panel_labels([ax_s, ax_f, ax_b] + snap_axes)
    save(fig, "fig6_stacking")
    np.savez(os.path.join(FIGDIR, "exp6_cache.npz"),
             **{f"traj_{l}": results[l] for l in lams},
             **{f"snap_{l}": snaps[l] for l in lams})
    for lam, p, fl in zip(lams, peak, final):
        print(f"[exp6] lambda={lam:g}: peak slow-set RMS {p:.3e}, final {fl:.3e}")


# ======================================================================
# 6.7  Momentum loss under uniform rescaling u_pred = c u, p_pred = d p.
#      Kovasznay: isolated minima at (0,0) and (1,1).
#      Poiseuille: valley of minima along d = c.
# ======================================================================

def exp4_uniform_rescaling(Re=40, ngrid=401, nxy=320):
    X, Y = np.meshgrid(np.linspace(-0.5, 1.0, nxy),
                       np.linspace(-0.5, 1.5, nxy), indexing="ij")
    f = kovasznay(Re, X, Y)
    adv1 = f["u1"] * f["u1_x"] + f["u2"] * f["u1_y"]
    adv2 = f["u1"] * f["u2_x"] + f["u2"] * f["u2_y"]

    cs = np.linspace(-0.5, 1.5, ngrid)
    ds = np.linspace(-0.5, 1.5, ngrid)
    C, D = np.meshgrid(cs, ds, indexing="ij")

    # R = c^2 (u.grad)u + d grad p - (c/Re) lap u;  L = mean |R|^2, a
    # polynomial in (c, d) with spatial means as coefficients
    A2 = np.mean(adv1 ** 2 + adv2 ** 2)
    P2 = np.mean(f["p_x"] ** 2 + f["p_y"] ** 2)
    V2 = np.mean(f["lap_u1"] ** 2 + f["lap_u2"] ** 2)
    AP = np.mean(adv1 * f["p_x"] + adv2 * f["p_y"])
    AV = np.mean(adv1 * f["lap_u1"] + adv2 * f["lap_u2"])
    PV = np.mean(f["p_x"] * f["lap_u1"] + f["p_y"] * f["lap_u2"])
    L_kov = (C ** 4 * A2 + D ** 2 * P2 + (C / Re) ** 2 * V2
             + 2 * C ** 2 * D * AP - 2 * C ** 3 / Re * AV - 2 * D * C / Re * PV)

    # Poiseuille: R = (2/Re)(c - d) x_hat
    L_poi = (2.0 / Re) ** 2 * (C - D) ** 2

    # direct check that both Kovasznay wells are zeros of the residual
    def rms_residual(c, d):
        r1 = c ** 2 * adv1 + d * f["p_x"] - c / Re * f["lap_u1"]
        r2 = c ** 2 * adv2 + d * f["p_y"] - c / Re * f["lap_u2"]
        return l2(r1, r2)

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.42 * TEXTWIDTH),
                           layout="constrained")
    for a, L in ((ax[0], L_kov), (ax[1], L_poi)):
        Z = np.clip(np.log10(L + 1e-12), -6.0, None)
        cf = a.contourf(C, D, Z, levels=24, cmap="magma")
        cbar(fig, a, cf, r"$\log_{10}\mathcal{L}_{\mathrm{phys}}$")
        a.set_xlabel(r"$s_u$")
        a.set_ylabel(r"$s_p$")
        a.set_aspect("equal")
    ax[0].plot([0, 1], [0, 1], "*", color="w", mec="k", mew=0.5, ms=9)
    ax[1].plot([-0.5, 1.5], [-0.5, 1.5], "w--", lw=0.8)
    panel_labels(ax)
    save(fig, "fig4_uniform_rescaling")
    print(f"[exp4] Kovasznay RMS residual at (0,0): {rms_residual(0, 0):.1e}, "
          f"at (1,1): {rms_residual(1, 1):.1e}")


# ======================================================================
# 6.8  Error-scale Reynolds number.  gamma = sin(x/ell) y_hat on Kovasznay
#      flow; A gamma = (u.grad)gamma + (gamma.grad)u - (1/Re) lap gamma
#      (pressure error omitted, so lambda* is a lower bound).
# ======================================================================

def exp5_crossover(nxy=512):
    X, Y = np.meshgrid(np.linspace(-0.5, 1.0, nxy),
                       np.linspace(0.0, 2.0, nxy), indexing="ij")
    Res = np.array([1, 3, 10, 30, 100, 300, 1000], dtype=float)
    kk = np.array([4, 8, 16, 32, 64], dtype=float)     # ell = 1/k

    ratio_pts, lam_pts = [], []
    for Re in Res:
        f = kovasznay(Re, X, Y)
        for k in kk:
            ell = 1.0 / k
            gy = np.sin(k * X)                          # divergence-free
            gy_x = k * np.cos(k * X)
            adv = f["u1"] * gy_x                        # (u.grad)gamma, y-comp.
            visc = (k ** 2 / Re) * gy                   # -(1/Re) lap gamma
            ratio_pts.append((Re * ell, l2(adv) / l2(visc)))

            A1 = gy * f["u1_y"]
            A2 = f["u1"] * gy_x + gy * f["u2_y"] + visc
            lam_star = (l2(gy) / l2(A1, A2)) ** 2
            lam_pts.append((Re * ell, lam_star / ell ** 2, Re))

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.4 * TEXTWIDTH),
                           layout="constrained")
    xs = np.array([p[0] for p in ratio_pts])
    ys = np.array([p[1] for p in ratio_pts])
    ax[0].loglog(xs, ys, "o", ms=3, mfc="none", mew=0.8, color="C0")
    xr = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 10)
    ax[0].loglog(xr, np.sqrt(2.0) * xr, "k--", lw=0.8)
    ax[0].axvline(1.0, color="0.6", lw=0.6)
    ax[0].set_xlabel(r"$Re\,\ell$")
    ax[0].set_ylabel("advective / viscous (RMS)")

    for Re in Res:
        xs = [p[0] for p in lam_pts if p[2] == Re]
        ys = [p[1] for p in lam_pts if p[2] == Re]
        ax[1].loglog(xs, ys, "o-", ms=3, lw=0.6, mfc="none", mew=0.8,
                     label=rf"$Re = {Re:g}$")
    # prefactors of the two asymptotes, fitted away from the knee and
    # excluding the largest error scale (ell = 1/4), where ell << 1 fails
    fit = [p for p in lam_pts if p[0] / p[2] < 0.1]      # ell < 1/4
    c_low = np.median([p[1] / p[0] ** 2 for p in fit if p[0] < 0.3])
    c_high = np.median([p[1] for p in fit if p[0] > 3.0])
    xr = np.logspace(-2.3, 0, 8)
    ax[1].loglog(xr, c_low * xr ** 2, "k--", lw=0.8)
    xr2 = np.logspace(0.3, 2.4, 8)
    ax[1].loglog(xr2, c_high * np.ones_like(xr2), "k:", lw=0.8)
    ax[1].axvline(1.0, color="0.6", lw=0.6)
    ax[1].set_xlabel(r"$Re\,\ell$")
    ax[1].set_ylabel(r"$\lambda^*/\ell^2$")
    ax[1].legend(ncol=2, fontsize=6)
    panel_labels(ax)
    save(fig, "fig5_crossover")
    print(f"[exp5] {len(ratio_pts)} (Re, ell) pairs; Re*ell from "
          f"{min(p[0] for p in ratio_pts):.2e} to {max(p[0] for p in ratio_pts):.2e}; "
          f"prefactors: lambda*/ell^2 ~ {c_low:.2f} (Re ell)^2 below, "
          f"~ {c_high:.2f} above")

def exp6_resolution_check(lam=1.6, T=6.0, runs=((128, 1.5e-4), (256, 3.75e-5))):
    """Resolution check for the transport experiment.  Nothing physical can
    reach the right-hand cells (x > pi), so any error there is numerical.
    Reports it alongside the slow-set peak for each resolution.  Writes no
    manuscript figure."""
    T *= TFAC
    t_snap = 0.6 * TFAC
    for N, dt in runs:
        g = Torus(N)
        u1, u2, psi = taylor_green(g)
        speed2 = u1 ** 2 + u2 ** 2
        a0 = 1.0 - 0.5 * gaussian_blob(g, np.pi / 2, 0.12, 0.22)
        cell = (g.X < np.pi) & (g.Y < np.pi)
        m_slow = cell & (speed2 < 0.15)
        right = g.X > np.pi

        def w(f):
            return advect_skew(g, u1, u2, f)

        def rhs(state):
            (a,) = state
            return (2.0 * (1.0 - a) * speed2 + 2.0 * lam * w(w(a)),)

        def mon(step, t, state):
            (a,) = state
            e = (1.0 - a) ** 2
            return (t, np.sqrt(np.mean(e[m_slow])), np.sqrt(np.mean(e[right])))

        n1 = max(int(round(t_snap / dt)), 1)
        (aS,), log1 = rk4(rhs, (a0.copy(),), dt, n1, mon, every=100)
        n2 = max(int(round(T / dt)) - n1, 1)
        _, log2 = rk4(rhs, (aS,), dt, n2, mon, every=100)
        log = np.array(log1 + log2[1:])
        right_snap = np.sqrt(np.mean((1.0 - aS[right]) ** 2))
        print(f"[exp6-res] N={N}: slow-set peak {log[1:, 1].max():.3e}; "
              f"right-cell RMS at t={t_snap:g} {right_snap:.2e}, "
              f"max over run {log[:, 2].max():.2e}")


# ======================================================================

EXPERIMENTS = {
    # "exp1": exp1_homogenisation,
    # "exp2a": exp2a_tg_streamlines,
    # "exp2b": exp2b_tg_locality,
    # "exp2c": exp2c_poiseuille,
    # "exp3": exp3_balance,
    "exp6": exp6_transport,
    # "exp4": exp4_uniform_rescaling,
    "exp5": exp5_crossover,
    # "exp6res": exp6_resolution_check,
}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(EXPERIMENTS)
    for name in names:
        EXPERIMENTS[name]()
    print("Figures written to", FIGDIR)