"""Core numerics for the function-space verification experiments.

Spectral differentiation on the 2-pi torus, analytic flow fields, and a
generic RK4 integrator for functional gradient flows

    d(phi)/dt = -dL/d(phi),

integrated directly in function space.  No networks are involved.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------
# Spectral machinery on [0, 2pi)^2, x = axis 0, y = axis 1
# ----------------------------------------------------------------------

class Torus:
    def __init__(self, N):
        self.N = N
        self.h = TWO_PI / N
        x = np.arange(N) * self.h
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        k = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers
        k[N // 2] = 0.0                   # zero Nyquist: antisymmetric derivative
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2

    def dx(self, f):
        return np.real(np.fft.ifft2(1j * self.KX * np.fft.fft2(f)))

    def dy(self, f):
        return np.real(np.fft.ifft2(1j * self.KY * np.fft.fft2(f)))

    def div(self, fx, fy):
        return self.dx(fx) + self.dy(fy)

    def curl(self, fx, fy):
        return self.dx(fy) - self.dy(fx)

    def heat_exact(self, f0, kappa, t):
        """Exact solution of df/dt = kappa * Laplacian(f) at time t, on the
        same resolved modes as the spectral derivatives."""
        return np.real(np.fft.ifft2(np.fft.fft2(f0) * np.exp(-kappa * self.K2 * t)))


def advect_skew(grid, u1, u2, f):
    """Skew-symmetric form of u.grad f:  0.5 * [u.grad f + div(u f)].

    Equal to u.grad f for divergence-free u.  With the spectral derivative
    antisymmetric (Nyquist zeroed), the discrete operator is exactly
    antisymmetric, so 2 w(w(f)) is exactly minus the gradient of the
    discretised loss mean(w(f)^2)."""
    return 0.5 * (u1 * grid.dx(f) + u2 * grid.dy(f)
                  + grid.dx(u1 * f) + grid.dy(u2 * f))


def rk4(rhs, state, dt, nsteps, monitor=None, every=50):
    """Classical RK4 for state = tuple of arrays; rhs(state) -> tuple.

    monitor(step, t, state) is called at the start, every `every` steps and
    at the end; its return values are collected and returned with the final
    state."""
    log = []
    t = 0.0
    if monitor is not None:
        log.append(monitor(0, t, state))
    for n in range(1, nsteps + 1):
        k1 = rhs(state)
        s2 = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
        k2 = rhs(s2)
        s3 = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
        k3 = rhs(s3)
        s4 = tuple(s + dt * k for s, k in zip(state, k3))
        k4 = rhs(s4)
        state = tuple(
            s + (dt / 6.0) * (a + 2 * b + 2 * c + d)
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
        t += dt
        if monitor is not None and (n % every == 0 or n == nsteps):
            log.append(monitor(n, t, state))
    return state, log


# ----------------------------------------------------------------------
# Flows
# ----------------------------------------------------------------------

def taylor_green(grid):
    """Steady Taylor-Green cells: psi = sin x sin y, u = (dpsi/dy, -dpsi/dx).

    Divergence-free, four cells per box, closed streamlines, stagnation
    points at cell centres and corners.  A steady Euler solution, and a
    steady Navier-Stokes solution only with a body force, so it is used for
    continuity-loss experiments only."""
    psi = np.sin(grid.X) * np.sin(grid.Y)
    u1 = np.sin(grid.X) * np.cos(grid.Y)
    u2 = -np.cos(grid.X) * np.sin(grid.Y)
    return u1, u2, psi


def kovasznay(Re, X, Y):
    """Kovasznay flow: exact steady Navier-Stokes solution.

    u1 = 1 - e^{lam x} cos(2 pi y),  u2 = (lam / 2 pi) e^{lam x} sin(2 pi y),
    p  = (1 - e^{2 lam x}) / 2,      lam = Re/2 - sqrt(Re^2/4 + 4 pi^2).

    Returns a dict of the fields and their analytic first and second
    derivatives."""
    lam = Re / 2.0 - np.sqrt(Re ** 2 / 4.0 + 4.0 * np.pi ** 2)
    E = np.exp(lam * X)
    c = np.cos(TWO_PI * Y)
    s = np.sin(TWO_PI * Y)
    return {
        "lam": lam,
        "u1": 1.0 - E * c,
        "u2": (lam / TWO_PI) * E * s,
        "u1_x": -lam * E * c,
        "u1_y": TWO_PI * E * s,
        "u2_x": (lam ** 2 / TWO_PI) * E * s,
        "u2_y": lam * E * c,
        # u1_xx = -lam^2 E c,  u1_yy = +4 pi^2 E c
        "lap_u1": (TWO_PI ** 2 - lam ** 2) * E * c,
        # u2_xx = (lam^3 / 2pi) E s,  u2_yy = -(2 pi lam) E s
        "lap_u2": (lam ** 3 / TWO_PI - lam * TWO_PI) * E * s,
        "p_x": -lam * np.exp(2.0 * lam * X),
        "p_y": np.zeros_like(X),
    }


# ----------------------------------------------------------------------
# Initial-field building blocks
# ----------------------------------------------------------------------

def gaussian_blob(grid, x0, y0, sigma, periodic=True):
    """Gaussian bump on the torus, summed over nearest periodic images."""
    g = np.zeros_like(grid.X)
    shifts = (-TWO_PI, 0.0, TWO_PI) if periodic else (0.0,)
    for sx in shifts:
        for sy in shifts:
            g += np.exp(-(((grid.X - x0 + sx) ** 2) + ((grid.Y - y0 + sy) ** 2))
                        / (2.0 * sigma ** 2))
    return g


def compact_bump(grid, x0, y0, R):
    """Smooth bump exp(1 - 1/(1 - r^2/R^2)) for r < R, identically zero
    outside.  C-infinity, peak value 1 at (x0, y0).  The support must lie
    inside the domain (no periodic images are summed)."""
    r2 = ((grid.X - x0) ** 2 + (grid.Y - y0) ** 2) / R ** 2
    b = np.zeros_like(grid.X)
    inside = r2 < 1.0
    b[inside] = np.exp(1.0 - 1.0 / (1.0 - r2[inside]))
    return b
