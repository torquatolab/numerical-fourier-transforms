#!/bin/bash
"""
Numerical Fourier transform in 1-, 2-, and 3-dimensions
Assumes that we are transforming a function $F(r)$ that is statistically isotropic in $r$

References
    F. Lado. Numerical Fourier Transforms in One, Two, and Three Dimensions for Liquid State Calculations. J. Comp. Phys., 8 (1971).

Sam Dawley
02/2026
"""

import numpy as np
from scipy.special import j0, j1, jn_zeros


# ==================================================
# FORWARD NUMERICAL FOURIER TRANSFORM
# ==================================================


def nft_1d(
    x: np.ndarray,
    y: np.ndarray,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Forward numerical Fourier transform in 1 dimension via the
    midpoint-rule quadrature of F. Lado, J. Comp. Phys. 8 (1971).

    Computes

    .. math::

        \tilde{G}(k) = 2 \int_0^R g(r)\,\cos(kr)\,\mathrm{d}r

    on the Lado grid, where :math:`R` is ``cutoff_radius`` and
    :math:`g(r) \equiv 0` for :math:`r \ge R`.  This is the full
    Fourier transform of an even function :math:`g(-r) = g(r)`.

    The input data ``(x, y)`` are linearly interpolated onto the Lado
    quadrature nodes :math:`r_j = (j - \tfrac{1}{2})\,\Delta r`,
    :math:`j = 1, \ldots, N`, before the transform is evaluated.

    Parameters
    ----------
        x : np.ndarray, shape ``(N,)`` or ``(N, 1)``
            Abscissa values at which ``y`` is sampled.
        y : np.ndarray, shape ``(N,)`` or ``(N, 1)``
            Ordinate values (the function to be transformed).
        cutoff_radius : float
            Maximum radius beyond which ``y`` is taken to vanish.
        n_intervals : int
            Number of quadrature intervals in ``[0, cutoff_radius]``.

    Returns
    -------
        k : np.ndarray, shape ``(n_intervals,)``
            Conjugate-space grid :math:`k_m = (m - \tfrac{1}{2})\,\Delta k`,
            :math:`m = 1, \ldots, N`.
        _ : np.ndarray, shape ``(n_intervals,)``
            Numerical Fourier transform evaluated at the :math:`k_m`.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    delta_r = cutoff_radius / (n_intervals - 0.5)
    delta_k = np.pi / cutoff_radius

    # ----- quadrature nodes -----
    m = np.arange(1, n_intervals + 1)
    r = (m - 0.5) * delta_r
    k = (m - 0.5) * delta_k

    # C_{mj} = cos(k_m r_j) with k_m r_j = (m-1/2)(j-1/2) pi/(N-1/2)
    g = np.interp(r, x, y, left=0.0, right=0.0)
    cos_matrix = np.cos(np.outer(k, r))

    return k, 2.0 * delta_r * (cos_matrix @ g)


def nft_2d(
    x: np.ndarray,
    y: np.ndarray,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Forward numerical Fourier transform in 2 dimensions via the
    Bessel-function quadrature of F. Lado, J. Comp. Phys. 8 (1971).

    For a radially symmetric function :math:`g(r)` in 2-D the full
    Fourier transform reduces to the zeroth-order Hankel transform

    .. math::

        \tilde{G}(k) = 2\pi \int_0^R g(r)\,J_0(kr)\,r\,\mathrm{d}r

    Lado discretises this using the zeros :math:`\lambda_n` of
    :math:`J_0`.  The quadrature nodes are
    :math:`r_i = \lambda_i\,R / \lambda_N` in real space and
    :math:`k_j = \lambda_j / R` in conjugate space
    (:math:`i,j = 1,\ldots,N{-}1`), where :math:`\lambda_N` is the
    :math:`N`-th zero of :math:`J_0`.

    The input data ``(x, y)`` are linearly interpolated onto the
    (non-uniform) quadrature nodes before the transform is evaluated.

    Parameters
    ----------
        x : np.ndarray, shape ``(M,)`` or ``(M, 1)``
            Abscissa values at which ``y`` is sampled.
        y : np.ndarray, shape ``(M,)`` or ``(M, 1)``
            Ordinate values (the function to be transformed).
        cutoff_radius : float
            Maximum radius beyond which ``y`` is taken to vanish.
        n_intervals : int
            Number of Bessel-function zeros used to build the
            quadrature.  The transform returns ``n_intervals - 1``
            points in each of real and conjugate space.

    Returns
    -------
        k : np.ndarray, shape ``(n_intervals - 1,)``
            Conjugate-space grid :math:`k_j = \lambda_j / R`.
        _ : np.ndarray, shape ``(n_intervals - 1,)``
            Numerical Fourier transform evaluated at the :math:`k_j`.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    zeros = jn_zeros(0, n_intervals)        # λ₁, …, λ_N
    lambda_N = zeros[-1]
    K = lambda_N / cutoff_radius

    r = zeros[:-1] * cutoff_radius / lambda_N   # r_i = λ_i R / λ_N,  i = 1…N-1
    k = zeros[:-1] / cutoff_radius              # k_j = λ_j / R,       j = 1…N-1

    g = np.interp(r, x, y, left=0.0, right=0.0)

    weights = 1.0 / j1(zeros[:-1]) ** 2          # 1 / [J₁(λ_i)]²

    j0_matrix = j0(np.outer(k, r))               # J₀(k_j r_i) = J₀(λ_j λ_i / λ_N)

    return k, (4.0 * np.pi / K ** 2) * (j0_matrix @ (g * weights))


def nft_3d(
    x: np.ndarray,
    y: np.ndarray,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Forward numerical Fourier transform in 3 dimensions via the
    sine-transform quadrature of F. Lado, J. Comp. Phys. 8 (1971).

    For a radially symmetric function :math:`g(r)` in 3-D the full
    Fourier transform reduces to

    .. math::

        \tilde{G}(k) = \frac{4\pi}{k}
            \int_0^R r\,g(r)\,\sin(kr)\,\mathrm{d}r

    Lado discretises this on an integer grid
    :math:`r_i = i\,\Delta r`, :math:`k_j = j\,\Delta k` with
    :math:`\Delta r = R/N` and :math:`\Delta k = \pi/R`
    (:math:`i,j = 1,\ldots,N{-}1`).  Integer (rather than
    half-integer) nodes are natural here because
    :math:`\sin(k_j\,r_N) = \sin(j\pi) = 0` and the
    :math:`r = 0` boundary contributes nothing to the integrand.

    The input data ``(x, y)`` are linearly interpolated onto the
    quadrature nodes before the transform is evaluated.

    Parameters
    ----------
        x : np.ndarray, shape ``(M,)`` or ``(M, 1)``
            Abscissa values at which ``y`` is sampled.
        y : np.ndarray, shape ``(M,)`` or ``(M, 1)``
            Ordinate values (the function to be transformed).
        cutoff_radius : float
            Maximum radius beyond which ``y`` is taken to vanish.
        n_intervals : int
            Number of intervals in ``[0, cutoff_radius]``.  The
            transform returns ``n_intervals - 1`` points in each of
            real and conjugate space.

    Returns
    -------
        k : np.ndarray, shape ``(n_intervals - 1,)``
            Conjugate-space grid :math:`k_j = j\,\Delta k`.
        _ : np.ndarray, shape ``(n_intervals - 1,)``
            Numerical Fourier transform evaluated at the :math:`k_j`.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    delta_r = cutoff_radius / n_intervals
    delta_k = np.pi / cutoff_radius

    # ----- quadrature nodes -----
    idx = np.arange(1, n_intervals)
    r = idx * delta_r
    k = idx * delta_k

    # sin(k_j r_i) = sin(i j π / n_intervals)
    g = np.interp(r, x, y, left=0.0, right=0.0)
    sin_matrix = np.sin(np.outer(k, r))

    return k, (4.0 * np.pi / k) * delta_r * (sin_matrix @ (r * g))


# ==================================================
# INVERSE NUMERICAL FOURIER TRANSFORM
# ==================================================


def inft_1d(
    x: np.ndarray,
    y: np.ndarray,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Inverse numerical Fourier transform in 1 dimension via the
    midpoint-rule quadrature of F. Lado, J. Comp. Phys. 8 (1971).

    Recovers a real-space function from its Fourier transform by
    evaluating

    .. math::

        g(r) = \frac{1}{\pi}
            \int_0^\infty \tilde{G}(k)\,\cos(kr)\,\mathrm{d}k

    on the same Lado grid used by :func:`nft_1d`.

    The input data ``(x, y)`` (sampled in conjugate space) are
    linearly interpolated onto the Lado quadrature nodes
    :math:`k_m = (m - \tfrac{1}{2})\,\Delta k` before the transform
    is evaluated.

    Parameters
    ----------
    x : np.ndarray, shape ``(M,)`` or ``(M, 1)``
        Conjugate-space abscissa values at which ``y`` is sampled.
    y : np.ndarray, shape ``(M,)`` or ``(M, 1)``
        Fourier-space ordinate values to be inverse-transformed.
    cutoff_radius : float
        Real-space cutoff :math:`R` that defines the grid spacing
        :math:`\Delta r = R/(N - \tfrac{1}{2})` and
        :math:`\Delta k = \pi/R`.  Use the same value passed to
        :func:`nft_1d` to ensure a consistent round-trip.
    n_intervals : int
        Number of quadrature intervals in ``[0, cutoff_radius]``.

    Returns
    -------
    r : np.ndarray, shape ``(n_intervals,)``
        Real-space grid :math:`r_j = (j - \tfrac{1}{2})\,\Delta r`,
        :math:`j = 1, \ldots, N`.
    g : np.ndarray, shape ``(n_intervals,)``
        Recovered real-space function evaluated at the :math:`r_j`.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    delta_r = cutoff_radius / (n_intervals - 0.5)
    delta_k = np.pi / cutoff_radius

    m = np.arange(1, n_intervals + 1)
    k = (m - 0.5) * delta_k
    r = (m - 0.5) * delta_r

    G = np.interp(k, x, y, left=0.0, right=0.0)

    cos_matrix = np.cos(np.outer(r, k))

    g = (delta_k / np.pi) * (cos_matrix @ G)

    return r, g


def inft_2d(
    x: np.ndarray,
    y: np.ndarray,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Inverse numerical Fourier transform in 2 dimensions via the
    Bessel-function quadrature of F. Lado, J. Comp. Phys. 8 (1971).

    Recovers a radially symmetric real-space function from its 2-D
    Fourier transform by evaluating

    .. math::

        g(r) = \frac{1}{2\pi}
            \int_0^\infty \tilde{G}(k)\,J_0(kr)\,k\,\mathrm{d}k

    on the same Lado grid used by :func:`nft_2d`.

    The input data ``(x, y)`` (sampled in conjugate space) are
    linearly interpolated onto the (non-uniform) Lado quadrature
    nodes :math:`k_j = \lambda_j / R` before the transform is
    evaluated.

    Parameters
    ----------
    x : np.ndarray, shape ``(M,)`` or ``(M, 1)``
        Conjugate-space abscissa values at which ``y`` is sampled.
    y : np.ndarray, shape ``(M,)`` or ``(M, 1)``
        Fourier-space ordinate values to be inverse-transformed.
    cutoff_radius : float
        Real-space cutoff :math:`R` that defines the grid.  Use the
        same value passed to :func:`nft_2d` to ensure a consistent
        round-trip.
    n_intervals : int
        Number of Bessel-function zeros used to build the
        quadrature.  The transform returns ``n_intervals - 1``
        points.

    Returns
    -------
    r : np.ndarray, shape ``(n_intervals - 1,)``
        Real-space grid :math:`r_i = \lambda_i\,R / \lambda_N`.
    g : np.ndarray, shape ``(n_intervals - 1,)``
        Recovered real-space function evaluated at the :math:`r_i`.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    R = cutoff_radius
    N = n_intervals

    zeros = jn_zeros(0, N)            # λ₁, …, λ_N
    lambda_N = zeros[-1]

    k = zeros[:-1] / R               # k_j = λ_j / R,       j = 1…N-1
    r = zeros[:-1] * R / lambda_N    # r_i = λ_i R / λ_N,   i = 1…N-1

    G = np.interp(k, x, y, left=0.0, right=0.0)

    weights = 1.0 / j1(zeros[:-1]) ** 2          # 1 / [J₁(λ_j)]²

    j0_matrix = j0(np.outer(r, k))               # J₀(r_i k_j) = J₀(λ_i λ_j / λ_N)

    g = (1.0 / (np.pi * R ** 2)) * (j0_matrix @ (G * weights))

    return r, g


def inft_3d(
    x: np.ndarray,
    y: np.ndarray,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Inverse numerical Fourier transform in 3 dimensions via the
    sine-transform quadrature of F. Lado, J. Comp. Phys. 8 (1971).

    Recovers a radially symmetric real-space function from its 3-D
    Fourier transform by evaluating

    .. math::

        g(r) = \frac{1}{2\pi^2 r}
            \int_0^\infty k\,\tilde{G}(k)\,\sin(kr)\,\mathrm{d}k

    on the same Lado grid used by :func:`nft_3d`.

    The input data ``(x, y)`` (sampled in conjugate space) are
    linearly interpolated onto the Lado quadrature nodes
    :math:`k_j = j\,\Delta k` before the transform is evaluated.

    Parameters
    ----------
    x : np.ndarray, shape ``(M,)`` or ``(M, 1)``
        Conjugate-space abscissa values at which ``y`` is sampled.
    y : np.ndarray, shape ``(M,)`` or ``(M, 1)``
        Fourier-space ordinate values to be inverse-transformed.
    cutoff_radius : float
        Real-space cutoff :math:`R` that defines the grid spacing
        :math:`\Delta r = R/N` and :math:`\Delta k = \pi/R`.  Use
        the same value passed to :func:`nft_3d` to ensure a
        consistent round-trip.
    n_intervals : int
        Number of intervals in ``[0, cutoff_radius]``.  The
        transform returns ``n_intervals - 1`` points.

    Returns
    -------
    r : np.ndarray, shape ``(n_intervals - 1,)``
        Real-space grid :math:`r_i = i\,\Delta r`.
    g : np.ndarray, shape ``(n_intervals - 1,)``
        Recovered real-space function evaluated at the :math:`r_i`.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    delta_r = cutoff_radius / n_intervals
    delta_k = np.pi / cutoff_radius

    idx = np.arange(1, n_intervals)
    r = idx * delta_r
    k = idx * delta_k

    G = np.interp(k, x, y, left=0.0, right=0.0)

    sin_matrix = np.sin(np.outer(r, k))

    g = (delta_k / (2.0 * np.pi ** 2 * r)) * (sin_matrix @ (k * G))

    return r, g


# ==================================================
# SUBROUTINES
# ==================================================


def nft(
    x: np.ndarray,
    y: np.ndarray,
    d: int,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward numerical Fourier transform in 1, 2, or 3 dimensions.

    Dispatches to :func:`nft_1d`, :func:`nft_2d`, or :func:`nft_3d`
    according to the spatial dimension ``d``.

    Parameters
    ----------
        x : np.ndarray
            Abscissa values at which ``y`` is sampled.
        y : np.ndarray
            Ordinate values (the function to be transformed).
        d : int
            Spatial dimension (1, 2, or 3).
        cutoff_radius : float
            Maximum radius beyond which ``y`` is taken to vanish.
        n_intervals : int
            Number of quadrature intervals (or Bessel zeros for d=2).

    Returns
    -------
        k : np.ndarray
            Conjugate-space grid.
        G : np.ndarray
            Numerical Fourier transform evaluated on that grid.
    """
    match d:
        case 1:
            return nft_1d(x, y, cutoff_radius, n_intervals)
        case 2:
            return nft_2d(x, y, cutoff_radius, n_intervals)
        case 3:
            return nft_3d(x, y, cutoff_radius, n_intervals)
        case _:
            raise ValueError(f"Dimension d={d} not supported (must be 1, 2, or 3)")


def inft(
    x: np.ndarray,
    y: np.ndarray,
    d: int,
    cutoff_radius: float,
    n_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse numerical Fourier transform in 1, 2, or 3 dimensions.

    Dispatches to :func:`inft_1d`, :func:`inft_2d`, or :func:`inft_3d`
    according to the spatial dimension ``d``.

    Parameters
    ----------
        x : np.ndarray
            Conjugate-space abscissa values at which ``y`` is sampled.
        y : np.ndarray
            Fourier-space ordinate values to be inverse-transformed.
        d : int
            Spatial dimension (1, 2, or 3).
        cutoff_radius : float
            Real-space cutoff that defines the grid.  Use the same value
            passed to the corresponding forward transform.
        n_intervals : int
            Number of quadrature intervals (or Bessel zeros for d=2).

    Returns
    -------
        r : np.ndarray
            Real-space grid.
        g : np.ndarray
            Recovered real-space function evaluated on that grid.
    """
    match d:
        case 1:
            return inft_1d(x, y, cutoff_radius, n_intervals)
        case 2:
            return inft_2d(x, y, cutoff_radius, n_intervals)
        case 3:
            return inft_3d(x, y, cutoff_radius, n_intervals)
        case _:
            raise ValueError(f"Dimension d={d} not supported (must be 1, 2, or 3)")


# ==================================================
# TESTING
# ==================================================


def _damped_cosine_ft(alpha: float, omega: float, d: int, k: np.ndarray) -> np.ndarray:
    r"""
    Analytical Fourier transform of :math:`g(r) = e^{-\alpha r}\cos(\omega r)`
    in *d* dimensions, evaluated at wavenumbers *k*.

    1-D (cosine transform of the even extension):

    .. math::

        \tilde{G}(k) = \frac{\alpha}{\alpha^2+(k-\omega)^2}
                      + \frac{\alpha}{\alpha^2+(k+\omega)^2}

    2-D (zeroth-order Hankel transform, using
    :math:`s = \alpha - i\omega`):

    .. math::

        \tilde{G}(k) = 2\pi\,\mathrm{Re}\!\left[
            \frac{s}{(s^2+k^2)^{3/2}}\right]

    3-D (Fourier--sine transform):

    .. math::

        \tilde{G}(k) = \frac{4\pi\alpha}{k}\!\left[
            \frac{k+\omega}{(\alpha^2+(k+\omega)^2)^2}
          + \frac{k-\omega}{(\alpha^2+(k-\omega)^2)^2}\right]
    """
    a = alpha
    w = omega
    if d == 1:
        return a / (a ** 2 + (k - w) ** 2) + a / (a ** 2 + (k + w) ** 2)
    if d == 2:
        s = a - 1j * w
        return 2.0 * np.pi * np.real(s / (s ** 2 + k ** 2) ** 1.5)
    if d == 3:
        kp = k + w
        km = k - w
        return (4.0 * np.pi * a / k) * (
            kp / (a ** 2 + kp ** 2) ** 2 + km / (a ** 2 + km ** 2) ** 2
        )
    raise ValueError(f"d={d}")


def main():
    """
    Test suite for the numerical Fourier transforms.

    Three families of test functions are used (all with known
    analytical transforms in 1-D, 2-D, and 3-D):

    1. **Gaussian** g(r) = exp(-r²/2)
    2. **Damped cosine** g(r) = exp(-αr) cos(ω₀r)  (intermediate ω₀)
    3. **Two-frequency damped cosine**
       g(r) = exp(-αr) [cos(ω₁r) + cos(ω₂r)]  (ω₁ ≪ ω₂)

    Each test compares
        (a) the forward transform against the analytical result, and
        (b) the round-trip inft(nft(g)) against the original g.
    """
    n_pass = 0
    n_fail = 0

    def _report(label: str, err: float, tol: float) -> None:
        nonlocal n_pass, n_fail
        status = "PASS" if err < tol else "FAIL"
        if err < tol:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  [{status}] {label:40s}  max|err| = {err:.2e}")

    def _run_tests(
        name: str, x: np.ndarray, y_func, analytical_func,
        R: float, N: int, tol_fwd: float, tol_rt: float,
    ) -> None:
        y = y_func(x)
        for d in (1, 2, 3):
            print(f"\n--- {d}-D {name} (R={R}, N={N}) ---")

            k, G_num = nft(x, y, d, R, N)
            G_exact = analytical_func(d, k)
            fwd_err = np.max(np.abs(G_num - G_exact))
            _report(f"nft_{d}d  forward vs analytical", fwd_err, tol_fwd)

            r_rt, g_rt = inft(k, G_num, d, R, N)
            rt_err = np.max(np.abs(g_rt - y_func(r_rt)))
            _report(f"inft_{d}d(nft_{d}d(.))  round-trip", rt_err, tol_rt)

    # ==========================================================
    #  Test 1: Gaussian   g(r) = exp(-r²/2)
    # ==========================================================
    R1, N1 = 6.0, 512
    x1 = np.linspace(0, R1, 4096)

    gauss_analytical = {
        1: lambda k: np.sqrt(2.0 * np.pi) * np.exp(-k ** 2 / 2.0),
        2: lambda k: 2.0 * np.pi * np.exp(-k ** 2 / 2.0),
        3: lambda k: (2.0 * np.pi) ** 1.5 * np.exp(-k ** 2 / 2.0),
    }

    _run_tests(
        "Gaussian", x1,
        y_func=lambda r: np.exp(-r ** 2 / 2.0),
        analytical_func=lambda d, k: gauss_analytical[d](k),
        R=R1, N=N1, tol_fwd=1e-4, tol_rt=1e-4,
    )

    # ==========================================================
    #  Test 2: Damped cosine   g(r) = exp(-αr) cos(ω₀r)
    #          intermediate frequency ω₀ = 5
    # ==========================================================
    alpha, omega_0 = 1.0, 5.0
    R2, N2 = 12.0, 1024
    x2 = np.linspace(0, R2, 8192)

    _run_tests(
        f"damped cosine (α={alpha}, ω₀={omega_0})", x2,
        y_func=lambda r: np.exp(-alpha * r) * np.cos(omega_0 * r),
        analytical_func=lambda d, k: _damped_cosine_ft(alpha, omega_0, d, k),
        R=R2, N=N2, tol_fwd=1e-3, tol_rt=1e-3,
    )

    # ==========================================================
    #  Test 3: Two-frequency damped cosine
    #          g(r) = exp(-αr) [cos(ω₁r) + cos(ω₂r)]
    #          ω₁ = 0.5  (slow),  ω₂ = 20  (fast)
    # ==========================================================
    omega_1, omega_2 = 0.5, 20.0
    R3, N3 = 12.0, 1024
    x3 = np.linspace(0, R3, 8192)

    _run_tests(
        f"two-freq cosine (ω₁={omega_1}, ω₂={omega_2})", x3,
        y_func=lambda r: np.exp(-alpha * r) * (
            np.cos(omega_1 * r) + np.cos(omega_2 * r)
        ),
        analytical_func=lambda d, k: (
            _damped_cosine_ft(alpha, omega_1, d, k)
            + _damped_cosine_ft(alpha, omega_2, d, k)
        ),
        R=R3, N=N3, tol_fwd=1e-3, tol_rt=1e-3,
    )

    # ==========================================================
    print(f"\n{'='*56}")
    print(f"  {n_pass} passed, {n_fail} failed")
    print(f"{'='*56}")

    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())
