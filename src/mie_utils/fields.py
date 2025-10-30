import scipy.special as sp
import numpy as np
from typing import Generator


def p_n_PW(N: int) -> np.ndarray:
    """Compute $p_n$ coefficients for the plane wave expansion.

    Args:
        N (int): Maximum order of the expansion.

    Returns:
        np.ndarray: 1-D array of $p_n$ coefficients.
    """
    n_array = np.arange(0, N + 1, 1)
    p_n = (1j) ** n_array * (2 * n_array + 1)
    return p_n


def p_n_SW(N: int, k: float, R: float) -> np.ndarray:
    """Compute $p_n$ coefficients for the spherical wave expansion.

    Args:
        N (int): Maximum order of the expansion.
        k (float): Wavenumber.
        R (float): Distance between the point source and the scatterer center.

    Returns:
        np.ndarray: 1-D array of $p_n$ coefficients.
    """
    n_array = np.arange(0, N + 1, 1)

    kR = k * R
    # Spherical Hankel function of the first kind
    h_n = sp.spherical_jn(n_array, kR) + 1j * sp.spherical_yn(n_array, kR)
    coef = kR / np.exp(1j * kR) * (1j) ** (n_array + 1) * h_n

    p_n = (1j) ** n_array * (2 * n_array + 1) * coef
    return p_n


def plane_p_b_generator(
    z: np.ndarray,
    x: np.ndarray,
    a: float,
    k: float,
    p_n: np.ndarray,
) -> Generator[np.ndarray, None, None]:
    """Generate terms of the background field expansion $p_b$ in the z–x plane.

    Args:
        z (np.ndarray): Array of z-coordinates where the field is evaluated.
        x (np.ndarray): Array of x-coordinates where the field is evaluated.
        a (float): The radius of the circle where the expansion is used.
            The field outside this circle is set to NaN.
        k (float): Wavenumber.
        p_n (np.ndarray): 1-D array of expansion coefficients $p_n$.

    Yields:
        np.ndarray: Individual terms of the expansion up to order `len(p_n) + 1`.
    """
    assert z.shape == x.shape, "z and x must have the same shape"

    theta = np.atan2(x, z)
    rho = np.sqrt(z**2 + x**2)

    N = p_n.shape[0]

    P = sp.legendre_p_all(N, np.cos(theta))[0]

    for n in range(N):
        term = p_n[n] * sp.spherical_jn(n, k * rho) * P[n]
        term[rho > a] = np.nan
        yield term


def plane_p_b_PW(
    z: np.ndarray,
    x: np.ndarray,
    a: float,
    k: float,
) -> np.ndarray:
    """Compute the background pressure field $p_b$ for a plane wave in the z–x plane.

    Args:
        z (np.ndarray): Array of z-coordinates where the field is evaluated.
        x (np.ndarray): Array of x-coordinates where the field is evaluated.
        a (float): Radius of the region where the expansion is valid.
        The field outside this circle is set to NaN.
        k (float): Wavenumber.

    Returns:
        np.ndarray: Pressure field.
    """
    assert z.shape == x.shape, "z and x must have the same shape"

    rho = np.sqrt(z**2 + x**2)

    p_b = np.exp(1j * (k * z))

    p_b[rho > a] = np.nan
    return p_b


def plane_p_b_SW(
    z: np.ndarray,
    x: np.ndarray,
    a: float,
    k: float,
    R: float,
) -> np.ndarray:
    """Compute the background pressure field $p_b$ for a spherical wave in the z–x plane.

    Args:
        z (np.ndarray): Array of z-coordinates where the field is evaluated.
        x (np.ndarray): Array of x-coordinates where the field is evaluated.
        a (float): Radius of the region where the expansion is valid.
            The field outside this circle is set to NaN.
        k (float): Wavenumber.
        R (float): Distance between the point source and the scatterer center.

    Returns:
        np.ndarray: Pressure field.
    """
    assert z.shape == x.shape, "z and x must have the same shape"

    r = np.sqrt((z + R) ** 2 + x**2)
    rho = np.sqrt(z**2 + x**2)

    p_b = R * np.exp(1j * (k * (r - R))) / r

    p_b[rho > a] = np.nan
    return p_b


def a_n_sphere(
    N: int,
    k: np.ndarray,
    a: float,
    c_host: float,
    rho_host: float,
    c_sphere: float,
    rho_sphere: float,
):
    """Compute Mie coefficients $a_n$ for acoustic scattering on a sphere.

    Args:
        N (int): Maximum order of the expansion.
        k (np.ndarray): Wavenumber array.
        a (float): Radius of the sphere.
        c_host (float): Speed of sound in the host medium.
        rho_host (float): Density of the host medium.
        c_sphere (float): Speed of sound in the sphere.
        rho_sphere (float): Density of the sphere.

    Returns:
        np.ndarray: Mie coefficients $a_n$.
    """
    alpha = rho_sphere * c_sphere / (rho_host * c_host)

    result = np.empty((N + 1, *k.shape), dtype=np.complex128)
    for n in range(N + 1):

        jn = sp.spherical_jn(n, k * a)
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        hn = jn + 1j * sp.spherical_yn(n, k * a)
        hnp = jnp + 1j * sp.spherical_yn(n, k * a, derivative=True)

        k_prime = k * c_host / c_sphere
        jn_prime = sp.spherical_jn(n, k_prime * a)
        jnp_prime = sp.spherical_jn(n, k_prime * a, derivative=True)

        up = alpha * jnp * jn_prime - jn * jnp_prime
        down = hn * jnp_prime - alpha * hnp * jn_prime

        result[n] = up / down

    return result


def line_p_s_generator(
    kz: np.ndarray,
    p_n: np.ndarray,
    a_n: np.ndarray,
) -> Generator[np.ndarray, None, None]:
    """Generate scattered pressure field components along the z-axis.

    Args:
        kz (np.ndarray): Array of products $kz$.
        p_n (np.ndarray): Array of $p_n$ expansion coefficients (first dimension corresponds to order N).
        a_n (np.ndarray): Array of Mie coefficients (first dimension corresponds to order N).

    Yields:
        np.ndarray: Individual terms of the scattered pressure field along the z-axis.
    """

    assert (
        p_n.shape[0] == a_n.shape[0]
    ), "p_n and a_n must have the same length of first dimension"

    N = p_n.shape[0]

    for n in range(N):
        # Spherical Hankel function of the first kind
        h_n = sp.spherical_jn(n, kz) + 1j * sp.spherical_yn(n, kz)

        yield p_n[n] * a_n[n] * h_n
