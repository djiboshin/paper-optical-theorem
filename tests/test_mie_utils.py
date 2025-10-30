from mie_utils import (
    p_n_PW,
    p_n_SW,
    plane_p_b_generator,
    plane_p_b_PW,
    plane_p_b_SW,
    a_n_sphere,
    line_p_s_generator,
)
import numpy as np
import pandas as pd
from pathlib import Path
import pytest


@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("k", [0.1, 0.5, 10])
@pytest.mark.parametrize("theta", [0, np.pi / 2, np.pi / 3])
def test_plane_p_b_PW(a, k, theta):
    r = np.linspace(0, 10, 100)
    z = r * np.cos(theta)
    x = r * np.sin(theta)

    plane_p_b_exact = np.exp(1j * k * z)
    plane_p_b_exact[r >= a] = np.nan

    assert np.allclose(plane_p_b_exact, plane_p_b_PW(z, x, a=a, k=k), equal_nan=True)


@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("k", [0.1, 0.5, 10])
@pytest.mark.parametrize("theta", [0, np.pi / 2, np.pi / 3])
@pytest.mark.parametrize("R", [4, 5, 6, 10])
def test_plane_p_b_SW(a, k, theta, R):
    r = np.linspace(0, 10, 100)
    z = r * np.cos(theta)
    x = r * np.sin(theta)

    _r = np.sqrt(x**2 + (z + R) ** 2)
    plane_p_b_exact = R * np.exp(1j * k * (_r - R)) / _r
    plane_p_b_exact[r >= a] = np.nan

    assert np.allclose(
        plane_p_b_exact, plane_p_b_SW(z, x, a=a, k=k, R=R), equal_nan=True
    )


@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("k", [0.1, 0.5, 10])
@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi])
def test_p_n_PW(a, k, theta):
    N = 50

    r = np.linspace(0, 10, 100)
    z = r * np.cos(theta)
    x = r * np.sin(theta)

    p_n = p_n_PW(N)
    p_b = sum(plane_p_b_generator(z, x, a, k, p_n))

    assert np.allclose(plane_p_b_PW(z, x, a, k), p_b, equal_nan=True)


@pytest.mark.parametrize("a", [0.5, 1.5, 2.5])
@pytest.mark.parametrize("k", [0.1, 0.5, 10])
@pytest.mark.parametrize("theta", [0, np.pi / 4, np.pi / 2])
@pytest.mark.parametrize("R", [4, 5, 6, 10])
def test_p_n_SW(a, k, theta, R):
    N = 80

    r = np.linspace(0, 10, 100)
    z = r * np.cos(theta)
    x = r * np.sin(theta)

    p_n = p_n_SW(N, k, R)
    p_b = sum(plane_p_b_generator(z, x, a, k, p_n))

    assert np.allclose(plane_p_b_SW(z, x, a, k, R), p_b, equal_nan=True)


def test_a_n_sphere():
    """Test the Mie coefficients against a precalculated extinction cross section."""
    N = 20

    df_comsol = pd.read_csv(Path(__file__).parent / "test_comsol_sphere_scattering.csv")
    k_list = 2 * np.pi * df_comsol["freq"].to_numpy() / 1.0

    a_n = a_n_sphere(N, k_list, 1, 1, 1, 0.3, 1)
    sigma_ext = -4 * np.pi / k_list**2 * np.real((2 * np.arange(N + 1) + 1.0) @ a_n)

    relative_error = np.mean(np.abs(sigma_ext / df_comsol["sigma_ext"].to_numpy() - 1))
    assert relative_error < 1e-2


def test_line_p_s_generator():
    """Test the calculation of the scattering field using the optical theorem for a plane wave."""
    N = 20

    k_list = np.linspace(0.01, 10, 100)

    a_n = a_n_sphere(N, k_list, 1, 1, 1, 0.3, 1)
    sigma_ext_exact = (
        -4 * np.pi / k_list**2 * np.real((2 * np.arange(N + 1) + 1.0) @ a_n)
    )

    z = 8e9  # far enough distance
    p_b = np.exp(1j * k_list * z)
    p_n = p_n_PW(N)
    p_s = np.sum(list(line_p_s_generator(k_list * z, p_n, a_n)), axis=0)
    sigma_ext_pw = 4 * np.pi * np.imag(p_s / p_b) / k_list * z

    relative_error = np.mean(np.abs(sigma_ext_pw / sigma_ext_exact - 1))
    assert relative_error < 1e-9
