from typing import Tuple

import numpy as np
from math import floor, ceil
from scipy.linalg import toeplitz


def chebyshev_matrices(x_span: Tuple[float, float], N: int):
    """
    Compute the Chebyshev spectral collocation points (corresponding to the extremal points)
    and the first order spectral differentiation matrix.
    Based on the implementation by Wiedeman & Reddy (http://dx.doi.org/10.1145/365723.365727)
    and pyddx (https://github.com/ronojoy/pyddx)
    :param x_span:
    :param N:
    :return:
    """
    n1 = floor(N / 2)
    n2 = ceil(N / 2)

    k = np.arange(N)
    theta = k * np.pi / (N - 1)

    # Compute the Chebyshev collocation points
    # x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way
    x = np.sin(
        np.pi * ((N - 1.0) - 2.0 * np.linspace(N - 1.0, 0, N)) / (2.0 * (N - 1.0))
    )  # W&R way
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(theta / 2, (N, 1))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)  # trigonometric identity
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))  # flipping trick
    DX[range(N), range(N)] = 1.0  # diagonals of D
    DX = DX.T

    C = toeplitz((-1.0) ** k)  # matrix with entries c(k)/c(j)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1.0 / DX  # Z contains entries 1/(x(k)-x(j))
    Z[range(N), range(N)] = 0.0  # with zeros on the diagonal.

    # set up differentiation matrix on [-1, 1]
    D = np.eye(N)

    D = Z * (C * np.tile(np.diag(D), (N, 1)).T - D)  # off-diagonals
    D[range(N), range(N)] = -np.sum(D, axis=1)  # negative sum trick

    # rescale for the arbitrary interval
    a, b = x_span
    x = (a + b) / 2.0 + (b - a) / 2.0 * x
    D = 2.0 * D / (b - a)

    return x, D


def adaptive_levin_(
    x_span: Tuple[float, float], f, theta, tol: float = 1e-7, chebyshev_order: int = 12
):
    """
    f should be an m-vector of non-rapidly oscillating functions (Levin 96 eq. 2.1)
    theta should be an (m x m)-matrix representing the phase matrix of the system (Levin 96's A or A^t matrix)
    :param x_span:
    :param f:
    :param theta:
    :param tol:
    :param chebyshev_order:
    :return:
    """
    grid, Dmat = chebyshev_matrices(x_span, chebyshev_order)

    # sample each component of f on the Chebyshev grid
    # columns of the resulting matrix carry the Chebyshev grid index
    # rows of the resulting matrix carry the f-vector index
    f_Cheb = np.column_stack([func(x) for x in grid] for func in f)

    # sample the phase function theta on the Chebyshev grid
    theta_Cheb = np.array(theta(x) for x in grid)

    # multiply theta by the spectral differentiation matrix Dmat in order to produce an estiamte of g'(x)
    # evaluated at the collocation points
    theta_prime_Cheb = np.matmul(Dmat, theta_Cheb)
