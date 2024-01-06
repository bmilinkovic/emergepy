#%% IMPORT
import numpy as np
import matplotlib.pyplot as plt

#%% FUNCTION DEFINITIONS

def iss_rand(n, m, rhoa, dis=False):
    assert rhoa < 1

    if not dis:
        dis = False

    A = specnorm(np.random.randn(m, m), rhoa)
    C = np.random.randn(n, m)
    K = np.random.randn(m, n)

    M = np.dot(K, C)
    rmin = speclim(A, M, -1, 0)
    rmax = speclim(A, M, +1, 0)

    r = rmin + (rmax - rmin) * np.random.rand()
    sqrtr = np.sqrt(np.abs(r))
    C = sqrtr * C
    K = np.sign(r) * sqrtr * K

    if n > 3:
        rhob = specnorm(A - np.dot(K, C))
    else:
        rhob = None

    if dis:
        nr = 1000
        ramax = 1.1 * max(rmax, -rmin)
        rr = np.linspace(-ramax, ramax, nr)
        rrhob = np.zeros(nr)
        for i in range(nr):
            rrhob[i] = specnorm(A - rr[i] * M)

        rholim = [0.9 * min(rrhob), 1.1 * max(rrhob)]
        plt.plot(rr, rrhob)
        plt.xlim([-ramax, ramax])
        plt.ylim(rholim)
        plt.axhline(1, color='k')
        plt.axvline(0, color='r')
        plt.axvline(r, color='g')
        plt.xlabel('r')
        plt.ylabel('rho')
        plt.legend(['rho(B)'])
        plt.title(f'rho(A) = {rhoa}, rho(B) = {specnorm(A - r * M)}')
        plt.show()

    return A, C, K, rhob


def speclim(A, M, r1, r2):
    assert specnorm(A - r1 * M) > 1 and specnorm(A - r2 * M) < 1

    while True:
        r = (r1 + r2) / 2
        rho = specnorm(A - r * M)
        if rho > 1:
            r1 = r
        else:
            r2 = r

        if np.abs(r1 - r2) < np.finfo(float).eps:
            break

    return r


def specnorm(A, newrho=None):
    if np.isscalar(A):
        p = int(A)
        p1 = p - 1
        A1 = np.concatenate((np.array([list(range(1, p + 1))]), np.concatenate((np.eye(p1), np.zeros((p1, 1))), axis=1)), axis=0)
    else:
        if A.ndim == 2:
            A = np.expand_dims(A, axis=2)  # Add third dimension of size one
        n, n1, p = A.shape
        assert n1 == n, 'VAR/VMA coefficients matrix has bad shape'
        pn1 = (p - 1) * n
        A1 = np.concatenate((A.reshape(n, p * n), np.concatenate((np.eye(pn1), np.zeros((pn1, n))), axis=1)), axis=0)

    # calculate spectral norm
    rho = np.max(np.abs(np.linalg.eig(A1)[0]))

    if newrho is None:
        assert np.prod(A.shape) == n * n1 * p, 'Too many output parameters'
        out1 = rho  # spectral norm
    else:
        out1 = var_decay(A, newrho / rho)  # adjusted coefficients
        out2 = rho  # previous value of spectral norm
        return out1, out2

    return out1

def var_decay(A, dfac):
    if np.isscalar(A):
        p = int(A)
        f = dfac
        for k in range(1, p + 1):
            A[k - 1] = f * A[k - 1]
            f = dfac * f
    else:
        p = A.shape[2]
        f = dfac
        for k in range(1, p + 1):
            A[:, :, k - 1] = f * A[:, :, k - 1].copy()
            f = dfac * f

    return A


#%% Example usage:
n_dim = 3
m_order = 4
spectral_norm_A = 0.8
display_plot = True

A, C, K, rhob = iss_rand(n_dim, m_order, spectral_norm_A, display_plot)

# %%
