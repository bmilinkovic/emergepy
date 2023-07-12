import numpy as np
from scipy.linalg import solve_discrete_are, cho_factor, cholesky, solve_triangular, cho_solve

def cak2ddx(L, CAK):
    # Calculate proxy dynamical dependence
    #
    # L        orthonormal subspace basis
    # A,C,K    ISS parameters
    #
    # For an innovations-form state-space model with parameters (A,C,K)
    #
    #     CAK_k, k = 1,...,r is the sequence of n x n matrices CA^{k-1}K
    #
    # where r is the ISS model order; see iss2cak.m.
    #
    # For a VAR model with coefficients sequence A, we may supply simply CAK = A.
    #
    # NOTE: assumes identity residuals covariance matrix
    
    r = CAK.shape[2]
    D = 0
    for k in range(r):
        LCAKk = np.matmul(L.T, CAK[:,:,k])
        LCAKLTk = np.matmul(LCAKk, L)
        D1k = LCAKk ** 2
        D2k = LCAKLTk ** 2
        D += np.sum(D1k) - np.sum(D2k) # this could be wrong.. will have to check.
    
    return D


def cak2ddxgrad(L, CAK):
    # Calculate gradient of proxy dynamical dependence
    #
    # L        orthonormal subspace basis
    # A,C,K    ISS parameters
    #
    # For an innovations-form state-space model with parameters (A,C,K)
    #
    #     CAK_k, k = 1,...,r is the sequence of n x n matrices CA^{k-1}K
    #
    # where r is the ISS model order; see iss2cak.m.
    #
    # For a VAR model with coefficients sequence A, we may supply simply CAK = A.
    #
    # NOTE: assumes identity residuals covariance matrix
    #
    # NOTE: the gradients are
    #
    # G <--- G-L*L'*G;   Grassmannian: Edelman-Arias-Smith, eq. (2.70)
    # G <--- G-L*G'*L;   Stiefel:      Edelman-Arias-Smith, eq. (2.53)
    
    r = CAK.shape[2]
    n = L.shape[0]
    P = L @ L.T
    g = np.zeros((n, n))
    
    for k in range(r):
        Q = CAK[:,:,k]
        QT = Q.T
        g = g + Q@QT - QT@P@Q - Q@P@QT
    
    G = 2 * g @ L
    G = G - P @ G
    
    if nargout > 1:
        mG = np.sqrt(np.sum(G**2))
        return G, mG
    
    return G, mG

def iss2cak(A, C, K):
    # Return the sequence CA^{k-1}K, k = 1,...,r of n x n matrices
    
    r = A.shape[0]
    n = C.shape[0]
    Ak = np.eye(r)
    CAK = np.zeros((n, n, r))
    CAK[:,:,0] = C @ K
    
    for k in range(1, r):
        Ak = Ak @ A
        CAK[:,:,k] = C @ Ak @ K
        
    return CAK


def iss2dd(L, A, C, K):
    """
    UNDER CONSTRUCTION!!

    Calculate dynamical dependence of projection L for
    innovations-form state-space model with parameters A, C, K.

    L:      orthonormal subspace basis
    A, C, K: ISS parameters
    
    NOTE: assumes identity residuals covariance matrix
    """
    # Calculate residuals covariance matrix V of projected model (solve DARE)
    # _, V, _ = solve_discrete_are(A, L.T @ C, K @ K.T, np.array([]), K @ L) # This could be wrong and needs to be converted to Lionels version.


    # Check if DARE failed
    if np.isnan(V).any():
        D = np.nan
        return D

    # D = log-determinant of residuals covariance matrix V
    try:
        L, lower = cho_factor(V)
        D = 2 * np.sum(np.log(cholesky(V, lower=lower, check_finite=False, overwrite_a=True).diagonal()))
    except:
        D = np.nan # fail: V not positive-definite
    
    return D


def orthonormalise(X):
    """
    Returns an orthonormal basis L for the range of X.
    That is, L.T @ L = I, the columns of L span the same space as
    the columns of X, and the number of columns of L is the
    rank of X. Optionally, also return orthonormal basis M for
    orthogonal subspace.

    Could use QR decomposition, but SVD may be more stable.
    """
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    L = U
    if len(s) < X.shape[1]: # X is rank deficient
        L = L[:, :len(s)]
    if np.linalg.matrix_rank(X) < X.shape[1]: # X is rank deficient
        M = VT[len(s):,:].T
    else:
        M = np.zeros((X.shape[1], 0))
    return L, M

import numpy as np

def rand_orthonormal(n, m, r=1):
    """
    Returns r random n x m orthonormal matrices, and optionally their n x (n-m) orthogonal complements.
    
    Parameters:
    n (int): Number of rows of the matrices
    m (int): Number of columns of the matrices
    r (int, optional): Number of matrices to generate (default=1)
    
    Returns:
    L (ndarray): Array of r n x m orthonormal matrices
    M (ndarray, optional): Array of r n x (n-m) orthogonal complements, returned only if nargout > 1
    """
    L = np.random.randn(n, m, r)
    if np.ndim(L) == 2:
        L = np.expand_dims(L, axis=2)
    
    if np.ndim(L) != 3:
        raise ValueError("L must be a 3D array.")
    
    if np.any(np.isinf(L)) or np.any(np.isnan(L)):
        raise ValueError("L contains infinite or NaN values.")
    
    if np.all(m == n):
        # L is square, so only need to compute orthonormal complements
        M = np.empty((n, n-m, r))
        for k in range(r):
            L, M = orthonormalise(L[:, :, k])
        return L, M
    else:
        # L is not square, so don't compute orthonormal complements
        for k in range(r):
            L, _ = orthonormalise(L[:, :, k])
        return L
    
def itransform_subspace(L, V0):
    """
    Inverse-transform projections (subspaces) back for correlated residuals

    Parameters:
    -----------
    L : numpy.ndarray
        Transformed orthonormal subspace basis
    V0 : numpy.ndarray
        Untransformed residuals covariance matrix

    Returns:
    --------
    L0 : numpy.ndarray
        Untransformed orthonormal subspace basis
    """

    assert L.ndim >= 2
    siz = L.shape
    xdims = np.prod(siz[2:])  # extra dimensions
    IV0LCT = cho_solve(cho_factor(V0, lower=True), np.eye(V0.shape[0]))
    L0 = L.reshape(siz[0], siz[1], xdims, order='F')
    for k in range(xdims):
        L0[:, :, k] = orthonormalise(IV0LCT @ L0[:, :, k])  # note that IV0LCT'*L will NOT generally be orthonormal!
    L0 = np.reshape(L0, siz)
    return L0

def transform_subspace(L0, V0):
    """
    Transform projections (subspaces) for correlated residuals
    
    L0 : array_like
        Untransformed orthonormal subspace basis
    V0 : array_like
        Untransformed residuals covariance matrix
    L : array_like
        Transformed orthonormal subspace basis
    """
    assert np.ndim(L0) >= 2
    siz = L.shape
    xdims = np.prod(siz[2:])  # extra dimensions
    V0LCT = np.linalg.cholesky(V0, lower=True)
    L = np.reshape(L0, (siz[0], siz[1], xdims))
    for k in range(xdims):
        L[:,:,k] = orthonormalise(V0LCT @ L[:,:,k]) # note that V0LCT.T @ L0 will NOT generally be orthonormal!
    L = np.reshape(L, siz)
    return L

def transform_ss(A, C, K, V):
    # Transform SS model parameters to decorrelate and normalise residuals;
    # i.e., so that residuals covariance matrix is the identity matrix.

    n = C.shape[0]
    SQRTV = np.linalg.cholesky(V, lower=True)
    ISQRTV = np.linalg.inv(SQRTV)
    # A is unchanged
    C = ISQRTV @ C
    K = K @ SQRTV
    V = np.eye(n)

    return A, C, K, V

def transform_var(A, V):
    """
    Transform VAR model parameters to decorrelate and normalise residuals;
    i.e., so that residuals covariance matrix is the identity matrix.

    Parameters:
        A (ndarray): VAR model coefficient matrices with shape (n, n, p).
        V (ndarray): Residual covariance matrix with shape (n, n).

    Returns:
        A (ndarray): Transformed VAR model coefficient matrices.
        V (ndarray): Identity matrix with shape (n, n).
    """

    n, _, p = A.shape
    SQRTV = np.linalg.cholesky(V, lower=True)
    ISQRTV = np.linalg.inv(SQRTV)

    for k in range(p):
        A[:,:,k] = ISQRTV @ A[:,:,k] @ SQRTV

    V = np.eye(n)
    return A, V

def trfun2dd(L, H):
    # Calculate spectral dynamical dependence of projection L
    # from transfer function H.
    #
    # L    orthonormal subspace basis
    # H    transfer function
    #
    # NOTE: assumes identity residuals covariance matrix

    h = H.shape[2]
    d = np.zeros(h)
    for k in range(h): # over [0,pi]
        Qk = H[:,:,k].T @ L
        d[k] = np.sum(np.log(np.diag(np.linalg.cholesky(Qk.T @ Qk, lower=True)))) # (log-determinant)/2
    D = np.sum(d[:-1] + d[1:])/(h-1) # integrate frequency-domain DD (trapezoidal rule) to get time-domain DD
    return D, d

def trfun2ddgrad(L, H):
    """
    Calculate gradient of spectral dynamical dependence
    of projection L from transfer function H.

    L    orthonormal subspace basis
    H    transfer function

    NOTE: assumes identity residuals covariance matrix
    """
    h = H.shape[2]
    n, m = L.shape
    g = np.zeros((n, m, h))
    for k in range(h): # over [0, pi]
        Hk = H[:,:,k]
        HLk = Hk.T @ L
        g[:,:,k] = np.real((Hk @ HLk) / (HLk.T @ HLk)) # grad/2

    # Integrate frequency-domain derivative (trapezoidal rule) and
    # subtract 2*L to get time-domain gradient (see note below)
    G = np.sum(g[:,:,0:-1] + g[:,:,1:], axis=2) / (h-1) - 2*L

    if len(np.shape(G)) == 1:
        mG = np.sqrt(np.sum(G ** 2))
    else:
        mG = np.sqrt(np.sum(G ** 2, axis=(0,1)))

    return G, mG

def ac2ces(G):
    """

    UNDER CONSTRUCTION!!!

    Calculate the CE Sigma_i matrices

    Args:
    G (ndarray): autocovariance sequence

    Returns:
    CESRC (ndarray): right (upper) Cholesky factors of the CE Sigma_i matrices
    CRC (ndarray): right (upper) Cholesky factor of covariance matrix

    Note:
    All calculations in UNTRANSFORMED coordinates!!!
    """

    CRC = cho_factor(G[:,:,0], lower=False, overwrite_a=True)
    n = G.shape[0]
    CESRC = np.zeros((n, n, n))

    for i in range(n):
        Gi = G[:, i, 1:]
        Gii = np.fliplr(G[i, i, :-1])
        Fi = solve_triangular(CRC[0], Gi, lower=False, trans='T')
        Fi = solve_triangular(CRC[0], Fi, lower=False)

        CESRC[:, :, i] = cho_factor(Gii, lower=False, overwrite_a=True)[0].T
        CESRC[:, :, i] = solve_triangular(CESRC[:, :, i], Fi, lower=False)
        CESRC[:, :, i] = cho_factor(np.dot(CRC[0].T, CRC[0]) - np.dot(Fi.T, Fi), lower=False, overwrite_a=True)[0].T

    return CESRC, CRC[0]

def logdet(A):
    """Compute the log determinant of a matrix."""
    return np.sum(np.log(np.linalg.eigvalsh(A)))

def ces2ce(L, H, VRC, CESRC, CRC):
    """Calculate causal emergence of projection L.
    
        Under construction!!!   

    Args:
        L (ndarray): Orthonormal subspace basis of shape (n, m).
        H (ndarray): Transfer function of shape (n, n, h).
        VRC (ndarray): Right (upper) Cholesky factor of residuals covariance matrix of shape (n, n).
        CESRC (ndarray): Right (upper) Cholesky factors of the CE Sigma_i matrices of shape (n, n, n).
        CRC (ndarray): Right (upper) Cholesky factor of covariance matrix of shape (n, n).
    
    Returns:
        CE (float): Causal emergence.
        DD (float, optional): Dynamical dependence.
    """
    # Calculate reduced residuals generalised covariance
    h = H.shape[2]
    v = np.zeros(h)
    for k in range(h):  # over [0, pi]
        RHLk = VRC @ H[:,:,k].T @ L
        v[k] = np.sum(np.log(np.diag(np.linalg.cholesky(RHLk.T @ RHLk, lower=True)))) / 2  # (log-determinant) / 2
    VRED = np.trapz(v, dx=np.pi/(h-1))  # integrate frequency-domain DD (trapezoidal rule) to get time-domain DD

    # Calculate causal emergence
    n = CRC.shape[0]
    CEH = np.zeros(n,1)
    for i in range(n):
        CESRCLi = CESRC[:,:,i] @ L
        CEH[i] = logdet(CESRCLi.T @ CESRCLi)
    CRCL = CRC @ L
    CE = -(n-1)*logdet(CRCL.T @ CRCL) + np.sum(CEH) - VRED

    # And, return dynamical dependence
    RHL = VRC @ L
    DD = VRED - 2*np.sum(np.log(np.diag(np.linalg.cholesky(RHL.T @ RHL, lower=True))))
    return CE, DD





