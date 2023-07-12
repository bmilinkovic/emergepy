import numpy as np
from dd.utils import orthonormalise, trfun2dd, trfun2ddgrad, cak2ddx, cak2ddxgrad
import time

def opt_gd1_dds(H, L0, maxiters, gdsig0, gdls, tol, hist):
    # Assumptions
    #
    # 1 - L is orthonormal
    # 2 - Residuals covariance matrix is identity
    
    if np.isscalar(gdls):
        ifac = gdls
        nfac = 1/ifac
    else:
        ifac = gdls[0]
        nfac = gdls[1]

    if np.isscalar(tol):
        stol = tol
        dtol = tol
        gtol = tol
    else:
        stol = tol[0]
        dtol = tol[1]
        gtol = tol[2]

    # Calculate dynamical dependence of initial projection
    L = L0
    G, g = trfun2ddgrad(L, H) # dynamical dependence gradient and magnitude
    dd = trfun2dd(L, H)
    sig = gdsig0

    if hist:
        dhist = np.zeros((maxiters, 3))
        dhist[0,:] = [dd, sig, g]
    else:
        dhist = []

    # Optimise: gradient descent (variant 1)
    converged = 0
    for iters in range(1, maxiters):
        # Move (hopefully) down gradient
        Ltry = orthonormalise(L - sig * (G/g)) # gradient descent
        ddtry = trfun2dd(Ltry, H)

        # If dynamical dependence smaller than current optimum, accept move and increase
        # step size; else reject move and decrease step size (similar to 1+1 ES)
        if ddtry < dd:
            L = Ltry
            G, g = trfun2ddgrad(L, H) # dynamical dependence gradient and magnitude
            dd = ddtry
            sig = ifac * sig
        else:
            sig = nfac * sig

        if hist:
            dhist[iters,:] = [dd, sig, g]

        # Test convergence
        if sig < stol:
            converged = 1
            break
        elif dd < dtol:
            converged = 2
            break
        elif g < gtol:
            converged = 3
            break

    if hist:
        dhist = dhist[:iters,:]

    return dd, L, converged, sig, iters, dhist

def opt_gd1_ddx(CAK, L0, maxiters, gdsig0, gdls, tol, hist):
    """
    Optimizes a projection matrix L using the gradient descent algorithm with the dynamical dependence
    metric calculated based on the CAK matrix.

    Arguments:
    CAK -- the CAK matrix
    L0 -- the initial projection matrix
    maxiters -- the maximum number of iterations
    gdsig0 -- the initial step size
    gdls -- a scalar or a tuple of two values representing the increase and decrease factors for the step size
    tol -- a scalar or a tuple of three values representing the stopping criteria for the step size, the dynamical dependence,
    and the gradient magnitude respectively
    hist -- whether to record convergence history or not

    Returns:
    dd -- the dynamical dependence of the optimized projection matrix
    L -- the optimized projection matrix
    converged -- a flag indicating the convergence status
    sig -- the final step size
    iters -- the number of iterations
    dhist -- a history of the dynamical dependence, step size and gradient magnitude at each iteration (if hist is True)
    """
    if isinstance(gdls, (int, float)):
        ifac = gdls
        nfac = 1 / ifac
    else:
        ifac, nfac = gdls

    if isinstance(tol, (int, float)):
        stol = dtol = tol
        gtol = tol/10
    else:
        stol, dtol, gtol = tol

    # Calculate proxy dynamical dependence of initial projection
    L = L0
    G, g = cak2ddxgrad(L, CAK)  # proxy dynamical dependence gradient and magnitude
    dd = cak2ddx(L, CAK)
    sig = gdsig0

    if hist:
        dhist = np.zeros((maxiters, 3))
        dhist[0] = [dd, sig, g]
    else:
        dhist = None

    # Optimise: gradient descent (variant 1)
    converged = 0
    for iters in range(1, maxiters):
        # Move (hopefully) down gradient
        Ltry = orthonormalise(L - sig * (G / g))  # gradient descent
        ddtry = cak2ddx(Ltry, CAK)

        # If dynamical dependence smaller than current optimum, accept move and increase
        # step size; else reject move and decrease step size (similar to 1+1 ES)
        if ddtry < dd:
            L = Ltry
            G, g = cak2ddxgrad(L, CAK)  # proxy dynamical dependence gradient and magnitude
            dd = ddtry
            sig *= ifac
        else:
            sig *= nfac

        if hist:
            dhist[iters] = [dd, sig, g]

        # Test convergence
        if sig < stol:
            converged = 1
            break
        elif dd < dtol:
            converged = 2
            break
        elif g < gtol:
            converged = 3
            break

    if hist:
        dhist = dhist[:iters + 1]

    return dd, L, converged, sig, iters, dhist


# gd2 algorithm is much faster but needs to compute the gradient at every step. By default, use gd2

def opt_gd2_dds(H, L0, maxiters, gdsig0, gdls, tol, hist):

    """
    Assumptions:
    1 - L is orthonormal
    2 - Residuals covariance matrix is identity
    """
    
    if isinstance(gdls, float):
        ifac = gdls
        nfac = 1/ifac
    else:
        ifac = gdls[0]
        nfac = gdls[1]

    if isinstance(tol, float):
        stol = tol
        dtol = tol
        gtol = tol
    else:
        stol = tol[0]
        dtol = tol[1]
        gtol = tol[2]

    # Calculate dynamical dependence of initial projection

    L     = L0
    G,g   = trfun2ddgrad(L,H) # dynamical dependence gradient and magnitude
    dd    = trfun2dd(L,H)
    sig   = gdsig0

    if hist:
        dhist = np.zeros((maxiters, 3))
        dhist[0,:] = [dd, sig, g]
    else:
        dhist = []

    # Optimise: gradient descent (variant 2)

    converged = 0
    for iters in range(1, maxiters):

        # Move (hopefully) down gradient

        L     = orthonormalise(L-sig*(G/g)) # gradient descent
        G,g   = trfun2ddgrad(L,H) # dynamical dependence gradient and magnitude
        ddnew = trfun2dd(L,H)

        # If dynamical dependence smaller than current optimum, update current optimum
        # and increase step size; else decrease step size (similar to 1+1 ES)

        if ddnew < dd:
            dd  = ddnew
            sig = ifac*sig
        else:
            sig = nfac*sig

        if hist:
            dhist[iters,:] = [dd, sig, g]

        # Test convergence

        if     sig < stol:
            converged = 1
            break
        elif dd < dtol:
            converged = 2
            break
        elif g  < gtol:
            converged = 3
            break

    if hist:
        dhist = dhist[:iters,:]

    return dd, L, converged, sig, iters, dhist

def opt_gd2_ddx(CAK, L0, maxiters, gdsig0, gdls, tol, hist):
    """
    Optimizes a projection matrix L using the gradient descent algorithm with the dynamical dependence
    metric calculated based on the CAK matrix.

    Arguments:
    CAK -- the CAK matrix
    L0 -- the initial projection matrix
    maxiters -- the maximum number of iterations
    gdsig0 -- the initial step size
    gdls -- a scalar or a tuple of two values representing the increase and decrease factors for the step size
    tol -- a scalar or a tuple of three values representing the stopping criteria for the step size, the dynamical dependence,
    and the gradient magnitude respectively
    hist -- whether to record convergence history or not

    Returns:
    dd -- the dynamical dependence of the optimized projection matrix
    L -- the optimized projection matrix
    converged -- a flag indicating the convergence status
    sig -- the final step size
    iters -- the number of iterations
    dhist -- a history of the dynamical dependence, step size and gradient magnitude at each iteration (if hist is True)
    """
    if isinstance(gdls, (int, float)):
        ifac = gdls
        nfac = 1 / ifac
    else:
        ifac, nfac = gdls

    if isinstance(tol, (int, float)):
        stol = dtol = tol
        gtol = tol/10
    else:
        stol, dtol, gtol = tol

    # Calculate proxy dynamical dependence of initial projection
    L = L0
    G, g = cak2ddxgrad(L, CAK)  # proxy dynamical dependence gradient and magnitude
    dd = cak2ddx(L, CAK)
    sig = gdsig0

    if hist:
        dhist = np.zeros((maxiters, 3))
        dhist[0] = [dd, sig, g]
    else:
        dhist = None

    # Optimise: gradient descent (variant 1)
    converged = 0
    for iters in range(1, maxiters):
        # Move (hopefully) down gradient
        L = orthonormalise(L - sig * (G / g))  # gradient descent
        G, g = cak2ddxgrad(L, CAK)  # proxy dynamical dependence gradient and magnitude
        ddnew = cak2ddx(L, CAK)

        # If dynamical dependence smaller than current optimum, accept move and increase
        # step size; else reject move and decrease step size (similar to 1+1 ES)
        if ddnew < dd:
            dd = ddnew
            sig *= ifac
        else:
            sig *= nfac

        if hist:
            dhist[iters] = [dd, sig, g]

        # Test convergence
        if sig < stol:
            converged = 1
            break
        elif dd < dtol:
            converged = 2
            break
        elif g < gtol:
            converged = 3
            break

    if hist:
        dhist = dhist[:iters, :]

    return dd, L, converged, sig, iters, dhist

# The next code is the optimisers above ran across multiple runs or restrarts:

def opt_gd_dds_mruns(H, L0, niters, gdes, gdsig0, gdls, gdtol, hist=False, pp=False):
    
    if pp:
        from joblib import Parallel, delayed
        
    nruns = L0.shape[2]
    
    dopt = np.zeros(nruns)
    Lopt = np.zeros_like(L0)
    conv = np.zeros(nruns, dtype=bool)
    iopt = np.zeros(nruns, dtype=int)
    sopt = np.zeros(nruns)
    cput = np.zeros(nruns)
    if hist:
        ohist = [None] * nruns
    else:
        ohist = []
    
    # DD optimisation (gradient descent)
    
    if pp:
        def run_opt_gd1_dds(H, L0k, niters, gdsig0, gdls, gdtol, hist):
            return opt_gd1_dds(H, L0k, niters, gdsig0, gdls, gdtol, hist)
        
        def run_opt_gd2_dds(H, L0k, niters, gdsig0, gdls, gdtol, hist):
            return opt_gd2_dds(H, L0k, niters, gdsig0, gdls, gdtol, hist)
        
        print_func = lambda k: f'GD/ES({gdes}) optimisation parallel run {k+1:4d} of {nruns:4d}: '
        opt_func = run_opt_gd1_dds if gdes == 1 else run_opt_gd2_dds
        
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_opt_gd1_dds if gdes == 1 else run_opt_gd2_dds)(
                H, L0[:, :, k], niters, gdsig0, gdls, gdtol, hist
            ) for k in range(nruns)
        )
        
        for k, result in enumerate(results):
            dopt[k], Lopt[:, :, k], conv[k], sopt[k], iopt[k], ohist[k] = result
            cput[k] = 0  # cputime is not supported in Python
        
            print(f'{print_func(k)} dd = {dopt[k]:.4e} : sig = {sopt[k]:.4e} : ' +
                  f"{'converged(' + str(conv[k]) + ')' if conv[k] > 0 else 'unconverged '}" +
                  f' in {iopt[k]:4d} iterations : CPU secs = {cput[k]:6.2f}')
            
    else:
        opt_func = opt_gd1_dds if gdes == 1 else opt_gd2_dds
        for k in range(nruns):
            print(f'GD/ES({gdes}) optimisation serial run {k+1:4d} of {nruns:4d}: ', end='')
            tstart = time.perf_counter()
            dopt[k], Lopt[:, :, k], conv[k], sopt[k], iopt[k], ohist[k] = opt_func(
                H, L0[:, :, k], niters, gdsig0, gdls, gdtol, hist
            )
            cput[k] = time.perf_counter() - tstart
            
            print(f'dd = {dopt[k]:.4e} : sig = {sopt[k]:.4e} : ' +
                  f"{'converged(' + str(conv[k]) + ')' if conv[k] > 0 else 'unconverged '}" +
                  f' in {iopt[k]:4d} iterations : CPU secs = {cput[k]:6.2f}')
            
    sidx = np.argsort(dopt)
    dopt = dopt[sidx]
    Lopt = Lopt[:,:,sidx]
    iopt = iopt[sidx]
    conv = conv[sidx]
    sopt = sopt[sidx]

    return dopt, Lopt, conv, sopt, iopt, ohist, cput

def opt_gd_ddx_mruns(CAK, L0, niters, gdes, gdsig0, gdls, gdtol, hist=False, pp=False):
    
    if pp:
        from joblib import Parallel, delayed
        
    nruns = L0.shape[2]
    
    dopt = np.zeros(nruns)
    Lopt = np.zeros_like(L0)
    conv = np.zeros(nruns, dtype=bool)
    iopt = np.zeros(nruns, dtype=int)
    sopt = np.zeros(nruns)
    cput = np.zeros(nruns)
    if hist:
        ohist = [None] * nruns
    else:
        ohist = []
    
    # DD optimisation (gradient descent)
    
    if pp:
        def run_opt_gd1_ddx(CAK, L0k, niters, gdsig0, gdls, gdtol, hist):
            return opt_gd1_ddx(CAK, L0k, niters, gdsig0, gdls, gdtol, hist)
        
        def run_opt_gd2_ddx(CAK, L0k, niters, gdsig0, gdls, gdtol, hist):
            return opt_gd2_ddx(CAK, L0k, niters, gdsig0, gdls, gdtol, hist)
        
        print_func = lambda k: f'GD/ES({gdes}) optimisation parallel run {k+1:4d} of {nruns:4d}: '
        opt_func = run_opt_gd1_ddx if gdes == 1 else run_opt_gd2_ddx
        
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_opt_gd1_ddx if gdes == 1 else run_opt_gd2_ddx)(
                CAK, L0[:, :, k], niters, gdsig0, gdls, gdtol, hist
            ) for k in range(nruns)
        )
        
        for k, result in enumerate(results):
            dopt[k], Lopt[:, :, k], conv[k], sopt[k], iopt[k], ohist[k] = result
            cput[k] = 0  # cputime is not supported in Python
        
            print(f'{print_func(k)} dd = {dopt[k]:.4e} : sig = {sopt[k]:.4e} : ' +
                  f"{'converged(' + str(conv[k]) + ')' if conv[k] > 0 else 'unconverged '}" +
                  f' in {iopt[k]:4d} iterations : CPU secs = {cput[k]:6.2f}')
            
    else:
        opt_func = opt_gd1_ddx if gdes == 1 else opt_gd2_ddx
        for k in range(nruns):
            print(f'GD/ES({gdes}) optimisation serial run {k+1:4d} of {nruns:4d}: ', end='')
            tstart = time.perf_counter()
            dopt[k], Lopt[:, :, k], conv[k], sopt[k], iopt[k], ohist[k] = opt_func(
                CAK, L0[:, :, k], niters, gdsig0, gdls, gdtol, hist
            )
            cput[k] = time.perf_counter() - tstart
            
            print(f'dd = {dopt[k]:.4e} : sig = {sopt[k]:.4e} : ' +
                  f"{'converged(' + str(conv[k]) + ')' if conv[k] > 0 else 'unconverged '}" +
                  f' in {iopt[k]:4d} iterations : CPU secs = {cput[k]:6.2f}')
            
    sidx = np.argsort(dopt)
    dopt = dopt[sidx]
    Lopt = Lopt[:,:,sidx]
    iopt = iopt[sidx]
    conv = conv[sidx]
    sopt = sopt[sidx]

    return dopt, Lopt, conv, sopt, iopt, ohist, cput