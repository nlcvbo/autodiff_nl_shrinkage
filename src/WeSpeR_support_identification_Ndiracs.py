import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
from numpy.polynomial import polynomial as poly
from WeSpeR_root_finding import root_brentq_m, root_brentq_p, root_brentq_r, root_brentq_l, root_brentq_lm, root_brentq_rm, m_idx_tab

def mld_Ndiracs(x, wd, d):
    if len(x.shape) == 1:
        return (wd[None,:]*d[None,:]/(d[None,:] - x[:, None])).sum(axis=1)
    else:
        return (wd[None, None,:]*d[None, None,:]/(d[None, None,:] - x[:, :, None])).sum(axis=2)

def mld_Ndiracsp(x, wd, d):
    if len(x.shape) == 1:
        return (wd[None,:]*d[None,:]/(d[None,:] - x[:, None])**2).sum(axis=1)
    else:
        return (wd[None, None,:]*d[None, None,:]/(d[None, None,:] - x[:, :, None])**2).sum(axis=2)

def mld_Ndiracspp(x, wd, d):
    if len(x.shape) == 1:
        return 2*(wd[None,:]*d[None,:]/(d[None,:] - x[:, None])**3).sum(axis=1)
    else:
        return 2*(wd[None, None,:]*d[None, None,:]/(d[None, None,:] - x[:, :, None])**3).sum(axis=2)

def tw(u, w, t, d, wd, c):
    return c*(w*t/(t - u)).sum()

def twp(u, w, t, d, wd, c):
    return c*(w*t/(t - u)**2).sum()

def twpp(u, w, t, d, wd, c):
    return 2*c*(w*t/(t - u)**3).sum()

def twppp(u, w, t, d, wd, c):
    return 6*c*(w*t/(t - u)**4).sum()

def mldm_root(u, k, w, t, d, wd, c, P, Q, eps=1e-2, verbose=False):
    g1 = tw(u, w, t, d, wd, c)
    if not np.isfinite(g1):
        if k < d.shape[0]:
            if g1 > 0:
                m = d[k-1]
            else:
                m = d[k]
        elif g1 >= 0:
            m = d[0]
        else:
            m = d[-1]
        return m
    
    f = lambda x: mld_Ndiracs(x*np.ones((1)), wd, d)[0] - g1
    try:
        if k < d.shape[0]:
            ul = d[k-1]
            ur = d[k]
            m = root_brentq_m(f,ul,ur,eps)
        elif g1 >= 0:
            ul = d[0] - 1
            ur = d[0]
            m = root_brentq_lm(f,ul,ur,eps)
        else:
            ul = d[-1]
            ur = d[-1] + 1
            m = root_brentq_r(f,ul,ur,eps)
        
    except Exception as error:
        if verbose:
            print("No roots found, u=",u,", t(u)=",g1)
        m = np.inf
    
    return m

def mldm_poly(u, k, w, t, d, wd, c, P, Q, verbose=False):
    g1 = tw(u, w, t, d, wd, c)
    if np.isfinite(g1):
        R = poly.polyadd(P, poly.polymul(poly.Polynomial((-g1)),Q))[0]
    else:
        idx = np.argmin(np.abs(u-t))
        if t[idx] - u > 0: # then g1 > 0
            R = poly.polymul(poly.Polynomial((-1)),Q)[0]
        else:
            R = Q
    try:
        roots =  np.sort(R.roots().real)
        if roots[0] < 0:
            roots = np.concatenate([roots[1:],roots[:1]])
        m = roots[k-1]
        
    except:
        if verbose:
            print("No roots found, t(u)=",g1)
        m = np.inf
        
    return m

def v(u, k, w, t, d, wd, c, m):
    return m

def vp(u, k, w, t, d, wd, c, m):
    return twp(u, w, t, d, wd, c)/(wd*d/(d - m)**2).sum()

def vpp(u, k, w, t, d, wd, c, m):
    return twpp(u, w, t, d, wd, c)/(wd*d/(d - m)**2).sum() - twp(u, w, t, d, wd, c)**2*2*(wd*d/(d - m)**3).sum()/(wd*d/(d - m)**2).sum()**3

def vppp(u, k, w, t, d, wd, c, m): #see p.135
    return twppp(u, w, t, d, wd, c)/(wd*d/(d - m)**2).sum() \
           - 3*twp(u, w, t, d, wd, c)*twpp(u, w, t, d, wd, c)*2*(wd*d/(d - m)**3).sum()/(wd*d/(d - m)**2).sum()**3 \
           + 3*twp(u, w, t, d, wd, c)**3*2*(wd*d/(d - m)**3).sum()**2/(wd*d/(d - m)**2).sum()**5 \
           - twp(u, w, t, d, wd, c)**3*6*(wd*d/(d - m)**4).sum()/(wd*d/(d - m)**2).sum()**4

def x_Ndiracs(k, w, t, d, wd, c, P, Q, mldm):
    def xk(u, return_m = False):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        rst = -u*tw(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m)
        if return_m:
            return rst, m
        return rst
    return xk

def x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm):
    def xkp(u):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        return -tw(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m) - u*twp(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m) - u*tw(u, w, t, d, wd, c)*vp(u,k, w, t, d, wd, c, m)
    return xkp

def x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldm):
    def xkpp(u):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        return - twp(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m) - tw(u, w, t, d, wd, c)*vp(u,k, w, t, d, wd, c, m) \
               - twp(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m) - u*twpp(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m) - u*twp(u, w, t, d, wd, c)*vp(u,k, w, t, d, wd, c, m) \
               - tw(u, w, t, d, wd, c)*vp(u,k, w, t, d, wd, c, m) - u*twp(u, w, t, d, wd, c)*vp(u,k, w, t, d, wd, c, m) - u*tw(u, w, t, d, wd, c)*vpp(u,k, w, t, d, wd, c, m)
    return xkpp

def x_Ndiracs_ppp(k, w, t, d, wd, c, P, Q, mldm):
    def xkpp(u):
        m = mldm(u, k, w, t, d, wd, c, P, Q)
        return  - 3*twpp(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m) - 6*twp(u, w, t, d, wd, c)*vp(u,k, w, t, d, wd, c, m) - 3*tw(u, w, t, d, wd, c)*vpp(u,k, w, t, d, wd, c, m) \
                - 3*u*twpp(u, w, t, d, wd, c)*vp(u,k, w, t, d, wd, c, m) - 3*u*twp(u, w, t, d, wd, c)*vpp(u,k, w, t, d, wd, c, m) \
                - u*twppp(u, w, t, d, wd, c)*v(u,k, w, t, d, wd, c, m) - u*tw(u, w, t, d, wd, c)*vppp(u,k, w, t, d, wd, c, m)
    return xkpp

def xX_Ndiracs(k, w, t, d, wd, c, P, Q, mldm):
    # X \mapsto x_Ndiracs(k, w, t, d, wd, c, P, Q)(-1/X)
    def xk(X, return_m = False):
        return x_Ndiracs(k, w, t, d, wd, c, P, Q, mldm)(-1/X, return_m = return_m)
    return xk

def xX_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm):
    def xkp(X):
        return x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm)(-1/X)/X**2
    return xkp

def xX_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldm):
    def xkpp(X):
        return x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldm)(-1/X)/X**4 -2/X**3*x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldm)(-1/X)
    return xkpp

def find_support_Ndiracs(c, wd, d, t, w, eps=1e-2, Kd = -1, Kt = -1, method = 'root', verbose = False):
    # On each interval ]tau_i, tau_{i+1}[, find the zero of x_k'' (for each k), and check if x_k' is >= 0 at this point, if yes, find the zeros of x_k' on the left and right sub-intervals
    # On ]-infty,tau_1[ and ]tau_p,infty[, find the zero of x_M' (for M only)
    # When p or M are too large, only check the largest intervals (on tau and delta) and x_M on the exterior interval
    M = d.shape[0]
    T = t.shape[0]
    support = []
    
    m_idx = m_idx_tab(Kd, d)
    t_idx = m_idx_tab(Kt, t)
    t_idx = [i-1 for i in t_idx]
    
    
    if method == 'poly':
        P = poly.Polynomial((0))
        for i in range(M):
            R = poly.Polynomial((wd[i]*d[i]))
            for j in range(M):
                if j != i:
                    R = poly.polymul(R, poly.Polynomial((d[j], -1)))
            P = poly.polyadd(P, R)[0]
        
        Q = poly.Polynomial((1))
        for i in range(M):
            Q = poly.polymul(Q, poly.Polynomial((d[i], -1)))[0]
        mldmf = mldm_poly
    
    elif method == 'root':
        P, Q = None, None
        mldmf = mldm_root
    
    else:
        P, Q = None, None
        mldmf = mldm_root
    
    for i in t_idx[:-1]:
        for k in m_idx:
            xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
            xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
            xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)
            
            try:
                u0 = root_brentq_p(xkpp,t[i],t[i+1],eps)
                if xkp(u0) == 0:
                    support += [xk(u0),xk(u0)]
                    if verbose:
                        print(u0,xk(u0),xkp(u0))
                elif xkp(u0) > 0:
                    try:
                        u1 = root_brentq_m(xkp,t[i],u0,eps)
                        u2 = root_brentq_p(xkp,u0,t[i+1],eps)
                        support += [xk(u1),xk(u2)]
                        if verbose:
                            print(u1,xk(u1),xkp(u1))
                            print(u2,xk(u2),xkp(u2))
                    except ValueError:
                        if verbose:
                            print("Error",t[i],t[i+1])
            except ValueError:
                if verbose:
                    print("Error:",k,t[i],t[i+1])
            # otherwise, no spectral gap
            
    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for all k != M
    for k in m_idx[:-1]:
        xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

        try:
            u0 = root_brentq_p(xkp,0,t[0],eps)
            support += [0,xk(u0)]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error",k,0,t[0])
            
            
    # for the right exterior intervals, ]tau_p, infty[, for all k != M, consider X = -1/u
    for k in m_idx[:-1]:        
        xkppX = xX_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkpX = xX_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xkX = xX_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)
        
        try:
            X0 = root_brentq_p(xkppX,-1/t[-1],0,eps)
            if xkpX(X0) == 0:
                support += [xkX(X0),xkX(X0)]
                if verbose:
                    print(X0,xkX(X0),xkX(X0))
            elif xkpX(X0) > 0:
                try:
                    X1 = root_brentq_m(xkpX,-1/t[-1],X0,eps)
                    X2 = root_brentq_p(xkpX,X0,0,eps)      
                    support += [xkX(X1),xkX(X2)]
                    if verbose:
                        print(-1/X1,-1/X2,xkX(X1),xkX(X2))
                except ValueError:
                    if verbose:
                        print("Error",t[i],t[i+1])
        except ValueError:
            if verbose:
                print("Error:",t[i],t[i+1])
            
    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c<1:
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)
        try:
            u0 = root_brentq_p(xkp,0,t[0],eps)
            xk0 = xk(u0)
            support += [0, xk0]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error",0,t[0])
    else:
        xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldmf)
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)
        
        try:
            u0 = root_brentq_l(xkp, 0, t[0], eps)
            xk0 = xk(u0)
            support += [0, xk0]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error:",-np.inf,t[0])
    
    # for the right exterior intervals, ]tau_p,infty[, for k == M
    xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldmf)
    xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
    xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)
    
    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1]+1,eps)
        support += [xk(u0)]
        if verbose:
            print(u0,xk(u0),xkp(u0))
    except ValueError:
        if verbose:
            print("Error:",M,t[-1],np.inf,eps,xkp(t[-1]*(1+eps)),xkp(2*(t[-1]*(1+eps))))
    
    support.sort()
    return support[1:]

def find_support_Ndiracs_u(c, wd, d, t, w, eps=1e-2, Kd = -1, Kt = -1, method = 'root', verbose = False):
    # On each interval ]tau_i, tau_{i+1}[, find the zero of x_k'' (for each k), and check if x_k' is >= 0 at this point, if yes, find the zeros of x_k' on the left and right sub-intervals
    # On ]-infty,tau_1[ and ]tau_p,infty[, find the zero of x_M' (for M only)
    # When p or M are too large, only check the largest intervals (on tau and delta) and x_M on the exterior interval
    M = d.shape[0]
    T = t.shape[0]
    support = []
    u = []
    mldm = []
    m = []
    tildeomegai = []
    
    m_idx = m_idx_tab(Kd, d, 1e-5)
    t_idx = m_idx_tab(Kt, t, 1e-5)
    t_idx = [i-1 for i in t_idx]
    
    if method == 'poly':
        P = poly.Polynomial((0))
        for i in range(M):
            R = poly.Polynomial((wd[i]*d[i]))
            for j in range(M):
                if j != i:
                    R = poly.polymul(R, poly.Polynomial((d[j], -1)))
            P = poly.polyadd(P, R)[0]
        
        Q = poly.Polynomial((1))
        for i in range(M):
            Q = poly.polymul(Q, poly.Polynomial((d[i], -1)))[0]
        mldmf = mldm_poly
    
    elif method == 'root':
        P, Q = None, None
        mldmf = mldm_root
    
    else:
        P, Q = None, None
        mldmf = mldm_root
    
    for i in t_idx[:-1]:
        for k in m_idx:
            xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
            xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
            xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)
            
            try:
                u0 = root_brentq_p(xkpp,t[i],t[i+1],eps)
                if xkp(u0) == 0:
                    xk0, m0 = xk(u0, return_m=True)
                    support += [xk0, xk0]
                    u += [u0, u0]
                    mldm += [m0, m0]
                    m += [k, k]
                    if verbose:
                        print(u0,xk(u0),xkp(u0))
                elif xkp(u0) > 0:
                    try:
                        u1 = root_brentq_m(xkp,t[i],u0,eps)
                        u2 = root_brentq_p(xkp,u0,t[i+1],eps)
                        xk1, m1 = xk(u1, return_m=True)
                        xk2, m2 = xk(u2, return_m=True)
                        support += [xk1,xk2]
                        u += [u1, u2]
                        mldm += [m1, m2]
                        m += [k, k]
                        if verbose:
                            print(u1,xk(u1),xkp(u1))
                            print(u2,xk(u2),xkp(u2))
                    except ValueError:
                        if verbose:
                            print("Error",t[i],t[i+1])
            except ValueError:
                if verbose:
                    print("Error:",k,t[i],t[i+1])
            # otherwise, no spectral gap
            
    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for all k != M
    for k in m_idx[:-1]:
        xkpp = x_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkp = x_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)

        try:
            u0 = root_brentq_p(xkp,0,t[0],eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            m += [k, k]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error",k,0,t[0])
            
            
    # for the right exterior intervals, ]tau_p, infty[, for all k != M, consider X = -1/u
    for k in m_idx[:-1]:        
        xkppX = xX_Ndiracs_pp(k, w, t, d, wd, c, P, Q, mldmf)
        xkpX = xX_Ndiracs_p(k, w, t, d, wd, c, P, Q, mldmf)
        xkX = xX_Ndiracs(k, w, t, d, wd, c, P, Q, mldmf)
        
        try:
            X0 = root_brentq_p(xkppX,-1/t[-1],0,eps)
            if xkpX(X0) == 0:
                xk0, m0 = xkX(X0, return_m=True)
                u0 = -1/X0
                support += [xk0,xk0]
                u += [u0, u0]
                mldm += [m0, m0]
                m += [k, k]
                if verbose:
                    print(X0,xkX(X0),xkX(X0))
            elif xkpX(X0) > 0:
                try:
                    X1 = root_brentq_m(xkpX,-1/t[-1],X0,eps)
                    X2 = root_brentq_p(xkpX,X0,0,eps)  
                    xk1, m1 = xkX(X1, return_m=True)
                    xk2, m2 = xkX(X2, return_m=True)
                    support += [xk1,xk2]
                    u += [-1/X1, -1/X2]
                    mldm += [m1, m2]
                    if verbose:
                        print(-1/X1,-1/X2,xkX(X1),xkX(X2))
                except ValueError:
                    if verbose:
                        print("Error",t[i],t[i+1])
        except ValueError:
            if verbose:
                print("Error:",t[i],t[i+1])
            
    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c<1:
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)
        try:
            u0 = root_brentq_p(xkp,0,t[0],eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            m += [M, M]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error",0,t[0])
    else:
        xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldm)
        xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldm)
        xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldm)
        
        try:
            u0 = root_brentq_l(xkp, 0, t[0], eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            m += [M, M]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error:",-np.inf,t[0])
    
    # for the right exterior intervals, ]tau_p,infty[, for k == M
    xkpp = x_Ndiracs_pp(M, w, t, d, wd, c, P, Q, mldmf)
    xkp = x_Ndiracs_p(M, w, t, d, wd, c, P, Q, mldmf)
    xk = x_Ndiracs(M, w, t, d, wd, c, P, Q, mldmf)
    
    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1]+1,eps)
        xk0, m0 = xk(u0, return_m=True)
        support += [xk0]
        u += [u0]
        mldm += [m0]
        m += [M]
        if verbose:
            print(u0,xk(u0),xkp(u0),tw(u0, w, t, d, wd, c), m0)
            
    except ValueError:
        if verbose:
            print("Error:",M,t[-1],np.inf,eps,xkp(t[-1]*(1+eps)),xkp(2*(t[-1]*(1+eps))))
    
    support, u, mldm, m = map(list, zip(*sorted(zip(support, u, mldm, m))))
            
    support = np.array(support[1:])
    u = np.array(u[1:])
    mldm = np.array(mldm[1:])
    m = np.array(m[1:]).astype(int)
    
    if support.shape[0] % 2 != 0:
        print("Support failed")
        support = np.concatenate([support, [t[-1]*d[-1]*(1+np.sqrt(c))**2]])
        u = np.concatenate([u, [t[-1]+1]])
        mldm = np.concatenate([mldm, [d[-1]+1]])
        m = np.concatenate([m, [M]])
        
        nu = support.shape[0]//2
        tildeomegai = np.ones(nu)/nu
        return support, u, mldm, tildeomegai
    
    nu = support.shape[0]//2
    tildeomegai = np.ones(nu)/nu
    for i in range(nu):
        m1, m2 = m[2*i]-1, m[2*i+1]-1
        if m1 == M and mldm[2*i] <= d[0]:
            m1 = 0
        if m2 == M and mldm[2*i+1] <= d[0]:
            m2 = 0 
        tildeomegai[i] = w[(t>u[2*i]) & (t<u[2*i+1])].sum() * wd[m1:m2+1].sum()
    if tildeomegai.sum() > 0:
        tildeomegai = tildeomegai/tildeomegai.sum()
    return support, u, mldm, tildeomegai
