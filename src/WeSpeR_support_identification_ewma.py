import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
from WeSpeR_root_finding import root_brentq_m, root_brentq_p, root_brentq_r, root_brentq_l, m_idx_tab


def mld_exp(x, alpha):
    beta = alpha/(1-np.exp(-alpha))
    rst = 1/alpha*np.log(1 + alpha/(beta*np.exp(-alpha)-x))
    return rst

def mld_expp(x,alpha):
    beta = alpha/(1-np.exp(-alpha))
    return 1/((beta*np.exp(-alpha)-x)**2 + alpha*(beta*np.exp(-alpha)-x))

def mld_exppp(x,alpha):
    beta = alpha/(1-np.exp(-alpha))
    return +(2*(beta*np.exp(-alpha)-x)+alpha)/((beta*np.exp(-alpha)-x)**2 + alpha*(beta*np.exp(-alpha)-x))**2

def x_exp(w,t,alpha,c):
    def xk(u, return_m=False):
        beta = alpha/(1-np.exp(-alpha))
        g1 = c*(w*t/(t-u)).sum()
        m = beta*np.exp(-alpha) + alpha/(1-np.exp(alpha*g1))
        
        rst = -u*g1*m
        if return_m:
            return rst, m
        return rst
    return xk

def x_exp_p(w,t,alpha,c):
    def xkp(u):
        beta = alpha/(1-np.exp(-alpha))
        g1 = c*(w*t/(t-u)).sum()
        m = beta*np.exp(-alpha) + alpha/(1-np.exp(alpha*g1))
        
        tw = c*(w*t/(t - u)).sum()
        twp = c*(w*t/(t - u)**2).sum()
        v = m
        vp = twp/mld_expp(m,alpha)
        
        rst = -tw*v - u*twp*v - u*tw*vp
        return rst
    return xkp

def x_exp_pp(w,t,alpha,c):
    def xkpp(u):
        beta = alpha/(1-np.exp(-alpha))
        g1 = c*(w*t/(t-u)).sum()
        m = beta*np.exp(-alpha) + alpha/(1-np.exp(alpha*g1))
        
        tw = c*(w*t/(t - u)).sum()
        twp = c*(w*t/(t - u)**2).sum()
        twpp = 2*c*(w*t/(t - u)**3).sum()
        v = m
        vp = twp/mld_expp(m,alpha)
        vpp = twpp/mld_expp(m,alpha) - twp**2*mld_exppp(m,alpha)/mld_expp(m,alpha)**3
        
        rst = - twp*v - tw*vp - twp*v - u*twpp*v - u*twp*vp - tw*vp - u*twp*vp - u*tw*vpp
        return rst
    return xkpp

def tw(u, w, t, d, wd, c):
    return c*(w*t/(t - u)).sum()

def twp(u, w, t, d, wd, c):
    return c*(w*t/(t - u)**2).sum()

def twpp(u, w, t, d, wd, c):
    return 2*c*(w*t/(t - u)**3).sum()

def twppp(u, w, t, d, wd, c):
    return 6*c*(w*t/(t - u)**4).sum()

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

def find_support_exp(c, alpha, t, w, Kt = -1, eps=1e-1, method = None, verbose = False):
    T = t.shape[0]
    support = []
    
    t_idx = m_idx_tab(Kt, t)
    t_idx = [i-1 for i in t_idx]
    
    for i in t_idx[:-1]:
        xkpp = x_exp_pp(w,t,alpha,c)
        xkp = x_exp_p(w,t,alpha,c)
        xk = x_exp(w,t,alpha,c)
            
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
                print("Error:",t[i],t[i+1])
        # otherwise, no spectral gap
        
    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c<1:
        xkp = x_exp_p(w,t,alpha,c)
        xk = x_exp(w,t,alpha,c)
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
        xkpp = x_exp_pp(w,t,alpha,c)
        xkp = x_exp_p(w,t,alpha,c)
        xk = x_exp(w,t,alpha,c)
        
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
    xkpp = x_exp_pp(w,t,alpha,c)
    xkp = x_exp_p(w,t,alpha,c)
    xk = x_exp(w,t,alpha,c)
    
    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1] + 1, eps)
        support += [xk(u0)]
        if verbose:
            print(u0,xk(u0),xkp(u0))
    except ValueError:
        if verbose:
            print("Error:",t[-1],np.inf)
    
    support.sort()
    support = support[1:]
    
    if support.shape[0] % 2 != 0:
        beta = alpha/(1-np.exp(-alpha))
        support = np.concatenate([support, [t[-1]*beta*(1+np.sqrt(c))**2]])
    return support

def find_support_exp_u(c, alpha, t, w, Kt = -1, eps=1e-1, method = None, verbose = False):
    T = t.shape[0]
    support = []
    u = []
    mldm = []
    tildeomegai = []
    
    t_idx = m_idx_tab(Kt, t, 1e-5)
    t_idx = [i-1 for i in t_idx]
    
    for i in t_idx[:-1]:
        xkpp = x_exp_pp(w,t,alpha,c)
        xkp = x_exp_p(w,t,alpha,c)
        xk = x_exp(w,t,alpha,c)
            
        try:
            u0 = root_brentq_p(xkpp,t[i],t[i+1],eps)
            if xkp(u0) == 0:
                xk0, m0 = xk(u0, return_m=True)
                support += [xk0, xk0]
                u += [u0, u0]
                mldm += [m0, m0]
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
                    if verbose:
                        print(u1,xk(u1),xkp(u1))
                        print(u2,xk(u2),xkp(u2))
                except ValueError:
                    if verbose:
                        print("Error",t[i],t[i+1])
        except ValueError:
            if verbose:
                print("Error:",t[i],t[i+1])
        # otherwise, no spectral gap
        
    # for the left exterior intervals, ]-infty,tau0[ -> in fact it is ]0, tau0[, for k == M if c<1
    if c<1:
        xkp = x_exp_p(w,t,alpha,c)
        xk = x_exp(w,t,alpha,c)
        try:
            u0 = root_brentq_p(xkp,0,t[0],eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error",0,t[0])
    else:
        xkpp = x_exp_pp(w,t,alpha,c)
        xkp = x_exp_p(w,t,alpha,c)
        xk = x_exp(w,t,alpha,c)
        
        try:
            u0 = root_brentq_l(xkp, 0, t[0], eps)
            xk0, m0 = xk(u0, return_m=True)
            support += [0, xk0]
            u += [0, u0]
            mldm += [0, m0]
            if verbose:
                print(u0,xk(u0),xkp(u0))
        except ValueError:
            if verbose:
                print("Error:",-np.inf,t[0])
    
    # for the right exterior intervals, ]tau_p,infty[, for k == M
    xkpp = x_exp_pp(w,t,alpha,c)
    xkp = x_exp_p(w,t,alpha,c)
    xk = x_exp(w,t,alpha,c)
    
    try:
        u0 = root_brentq_r(xkp, t[-1], t[-1] + 1, eps)
        xk0, m0 = xk(u0, return_m=True)
        support += [xk0]
        u += [u0]
        mldm += [m0]
        if verbose:
            print(u0,xk(u0),xkp(u0))
    except ValueError:
        if verbose:
            print("Error:",t[-1],np.inf)
    
    if len(support) != 0:
        support, u, mldm = map(list, zip(*sorted(zip(support, u, mldm))))
        
        support = np.array(support[1:])
        u = np.array(u[1:])
        mldm = np.array(mldm[1:])
        
        if support.shape[0] % 2 != 0:
            beta = alpha/(1-np.exp(-alpha))
            support = np.concatenate([support, [t[-1]*beta*(1+np.sqrt(c))**2]])
            u = np.concatenate([u, [t[-1]+1]])
            mldm = np.concatenate([mldm, [beta+1]])
            
            nu = support.shape[0]//2
            tildeomegai = np.ones(nu)/nu
            return support, u, mldm, tildeomegai
        
        nu = support.shape[0]//2
        tildeomegai = np.ones(nu)
        for i in range(nu):
            tildeomegai[i] = w[(t>u[2*i]) & (t<u[2*i+1])].sum()
        if tildeomegai.sum() > 0:
            tildeomegai = tildeomegai/tildeomegai.sum()
        return support, u, mldm, tildeomegai
    else:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
