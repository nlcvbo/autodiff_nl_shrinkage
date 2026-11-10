import numpy as np
from scipy.optimize import minimize, root_scalar

def m_idx_tab(K, d, eps):
    M = d.shape[0]
    d_diff = np.diff(d, axis=0)
    if K is None or K <= 0:
        idx = np.ones(d.shape, dtype=int).cumsum(axis=0)-1
        return (idx[:-1][d_diff > eps]+1).tolist() + [M]
    
    d_idx = np.argsort(d_diff, axis=0)
    if K >= 2:
        rst = (d_idx[-K+1:][d_diff[d_idx[-K+1:]] > eps]+1).tolist() + [M]
    else:
        rst = [M]
    return rst

def line_search_p(f,a,b,eps=1e-1):
    max_iter = 1e3
    n_iter = 0
    ur = (a+b)/2
    ul = (a+b)/2
    while f(ul) < 0 and n_iter < max_iter:
        ul  = a + (ul-a)/(1+eps)
        n_iter += 1
    
    while f(ur) > 0 and n_iter < max_iter:
        ur  = b - (b-ur)/(1+eps)
        n_iter += 1
    
    return ur, ul, n_iter < max_iter

def line_search_m(f,a,b,eps=1e-1):
    max_iter = 1e3
    
    um = (a+b)/2
    n_iter0 = 0
    while not np.isfinite(f(um)) and n_iter0 < max_iter:
        um = b - (b-um)/(1+eps)
        n_iter0 += 1
    um = (a+b)/2
    n_iter0 = 0
    while not np.isfinite(f(um)) and n_iter0 < max_iter:
        um = a + (um-a)/(1+eps)
        n_iter0 += 1
        
    n_iter1 = 0
    n_iter2 = 0
    ur = um
    ul = um
    while f(ul) > 0 and n_iter1 < max_iter:
        ul  = a + (ul-a)/(1+eps)
        n_iter1 += 1
    
    while f(ur) < 0 and n_iter2 < max_iter:
        ur  = b - (b-ur)/(1+eps)
        n_iter2 += 1
    return ur, ul, (n_iter1 < max_iter) and (n_iter2 < max_iter)

def line_search_r(f,a,b,eps=1e-1):
    max_iter = 1e3
    um = (a+b)/2
    n_iter0 = 0
    while not np.isfinite(f(um)) and n_iter0 < max_iter:
        um  = a + (um-a)*(1+eps)
        n_iter0 += 1
        
    ur = um
    ul = um
    
    n_iter1 = 0
    while f(ul) > 0 and n_iter1 < max_iter:
        ul  = a + (ul-a)/(1+eps)
        n_iter1 += 1
    
    n_iter2 = 0
    while f(ur) < 0 and n_iter2 < max_iter:
        ur = a + (ur- a)*(1+eps)
        n_iter2 += 1
    
    return ur, ul, (n_iter1 < max_iter) and (n_iter2 < max_iter)

def line_search_l(f,a,b,eps=1e-1):
    max_iter = 1e3
    um = (a+b)/2
    n_iter0 = 0
    while not np.isfinite(f(um)) and n_iter0 < max_iter:
        um = b - (b-um)*(1+eps)
        n_iter0 += 1
        
    ur = um
    ul = um
    
    n_iter1 = 0
    while f(ul) < 0 and n_iter1 < max_iter:
        ul = b - (b-ul)*(1+eps)
        n_iter1 += 1
    
    n_iter2 = 0
    while f(ur) > 0 and n_iter2 < max_iter:
        ur = b - (b-ur)/(1+eps)
        n_iter2 += 1
            
    return ur, ul, (n_iter1 < max_iter) and (n_iter2 < max_iter)

def root_brentq_p(f,a,b,eps):
    ur, ul, success = line_search_p(f,a,b,eps)
    try:
        u0 = root_scalar(f, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

def root_brentq_m(f,a,b,eps):
    ur, ul, success = line_search_m(f,a,b,eps)
    try:
        u0 = root_scalar(f, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

def root_brentq_r(f,a,b,eps):
    ur, ul, success = line_search_r(f,a,b,eps)
    try:
        u0 = root_scalar(f, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

def root_brentq_l(f,a,b,eps):
    ur, ul, success = line_search_l(f,a,b,eps)
    try:
        u0 = root_scalar(f, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

def root_brentq_lm(f,a,b,eps):
    g = lambda x: -f(x)
    ur, ul, success = line_search_l(g,a,b,eps)
    try:
        u0 = root_scalar(g, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0

def root_brentq_rm(f,a,b,eps):
    g = lambda x: -f(x)
    ur, ul, success = line_search_r(g,a,b,eps)
    try:
        u0 = root_scalar(g, bracket=[ul,ur], method='brentq').root
    except ValueError:
        raise ValueError
    return u0
