import torch
import scipy as sp
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from nl_utils import m_X_w_ewma, m_X_w_unif, m_X_w
import time

def nl_cov_shrinkage(lambda_, t, wt, d, wd, c = 0.1, weights = 'ewma', w_args = None, method = 'root', verbose = True):
    lambda_ = np.sort(lambda_)
    p = lambda_.shape[0]
    n = int(np.ceil(p/c))
    
    if c > 1:
        if weights == "ewma":
            alpha = w_args[0]
            m, X0 = m_X_w_ewma(torch.zeros(1), alpha, c, t, wt)
        elif weights == "unif":
            alpha = w_args[0]
            m, X0 = m_X_w_unif(torch.zeros(1), alpha, c, t, wt)
        elif weights == "Ndiracs":
            m, X0 = m_X_w(torch.zeros(1), d, wd, c, t, wt)
        else:
            m, X0 = m_X_w(torch.zeros(1), d, wd, c, t, wt)
            
        lambda_p = lambda_[lambda_ > 0][-n:]
        h_lambda = 1/(c-1)/X0*torch.ones(lambda_.shape)
    else:
        lambda_p = lambda_[lambda_ > 0]
        h_lambda = torch.zeros(lambda_.shape)
        
    if weights == "ewma":
        alpha = w_args[0]
        m, X_lambda = m_X_w_ewma(lambda_p, alpha, c, t, wt)
    elif weights == "unif":
        alpha = w_args[0]
        m, X_lambda = m_X_w_unif(lambda_p, alpha, c, t, wt)
    elif weights == "Ndiracs":
        m, X_lambda = m_X_w(lambda_p, d, wd, c, t, wt)
    else:
        m, X_lambda = m_X_w(lambda_p, d, wd, c, t, wt)
    
    numerator = (wt[None,:]*t[None,:]**2/torch.abs(t[None,:]*X_lambda[:,None] + 1)**2).sum(axis=1)
    denominator = (wt[None,:]*t[None,:]/torch.abs(t[None,:]*X_lambda[:,None] + 1)**2).sum(axis=1)
    h_lambda[-lambda_p.shape[0]:] = numerator/denominator
    
    return h_lambda

def nl_prec_shrinkage(lambda_, t, wt, d, wd, c = 0.1, weights = 'ewma', w_args = None, method = 'root', verbose = True):
    lambda_ = np.sort(lambda_)
    p = lambda_.shape[0]
    n = int(np.ceil(p/c))
    
    if c > 1:
        if weights == "ewma":
            alpha = w_args[0]
            m, X0 = m_X_w_ewma(torch.zeros(1), alpha, c, t, wt)
        elif weights == "unif":
            alpha = w_args[0]
            m, X0 = m_X_w_unif(torch.zeros(1), alpha, c, t, wt)
        elif weights == "Ndiracs":
            m, X0 = m_X_w(torch.zeros(1), d, wd, c, t, wt)
        else:
            m, X0 = m_X_w(torch.zeros(1), d, wd, c, t, wt)
            
        lambda_p = lambda_[lambda_ > 0][-n:]
        t_lambda = (-X0 + c/(c-1)*(wt/t).sum())*torch.ones(lambda_.shape)
    else:
        lambda_p = lambda_[lambda_ > 0]
        t_lambda = torch.zeros(lambda_.shape)
        
    if weights == "ewma":
        alpha = w_args[0]
        m, X_lambda = m_X_w_ewma(lambda_p, alpha, c, t, wt)
    elif weights == "unif":
        alpha = w_args[0]
        m, X_lambda = m_X_w_unif(lambda_p, alpha, c, t, wt)
    elif weights == "Ndiracs":
        m, X_lambda = m_X_w(lambda_p, d, wd, c, t, wt)
    else:
        m, X_lambda = m_X_w(lambda_p, d, wd, c, t, wt)
    
    numerator = (wt[None,:]/torch.abs(t[None,:]*X_lambda[:,None] + 1)**2).sum(axis=1)
    denominator = (wt[None,:]*t[None,:]/torch.abs(t[None,:]*X_lambda[:,None] + 1)**2).sum(axis=1)
    t_lambda[-lambda_p.shape[0]:] = numerator/denominator
    
    return t_lambda
