import torch
import scipy as sp
import numpy as np
from WeSpeR_support_identification_unif import find_support_unif_u, mld_unif, mld_unifp
from WeSpeR_support_identification_ewma import find_support_exp_u, mld_exp, mld_expp
from WeSpeR_support_identification_Ndiracs import find_support_Ndiracs_u, mld_Ndiracs, mld_Ndiracsp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

def m_tilde_loss_w(ab, x, w, ww, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    u = 1 + c*w[None,:]*(wt[None,:]*(t[None,:]*a[:,None]-x[:,None])*t[None,:]/((t[None,:]*a[:,None]-x[:,None])**2+t[None,:]**2*(-torch.exp(b)[:,None])**2)).sum(axis=1)[:,None]
    v = - c*w[None,:]*(wt[None,:]*t[None,:]**2*(-torch.exp(b)[:,None])/((t[None,:]*a[:,None]-x[:,None])**2+t[None,:]**2*(-torch.exp(b)[:,None])**2)).sum(axis=1)[:,None]
    loss = (a - (w[None,:]*ww[None,:]*u/(u**2+v**2)).sum(axis=1))**2 + (-torch.exp(b) + (w[None,:]*ww[None,:]*v/(u**2+v**2)).sum(axis=1))**2
    return loss.mean().item()

def X_loss_w(ab, x, w, ww, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    X = a + np.exp(b)*1j
    loss = torch.abs(X + (ww[None,:]*w[None,:]/(x[:,None] - c*w[None,:]*(wt[None,:]*t[None,:]/(1 + t[None,:]*X[:,None])).sum(axis=1)[:,None])).sum(axis=1))**2
    return loss.mean().item()

def m_X_w(x, w, ww, c, t, wt):
    X = torch.zeros(x.shape, dtype=torch.cfloat)
    m_tilde = torch.zeros(x.shape, dtype=torch.cfloat)
    x_last = np.zeros(2)
    for i in range(x.shape[0]):
        x0 = x_last
        if x[i] != 0:
            m = minimize(m_tilde_loss_w, x0 = x0, args = (x[i:i+1], w, ww, c, t, wt), options={'disp':False})
            if m.fun < 1e-4:
                ab = torch.from_numpy(m.x)
                x_last = m.x
            else:
                ab = torch.tensor([0,-1e1], dtype=torch.float)
                x_last = np.zeros(2)
            m_tilde[i] = ab[0] - torch.exp(ab[1])*1j
            X[i] = -m_tilde[i]/x[i]
        else:
            z = 1e-5*1j*torch.ones(1, dtype=torch.cfloat)
            m = minimize(X_loss_w, x0 = np.zeros(2), args = (z, w, ww, c, t, wt), options={'disp':False})
            x0 = m.x[:1]
            m = minimize(X_loss_w, x0 = x0, args = (x[i:i+1], w, ww, c, t, wt), options={'disp':False})
            ab = torch.from_numpy(m.x)
            x_last = np.zeros(2)
            m_tilde[i] = 0
            X[i] = ab[0]
    m = torch.zeros(x.shape, dtype=torch.cfloat)
    m[m_tilde != 0] = (wt[None,:]/(t[None,:]*m_tilde[:,None] - x[:,None]).to(torch.cfloat)[m_tilde != 0]).sum(axis=1)
    return m, X

def m_tilde_loss_w_ewma(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    m_tilde = a - torch.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*m_tilde[:,None]-x[:,None])).sum(axis=1)
    integral = mld_exp(-1/theta, alpha)/theta 
    loss = torch.abs(m_tilde - integral)**2
    return loss.mean().item()

def X_loss_w_ewma(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    X = a + np.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*X[:,None] + 1)).sum(axis=1)
    integral = mld_exp(x/theta, alpha)/theta 
    loss = torch.abs(X - integral)**2
    return loss.mean().item()

def m_X_w_ewma(x, alpha, c, t, wt):
    X = torch.zeros(x.shape, dtype=torch.cfloat)
    m_tilde = torch.zeros(x.shape, dtype=torch.cfloat)
    x_last = np.zeros(2)
    for i in range(x.shape[0]):
        x0 = x_last
        if x[i] != 0:
            m = minimize(m_tilde_loss_w_ewma, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            if m.fun < 1e-4:
                ab = torch.from_numpy(m.x)
                x_last = m.x
            else:
                ab = torch.tensor([0,-1e1], dtype=torch.float)
                x_last = np.zeros(2)
            m_tilde[i] = ab[0] - torch.exp(ab[1])*1j
            X[i] = -m_tilde[i]/x[i]
        else:
            z = 1e-3*1j*torch.ones(1, dtype=torch.cfloat)
            m = minimize(X_loss_w_ewma, x0 = np.array([float((wt*t).sum()), 0.0]), args = (z, alpha, c, t, wt), options={'disp':False})
            x0 = m.x[:1]
            m = minimize(X_loss_w_ewma, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            ab = torch.from_numpy(m.x)
            x_last = np.zeros(2)
            m_tilde[i] = 0
            X[i] = ab[0]
    m = torch.zeros(x.shape, dtype=torch.cfloat)
    m[m_tilde != 0] = (wt[None,:]/(t[None,:]*m_tilde[:,None] - x[:,None]).to(torch.cfloat)[m_tilde != 0]).sum(axis=1)
    return m, X

def m_tilde_loss_w_unif(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    m_tilde = a - torch.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*m_tilde[:,None]-x[:,None])).sum(axis=1)
    integral = mld_unif(-1/theta, alpha)/theta 
    loss = torch.abs(m_tilde - integral)**2
    return loss.mean().item()

def X_loss_w_unif(ab, x, alpha, c, t, wt):
    ab = torch.from_numpy(ab)
    a = ab[:ab.shape[0]//2]
    b = ab[ab.shape[0]//2:]
    X = a + np.exp(b)*1j
    theta = c*(wt[None,:]*t[None,:]/(t[None,:]*X[:,None] + 1)).sum(axis=1)
    integral = mld_unif(x/theta, alpha)/theta 
    loss = torch.abs(X - integral)**2
    return loss.mean().item()

def m_X_w_unif(x, alpha, c, t, wt):
    X = torch.zeros(x.shape, dtype=torch.cfloat)
    m_tilde = torch.zeros(x.shape, dtype=torch.cfloat)
    x_last = np.zeros(2)
    for i in range(x.shape[0]):
        x0 = x_last
        if x[i] != 0:
            m = minimize(m_tilde_loss_w_unif, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            if m.fun < 1e-4:
                ab = torch.from_numpy(m.x)
                x_last = m.x
            else:
                ab = torch.tensor([0,-1e1], dtype=torch.float)
                x_last = np.zeros(2)
            m_tilde[i] = ab[0] - torch.exp(ab[1])*1j
            X[i] = -m_tilde[i]/x[i]
        else:
            z = 1e-5*1j*torch.ones(1, dtype=torch.cfloat)
            m = minimize(X_loss_w_unif, x0 = np.zeros(2), args = (z, alpha, c, t, wt), options={'disp':False})
            x0 = m.x[:1]
            m = minimize(X_loss_w_unif, x0 = x0, args = (x[i:i+1], alpha, c, t, wt), options={'disp':False})
            ab = torch.from_numpy(m.x)
            x_last = np.zeros(2)
            m_tilde[i] = 0
            X[i] = ab[0]
    m = torch.zeros(x.shape, dtype=torch.cfloat)
    m[m_tilde != 0] = (wt[None,:]/(t[None,:]*m_tilde[:,None] - x[:,None]).to(torch.cfloat)[m_tilde != 0]).sum(axis=1)
    return m, X

def WeSpeR_support_u(tau, wtau, d, wd, c, weights, w_args, method = None, verbose = True):
    t = np.unique(tau)
    wt = np.zeros(t.shape)
    for i in range(len(t)):
        wt[i] = wtau[tau == t[i]].sum()
    
    if weights == "ewma":
        alpha = w_args[0]
        Kt = w_args[1]
        support, u, mldm, tildeomegai  = find_support_exp_u(c, alpha, t, wt, Kt = Kt, eps=1e-1, method=method, verbose = verbose)
        
    elif weights == "unif":
        alpha = w_args[0]
        Kt = w_args[1]
        support, u, mldm, tildeomegai  = find_support_unif_u(c, alpha, t, wt, Kt = Kt, eps=1e-1, method=method, verbose = verbose)
        
    elif weights == "Ndiracs":
        Kd = w_args[0]
        Kt = w_args[1]
        try:
            assert wd.shape == d.shape
        except:
            wd = np.ones(d.shape[0])/d.shape[0]
        support, u, mldm, tildeomegai = find_support_Ndiracs_u(c, wd, d, t, wt, Kt = Kt, Kd = Kd, eps=1e-2, method=method, verbose = verbose)
    elif weights == "test":
        support, u, mldm, tildeomegai = np.array([0.1, 10]), \
                                    np.array([-1/tau[0]-1,-1/tau[-1]+1]), \
                                    np.array([1-1/(-1/tau[0]-1),1-1/(-1/tau[-1]+1)]), \
                                    np.ones(1)
                                    
    return torch.from_numpy(support), torch.from_numpy(u), torch.from_numpy(mldm), torch.from_numpy(tildeomegai)

def WeSpeR_density(omega, support, d, wd, c, tau, tildeomegai, mu=0.1, wt = None, weights = 'test', w_args = None, verbose = True):
    if support.shape[0] % 2 != 0:
        support = np.concatenate([support, [tau[-1]*d[-1]*(1+np.sqrt(c))**2]])
        
    nu = support.shape[0]//2
    
    mu = max(0,min(mu,1))
    
    # Number of points
    omegai = (mu/nu + (1-mu)*tildeomegai)*omega
    
    # Discretization
    if omegai.sum() > 0:
        omegai = torch.round(omegai*omega/omegai.sum()+0.5).to(int)
    else:
        omegai = torch.round(omega*torch.ones(nu)/nu+0.5).to(int)
    
    try:
        assert wt.shape == tau.shape
    except:
        wt = torch.ones(tau.shape[0])/tau.shape[0]
    
    # Grid
    grid = torch.zeros(0)
    density = torch.zeros(0)
    X = torch.zeros(0)
    i_idx = torch.zeros(0, dtype=int)
    j_idx = torch.zeros(0, dtype=int)
    sin2 = torch.zeros(0)
    for i in range(nu):
        ul = support[2*i]
        ur = support[2*i+1]
        j = torch.ones(omegai[i]).cumsum(axis=0)
        xij = ul + (ur-ul)*torch.sin(np.pi*j/2/(omegai[i]+1))**2
        
        if weights == "ewma":
            alpha = w_args[0]
            m, Xij = m_X_w_ewma(xij, alpha, c, tau, wt)
            
        elif weights == "unif":
            alpha = w_args[0]
            m, Xij = m_X_w_unif(xij, alpha, c, tau, wt)
            
        elif weights == "Ndiracs":
            m, Xij = m_X_w(xij, d, wd, c, tau, wt)
            
        else:
            m, Xij = m_X_w(xij, d, wd, c, tau, wt)
            
        f = torch.abs(m.imag)/np.pi
        
        grid = torch.concatenate([grid, ul*torch.ones(1), xij, ur*torch.ones(1)])
        density = torch.concatenate([density, torch.zeros(1), f, torch.zeros(1)])
        X = torch.concatenate([X, torch.zeros(1), Xij, torch.zeros(1)])
        j_idx = torch.concatenate([j_idx, torch.ones(omegai[i]+2, dtype=int).cumsum(axis=0) - 1])
        i_idx = torch.concatenate([i_idx, 2*i*torch.ones(omegai[i]+2, dtype=int)])
        sin2 = torch.concatenate([sin2,  torch.zeros(1), torch.sin(np.pi*j/2/(omegai[i]+1))**2, torch.ones(1)])
    
    xi = grid.to(tau.dtype)
    f = density.to(tau.dtype)
    X = X
    
    F0 = max(0,1-1/c)
    G = torch.zeros(xi.shape, dtype=tau.dtype)
    F = torch.zeros(xi.shape, dtype=tau.dtype)
    G[0] = F0
    F[0] = F0
    for i in range(nu):
        j = torch.ones(omegai[i]+1, dtype=int).cumsum(0)
        offset = omegai[:i].sum(axis=0) + 2*i
        if offset > 0:
            G[offset] = F[offset-1]
            F[offset] = F[offset-1]
        G[j+offset] = G[offset] + 1/2*((xi[j+offset] - xi[j-1+offset])*(f[j+offset] + f[j-1+offset])).cumsum(axis=0)
        if tildeomegai.sum(axis=0) > 0:
            if (G[offset+j][-1] - F[offset]).to(float) > 0:
                F[j+offset] = F[offset] + tildeomegai[i]*(1-F[0])*(G[offset+j] - F[offset])/(G[offset+j][-1] - F[offset])
            else:
                F[j+offset] = F[offset] + tildeomegai[i]*(1-F[0])*j/j[-1]
        else:
            F[j+offset] = G[j+offset]
    if tildeomegai.sum(axis=0) == 0:
        F = G/G[-1]
    return xi, f, F, G, X, omegai, i_idx, j_idx, sin2, nu

class f_xi(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, wt = None, d = torch.ones(1), wd = torch.ones(1), c = 0.1, mu = 0.1, weights = 'test', w_args = None, omega = 100, method = 'root', verbose = True):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        tau = input
        
        if d == None:
            d = torch.ones(1)
        if wd == None or wd.shape != d.shape:
            wd = torch.ones(d.shape[0])/d.shape[0]
        if wt == None:
            wt = torch.ones(tau.shape[0])/tau.shape[0]
        
        
        
        support, u, mldm, tildeomegai = WeSpeR_support_u(tau.detach().numpy(), wt.numpy(), d.numpy(), wd.numpy(), c, weights, w_args, method = method, verbose = verbose)
        
        xi, f, F, G, X, omegai, i_idx, j_idx, sin2, nu = WeSpeR_density(omega, support, d, wd, c, tau, tildeomegai, mu=mu, wt = wt, weights = weights, w_args = w_args, verbose = verbose)
        
        output = torch.concatenate([f, F, xi, omegai, nu*torch.ones(1)])
        ctx.save_for_backward(input, output)
        ctx.wt = wt
        ctx.d = d
        ctx.wd = wd
        ctx.c = c
        ctx.weights = weights
        ctx.w_args = w_args
        ctx.omega = omega
        ctx.u = u
        ctx.mldm = mldm
        ctx.tildeomegai = tildeomegai
        ctx.omegai = omegai
        ctx.i_idx = i_idx
        ctx.j_idx = j_idx
        ctx.sin2 = sin2
        ctx.X = X
        ctx.xi = xi
        ctx.f = f
        ctx.G = G
        ctx.F = F
        ctx.nu = nu
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, output = ctx.saved_tensors
        tau = input
        wt = ctx.wt
        d = ctx.d
        wd = ctx.wd
        c = ctx.c
        weights = ctx.weights
        w_args = ctx.w_args
        omega = ctx.omega
        u = ctx.u
        mldm = ctx.mldm
        omegai = ctx.omegai
        tildeomegai = ctx.tildeomegai
        i_idx = ctx.i_idx
        j_idx = ctx.j_idx
        sin2 = ctx.sin2
        X = ctx.X
        f = ctx.f
        xi = ctx.xi        
        nu = ctx.nu
        G = ctx.G
        F = ctx.F
            
        tu = c*(wt[None,:]*tau[None,:]/(tau[None,:]-u[:,None])).sum(axis=1)
        theta = c*(tau[None,:]*wt[None,:]/(tau[None,:]*X[:,None]+1)).sum(axis=1)
        
        if weights == "ewma":
            alpha = w_args[0]
            mldmp = mld_expp(mldm,alpha)
            integral1 = (mld_exp(xi/theta,alpha)+xi/theta*mld_expp(xi/theta,alpha))/theta**2
            integral3 = mld_expp(xi/theta,alpha)/theta**2
            
        elif weights == "unif":
            alpha = w_args[0]
            mldmp = mld_unifp(mldm,alpha)
            integral1 = (mld_unif(xi/theta,alpha)+xi/theta*mld_unifp(xi/theta,alpha))/theta**2
            integral3 = mld_unifp(xi/theta,alpha)/theta**2
            
        elif weights == "Ndiracs":
            mldmp = mld_Ndiracsp(mldm, wd, d)
            integral1 = (mld_Ndiracs(xi/theta, wd, d)+xi/theta*mld_Ndiracsp(xi/theta, wd, d))/theta**2
            integral3 = mld_Ndiracsp(xi/theta, wd, d)/theta**2
            
        else:
            mldmp = (wd[None,:]*d[None,:]/(d[None,:]-mldm[:,None])**2).sum(axis=1)
            integral1 = (wd[None,:]*d[None,:]**2/(xi[:,None] - d[None,:]*theta[None,:])**2).sum(axis=1)
            integral3 = (wd[None,:]*d[None,:]/(xi[:,None] - d[None,:]*theta[None,:])**2).sum(axis=1)
                
        dsidtk = c*u[:,None]**2*wt[None,:]/(tau[None,:] - u[:,None])**2*(mldm[:,None] + tu[:,None]/mldmp[:,None])
        
        dxiidtk = (1-sin2[:,None])*dsidtk[i_idx,:] + sin2[:,None]*dsidtk[i_idx+1,:]
        
        integral2 = c*(wt[None,:]*tau[None,:]**2/(tau[None,:]*X[:,None]+1)**2).sum(axis=1)
        dhatGammaidtk = c*wt[None,:]/(tau[None,:]*X[:,None]+1)**2*integral1[:,None]
        dhatGammaidXi = 1 - integral2*integral1
        dhatGammaidxii = -integral3
        dhatXidtk = - dhatGammaidtk/dhatGammaidXi[:,None]
        dXidxii = - dhatGammaidxii/dhatGammaidXi
        dXidtk = dhatXidtk + dXidxii[:,None]*dxiidtk
        
        dXidtk[torch.logical_not(torch.isfinite(dXidtk))] = 0
        
        integral4 = (tau[None,:]*wt[None,:]/(tau[None,:]*X[:,None]+1)**2).sum(axis=1)
        integral5 = (wt[None,:]/(tau[None,:]*X[:,None]+1)).sum(axis=1)
        dtildefidXi = integral4/xi
        dtildefidxii = integral5/xi**2
        dtildefidtk = wt[None,:]*X[:,None]/(tau[None,:]*X[:,None]+1)**2/xi[:,None]
        dfidtk = (dtildefidXi[:,None]*dXidtk + dtildefidxii[:,None]*dxiidtk + dtildefidtk).imag/np.pi
        
        dfidtk[torch.logical_not(torch.isfinite(dfidtk))] = 0
        
        dGidtk = torch.zeros(dxiidtk.shape, dtype=dxiidtk.dtype)
        dFidtk = torch.zeros(dxiidtk.shape, dtype=dxiidtk.dtype)
        for i in range(nu):
            j = torch.ones(omegai[i]+1, dtype=int).cumsum(0)
            offset = omegai[:i].sum(axis=0) + 2*i
            dGidtk[j+offset] =  1/2*((dxiidtk[j+offset,:] - dxiidtk[j-1+offset,:])*(f[j+offset,None] + f[j-1+offset,None])).cumsum(axis=0) + 1/2*((xi[j+offset,None] - xi[j-1+offset,None])*(dfidtk[j+offset,:] + dfidtk[j-1+offset,:])).cumsum(axis=0)
            if tildeomegai.sum(axis=0) > 0:
                if (G[offset+j][-1] - F[offset]).to(float) > 0:
                    dFidtk[j+offset] = tildeomegai[i]*(1-F[0])/(G[offset+j[-1]] - F[offset])*dGidtk[j+offset,:] - tildeomegai[i]*(1-F[0])/(G[offset+j[-1]] - F[offset])**2*(G[offset+j,None] - F[offset])*dGidtk[offset+j[-1:],:]
                else:
                    dFidtk[j+offset] = 0
                dFidtk[j[-1]+offset] = 0
            else:
                dFidtk[j+offset] = dGidtk[j+offset]
        if tildeomegai.sum(axis=0) == 0:
            dFidtk = dGidtk/G[-1] - G[:,None]/G[-1]**2*dGidtk[-1:,:]
        
        domegaidtk = torch.zeros((omegai.shape[0]+1, wt.shape[0]))
        doutidtk = torch.concatenate([dfidtk, dFidtk, dxiidtk, domegaidtk], axis=0)
        
        return grad_output[:,None] * doutidtk, None, None, None, None, None, None, None, None, None, None

class WeSpeR_MD_model(torch.nn.Module):
    def __init__(self, tau_init, wt = None, d = torch.ones(1), wd = torch.ones(1), c = 0.1, mu = 0.1, weights = 'test', w_args = None, omega = 100, method = 'root', verbose = True):
        """
            Inputs:
                tau_init: float tensor of shape (p), initial guess of population spectrum
                d: float tensor of sahpe (n), weight vector (the diagonal of the weight matrix)
        """
        super().__init__()
        self.tau_mean = tau_init.clone().mean(axis=0)
        self.logtau = torch.nn.Parameter(torch.log(tau_init.clone()/self.tau_mean).type(torch.float64))
        self.wt = wt
        self.d = d
        self.wd = wd
        self.c = c
        self.mu = mu
        self.weights = weights
        self.w_args = w_args
        self.omega = omega
        self.method = method
        self.verbose = verbose
    
    def get_tau(self):
        return torch.exp(self.logtau)*self.tau_mean
        
    def forward(self):
        """
            Inputs:
            Outputs:
                
        """    
        F = f_xi.apply
        idx = torch.argsort(self.logtau)
        if self.wt is None:
            output = F(torch.exp(self.logtau[idx])*self.tau_mean, self.wt, self.d, self.wd, self.c, self.mu,self.weights, self.w_args, self.omega, self.method, self.verbose) 
        else:
            output = F(torch.exp(self.logtau[idx])*self.tau_mean, self.wt[idx], self.d, self.wd, self.c, self.mu,self.weights, self.w_args, self.omega, self.method, self.verbose) 
        return output


def loss_wasserstein_1D_cdf(p=2, regu1 = 1, regu2 = 1):
    def loss_wasserstein_p(u_values, v_values, u_cumul = None, v_cumul = None):
        # Inspired from Scipy 1D Wasserstein distance: https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L9733-L9807https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L9733-L9807
        
        if u_cumul is None:
            u_cumul = torch.ones(u_values.shape).cumsum(axis=0)-1
            u_cumul = u_cumul / u_cumul[-1]
        if v_cumul is None:
            v_cumul = torch.ones(v_values.shape).cumsum(axis=0)-1
            v_cumul = v_cumul / v_cumul[-1]
        
        u_weights = torch.diff(torch.cat((u_cumul[:1],u_cumul)))
        v_weights = torch.diff(torch.cat((v_cumul[:1],v_cumul)))
        
        u_sorter = torch.argsort(u_values, axis=0)
        v_sorter = torch.argsort(v_values, axis=0)
        
        all_values = torch.cat((u_values, v_values), axis=0)
        all_values = all_values.sort(axis=0)[0]
        
        # Compute the differences between pairs of successive values of u and v.
        deltas = torch.diff(all_values, axis=0)
        
        # Get the respective positions of the values of u and v among the values of both distributions.
        u_values_sort = u_values[u_sorter].contiguous()
        v_values_sort = v_values[v_sorter].contiguous()
        
        u_cdf_indices = torch.searchsorted(all_values[:-1].contiguous(), u_values_sort, right = True)
        v_cdf_indices = torch.searchsorted(all_values[:-1].contiguous(), v_values_sort, right = True)
        
        # Calculate the CDFs of u and v using their weights, if specified.
        u_cdf = torch.zeros(all_values.shape, dtype=torch.float)
        v_cdf = torch.zeros(all_values.shape, dtype=torch.float)
        
        u_cdf[u_cdf_indices] = u_weights[u_sorter].to(torch.float)
        u_cdf = u_cdf.cumsum(axis=0)
        
        v_cdf[v_cdf_indices] = v_weights[v_sorter].to(torch.float)
        v_cdf = v_cdf.cumsum(axis=0)
        
        
        # We do not normalize the power by 1/p at the end, to make it differentiable
        distance1 = ((torch.abs(u_cdf - v_cdf)[:-1]**p * deltas)).sum(axis=0)
        
        distance2 = regu1*((u_values*u_weights).sum(axis=0) - (v_values*v_weights).sum(axis=0))**2 
        
        if u_values.shape[0] > 1:
            u_max = torch.max(u_values[u_weights > 0])
            u_min = torch.min(u_values[u_weights > 0])
            q_max = u_cumul[-2]
            q_min = u_cumul[2]
            v_max_idx = v_cumul >= q_max
            v_min_idx = v_cumul <= q_min
            v_max = torch.max(v_values[v_weights > 0])
            v_min = torch.min(v_values[v_weights > 0])
            # if v_values[v_max_idx].shape[0] > 0:
            #     v_max = v_values[v_max_idx].mean()
            # else:
            #     v_max = v_values[-1]
            # if v_values[v_min_idx].shape[0] > 0:
            #     v_min = v_values[v_min_idx].mean()
            # else:
            #     v_min = v_values[0]
            
            distance3 = regu2*((v_min - u_min)**2 + (v_max - u_max)**2)
        else:
            distance3 = 0
        return (distance1 + distance2 + distance3).mean()
    
    return loss_wasserstein_p

def loss_LW(regu1 = 0., regu2 = 0.):
    def loss_fn(lambda_, F, xi, c, omegai):
        
        
        p = lambda_.shape[0]
        n = int(p/c)
        lambda_k = torch.zeros(p)
        X_k = 0
        X_kp = 0
        
        idx = torch.ones(F.shape[0], dtype=int).cumsum(axis=0) - 1
        omegai_cum = omegai.cumsum(axis=0)
        for kappa in range(max(p-n,0), p):
            kappap = kappa + 1
            
            j_idx = idx[torch.logical_and(F >= kappa/p, F < kappap/p)]
            
            if j_idx.shape[0] == 0:
                j_kp = max(torch.argmax(((kappap - p*F) < 0).to(int)) - 1, 0)
                j_k = max(torch.argmax(((kappap - p*F) < 0).to(int)) - 1, 0)
                i = torch.argmax(((j_k - omegai_cum) < 0).to(int))                
                ip = torch.argmax(((j_kp - omegai_cum) < 0).to(int))
                delta1 = (kappap/p - F[j_kp])*xi[j_kp] + (kappap/p - F[j_kp])**2/2/(F[j_kp+1] - F[j_kp])*(xi[j_kp+1] - xi[j_kp])
                delta2 =  -(kappa/p - F[j_k])*xi[j_k] - (kappa/p - F[j_k])**2/2/(F[j_k+1] - F[j_k])*(xi[j_k+1] - xi[j_k])
                delta3 = torch.zeros(1)
                
            else:
                j_kp = j_idx[-1]
                ip = torch.argmax(((j_kp - omegai_cum) < 0).to(int))
            
                j_k = j_idx[0]
                i = torch.argmax(((j_k - omegai_cum) < 0).to(int))
            
                if j_kp+1 < F.shape[0] and F[j_kp+1] - F[j_kp] > 0:
                    delta1 = (kappap/p - F[j_kp])*xi[j_kp] + (kappap/p - F[j_kp])**2/2/(F[j_kp+1] - F[j_kp])*(xi[j_kp+1] - xi[j_kp])
                else:
                    delta1 = torch.zeros(1)
                    
                if j_k-1 >= 0 and F[j_k] - F[j_k-1] > 0:
                    delta2 = (F[j_k] - kappa/p)*xi[j_k] - (F[j_k] - kappa/p)**2/2/(F[j_k] - F[j_k-1])*(xi[j_k] - xi[j_k-1])
                else:
                    delta2 = torch.zeros(1)
                
                if j_kp > j_k:
                    l = torch.ones(j_kp - j_k, dtype=int).cumsum(axis=0) - 1 + j_k
                    delta3 = ((F[l+1] - F[l])*(xi[l+1] + xi[l])/2).sum(axis=0)
                else:
                    delta3 = torch.zeros(1)
                
                    
            
            X_k = X_kp
            X_kp = X_k + delta1 + delta2 + delta3
            
            if kappa > 0:        
                X_kp = max(X_k + lambda_k[kappa-1]/p, X_kp)
            lambda_k[kappa] = (X_kp - X_k)*p
            
            if i != ip:
                j = torch.argmin((lambda_k[kappa] - xi)**2)
                lambda_k[kappa] = xi[j]
            
        # first penalization on the mean
        pen1 = regu1*(lambda_.mean(axis=0) - lambda_k.mean(axis=0))**2
        # second penalization on the extremums
        idx = torch.ones(lambda_.shape, dtype=int).cumsum(axis=0)-1
        weights = torch.cos(idx*np.pi/idx[-1])**2
        pen2 = regu2*(weights*(lambda_ - lambda_k)**2).mean()
            
            
        return (((lambda_ - lambda_k)**2).mean() + pen1 + pen2)/(1 + regu1 + regu2)
    return loss_fn

def loss_LWO():
    def loss_fn(lambda_, F, xi, c):
        p = lambda_.shape[0]
        n = int(p/c)
        lambda_k = torch.zeros(p)
        for kappa in range(max(p-n,0), p):
            j_k = max(torch.argmax((((kappa+1/2) - p*F) < 0).to(int)) - 1, 0)
            j_kp = max(torch.argmax((((kappa+1/2) - p*F) < 0).to(int)), 0)
            if F[j_kp] - F[j_k] > 0:
                lambda_k[kappa] = (((kappa+1/2)/p - F[j_k])*xi[j_k] + (F[j_kp] - (kappa+1/2)/p)*xi[j_kp])/(F[j_kp] - F[j_k])
            else:
                lambda_k[kappa] = (xi[j_k]+xi[j_kp])/2
        
        return ((lambda_ - lambda_k)**2).mean()
    return loss_fn

def WeSpeR_MD_minimization(lambda_, tau_init, wt = None, d = torch.ones(1), wd = torch.ones(1), c = 0.1, mu = 0.1, weights = 'test', w_args = None, omega = 100, lr = 1e-2, n_epochs = 100, decay = 1., loss_type="w2", method = 'root', save_all_tau = False, verbose = True, plot_loss = False):   
    model = WeSpeR_MD_model(tau_init, wt = wt, d = d, wd = wd, c =c, mu = mu, weights = weights, w_args = w_args, omega = omega, method = method, verbose = False)
    if loss_type == "w2":
        loss_fn = loss_wasserstein_1D_cdf(p=2, regu1=0.1, regu2=0.1) # we can also use loss_composite(p=2, regu=1)
    elif loss_type == "trap":
        loss_fn = loss_LW(regu1=0.1, regu2=1.)
    elif loss_type == "euler":
        loss_fn = loss_LWO()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    
    # Train loop
    if save_all_tau:
        all_tau = [model.get_tau().detach()]
    else:
        all_tau = []
    running_loss = []
    best_loss = np.inf
    best_tau = tau_init.type(torch.float64)
    
    eps = 1e-5
    
    model.train(True)
    for i in range(n_epochs):
        optimizer.zero_grad()
        output = model()
            
        nu = output[-1].to(int)
        omegai = output[-nu-1:-1].to(int)
        output = output[:-nu-1]
        f = output[:output.shape[0]//3]
        F = output[output.shape[0]//3:2*output.shape[0]//3]
        xi = output[2*output.shape[0]//3:]
        u_cumul = None
        v_cumul = F
        if loss_type == "w2":
            loss = loss_fn(torch.cat([torch.zeros(1),lambda_]), xi, u_cumul, v_cumul)
        elif loss_type == "trap":
            loss = loss_fn(lambda_, F, xi, c, omegai)
        elif loss_type == "euler":
            loss = loss_fn(lambda_, F, xi, c)
        loss.backward()
            
        optimizer.step()
        scheduler.step()        
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_tau = model.get_tau().clone()
        
        if save_all_tau:
            all_tau += [model.get_tau().detach()]
        running_loss += [loss.item()]
        if verbose:
            if n_epochs < 11 or i % min((n_epochs // 10),10) == 0:
                print("Loss epoch", i, ":", loss.item())
        
    if plot_loss:
        print("Final loss :", best_loss)
        plt.figure()
        plt.semilogy(np.array(running_loss))
        plt.title("Running loss in function of epochs")
        plt.xlabel("epochs")
        plt.ylabel("training loss")
        plt.show()
    
    return model, best_tau, np.array(running_loss), all_tau


def m_idx(K, d):
    M = d.shape[0]
    if K is None or K <= 0:
        return list(range(1,M+1))
    d_diff = torch.diff(d, axis=0)
    d_idx = torch.argsort(d_diff, axis=0)
    if K >= 2:
        rst = (d_idx[-K+1:]+1).tolist() + [M]
    else:
        rst = [M]
    return rst

