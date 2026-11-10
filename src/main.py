import torch
import scipy as sp
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import warnings

from utils import WeSpeR_LD_minimization
from autodiff import WeSpeR_LD
from utils import f_xi, m_idx, WeSpeR_MD_minimization

from nl_formulas import nl_cov_shrinkage, nl_prec_shrinkage
from weighted_LWO_estimator import wLWO_estimator

def WeSpeR_experiment_1(tau_init = None, p = 2000, n = 20000, plots = False):
    LD = True
    MD = True

    c = p/n
    
    t = torch.tensor([1.,3.,10.])
    wt = torch.tensor([0.2,0.4,0.4])
    Kt = 30
    
    mu = 0.1
    alpha = 5.
    weights = 'ewma'   
    method = 'root'
    Kd = 1
    
    Nd = 200
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    n_epochs_LD = 10
    n_epochs_MD = 10
    lr = 5e-2
    omega = min(400, min(p,n))
    p_tau = p
    loss_type = "trap"
    b = 1 # for LD
    verbose = True
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    
    if weights == "ewma":
        beta = alpha/(1-np.exp(-alpha))
        W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    elif weights == "unif":
        # alpha \in [0,2]
        W = alpha*(np.ones(n).cumsum()-(n+1)/2)/n + 1
    elif weights == "2diracs":
        # alpha \in (0,1)
        W = np.array([1-alpha]*(int(n*wd))+[(1-wd*(1-alpha))/(1-wd)]*(n-int(n*wd)))
    elif weights == "Ndiracs":
        W = []
        for j in range(1,d.shape[0]):
            W += [d[j]]*int(n*wd[j])
        W += [d[0]]*(n - len(W))
        W = np.array(W)
    
    # Sampling
    if n <= p:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))  
        S = X_sample.T @  (W[:, None] * X_sample)/n
        SLWO = wLWO_estimator(X_sample, W/n, assume_centered = True, S_r = None, est = False)
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")

        e_SLWO, _ = np.linalg.eigh(SLWO)
    else:
        n_left = n
        n_used = 0
        X2sum = np.zeros(n)
        S = np.zeros((p,p))
        while n_left > 0:
            n_sample = min(n_left, p)
            X_sample = np.random.normal(size = (n_sample,p)) @ np.sqrt(np.diag(tau_pop))  
            S += X_sample.T @  (W[n_used:n_used+n_sample, None] * X_sample)/n
            X2sum[n_used:n_used+n_sample] = (X_sample**2).sum(axis=1)
            n_used += n_sample
            n_left -= n_sample

        mu_S = np.trace(S @ np.eye(p))/p
        delta2 = np.linalg.norm(S-mu_S*np.eye(p), ord = 'fro')**2/p
        S2 = np.linalg.norm(S, ord = 'fro')**2/p
        beta2 = (X2sum**2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
        beta2 = beta2/(1-(W**2).sum()/n**2)
        beta2 = min(beta2, delta2)
        
        shrinkage = beta2/delta2
        SLWO = shrinkage*mu_S*np.eye(p) + (1 - shrinkage)*S
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")
    
        e_SLWO = shrinkage*mu_S + (1 - shrinkage)*lambda_

    if tau_init is None:
        if c >= 1:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        else:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        tau_init_LD = np.sort(np.random.choice(tau_init, p))
        tau_init_LD = torch.tensor(tau_init)
        tau_init_MD = np.sort(np.random.choice(tau_init, p_tau))
        tau_init_MD = torch.tensor(tau_init)
    
    # Covariance estimation
    start_time = time.time()
    h_oracle = nl_cov_shrinkage(lambda_, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    last_time = time.time()
    print("NL shrinkage time:", last_time - start_time, "s.")
    Soracle = U @ np.diag(h_oracle) @ U.T
    cov = np.diag(tau_pop)
    norm_cov = np.linalg.norm(cov, ord = 'fro')**2/p
    loss_S = np.linalg.norm(S - cov, ord = 'fro')**2/p
    loss_SLWO = np.linalg.norm(SLWO - cov, ord = 'fro')**2/p
    loss_Soracle = np.linalg.norm(Soracle - cov, ord = 'fro')**2/p
    PRIALS_oracle = 1-loss_Soracle/loss_S
    
    # Precision estimation
    prec = np.diag(1/tau_pop)
    t_oracle = nl_prec_shrinkage(lambda_, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    Poracle = U @ np.diag(t_oracle) @ U.T
    norm_prec = np.linalg.norm(prec, ord = 'fro')**2/p
    loss_P = np.linalg.norm(np.linalg.pinv(S) - prec, ord = 'fro')**2/p
    loss_PLWO = np.linalg.norm(np.linalg.pinv(SLWO) - prec, ord = 'fro')**2/p
    loss_Poracle = np.linalg.norm(Poracle - prec, ord = 'fro')**2/p
    PRIALP_oracle = 1-loss_Poracle/loss_P
    
    # Optimize tau
    if LD:
        print("WeSpeR_LD:", LD)
        start_time = time.time()
        estimator_LD = WeSpeR_LD(bias=True, assume_centered=True).fit_from_S(S, W, tau_init = tau_init_LD, method = 'Adam', n_epochs = n_epochs_LD, b = b, assume_centered = True, lr = lr, momentum=0., verbose = True)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        Snl = estimator_LD.get_covariance(d = d, wd = wd, weights = weights, w_args = w_args, method = 'root', verbose = False)
        loss_Snlb = np.linalg.norm(Snl -  np.diag(tau_pop), ord = 'fro')**2/p
        Pnl = estimator_LD.get_precision(d = d, wd = wd, weights = weights, w_args = w_args, method = 'root', verbose = False)
        loss_Pnlb = np.linalg.norm(Pnl -  np.diag(1/tau_pop), ord = 'fro')**2/p
        tau_fit_LD = estimator_LD.get_tau()
    else:
        estimator_LD = None

    if MD:
        print("WeSpeR_MD:", MD) 
        start_time = time.time()
        estimator_MD = WeSpeR_MD(bias=True, assume_centered=True).fit_from_F(torch.tensor(lambda_), U, W, tau_init = tau_init_MD, wt = None, d = d, wd = wd, c = c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs_MD, loss_type = loss_type, method = method, verbose = verbose)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        Snl = estimator_MD.get_covariance(d = d, wd = wd, weights = weights, w_args = w_args, method = 'root', verbose = False)
        loss_Snlq = np.linalg.norm(Snl -  np.diag(tau_pop), ord = 'fro')**2/p
        Pnl = estimator_MD.get_precision(d = d, wd = wd, weights = weights, w_args = w_args, method = 'root', verbose = False)
        loss_Pnlq = np.linalg.norm(Pnl -  np.diag(1/tau_pop), ord = 'fro')**2/p
        tau_fit_MD = estimator_MD.get_tau()
    else:
        estimator_MD = None
    
    print("Norm cov:       \t", np.around(norm_cov,3))
    print("Loss/PRIAL S:   \t", np.around(loss_S,3), "\t|\t", 0)
    print("Loss/PRIAL SLWO:\t", np.around(loss_SLWO,3), "\t|\t", np.around(1-loss_SLWO/loss_S,3))
    if LD:
        print("Loss/PRIAL SLD: \t", np.around(loss_Snlb,3), "\t|\t", np.around(1-loss_Snlb/loss_S,3))
    if MD:
        print("Loss/PRIAL SMD: \t", np.around(loss_Snlq,3), "\t|\t", np.around(1-loss_Snlq/loss_S,3))
    print("Loss/PRIAL Sor: \t", np.around(loss_Soracle,3), "\t|\t", np.around(1-loss_Soracle/loss_S,3))
    print()
    
    print("Norm prec:      \t", np.around(norm_prec,3))
    print("Loss/PRIAL P:   \t", np.around(loss_P,3), "\t|\t", 0)
    print("Loss/PRIAL PLWO:\t", np.around(loss_PLWO,3), "\t|\t", np.around(1-loss_PLWO/loss_P,5))
    if LD:
        print("Loss/PRIAL PLD: \t", np.around(loss_Pnlb,3), "\t|\t", np.around(1-loss_Pnlb/loss_P,3))
    if MD:
        print("Loss/PRIAL PMD: \t", np.around(loss_Pnlq,3), "\t|\t", np.around(1-loss_Pnlq/loss_P,3))
    print("Loss/PRIAL Por: \t", np.around(loss_Poracle,3), "\t|\t", np.around(1-loss_Poracle/loss_P,5))
    
    if plots:
        if LD:
            # Plot the results        
            Fx = f_xi.apply
            output = Fx(tau_fit_LD, None, d, wd, c, mu, weights, w_args, omega, method, False) 
            nu = output[-1].to(int)
            omegai = output[-nu-1:-1].to(int)
            output = output[:-nu-1]
            f = output[:output.shape[0]//3]
            F = output[output.shape[0]//3:2*output.shape[0]//3]
            xi = output[2*output.shape[0]//3:]
            plt.figure()
            plt.plot(xi.detach().numpy(), f.detach().numpy(), label="(LD) Asymptotic sample density")
            plt.hist(lambda_, bins=100, density=True, label="Sample density")
            plt.hist(tau_fit_LD.detach().numpy(), weights=3*np.ones(tau_fit_LD.shape[0])/tau_fit_LD.shape[0]*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=100, label="(LD) Asymptotic population density")
            plt.hist(t.detach().numpy(), weights=4*wt.detach().numpy()*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=150, label="Population density")
            plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
            plt.legend()
            plt.show()
            
            output_i = Fx(torch.tensor(tau_init), None, d, wd, c, mu, weights, w_args, omega, method, False) 
            nu = output_i[-1].to(int)
            omegai = output_i[-nu-1:-1].to(int)
            output_i = output_i[:-nu-1]
            f_i = output_i[:output_i.shape[0]//3]
            F_i = output_i[output_i.shape[0]//3:2*output_i.shape[0]//3]
            xi_i = output_i[2*output_i.shape[0]//3:]
            
            plt.figure()
            plt.plot(xi.detach().numpy(), F.detach().numpy(), label="Fit (LD) sample cdf")
            plt.plot(xi_i.detach().numpy(), F_i.detach().numpy(), label="Init sample cdf")
            plt.hist(lambda_, bins=200, cumulative = True, histtype="step", density=True, label="Sample cdf")
            plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.hist(tau_fit_LD.detach().numpy(), bins=200, cumulative = True, histtype="step", density=True, label="Fit (LD) population cdf")
            plt.hist(tau_init, bins=200, cumulative = True, histtype="step", density=True, label="Init population cdf")
            plt.hist(t.detach().numpy(), weights=wt.detach().numpy(), cumulative = True, histtype="step", bins=200, label="Population density")
            plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
            plt.legend()
            plt.show()    
        if MD:
            # Plot the results        
            Fx = f_xi.apply
            output = Fx(tau_fit_MD, None, d, wd, c, mu, weights, w_args, omega, method, False) 
            nu = output[-1].to(int)
            omegai = output[-nu-1:-1].to(int)
            output = output[:-nu-1]
            f = output[:output.shape[0]//3]
            F = output[output.shape[0]//3:2*output.shape[0]//3]
            xi = output[2*output.shape[0]//3:]
            plt.figure()
            plt.plot(xi.detach().numpy(), f.detach().numpy(), label="(MD) Asymptotic sample density")
            plt.hist(lambda_, bins=100, density=True, label="Sample density")
            plt.hist(tau_fit_MD.detach().numpy(), weights=3*np.ones(tau_fit_MD.shape[0])/tau_fit_MD.shape[0]*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=100, label="(MD) Asymptotic population density")
            plt.hist(t.detach().numpy(), weights=4*wt.detach().numpy()*np.max(f.detach().numpy())/np.max(wt.detach().numpy()), bins=150, label="Population density")
            plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
            plt.legend()
            plt.show()
            
            output_i = Fx(torch.tensor(tau_init), None, d, wd, c, mu, weights, w_args, omega, method, False) 
            nu = output_i[-1].to(int)
            omegai = output_i[-nu-1:-1].to(int)
            output_i = output_i[:-nu-1]
            f_i = output_i[:output_i.shape[0]//3]
            F_i = output_i[output_i.shape[0]//3:2*output_i.shape[0]//3]
            xi_i = output_i[2*output_i.shape[0]//3:]
            
            plt.figure()
            plt.plot(xi.detach().numpy(), F.detach().numpy(), label="Fit (MD) sample cdf")
            plt.plot(xi_i.detach().numpy(), F_i.detach().numpy(), label="Init sample cdf")
            plt.hist(lambda_, bins=200, cumulative = True, histtype="step", density=True, label="Sample cdf")
            plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
            plt.legend()
            plt.show()
            
            plt.figure()
            plt.hist(tau_fit_MD.detach().numpy(), bins=200, cumulative = True, histtype="step", density=True, label="Fit (MD) population cdf")
            plt.hist(tau_init, bins=200, cumulative = True, histtype="step", density=True, label="Init population cdf")
            plt.hist(t.detach().numpy(), weights=wt.detach().numpy(), cumulative = True, histtype="step", bins=200, label="Population density")
            plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
            plt.legend()
            plt.show()   
    
    return estimator_LD, estimator_MD, tau_init, S

def WeSpeR_experiment_2(tau_init = None, p = 2000, n = 20000):
    dtype = np.float64
    LD = True
    MD = True
    HD = True
    plots = False

    c = p/n
    
    t = torch.tensor([1.,3.,10.])
    wt = torch.tensor([0.2,0.4,0.4])
    Kt = 30
    
    mu = 0.1
    alpha = 5.
    weights = 'ewma'   
    method = 'root'
    Kd = 1
    
    Nd = 200
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    n_epochs_LD = 10
    n_epochs_MD = 10
    n_epochs_HD = 10
    lr = 5e-2
    omega = min(400, min(p,n))
    K = min(p,1000)
    p_tau = K
    loss_type = "trap"
    b = 1 # for LD
    num_probes = 1
    ortho = True
    verbose = True
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    
    # Draw v for regression
    v = np.random.normal(size=p).astype(dtype)
    v = v / np.linalg.norm(v)

    tau_pop = np.array([1]*(p//5)+[3]*(2*p//5)+[10]*(p - p//5 - 2*p//5))
    
    if weights == "ewma":
        beta = alpha/(1-np.exp(-alpha))
        W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    elif weights == "unif":
        # alpha \in [0,2]
        W = alpha*(np.ones(n).cumsum()-(n+1)/2)/n + 1
    elif weights == "2diracs":
        # alpha \in (0,1)
        W = np.array([1-alpha]*(int(n*wd))+[(1-wd*(1-alpha))/(1-wd)]*(n-int(n*wd)))
    elif weights == "Ndiracs":
        W = []
        for j in range(1,d.shape[0]):
            W += [d[j]]*int(n*wd[j])
        W += [d[0]]*(n - len(W))
        W = np.array(W)
    
    # Sampling
    if n <= p and p <= 10000:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))  
        S = X_sample.T @  (W[:, None] * X_sample)/n
        SLWO, c = wLWO_estimator(X_sample, W/n, assume_centered = True, S_r = None, est = True)
        shrinkage = 1 - c[0]

        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")
    else:
        n_left = n
        n_used = 0
        X2sum = np.zeros(n)
        S = np.zeros((p,p))
        while n_left > 0:
            n_sample = min(n_left, p)
            X_sample = np.random.normal(size = (n_sample,p)) @ np.sqrt(np.diag(tau_pop))  
            S += X_sample.T @  (W[n_used:n_used+n_sample, None] * X_sample)/n
            X2sum[n_used:n_used+n_sample] = (X_sample**2).sum(axis=1)
            n_used += n_sample
            n_left -= n_sample

        mu_S = np.trace(S @ np.eye(p))/p
        delta2 = np.linalg.norm(S-mu_S*np.eye(p), ord = 'fro')**2/p
        S2 = np.linalg.norm(S, ord = 'fro')**2/p
        beta2 = (X2sum**2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
        beta2 = beta2/(1-(W**2).sum()/n**2)
        beta2 = min(beta2, delta2)
        
        shrinkage = beta2/delta2
    
    # Optimize tau    
    if LD:
        print("WeSpeR_LD:", LD)

        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")
        e_SLWO = shrinkage*mu_S + (1 - shrinkage)*lambda_
        if tau_init is None:
            if c >= 1:
                tau_init = np.unique(np.around(e_SLWO,6))
                tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
            else:
                tau_init = np.unique(np.around(e_SLWO,6))
                tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
            tau_init = np.sort(np.random.choice(tau_init, p))
            tau_init = torch.tensor(tau_init)

        start_time = time.time()
        estimator_LD = WeSpeR_LD(bias=True, assume_centered=True).fit_from_S(S, W, tau_init = tau_init, method = 'Adam', n_epochs = n_epochs_LD, b = b, assume_centered = True, lr = lr, momentum=0., verbose = True)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        start_time = time.time()
        wnl_LD = estimator_LD.get_precision(d = d, wd = wd, weights = weights, w_args = w_args, method = 'root', verbose = False) @ v
        last_time = time.time()
        print("wnl_LD compute time:", last_time - start_time, "s.")
    else:
        estimator_LD = None

    if MD:
        print("WeSpeR_MD:", MD) 

        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")
        e_SLWO = shrinkage*mu_S + (1 - shrinkage)*lambda_
        if tau_init is None:
            if c >= 1:
                tau_init = np.unique(np.around(e_SLWO,6))
                tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
            else:
                tau_init = np.unique(np.around(e_SLWO,6))
                tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
                tau_init = np.sort(np.concatenate([tau_init, tau_add]))
                tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
            tau_init = np.sort(np.random.choice(tau_init, p_tau))
            tau_init = torch.tensor(tau_init)

        start_time = time.time()
        estimator_MD = WeSpeR_MD(bias=True, assume_centered=True).fit_from_F(torch.tensor(lambda_), U, W, tau_init = tau_init, wt = None, d = d, wd = wd, c = c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs_MD, loss_type = loss_type, method = method, verbose = verbose)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        start_time = time.time()
        wnl_MD = estimator_MD.get_precision(d = d, wd = wd, weights = weights, w_args = w_args, method = 'root', verbose = False) @ v
        last_time = time.time()
        print("wnl_MD compute time:", last_time - start_time, "s.")
    else:
        estimator_MD = None
    
    if HD:
        print("WeSpeR_HD:", HD) 
        start_time = time.time()
        estimator_HD = WeSpeR_HD(bias=True, assume_centered=True).fit_from_S(S, W, shrinkage = shrinkage, num_probes = num_probes, K = K, ortho = ortho, tau_init = tau_init, wt = None, d = d, wd = wd, c = c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs_HD, loss_type = loss_type, method = method, verbose = verbose)
        last_time = time.time()
        print("Compute time:", last_time - start_time, "s.")

        start_time = time.time()
        wnl_HD = estimator_HD.get_precision_operator(K = K, ortho = ortho, d = d, wd = wd, weights = weights, w_args = w_args, method = method, verbose = False)(v)
        last_time = time.time()
        print("wnl_HD compute time:", last_time - start_time, "s.")
    else:
        estimator_HD = None

    start_time = time.time()
    f = lambda x: nl_prec_shrinkage(x, t, wt, d, wd, c = c, weights = weights, w_args = w_args, method = method, verbose = False).numpy()
    wor = lanczos_f(S, v, K, f, ortho = ortho)
    last_time = time.time()
    print("wor compute time:", last_time - start_time, "s.")
    
    start_time = time.time()
    wna = np.linalg.solve(S, v)
    last_time = time.time()
    print("wna compute time:", last_time - start_time, "s.")
    
    # SLWO = shrinkage*mu*np.eye(p) + (1 - shrinkage)*S
    S *= 1 - shrinkage
    S[np.diag_indices(p)] += shrinkage*mu_S
    start_time = time.time()
    wls = np.linalg.solve(S, v)
    last_time = time.time()
    print("wls compute time:", last_time - start_time, "s.")

    wpop = v/tau_pop
    
    print("norm:", np.linalg.norm(wpop)**2/p)
    if LD:
        print("wnl_LD: \t", np.around(np.linalg.norm(wnl_LD-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wnl_LD-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    if MD:
        print("wnl_MD: \t", np.around(np.linalg.norm(wnl_MD-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wnl_MD-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    if HD:
        print("wnl_HD: \t", np.around(np.linalg.norm(wnl_HD-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wnl_HD-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    print("wor:    \t", np.around(np.linalg.norm(wor-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wor-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    print("wls:    \t", np.around(np.linalg.norm(wls-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wls-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))
    print("wna:    \t", np.around(np.linalg.norm(wna-wpop)**2/np.linalg.norm(wpop)**2, 8), " \t | \t ", np.around(1-np.linalg.norm(wna-wpop)**2/np.linalg.norm(wna-wpop)**2, 6))

    return estimator_LD, estimator_MD, estimator_HD, tau_init, S

def WeSpeR_experiment_3(H_distrib = 1, tau_init = None, p = 2000, n = 20000):
    plots = False

    c = p/n
    
    x = np.linspace(0,1,p)
    H1 = 1 - (1-x**3)**(1/3)
    H2 = (1 - (1-x)**3)**(1/3)
    H3 = (1 - np.abs(1-2*x)**3)**(1/3)/2
    H3[x>1/2] = 1 - H3[x>1/2]
    H4 = 1/2 - (1 - (2*x)**3)**(1/3)/2
    H4[x>1/2] = 1/2 + (1 - (2-2*x[x>1/2])**3)**(1/3)/2
    if H_distrib == 1:
        H = H1
    elif H_distrib == 2:
        H = H2
    elif H_distrib == 3:
        H = H3
    else:
        H = H4
        
    t = torch.zeros(p)
    for kappa in range(p):
        j_k = max(np.argmax((((kappa+1/2) - p*H) < 0).astype(int)) - 1, 0)
        j_kp = max(np.argmax((((kappa+1/2) - p*H) < 0).astype(int)), 0)
        if H[j_kp] - H[j_k] > 0:
            t[kappa] = (((kappa+1/2)/p - H[j_k])*x[j_k] + (H[j_kp] - (kappa+1/2)/p)*x[j_kp])/(H[j_kp] - H[j_k])
        else:
            t[kappa] = (x[j_k]+x[j_kp])/2
    t = torch.tensor(1 + 9*t)
    wt = torch.ones(p)/p
    Kt = 30
    
    mu = 0.1
    alpha = 1.
    weights = 'unif'   
    method = 'root'
    Kd = 1
    
    Nd = 200
    d = torch.tensor(np.linspace(1,10,Nd))
    wd = torch.tensor(np.ones(Nd))
    wd = wd/wd.sum(axis=0)
    d = d/(d*wd).sum(axis=0)
    
    n_epochs_LD = 200
    n_epochs_MD = 400
    lr = 2e-3
    omega = min(400, min(p,n))
    p_tau = p
    loss_type = "trap"
    b = 1 # for LD
    verbose = True
    
    if weights == "Ndiracs":
        w_args = [Kd, Kt]
    else:
        w_args = [alpha, Kt]
    
    tau_pop = t
    plt.figure()
    plt.hist(tau_pop.numpy(), bins=100, density=True, label="H1 histogram")
    plt.show()
    
    if weights == "ewma":
        beta = alpha/(1-np.exp(-alpha))
        W = np.exp(-alpha*((np.ones(n).cumsum()-1)/n))*beta
    elif weights == "unif":
        # alpha \in [0,2]
        W = alpha*(np.ones(n).cumsum()-(n+1)/2)/n + 1
    elif weights == "2diracs":
        # alpha \in (0,1)
        W = np.array([1-alpha]*(int(n*wd))+[(1-wd*(1-alpha))/(1-wd)]*(n-int(n*wd)))
    elif weights == "Ndiracs":
        W = []
        for j in range(1,d.shape[0]):
            W += [d[j]]*int(n*wd[j])
        W += [d[0]]*(n - len(W))
        W = np.array(W)
    
    # Sampling
    if n <= p:
        X_sample = np.random.normal(size = (n,p)) @ np.sqrt(np.diag(tau_pop))  
        S = X_sample.T @  (W[:, None] * X_sample)/n
        SLWO = wLWO_estimator(X_sample, W/n, assume_centered = True, S_r = None, est = False)
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")

        e_SLWO, _ = np.linalg.eigh(SLWO)
    else:
        n_left = n
        n_used = 0
        X2sum = np.zeros(n)
        S = np.zeros((p,p))
        while n_left > 0:
            n_sample = min(n_left, p)
            X_sample = np.random.normal(size = (n_sample,p)) @ np.sqrt(np.diag(tau_pop))  
            S += X_sample.T @  (W[n_used:n_used+n_sample, None] * X_sample)/n
            X2sum[n_used:n_used+n_sample] = (X_sample**2).sum(axis=1)
            n_used += n_sample
            n_left -= n_sample

        mu_S = np.trace(S @ np.eye(p))/p
        delta2 = np.linalg.norm(S-mu_S*np.eye(p), ord = 'fro')**2/p
        S2 = np.linalg.norm(S, ord = 'fro')**2/p
        beta2 = (X2sum**2*W**2).sum()/p/n**2 - S2*(W**2).sum()/n**2
        beta2 = beta2/(1-(W**2).sum()/n**2)
        beta2 = min(beta2, delta2)
        
        shrinkage = beta2/delta2
        SLWO = shrinkage*mu_S*np.eye(p) + (1 - shrinkage)*S
        
        start_time = time.time()
        lambda_, U = np.linalg.eigh(S)
        last_time = time.time()
        print("Diagonalization time:", last_time - start_time, "s.")
    
        e_SLWO = shrinkage*mu_S + (1 - shrinkage)*lambda_

    if tau_init is None:
        if c >= 1:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        else:
            tau_init = np.unique(np.around(e_SLWO,6))
            tau_add = np.linspace(np.min(e_SLWO), np.max(e_SLWO), e_SLWO.shape[0]-tau_init.shape[0]+2)[1:-1]
            tau_init = np.sort(np.concatenate([tau_init, tau_add]))
            tau_init = tau_init/tau_init.mean()*e_SLWO.mean()
        tau_init_LD = np.sort(np.random.choice(tau_init, p))
        tau_init_LD = torch.tensor(tau_init_LD)
        tau_init_MD = np.sort(np.random.choice(tau_init, p_tau))
        tau_init_MD = torch.tensor(tau_init_MD)
    
    # Optimize tau LD
    print("WeSpeR_LD:")
    start_time = time.time()
    estimator_LD = WeSpeR_LD(bias=True, assume_centered=True).fit_from_S(S, W, tau_init = tau_init_LD, method = 'Adam', n_epochs = n_epochs_LD, b = b, assume_centered = True, lr = lr, momentum=0., verbose = True)
    last_time = time.time()
    print("Compute time:", last_time - start_time, "s.")
    
    tau_fit = estimator_LD.get_tau()
    # Q-Q plots:
    plt.figure()
    plt.plot(tau_pop, np.sort(tau_fit.detach().numpy()), label = '(LD) algorithm')

    # Optimize tau MD
    print("WeSpeR_MD:") 
    start_time = time.time()
    estimator_MD = WeSpeR_MD(bias=True, assume_centered=True).fit_from_F(torch.tensor(lambda_), U, W, tau_init = tau_init_MD, wt = None, d = d, wd = wd, c = c, mu = mu, weights = weights, w_args = w_args, omega = omega, lr = lr, n_epochs = n_epochs_MD, loss_type = loss_type, method = method, verbose = verbose)
    last_time = time.time()
    print("Compute time:", last_time - start_time, "s.")

    tau_fit = estimator_MD.get_tau()

    # Q-Q plots:
    plt.plot(tau_pop, np.sort(tau_fit.detach().numpy()), label = '(MD) algorithm')
    plt.plot(tau_pop, np.sort(lambda_), label = r'Sample spectrum')
    plt.plot(tau_pop, tau_pop, label = 'Population spectrum')
    plt.xlabel(r"$H$")
    plt.ylabel(r"$\hat H$")
    if H_distrib == 1:
        plt.title(r"Q-Q plot for $H_1$")
    elif H_distrib == 2:
        plt.title(r"Q-Q plot for $H_2$")
    elif H_distrib == 3:
        plt.title(r"Q-Q plot for $H_3$")
    else:
        plt.title(r"Q-Q plot for $H_4$")
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(estimator_LD.get_tau().detach().numpy(), bins=200, cumulative = False, histtype="step", density=True, label="Fit (LD) population pdf")
    plt.hist(estimator_MD.get_tau().detach().numpy(), bins=200, cumulative = False, histtype="step", density=True, label="Fit (MD) population pdf")
    plt.hist(lambda_, bins=200, cumulative = False, histtype="step", density=True, label="Sample pdf")
    plt.hist(tau_pop.detach().numpy(), weights=wt.detach().numpy(), cumulative = False, density=True, histtype="step", bins=200, label="Population density")
    plt.title("c="+str(c)+", "+weights+", alpha="+str(w_args[0])+", omega="+str(omega))
    plt.legend()
    plt.show()   
    
    return estimator_LD, estimator_MD

if __name__ == '__main__':
    try:
        matplotlib.use('tkagg')
        matplotlib.pyplot.ion()
    except:
        pass
    
    warnings.filterwarnings('ignore')

    # estimator_LD, estimator_MD, tau_init, S =  WeSpeR_experiment_1(tau_init = None, p = 100, n = 1000, plots = True)
    # estimator_LD, estimator_MD, estimator_HD, tau_init, S =  WeSpeR_experiment_2(tau_init = None, p = 100, n = 1000)
    estimator_LD, estimator_MD = WeSpeR_experiment_3(H_distrib = 1, tau_init = None, p = 2000, n = 20000)
    # estimator_LD, estimator_MD = WeSpeR_experiment_3(H_distrib = 2, tau_init = None, p = 2000, n = 20000)
    # estimator_LD, estimator_MD = WeSpeR_experiment_3(H_distrib = 3, tau_init = None, p = 2000, n = 20000)
    # estimator_LD, estimator_MD = WeSpeR_experiment_3(H_distrib = 4, tau_init = None, p = 2000, n = 20000)