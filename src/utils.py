"""Building blocks of the autodiff estimator.

This module contains the differentiable sampling model, the losses and the
optimisation loop used to fit the population spectrum ``tau`` by
back-propagating through the sample eigenvalues.
"""

import numpy as np
import torch


def initial_tau_guess(eigenvalues, size=None):
    """Build an initial guess of the population spectrum.

    The distinct eigenvalues are completed by evenly spaced values so that the
    guess has as many entries as ``eigenvalues``; it is rescaled to have the
    same mean and then, if ``size`` is given, resampled (with replacement) to
    ``size`` atoms.

    Parameters
    ----------
    eigenvalues : ndarray of shape (p,)
        Eigenvalues of a (possibly shrunk) sample covariance matrix.
    size : int, optional
        Number of atoms of the returned spectrum (default: no resampling).

    Returns
    -------
    tau_init : torch.Tensor
        Sorted initial population eigenvalues.
    """
    tau = np.unique(np.around(eigenvalues, 6))
    n_missing = eigenvalues.shape[0] - tau.shape[0] + 2
    tau_add = np.linspace(np.min(eigenvalues), np.max(eigenvalues), n_missing)[1:-1]
    tau = np.sort(np.concatenate([tau, tau_add]))
    tau = tau / tau.mean() * eigenvalues.mean()
    if size is not None:
        tau = np.sort(np.random.choice(tau, size))
    return torch.tensor(tau)


class AutodiffModel(torch.nn.Module):
    """Differentiable sampler of the eigenvalues of a weighted sample covariance.

    The learnable parameter is the logarithm of the population spectrum.
    Each forward pass draws ``b`` Gaussian samples with covariance
    ``diag(tau)`` and weights ``Wsq**2``, and returns the sorted eigenvalues
    of the resulting sample covariance matrices.

    Parameters
    ----------
    tau_init : torch.Tensor of shape (p,)
        Initial population eigenvalues.
    Wsq : torch.Tensor of shape (n,)
        Square root of the sample weights.
    p, n : int
        Dimension and number of samples.
    b : int
        Batch size (number of simulated matrices per forward pass).
    """

    def __init__(self, tau_init, Wsq, p, n, b):
        super().__init__()
        self.logtau = torch.nn.Parameter(torch.log(tau_init).clone().type(torch.float64))
        self.p = p
        self.n = n
        self.b = b
        self.Wsq = Wsq

    def get_tau(self):
        """Return the current population eigenvalues."""
        return torch.exp(self.logtau)

    def forward(self):
        """Simulate sample eigenvalues.

        Returns
        -------
        lambda_ : torch.Tensor of shape (b, p)
            Sorted sample eigenvalues.
        """
        sq_cov = torch.diag(torch.sqrt(torch.exp(self.logtau).sort()[0]))[None, :, :]
        if self.n > self.p:
            # Accumulate the sample covariance by blocks of at most p samples
            # to bound memory usage.
            n_left = self.n
            n_used = 0
            S = torch.zeros(size=(self.b, self.p, self.p), dtype=torch.float64)
            while n_left > 0:
                n_sample = min(n_left, self.p)
                X_sample = (
                    torch.empty(size=(self.b, n_sample, self.p), dtype=torch.float64).normal_()
                    * self.Wsq[None, n_used : n_used + n_sample, None]
                )
                S += sq_cov @ X_sample.transpose(-2, -1) @ X_sample @ sq_cov / self.n
                n_used += n_sample
                n_left -= n_sample
        else:
            X = torch.empty(size=(self.b, self.n, self.p), dtype=torch.float64).normal_() * self.Wsq[None, :, None]
            S = sq_cov @ X.transpose(-2, -1) @ X @ sq_cov / self.n
        # torch.linalg.eigvals segfaults when there are NaNs.
        if S.isnan().sum() > 0:
            raise ValueError("NaN not supported in torch.linalg.eigvals")
        lambda_ = torch.linalg.eigvals(S).real
        return lambda_.sort()[0]


def loss_wasserstein_1D(p=2, regu=1):
    """Return a differentiable 1D Wasserstein-``p`` loss between two batches.

    The returned function compares two tensors of shape ``(b, k)`` of sample
    points and returns the batch-averaged ``p``-th power of the Wasserstein
    distance, plus ``regu`` times the squared difference of the means.
    """

    def loss_wasserstein_p(u_values, v_values, u_weights=None, v_weights=None):
        # Inspired from the SciPy 1D Wasserstein distance:
        # https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L9733-L9807
        if u_weights is None:
            u_weights = torch.ones(u_values.shape)
            u_weights = u_weights / u_weights.sum(axis=1)[:, None]
        if v_weights is None:
            v_weights = torch.ones(v_values.shape)
            v_weights = v_weights / v_weights.sum(axis=1)[:, None]

        u_sorter = torch.argsort(u_values, axis=1)
        u_sorter = (torch.ones(u_values.shape, dtype=int).cumsum(0) - 1, u_sorter)
        v_sorter = torch.argsort(v_values, axis=1)
        v_sorter = (torch.ones(v_values.shape, dtype=int).cumsum(0) - 1, v_sorter)

        all_values = torch.cat((u_values, v_values), axis=1)
        all_values = all_values.sort(axis=1)[0]

        # Differences between pairs of successive values of u and v.
        deltas = torch.diff(all_values, axis=1)

        # Positions of the values of u and v among the values of both distributions.
        u_values_sort = u_values[u_sorter].contiguous()
        v_values_sort = v_values[v_sorter].contiguous()

        u_cdf_indices = torch.searchsorted(all_values[:, :-1].contiguous(), u_values_sort, right=True)
        u_cdf_indices = (torch.ones(u_cdf_indices.shape, dtype=int).cumsum(0) - 1, u_cdf_indices)
        v_cdf_indices = torch.searchsorted(all_values[:, :-1].contiguous(), v_values_sort, right=True)
        v_cdf_indices = (torch.ones(v_cdf_indices.shape, dtype=int).cumsum(0) - 1, v_cdf_indices)

        # CDFs of u and v, using their weights.
        u_cdf = torch.zeros(all_values.shape)
        v_cdf = torch.zeros(all_values.shape)

        u_cdf[u_cdf_indices] = u_weights[u_sorter]
        u_cdf = u_cdf[:, 1:].cumsum(axis=1)
        u_cdf = u_cdf / u_cdf[:, -1:]

        v_cdf[v_cdf_indices] = v_weights[v_sorter]
        v_cdf = v_cdf[:, 1:].cumsum(axis=1)
        v_cdf = v_cdf / v_cdf[:, -1:]

        # The power is not normalised by 1/p at the end, to keep it differentiable.
        distance = (torch.abs(u_cdf - v_cdf) ** p * deltas).sum(axis=1) + regu * (u_values.mean(axis=1) - v_values.mean(axis=1)) ** 2
        return distance.mean()

    return loss_wasserstein_p


def loss_composite(p=2, regu=1):
    """Return the training loss: Wasserstein-``p`` plus the mean squared error."""

    def loss(u_values, v_values):
        return loss_wasserstein_1D(p, regu)(u_values, v_values) + ((u_values - v_values) ** 2).mean()

    return loss


def autodiff_minimization(
    S,
    n,
    p,
    tau_init,
    Wsq,
    method="Adam",
    n_epochs=1000,
    b=64,
    lr=1e-2,
    momentum=0.0,
    save_all_tau=False,
    verbose=True,
):
    """Fit the population spectrum by gradient descent through the eigenvalues.

    Parameters
    ----------
    S : torch.Tensor of shape (p, p)
        Weighted sample covariance matrix whose spectrum is matched.
    n : int
        Number of samples.
    p : int
        Dimension.
    tau_init : torch.Tensor of shape (p,)
        Initial guess of the population spectrum.
    Wsq : torch.Tensor of shape (n,)
        Square root of the sample weights.
    method : {"Adam", "SGD"}, default="Adam"
        Optimiser (anything other than ``"SGD"`` falls back to Adam).
    n_epochs : int, default=1000
        Number of optimisation steps.
    b : int, default=64
        Batch size of the simulated sample spectra.
    lr : float, default=1e-2
        Learning rate.
    momentum : float, default=0.0
        Momentum (SGD only).
    save_all_tau : bool, default=False
        If True, keep the spectrum at every epoch.
    verbose : bool, default=True
        If True, print the loss about ten times during training.

    Returns
    -------
    model : AutodiffModel
        The trained model.
    best_tau : torch.Tensor of shape (p,)
        Population spectrum reaching the lowest loss.
    running_loss : ndarray of shape (n_epochs,)
        Training loss at each epoch.
    all_tau : list of torch.Tensor
        Spectrum at each epoch (empty unless ``save_all_tau``).
    train_lambda_ : torch.Tensor of shape (p,)
        Eigenvalues of ``S``.
    V : torch.Tensor of shape (p, p)
        Eigenvectors of ``S``.
    """
    train_lambda_, V = torch.linalg.eigh(S)

    model = AutodiffModel(tau_init, Wsq, p, n, b)
    loss_fn = loss_composite(p=2, regu=1)
    if method == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = []
    best_loss = np.inf
    best_tau = tau_init.type(torch.float64)
    all_tau = [model.get_tau().detach()] if save_all_tau else []
    train_lambda_batch = train_lambda_.tile((b, 1))

    model.train(True)
    for i in range(n_epochs):
        optimizer.zero_grad()
        lambda_ = model()
        loss = loss_fn(lambda_, train_lambda_batch)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_tau = model.get_tau().detach()

        running_loss += [loss.item()]
        if save_all_tau:
            all_tau += [model.get_tau().detach()]

        if verbose and (n_epochs < 10 or i % (n_epochs // 10) == 0):
            print("Loss epoch", i, ":", loss.item())

    if verbose:
        print("Final loss :", best_loss)

    return model, best_tau, np.array(running_loss), all_tau, train_lambda_, V


def make_weight_distribution(n_atoms=200):
    """Return the support ``d`` and probabilities ``wd`` of the weight law.

    The support is uniform on ``[1, 10]``, rescaled to have unit mean.
    """
    d = torch.tensor(np.linspace(1, 10, n_atoms))
    wd = torch.ones(n_atoms, dtype=torch.float64)
    wd = wd / wd.sum(axis=0)
    d = d / (d * wd).sum(axis=0)
    return d, wd


def make_weights(weights, n, alpha, d, wd):
    """Build the vector of ``n`` sample weights.

    Parameters
    ----------
    weights : {"ewma", "unif", "Ndiracs"}
        ``"ewma"``: exponentially decaying weights with rate ``alpha``;
        ``"unif"``: linear weights, ``alpha`` in ``[0, 2]``;
        ``"Ndiracs"``: weights drawn from the discrete law ``(d, wd)``.
    n : int
        Number of samples.
    alpha : float
        Parameter of the ``"ewma"`` and ``"unif"`` families.
    d, wd : torch.Tensor
        Support and probabilities of the ``"Ndiracs"`` law.

    Returns
    -------
    W : ndarray of shape (n,)
        Sample weights, with unit mean.
    """
    index = np.arange(n)
    if weights == "ewma":
        beta = alpha / (1 - np.exp(-alpha))
        return np.exp(-alpha * index / n) * beta
    if weights == "unif":
        return alpha * (index + 1 - (n + 1) / 2) / n + 1
    if weights == "Ndiracs":
        d, wd = d.numpy(), wd.numpy()
        W = []
        for j in range(1, d.shape[0]):
            W += [d[j]] * int(n * wd[j])
        W += [d[0]] * (n - len(W))
        return np.array(W)
    raise ValueError(f"Unknown weights family: {weights!r}")
