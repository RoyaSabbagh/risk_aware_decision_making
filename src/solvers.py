#!/usr/bin/env python3
"""
cem_solver.py

Implementation of Cross-Entropy Method (CEM) solver.

Author: Adam Conkey
"""
import sys
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from sklearn import mixture


class CEMSolver():
    """
    Cross-Entropy Method (CEM) solver initially developed for rare-event simulation [1] and
    widely used in robotics, e.g. motion planning [2].

    Implements the basic steps of:
        1. Generate N samples.
        2. Evaluate the cost of each sample.
        3. Take the subset of top-K samples with the lowest cost (a.k.a. the elite set).
        4. Re-fit the distribution with the top-K elite samples.
        5. Repeat until distribution converges on low-cost region.

    [1] Rubinstein, Reuven Y., and Dirk P. Kroese. The cross-entropy method: a unified
        approach to combinatorial optimization, Monte-Carlo simulation and machine learning.
        Springer Science & Business Media, 2013.

    [2] Kobilarov, Marin. "Cross-entropy motion planning." The International Journal of
        Robotics Research 31.7 (2012): 855-871.

    TODO: Add options for other distribution types (e.g. GMM), right now only Gaussian is supported.
    """

    def __init__(self, max_iterations=100, kl_epsilon=2, n_elite=7, n_samples=50):
        """
        Args:
            max_iterations (int): Maximum number of iterations to run solver.
            kl_epsilon (flt): Threshold on KL divergence between distributions at consecutive
                              iterations, solver converges when value goes below threshold.
            n_samples (int): Number of samples to generate at each iteration.
            n_elite (int): Number of elite samples to re-fit distribution with at each iteration.
        """
        self.max_iterations = max_iterations
        self.kl_epsilon = kl_epsilon
        self.n_elite = n_elite
        self.n_samples = n_samples

    def optimize(self, f, x0, sigma0=None, sigma_reg=None):
        """
        Runs CEM optimization up to convergence or terminates at max_iterations.

        Args:
            f (func): Cost function to optimized.
            x0 (ndarray): Initial distribution mean, shape (n_dims,), defaults to all zeros.
            sigma0 (ndarray): Initial distribution covariance, shape (n_dims, n_dims), defaults
                              to identity matrix.
            sigma_reg (ndarray): Additive regularization covariance to ensure well-conditioned,
                                 shape (n_dims, n_dims), defaults to small-scaled identity matrix.

        """
        if sigma0 is None:
            sigma0 = 3*np.eye(3)
        if sigma_reg is None:
            sigma_reg = 1e-5 * np.eye(3)

        mu = torch.Tensor(x0)
        sigma = torch.Tensor(sigma0) + torch.Tensor(sigma_reg)
        distribution = MultivariateNormal(mu, sigma)
        converged = False
        cost = sys.maxsize
        x = None
        iterates = []
        sample_iterates = []
        for i in range(self.max_iterations):
            # Generate samples and evaluate costs to find elite set
            samples = distribution.sample((self.n_samples,)).data.numpy()
            sample_iterates.append(samples)

            costs = [f(s) for s in samples]
            sorted_samples = [s for _, s in sorted(zip(costs, samples), key=lambda x: x[0])]
            elite = np.vstack(sorted_samples[:self.n_elite])

            # Re-fit the distribution based on elite set
            prev_distribution = distribution
            mu = torch.Tensor(np.mean(elite, axis=0))
            sigma = torch.Tensor(np.cov(elite.T)) + torch.Tensor(sigma_reg)
            distribution = MultivariateNormal(mu, sigma)
            # Check convergence based on KL-divergence between previous and current distributions
            kl = kl_divergence(prev_distribution, distribution).item()
            # print("kl:", kl)
            x = mu.data.numpy()
            iterates.append(x)

            if kl < self.kl_epsilon:
                converged = True
                cost = f(x)
                break

        if converged:
            print("\nCEM converged after {} iterations!".format(i))
            print("Solution: {}, Cost: {}\n".format(x, cost))
        else:
            print("\nCEM failed to converge after {} iterations. :(\n"
                  "".format(self.max_iterations))

        return x, cost, converged, np.array(iterates), np.array(sample_iterates)

    def optimize_GMM(self, f, x0, sigma0=None, sigma_reg=None):
        """
        Runs CEM optimization up to convergence or terminates at max_iterations.

        Args:
            f (func): Cost function to optimized.
            x0 (ndarray): Initial distribution mean, shape (n_dims,), defaults to all zeros.
            sigma0 (ndarray): Initial distribution covariance, shape (n_dims, n_dims), defaults
                              to identity matrix.
            sigma_reg (ndarray): Additive regularization covariance to ensure well-conditioned,
                                 shape (n_dims, n_dims), defaults to small-scaled identity matrix.

        """
        if sigma0 is None:
            sigma0 = 2*np.eye(3)
        if sigma_reg is None:
            sigma_reg = 1e-5 * np.eye(3)

        self.kl_epsilon = 10

        mu = torch.Tensor(x0)
        sigma = torch.Tensor(sigma0) + torch.Tensor(sigma_reg)
        distribution = MultivariateNormal(mu, sigma)
        samples = distribution.sample((self.n_samples,)).data.numpy()

        GMM = mixture.GaussianMixture(n_components= 3 , covariance_type='full'  )
        data = samples
        GMM.fit(data)
        #
        # m = GMM.means_
        # w = GMM.weights_
        # cov = GMM.covariances_

        converged = False
        cost = sys.maxsize
        x = None
        iterates = []
        sample_iterates = []
        for i in range(self.max_iterations):
            # Generate samples and evaluate costs to find elite set
            samples = GMM.sample(self.n_samples)[0]
            # print(samples)
            sample_iterates.append(samples)

            costs = [f(s) for s in samples]
            sorted_samples = [s for _, s in sorted(zip(costs, samples), key=lambda x: x[0])]
            elite = np.vstack(sorted_samples[:self.n_elite])

            # Re-fit the distribution based on elite set
            prev_GMM = GMM
            GMM = mixture.GaussianMixture(n_components= 2 , covariance_type='full'  )
            GMM.fit(elite)
            # Check convergence based on KL-divergence between previous and current distributions
            kl = self.gmm_kl(prev_GMM, GMM)
            print("kl:", kl)
            x = GMM.means_
            iterates.append(x)

            if kl < self.kl_epsilon:
                converged = True
                cost = [f(s) for s in x]
                break

        if converged:
            print("\nCEM converged after {} iterations!".format(i))
            print("Solution: {}, Cost: {}\n".format(x, cost))
        else:
            print("\nCEM failed to converge after {} iterations. :(\n"
                  "".format(self.max_iterations))

        return x, cost, converged, np.array(iterates), np.array(sample_iterates)

    def gmm_kl(self, gmm_p, gmm_q, n_samples=10**5):
        X = gmm_p.sample(n_samples)[0]
        log_p_X = gmm_p.score(X)
        log_q_X = gmm_q.score(X)
        # print(log_p_X, log_q_X)
        return log_p_X.mean() - log_q_X.mean()

class DeterministicSolver():
    def __init__(self):
        """
        Args:
            max_iterations (int): Maximum number of iterations to run solver.
            kl_epsilon (flt): Threshold on KL divergence between distributions at consecutive
                              iterations, solver converges when value goes below threshold.
        """


    def optimize(self, f, traj):
        costs = []
        for i in range(1, len(traj)-1):
            costs.append(f(traj, i))

        best = costs.index(min(costs))
        # print("costs:", costs)

        return [traj[best+1][0], traj[best+1][1], (best+1)*0.5], costs[best]
