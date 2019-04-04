#!/usr/bin/env python
"""
Testing the recovery of a multi-dimensional
Gaussian with zero mean and unit variance
"""
from __future__ import division
import bilby
import numpy as np
from dynesty import plotting as dyplot

# A few simple setup steps
label = 'multidim_gaussian'
outdir = 'outdir'

# Making simulated data: generating n-dim Gaussian

dim = 5
mean = np.zeros(dim)
cov = np.ones((dim,dim))
data = np.random.multivariate_normal(mean, cov, 100)

class MultidimGaussianLikelihood(bilby.Likelihood):
    """
        A very simple multivariate Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
    def __init__(self, data, dim):
        self.dim = dim
        self.data = np.array(data)
        self.N = len(data)
        self.parameters = {}

            
    def log_likelihood(self):
        mu = np.array(
            [self.parameters['mu_{0}'.format(i)] for i in range(self.dim)])
        sigma = np.array(
            [self.parameters['sigma_{0}'.format(i)] for i in range(self.dim)])
        p = np.array(
            [(self.data[n,:] - mu)/sigma for n in range(self.N)])
        return np.sum(-0.5 * (np.sum(p**2) + self.N * np.log(2 * np.pi * sigma**2)))

likelihood = MultidimGaussianLikelihood(data, dim)
priors = bilby.core.prior.PriorDict()
priors.update(
    {"mu_{0}".format(i): bilby.core.prior.Uniform(-5, 5, 'mu') for i in range(dim)} )       
priors.update(
    {"sigma_{0}".format(i): bilby.core.prior.LogUniform(0.2, 5, 'sigma')
    for i in range(dim)})

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    walks=10, outdir=outdir, label=label)
res = result
#result.plot_corner()

# Analytic evidence solution
logz_real = dim * np.log(2 * 5)
fig, axes = dyplot.runplot(res, lnz_truth=logz_real)
