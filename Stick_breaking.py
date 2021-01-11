# -*- coding: utf-8 -*-
"""
@author: William
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

N = 20
K = 30
alpha = 2.
P0 = sp.stats.norm

beta = sp.stats.beta.rvs(1, alpha, size = (N, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, : -1]).cumprod(axis = 1)

omega = P0.rvs(size=(N, K))

x_plot = np.linspace(-3, 3, 200)

sample_cdfs = (w[..., np.newaxis] * np.less.outer(omega, x_plot)).sum(axis = 1)

plt.plot(x_plot, sample_cdfs[0], c = 'gray', alpha = 0.75, label = 'Sampled CDFs')
plt.plot(x_plot, sample_cdfs[1:].T, c = 'gray', alpha = 0.75)
plt.plot(x_plot, P0.cdf(x_plot), c = 'k', label = 'Base CDF')

plt.title(r'$\alpha = {}$'.format(alpha))
plt.legend(loc = 2)
plt.grid()
plt.show()