# Statistical Rethinking
# Chapter 2: Small Worlds and Larger Worlds
# Examples and Exercises

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')

# 2.2: Building a model: binomial likelihood with (flat) beta prior
# data: WLWWWLWLW
a0 = b0 = 1
beta.stats(a0, b0, moments='mvsk')    # mean, var, skew, kurtosis (mvsk)

# update: {[p ^ x] * [(1 - p) ^ (n - x)]} * {[ p ^ (a - 1)] * [(1-p) ^ (b - 1)]} / B(a,b)
pRange = np.arange(0, 1, 0.01)
post0 = beta.pdf(pRange, a0, b0)
a1 = a0 + 1
b1 = b0
a2 = a1 + 0
b2 = b1 + 1

plt.figure(1)
plt.subplot(331)
plt.plot(pRange, beta.pdf(pRange, a0, b0), 'b--',
         pRange, beta.pdf(pRange, a1, b1), 'r-')
plt.ylabel('plausibility')
plt.title(r'\textcolor{red}{W} ' + r'\textcolor{black}{L W W W L W L W}', fontsize=8)
plt.text(.05, max(beta.pdf(pRange, a1, b1)) - .15, 'n = 1')
plt.subplot(332)
plt.plot(pRange, beta.pdf(pRange, a1, b1), 'b--',
         pRange, beta.pdf(pRange, a2, b2), 'r-')
plt.title(r'\textcolor{black}{W} ' +
          r'\textcolor{red}{L} ' +
          r'\textcolor{black}{W W W L W L W}', fontsize=8)
plt.text(.05, max(beta.pdf(pRange, a1, b1)) - .15, 'n = 2')
plt.show()
plt.savefig('fig22a.pdf')