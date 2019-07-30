# Statistical Rethinking
# Chapter 2: Small Worlds and Larger Worlds
# Examples and Exercises

from scipy.stats import beta
from scipy.stats import binom
from scipy.stats import uniform
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
# maybe check out AlogPy

# 2.2: Building a model: binomial likelihood with (flat) beta prior
data22 = ['W', 'L', 'W', 'W', 'W', 'L', 'W', 'L', 'W']
aPost = np.cumsum([1 if x == 'W' else 0 for x in data22]) + 1
aPost.shape
a = np.insert(aPost, 0, 1)
bPost = np.cumsum([0 if x == 'W' else 1 for x in data22]) + 1
b = np.insert(bPost, 0, 1)

beta.stats(a[0], b[0], moments='mvsk')    # mean, var, skew, kurtosis (mvsk)

# update: {[p ^ x] * [(1 - p) ^ (n - x)]} * {[ p ^ (a - 1)] * [(1-p) ^ (b - 1)]} / B(a,b)
pRange = np.arange(0, 1, 0.01)

yVals = beta.pdf(pRange, a[0], b[0])
for i in range(1, len(a)):
    yVals = np.append(yVals, beta.pdf(pRange, a[i], b[i]))
yMax = max(yVals)

plt.figure(1)
for i in range(0, len(a) - 1):
    plotloc = '33' + str(i + 1)
    plt.subplot(plotloc)
    plt.plot(pRange, beta.pdf(pRange, a[i], b[i]), 'b--',
             pRange, beta.pdf(pRange, a[i + 1], b[i + 1]), 'r-')
    plt.axis([0, 1, 0, yMax + .1])
    plt.tight_layout()
    if i in [6, 7, 8]:
        plt.xlabel('proportion of water')
    if i in [0, 3, 6]:
        plt.ylabel('plausibility')
    for j in range(0, len(a) - 1):
        tmpCol = 'black'
        if i >= j:
            tmpCol = 'red'
        tmpX = .05 + (j + 1) * .09
        plt.text(tmpX, yMax + .3, data22[j], color=tmpCol)
plt.show()
plt.savefig('fig22a.pdf')

# Section 2.4: R code 2.3, 2.4, & 2.5
p_grid = np.arange(0, 1, .05)
prior = np.repeat(1, 20)
likelihood = binom.pmf(k=6, n=9, p=p_grid)
unstd_posterior = likelihood * prior
posterior = unstd_posterior / sum(unstd_posterior)
plt.plot(p_grid, posterior, 'bo-')

prior2 = [0 if x < 0.5 else 1 for x in p_grid]
unstd_posterior2 = likelihood * prior2
posterior2 = unstd_posterior2 / sum(unstd_posterior2)

prior3 = np.exp(-5*abs(p_grid - 0.5))
unstd_posterior3 = likelihood * prior3
posterior3 = unstd_posterior3 / sum(unstd_posterior3)

plt.figure(1)
plt.subplot(131)
plt.plot(p_grid, posterior, 'o-')
plt.title('flat prior')
plt.axis([0, 1, 0, .2])
plt.subplot(132)
plt.plot(p_grid, posterior2, 'o-')
plt.axis([0, 1, 0, .2])
plt.title('0/1 prior')
plt.subplot(133)
plt.plot(p_grid, posterior3, 'o-')
plt.axis([0, 1, 0, .2])
plt.title('normalish prior')

# Section 2.6: quadratic approximation
def negLogLike ( x ):
    p = -1 * (binom.logpmf(k=6, n=9, p=x) + uniform.logpdf(x, 0, 1))
    return p

negLogLike(.4)
result = minimize(negLogLike, .5, method='nelder-mead', options={'disp':True})
result.x