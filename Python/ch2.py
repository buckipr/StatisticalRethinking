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
    plt.ylabel('plausibility')
    plt.axis([0, 1, 0, yMax + .1])
    plt.title(r'\textcolor{red}{W} ' + r'\textcolor{black}{L W W W L W L W}', fontsize=8)
    plt.text(.05, yMax - .2, 'n = ' + str(i + 1))
    # plt.subplot(332)
    # plt.plot(pRange, beta.pdf(pRange, a1, b1), 'b--',
    #          pRange, beta.pdf(pRange, a2, b2), 'r-')
    # plt.title(r'\textcolor{black}{W} ' +
    #           r'\textcolor{red}{L} ' +
    #           r'\textcolor{black}{W W W L W L W}', fontsize=8)
    # plt.text(.05, max(beta.pdf(pRange, a1, b1)) - .15, 'n = 2')
plt.show()
plt.savefig('fig22a.pdf')
