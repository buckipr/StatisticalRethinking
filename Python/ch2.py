# Statistical Rethinking
# Chapter 2: Small Worlds and Larger Worlds
# Examples and Exercises

from scipy import stats
import numpy
import pandas as pd

# 2.2: Building a model: binomial likelihood with beta prior
a0 = b0 = 1
beta.stats(a0, b0, moments = 'mvsk') # mean, var, skew, kurtosis (mvsk)

pRange = numpy.arange(0, 1, 0.01)
pRange
dir(pRange)  ## list attributes of the numpy.array
type(pRange)
pRange.ndim
pRange.shape
pRange.size
pRange.dtype
pRange[0]
pRange[0:10:1]

## store posterior (proportional) probs in array
stats.beta.pdf(pRange, a0, b0)
#r = beta.rvs(a, b, size=1000) # generate random numbers


