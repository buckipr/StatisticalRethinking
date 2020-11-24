### Statistical Rethinking  Chapter 6: Overfitting, Regularization, and Information Criteria

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
import pymc3 as pm
from scipy import stats
import arviz as az

## note: statsmodels is an extensive stats package

# 6.5 Using information criteria

# prepare data

milk = pd.read_csv('Data/milk.csv', sep=';')
milk.shape
d = milk.dropna().copy()
d.shape
d.columns
d['neocortex'] = d['neocortex.perc']/100
d[['neocortex', 'neocortex.perc']]
d['lmass'] = np.log(d['mass'])

# fit models
d['kcal.per.g'].describe()
#m6_11 = pm.Model()
with pm.Model() as m6_11:
    alpha = pm.Uniform('alpha', 0, 5)
    log_sigma = pm.Uniform('log_sigma', -10, 10)
    mu = alpha
    y_obs = pm.Normal('y_obs', mu=mu, sigma=np.exp(log_sigma), observed=d['kcal.per.g'])

pm.find_MAP(model=m6_11, method='BFGS')

with m6_11:
    trace = pm.sample(2000, return_inferencedata=True, chains=2)
pm.summary(trace)
az.summary(trace)
#pm.gelman_rubin(trace)
with m6_11:
    az.plot_trace(trace)
plt.show()
az.plot_autocorr(trace)
plt.show()
az.plot_density(trace)
plt.show()
az.plot_forest(trace)
plt.show()

# might need to multiply by -2 to compare with McElreath
with m6_11:
    print(pm.waic(trace))
    print(pm.loo(trace))


#m6_13 = pm.Model()
with pm.Model() as m6_13:
    alpha = pm.Uniform('alpha', 0, 5)
    bm = pm.Uniform('bm', -10, 10)
    log_sigma = pm.Uniform('log_sigma', -10, 10)
    mu = alpha + bm*d['lmass']
    y_obs = pm.Normal('y_obs', mu=mu, sigma=np.exp(log_sigma), observed=d['kcal.per.g'])
    trace = pm.sample(2000, return_inferencedata=True, chains=2)

with m6_13:
    print(pm.summary(trace))
    print(pm.waic(trace))
    print(pm.loo(trace))


with m6_13:
    print(-2*pm.loo(trace))

pm.compare({m6_11: trace, m6_13: trace})
pm.compare({m6_11: trace, m6_13: trace}, ic='WAIC')
