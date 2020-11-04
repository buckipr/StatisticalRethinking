### Statistical Rethinking  Chapter 4: Linear Models  Examples and Exercises

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3

### 4.3: A Gaussian model of height
Howell1 = pd.read_csv('Data/Howell1.csv', sep = ';')
type(Howell1)
Howell1.columns
Howell1.shape

type(Howell1['height'])
type(Howell1[['height']])
d2 = Howell1[['height']][Howell1['age'] >= 18]
type(d2)
d2.describe()

# plot observed data
x_plot = np.linspace(136, 180, len(d2))[:, np.newaxis]
kde = KernelDensity(kernel = 'gaussian', bandwidth = 2)
kde.fit(d2)
y = np.exp(kde.score_samples(x_plot))
plt.plot(x_plot, y)
plt.show()

# code chunk 4.13 (set up prior)
sample_mu = np.random.normal(loc=178, scale=20, size=1000)
sample_sigma = np.random.uniform(0, 50, 1000)
prior_h = np.random.normal(sample_mu, sample_sigma, 1000)
sns.set_theme(style='darkgrid')
sns.kdeplot(prior_h, bw=2, label = 'Prior')
plt.show()

# code chunk 4.14 (grid estimation)


# pymc3
size = 50
true_intercept = 1
true_slope = 2
x = np.linspace(0, 1, size)
y = true_intercept + x*true_slope + np.random.normal(scale=.5, size=size)
data = {'x': x, 'y': y}

with Model() as model:
    lm = glm.LinearComponent.from_formula('y ~ x', data)
    sigma = Uniform('sigma', 0, 20)
    y_obs = Normal('y_obs', mu=lm.y_est, sigma=sigma, observed=y)
    trace = sample(2000, cores=2)

plt.figure(figsize=(5, 5))
plt.plot(x, y, 'x')
plot_posterior_predictive_glm(trace)
