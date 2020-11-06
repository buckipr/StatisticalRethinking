### Statistical Rethinking  Chapter 4: Linear Models  Examples and Exercises

from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import describe
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
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
kde = KernelDensity(kernel='gaussian', bandwidth=2)
kde.fit(d2)
y = np.exp(kde.score_samples(x_plot))
plt.plot(x_plot, y)
plt.show()

pymc3.kdeplot(d2)
plt.xlabel('height')
plt.ylabel('density')
plt.title('Prior')
plt.show()

# code chunk 4.13 (set up prior)
sample_mu = norm.rvs(loc=178, scale=20, size=1000)
sample_sigma = uniform.rvs(0, 50, 1000)
prior_h = norm.rvs(sample_mu, sample_sigma, 1000)
sns.set_theme(style='darkgrid')
ax = sns.kdeplot(prior_h, bw=2)
ax.set(xlabel='height', title='Prior')
plt.show()

# code chunk 4.14 (grid estimation)
mu_grid = np.linspace(140, 160, 200)
sigma_grid = np.linspace(4, 9, 200)
post_list = [sum(norm.logpdf(d2,m,s)) for m in mu_grid for s in sigma_grid]
post_ll = np.concatenate(post_list, axis=0)

mu_grid_rep = np.repeat(mu_grid, 200)
sigma_grid_rep = np.tile(sigma_grid, 200)
len(post_ll) == len(mu_grid_rep) and len(post_ll) == len(sigma_grid_rep)
post_log_prob = post_ll + norm.logpdf(mu_grid_rep, 178, 20) + uniform.logpdf(sigma_grid_rep, 0, 50)
post_prob = np.exp(post_log_prob - max(post_log_prob))
X, Y = np.meshgrid(mu_grid, sigma_grid)
Z = post_prob.reshape(200,200)
plt.contour(X,Y,Z)
plt.show()

post_prob_std = post_prob/sum(post_prob)
sample_rows = np.random.choice(range(len(post_prob)), size=1000, replace=True, p=post_prob_std)
sample_mu = mu_grid_rep[sample_rows]
sample_sigma = sigma_grid_rep[sample_rows]
plt.scatter(sample_mu, sample_sigma)
plt.show()

sns.kdeplot(x=sample_mu, y=sample_sigma, color="b", linewidths=1)
plt.xlabel('mean')
plt.ylabel('sd')
plt.title('Posterior')
plt.show()

g = sns.JointGrid(x=sample_mu, y=sample_sigma, space=0)
g.plot_joint(sns.kdeplot,
             fill=True, #clip=((2200, 6800), (10, 25)),
             thresh=0, levels=100, cmap="rocket")
g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
plt.title('Posterior')
plt.show()

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
