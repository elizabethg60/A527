import numpy as np
import matplotlib.pyplot as plt

#read in data
with open('hw2_fitting.dat') as f:
    x = np.array([float(line.split()[0]) for line in f])
with open('hw2_fitting.dat') as f:
    y = np.array([float(line.split()[1]) for line in f])
with open('hw2_fitting.dat') as f:
    sigma = np.array([float(line.split()[2]) for line in f])

def model(xi, mu, alpha, A):
#returns the Gaussian model value provided: x, mu, alpha, and A
    return (A/alpha)*(np.sqrt(np.log(2)/np.pi))*np.exp((-np.log(2)*(xi-mu)**2)/alpha**2)

#visualizing the data with a given model 
plt.figure(figsize=(8, 6), dpi=80)
plt.errorbar(x, y, yerr = sigma, fmt = 'o', markeredgecolor='green', markerfacecolor='green', c = 'black', label = 'Data')
plt.scatter(x, model(x, 45, 14, 0.9), label = 'Model')
plt.title("initial determined parameters", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.savefig("Figures/midterm/guess.pdf")
plt.show()

"""
Construct a Markov Chain to sample from the posterior, assuming flat prior for μ, αD and
A, and using a uniform proposal distribution. Write a computer program to Implement the
Markov Chain.
"""

def log_like(xi, yi, sigmai, mui, alphai, Ai): 
#returns calculated loglikelihood given parameters 
    return np.sum(-((yi - model(xi, mui, alphai, Ai))/(np.sqrt(2)*sigmai))**2)

#number of mcmc iterations
nm = 10000

#initialize the MCMC + create list to store states 
mu = [np.random.uniform(0,100)]*(nm)
alpha = [np.random.uniform(0,50)]*(nm)
A = [np.random.uniform(0,2)]*(nm)

#loglikelihood stores log(likelihood)
loglikelihood = [log_like(x, y, sigma, mu[0], alpha[0], A[0])]*(nm)

for i in range(1,nm):
    #find next step with a uniform proposal
    mu[i] = mu[i-1] + 2*(np.random.uniform(0,1)-0.5) * 0.06
    alpha[i] = alpha[i-1] + 2*(np.random.uniform(0,1)-0.5) * 0.04
    A[i] = A[i-1] + 2*(np.random.uniform(0,1)-0.5) * 0.0025

    #compute the log of likelihood 
    loglikelihood[i] = log_like(x, y, sigma, mu[i], alpha[i], A[i])
    #compare with previous likelihood
    dy = loglikelihood[i]-loglikelihood[i-1]

    #accept prob is exp(dy) i.e., curr_posterior/prev_posterior
    a = np.exp(min(0.0,dy))
    b = np.random.uniform(0,1) 
    if a < b:
        #reject, use the previous state
        mu[i] = mu[i-1]
        alpha[i] = alpha[i-1]
        A[i] = A[i-1]
        loglikelihood[i] = loglikelihood[i-1]

"""
Run the MCMC program and experiment with different starting values of the parameters
and the width of the proposal distribution function, and choose the most appropriate ones
that result in a well-mixed Markov Chain. Plot μ, αD, and A along the Markov Chain
history, and determine the burn-in period. Plot the distribution functions of μ, αD, and A
using the Markov Chain after discarding the burn-in period.
"""
ix = np.arange(nm)

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(ix, mu)
plt.xlabel('mcmc iteration', fontsize=15)
plt.ylabel('mu', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("Figures/midterm/mu_chain2.pdf")
plt.show()

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(ix, alpha)
plt.xlabel('mcmc iteration', fontsize=15)
plt.ylabel('alpha', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("Figures/midterm/alpha_chain2.pdf")
plt.show()

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(ix, A)
plt.xlabel('mcmc iteration', fontsize=15)
plt.ylabel('A', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("Figures/midterm/A_chain2.pdf")
plt.show()

#visualizing the data with best fit model 
plt.figure(figsize=(8, 6), dpi=80)
plt.errorbar(x, y, yerr = sigma, fmt = 'o', markeredgecolor='green', markerfacecolor='green', c = 'black', label = 'Data')
plt.scatter(x, model(x, mu[-1], alpha[-1], A[-1]), label = 'Model')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.savefig("Figures/midterm/best.pdf")
plt.show()
