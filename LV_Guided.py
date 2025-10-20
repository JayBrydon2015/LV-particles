# %%

# -*- coding: utf-8 -*-

""" Implements a LV SSM and runs a particle filter for state estimation. """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Particles package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists

def get_obs(t):
    """ Returns true if an observation is aquired at this time. """
    return True #t % 10 == 0 and t > 0

class LotkaVolterra(ssm.StateSpaceModel):
    """
    Predator-Prey Model

    Parameters:
        alpha, beta, gamma, delta are the model rate parameters.
        h is the constant change in time.
        n0 is the mean initial predator population; n1 the mean initial prey population.
        sigmaPrime and tauPrime are the variances of the initial log-predator and log-prey populations.
        sigma and tau are the variances of the transition densities.
        theta is the probability of observing a particular prey.
        Note: alpha, beta, gamma, delta > 0. n0, n1 > 0.
    """

    def PX0(self):
        d = dists.MvNormal(loc=np.array([np.log(self.n0), np.log(self.n1)]),
                           scale=np.array([(self.sigmaPrime), (self.tauPrime)])) 
        return d

    def PX(self, t, xp):
        mu0 = self.h * (self.beta * np.exp(xp[:,1]) - self.alpha) + xp[:,0]
        mu1 = self.h * (self.gamma - self.delta * np.exp(xp[:,0])) + xp[:,1]
        d = dists.MvNormal(loc=np.vstack((mu0,mu1)).T,
                           scale=np.array([self.sigma, self.tau]))
        return d

    def PY(self, t, xp, x):
        if get_obs(t):
            return dists.Binomial(n=np.rint(np.exp(x[:,0])).astype(int), p=self.theta)
        else:
            return dists.FlatNormal(loc=np.zeros(len(x)))

class LotkaVolterra_proposal(LotkaVolterra):
    def proposal0(self, data):
        return self.PX0() # No data at t=0
    def proposal(self, t, xp, data):
        if np.isnan(data[t]): # data[t]?
            return self.PX(t, xp)
        mu0 = self.h * (self.beta * np.exp(xp[:,1]) - self.alpha) + xp[:,0]
        mu1 = self.h * (self.gamma - self.delta * np.exp(xp[:,0])) + xp[:,1]
        K = 0.8
        nudge = K * (np.log(data[t] / self.theta) - mu0)
        new_mu0 = mu0 + nudge
        return dists.MvNormal(loc=np.vstack((new_mu0,mu1)).T,
                              scale=np.array([self.sigma, self.tau]))

## Define parameters ##
alpha = 0.5; beta = 0.02; gamma = 0.8; delta = 0.01
h = 0.002
n0 = 20; n1 = 40
sigmaPrime = 0.5; tauPrime = 0.6
sigma = 0.02; tau = 0.01 # Need to go smaller for smaller h
theta = 1

## Define time period ##
T = 8000 # t = 0, 1, ..., T
t_obs = [t for t in range(T+1) if get_obs(t)]
t_system = range(T+1)

## Run SSM simulation ##
lv_ssm = LotkaVolterra_proposal(alpha=alpha, beta=beta, gamma=gamma,
                                delta=delta, h=h, n0=n0, n1=n1,
                                sigmaPrime=sigmaPrime, tauPrime=tauPrime,
                                sigma=sigma, tau=tau, theta=theta)
true_states, data = lv_ssm.simulate(T+1)
data_clean = [val for val in data if not np.isnan(val)] # For plotting
pred_vals = []
prey_vals = []
for i in range(T+1): # pop or log-pop
        pred, prey = true_states[i][0]
        pred_vals.append(pred)
        prey_vals.append(prey)

#%%
## Plot simulation results ##

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,6))

# Top plot: observed predator population
# ax1.plot(t_obs, data_clean, '.-', label="pred obs") # If obs are sparse
ax1.plot(t_obs, data_clean, label="pred obs")
ax1.set_ylabel("population")
ax1.legend()
ax1.grid(True)

# Bottom plot: true predator + prey (log-)populations
ax2.plot(t_system, pred_vals, label="Predator", color="red")
ax2.plot(t_system, prey_vals, label="Prey", color="blue")
# Add vertical lines at observation times onto bottom plot
#for t in t_obs:
#    ax2.axvline(x=t, color="green", linestyle="--", alpha=0.2)
ax2.set_xlabel("t")
ax2.set_ylabel("log-population")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()

#%%
## Bootstrap Particle Filter ##

# Create and run particle filter
fk_boot = ssm.Bootstrap(ssm=lv_ssm, data=data)
pf_boot = particles.SMC(fk=fk_boot, N=1000, resampling='stratified', store_history=True)
pf_boot.run()

#%%
## Guided Particle Filter ##

fk_guided = ssm.GuidedPF(ssm=lv_ssm, data=data)
pf_guided = particles.SMC(fk=fk_guided, N=1000, resampling='stratified', store_history=True)
pf_guided.run()

#%%

#pf_both = particles.multiSMC(fk={'boot':fk_boot, 'guided':fk_guided},
#                             nruns=20, nprocs=1, out_func=outf)

#%%

## Plot histograms, box-plots, and KDEs at t = n ##
n = 21

## Box plots ##

# Predator
plt.boxplot([pf_boot.hist.X[n][:, 0], pf_guided.hist.X[n][:, 0]],
            labels=["Boot PF", "Guided PF"])
plt.title('Predator Log-Population Filtering Dists @ t=n')
plt.ylabel('Log-pop')
plt.show()

# Prey
plt.boxplot([pf_boot.hist.X[n][:, 1], pf_guided.hist.X[n][:, 1]],
            labels=["Boot PF", "Guided PF"])
plt.title('Prey Log-Population Filtering Dists @ t=n')
plt.ylabel('Log-pop')
plt.show()

## KDEs ##

# Predator
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(pf_boot.hist.X[n][:, 0], ax=ax, fill=True,
            color="skyblue", label="Boot")
sns.kdeplot(pf_guided.hist.X[n][:, 0], ax=ax, fill=True,
            color="lightcoral", label="Guided")
ax.set_xlabel("Log-pop")
ax.set_ylabel("Density")
ax.set_title("Predator Log-Population Filtering Dists @ t=n")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Prey
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(pf_boot.hist.X[n][:, 1], ax=ax, fill=True,
            color="skyblue", label="Boot")
sns.kdeplot(pf_guided.hist.X[n][:, 1], ax=ax, fill=True,
            color="lightcoral", label="Guided")
ax.set_xlabel("Log-pop")
ax.set_ylabel("Density")
ax.set_title("Prey Log-Population Filtering Dists @ t=n")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

## Histograms ##

# Predator
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(pf_boot.hist.X[n][:, 0], bins=30, alpha=0.5, label='Boot',
        color='skyblue', density=True)
ax.hist(pf_guided.hist.X[n][:, 0], bins=30, alpha=0.5, label='Guided', 
        color='salmon', density=True)
ax.set_title("Predator Log-Population Filtering Dists @ t=n")
ax.set_xlabel("Log-pop")
ax.set_ylabel('Density')
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Prey
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(pf_boot.hist.X[n][:, 1], bins=30, alpha=0.5, label='Boot',
        color='skyblue', density=True)
ax.hist(pf_guided.hist.X[n][:, 1], bins=30, alpha=0.5, label='Guided', 
        color='salmon', density=True)
ax.set_title("Prey Log-Population Filtering Dists @ t=n")
ax.set_xlabel("Log-pop")
ax.set_ylabel('Density')
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

