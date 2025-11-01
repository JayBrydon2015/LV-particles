# %%

# -*- coding: utf-8 -*-

""" Simple exponential growth """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Particles package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments

## Constants ##
DELTA_T = 2 # Data collected every DELTA_T-th time.
K = 0.8 # Nudge factor of guided P

def get_obs(t):
    """ Returns true if an observation is aquired at this time. """
    return t > 0 and t % DELTA_T == 0

class SimpleExpGrowth(ssm.StateSpaceModel):
    """
    Simple exponential growth SSM
    """

    def PX0(self):
        return dists.Normal(loc=self.mu0, scale=self.sigma0)

    def PX(self, t, xp):
        return dists.Normal(loc=self.alpha * xp, scale=self.sigma)

    def PY(self, t, xp, x):
        if get_obs(t):
            return dists.Normal(loc=x, scale=self.gamma)
        else:
            return dists.FlatNormal(loc=np.zeros(len(x)))

class SimpleExpGrowth_proposal(SimpleExpGrowth): # A non-optimal proposal
    """ A non-optimal proposal """
    
    def proposal0(self, data): # t = 0
        if np.isnan(data[0]):
            return self.PX0()
        else:
            return self.PX0() # EDIT if data exists at t = 0
    def proposal(self, t, xp, data): # t >= 1
        if np.isnan(data[t]):
            return self.PX(t, xp)
        else:
            mu = self.alpha * xp
            nudge = K * (data[t] - mu)
            proposal_sigma = self.sigma + self.gamma
            return dists.Normal(loc = nudge + mu, scale = proposal_sigma)

class SimpleExpGrowth_proposal_lf(SimpleExpGrowth): # A non-optimal proposal
    """ A non-optimal proposal with look forward (for sparse observations) """
    
    def proposal0(self, data): # t = 0
        if np.isnan(data[0]):
            return self.PX0()
        else:
            return self.PX0() # EDIT if data[0] exists
    def proposal(self, t, xp, data): # t >= 1
        if np.isnan(data[t]):
            k = DELTA_T - t % DELTA_T
            prop_ps = []
            for p in xp:
                mu = self.alpha * p
                prop_ps.append(mu)
            prop_ps_t = np.array([p for p in prop_ps])
            for _ in range(k):
                for i in range(len(prop_ps)):
                    prop_ps[i] *= self.alpha
            weights = []
            if not get_obs(t+k): # To check
                raise Exception("t + k is not where data is!")
            for p in prop_ps:
                weights.append(stats.norm.pdf(data[t+k],
                                               loc=p,
                                               scale=self.gamma))
            weights = np.array(weights).reshape(len(xp),)
            weights = weights / np.sum(weights) # Normalise weights
            data_t_est = np.average(prop_ps_t, weights=weights)
            
            mu = self.alpha * xp
            nudge = K * (data_t_est - mu)
            proposal_sigma = self.sigma + self.gamma
            return dists.Normal(loc = nudge + mu, scale = proposal_sigma)
        else:
            mu = self.alpha * xp
            nudge = K * (data[t] - mu)
            proposal_sigma = self.sigma + self.gamma
            return dists.Normal(loc = nudge + mu, scale = proposal_sigma)

## Define parameters ##
mu0 = 3; sigma0 = 0.8 # Initial dist. parameters
sigma = 0.3 # Propagation noise
gamma = 0.05 # Observation noise
alpha = 1.05 # Growth rate

## Define time period ##
T = 10 # t = 0, 1, ..., T
t_obs = [t for t in range(T+1) if get_obs(t)]
t_system = range(T+1)

## Number of particles for PFs ##
N = 100

seg_ssm = SimpleExpGrowth_proposal(mu0 = mu0, sigma0 = sigma0, 
                          sigma = sigma, gamma = gamma, 
                          alpha = alpha)
seg_ssm_lf = SimpleExpGrowth_proposal_lf(mu0 = mu0, sigma0 = sigma0, 
                          sigma = sigma, gamma = gamma, 
                          alpha = alpha)

true_states, data = seg_ssm.simulate(T+1)
data_clean = [val for val in data if not np.isnan(val)] # For plotting

# %%

plt.plot(t_system, true_states, label="true state", color='red')
plt.plot(t_obs, data_clean, label="observation", color='blue', marker="o",
         alpha=0.7)
plt.xlabel("t")
plt.ylabel("Value")
plt.title("True state vs. Observations")
plt.legend()
plt.grid(True)
plt.show()

# %%

## Bootstrap PF ##

fk_boot = ssm.Bootstrap(ssm=seg_ssm, data=data)
pf_boot = particles.SMC(fk=fk_boot, N=N, resampling='stratified', 
                        store_history=True, collect=[Moments()])
pf_boot.run()

# %%

## Guided PF no LF ##

fk_guided = ssm.GuidedPF(ssm=seg_ssm, data=data)
pf_guided = particles.SMC(fk=fk_guided, N=N, resampling='stratified', 
                          store_history=True, collect=[Moments()])
pf_guided.run()

# %%

## Guided PF with LF ##

fk_guided_lf = ssm.GuidedPF(ssm=seg_ssm_lf, data=data)
pf_guided_lf = particles.SMC(fk=fk_guided_lf, N=N, resampling='stratified', 
                             store_history=True, collect=[Moments()])
pf_guided_lf.run()

# %%

#### Filtering band plots ####

## Bootstrap ##

means_boot =  np.stack([m['mean'] for m in pf_boot.summaries.moments])
vars_boot = np.stack([m['var'] for m in pf_boot.summaries.moments])

plt.plot(t_system, true_states, label="true state", color='red', alpha=0.7)
plt.plot(means_boot, color="green", label="PF mean", alpha=0.7)
plt.plot(t_obs, data_clean, label="observation", color='blue', alpha=0.3,
         marker='o')
plt.fill_between(t_system, 
                 y1=means_boot-2*np.sqrt(vars_boot), 
                 y2=means_boot+2*np.sqrt(vars_boot), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Bootstrap PF band plot")

# %%
## Guided no LF ##

means_guided =  np.stack([m['mean'] for m in pf_guided.summaries.moments])
vars_guided = np.stack([m['var'] for m in pf_guided.summaries.moments])

plt.plot(t_system, true_states, label="true state", color='red', alpha=0.7)
plt.plot(means_guided, color="green", label="PF mean", alpha=0.7)
plt.plot(t_obs, data_clean, label="observation", color='blue', alpha=0.3,
         marker='o')
plt.fill_between(t_system, 
                 y1=means_guided-2*np.sqrt(vars_guided), 
                 y2=means_guided+2*np.sqrt(vars_guided), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Guided PF band plot")

# %%
## Guided with LF ##

means_guided_lf =  np.stack([m['mean'] for m in pf_guided_lf.summaries.moments])
vars_guided_lf = np.stack([m['var'] for m in pf_guided_lf.summaries.moments])

plt.plot(t_system, true_states, label="true state", color='red', alpha=0.7)
plt.plot(means_guided_lf, color="green", label="PF mean", alpha=0.7)
plt.plot(t_obs, data_clean, label="observation", color='blue', alpha=0.3,
         marker='o')
plt.fill_between(t_system, 
                 y1=means_guided_lf-2*np.sqrt(vars_guided_lf), 
                 y2=means_guided_lf+2*np.sqrt(vars_guided_lf), 
                 color="green", alpha=0.3)
plt.legend()
plt.xlabel("t")
plt.ylabel("Value")
plt.title("Guided-LF PF band plot")

# %%

## Plot histograms/KDEs and box-plots at t = n ##
n = 1

## Box plots ##

plt.boxplot([pf_boot.hist.X[n], pf_guided.hist.X[n], pf_guided_lf.hist.X[n]],
            tick_labels=["Boot PF", "Guided PF", "Guided-LF PF"])
plt.scatter([1, 2, 3], [true_states[n], true_states[n], true_states[n]], color='red', 
            marker='x', s=100, label='True state')
plt.title('Filtering Dists @ t=n')
plt.ylabel('Value')
plt.legend()
plt.show()

## KDEs ##

fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(pf_boot.hist.X[n], ax=ax, fill=True,
            color="skyblue", label="Boot")
sns.kdeplot(pf_guided.hist.X[n], ax=ax, fill=True,
            color="lightcoral", label="Guided")
sns.kdeplot(pf_guided_lf.hist.X[n], ax=ax, fill=True,
            color="gold", label="Guided-LF")
ax.axvline(x=true_states[n], color='red', linestyle=':', linewidth=1.5, 
           label='True state')
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Filtering Dists @ t=n")
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



