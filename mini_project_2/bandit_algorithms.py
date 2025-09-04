import numpy as np
from scipy.stats import norm, beta
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class BanditAlgorithm:
    """Base class for bandit algorithms"""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.reset()
    
    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.t = 0
    
    def select_arm(self):
        raise NotImplementedError
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.t += 1

class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-Greedy algorithm"""
    def __init__(self, n_arms, epsilon):
        super().__init__(n_arms)
        self.epsilon = epsilon
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            # Handle case where no arm has been pulled
            if np.sum(self.counts) == 0:
                return np.random.randint(self.n_arms)
            
            # Choose arm with highest average reward
            avg_rewards = np.divide(self.rewards, self.counts, 
                                  out=np.zeros_like(self.rewards), 
                                  where=self.counts!=0)
            return np.argmax(avg_rewards)

class UCB(BanditAlgorithm):
    """Upper Confidence Bound algorithm"""
    def __init__(self, n_arms):
        super().__init__(n_arms)
    
    def select_arm(self):
        # First, pull each arm once
        if self.t < self.n_arms:
            return self.t
        
        # Calculate UCB values
        avg_rewards = self.rewards / self.counts
        confidence = np.sqrt(2 * np.log(self.t) / self.counts)
        ucb_values = avg_rewards + confidence
        
        return np.argmax(ucb_values)

class ThompsonSamplingBernoulli(BanditAlgorithm):
    """Thompson Sampling for Bernoulli rewards"""
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
    
    def select_arm(self):
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        super().update(arm, reward)
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

class ThompsonSamplingGaussian(BanditAlgorithm):
    """Thompson Sampling for Gaussian rewards"""
    def __init__(self, n_arms, sigma=1.0):
        super().__init__(n_arms)
        self.sigma = sigma  # Known variance
        self.sum_rewards = np.zeros(n_arms)
    
    def select_arm(self):
        # Sample from posterior distribution (Normal-Normal conjugate)
        samples = np.zeros(self.n_arms)
        
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                # Prior: N(0, 1)
                samples[i] = np.random.normal(0, 1)
            else:
                # Posterior mean and variance
                posterior_var = 1 / (1 + self.counts[i] / (self.sigma**2))
                posterior_mean = posterior_var * self.sum_rewards[i] / (self.sigma**2)
                
                samples[i] = np.random.normal(posterior_mean, np.sqrt(posterior_var))
        
        return np.argmax(samples)
    
    def update(self, arm, reward):
        super().update(arm, reward)
        self.sum_rewards[arm] += reward

class KL_UCB(BanditAlgorithm):
    """KL-UCB algorithm for Bernoulli rewards"""
    def __init__(self, n_arms):
        super().__init__(n_arms)
    
    def kl_divergence(self, p, q):
        """KL divergence between two Bernoulli distributions"""
        if p == 0:
            return -np.log(1 - q)
        elif p == 1:
            return -np.log(q)
        else:
            return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    def select_arm(self):
        # First, pull each arm once
        if self.t < self.n_arms:
            return self.t
        
        # Calculate KL-UCB values
        kl_ucb_values = np.zeros(self.n_arms)
        
        for i in range(self.n_arms):
            p_hat = self.rewards[i] / self.counts[i]
            threshold = np.log(self.t) / self.counts[i]
            
            # Find q that satisfies KL(p_hat, q) = threshold
            if p_hat == 1:
                kl_ucb_values[i] = 1
            else:
                def objective(q):
                    if q <= p_hat or q >= 1:
                        return np.inf
                    return (self.kl_divergence(p_hat, q) - threshold)**2
                
                result = minimize_scalar(objective, bounds=(p_hat + 1e-6, 1 - 1e-6), 
                                       method='bounded')
                kl_ucb_values[i] = result.x
        
        return np.argmax(kl_ucb_values)

class SuccessiveElimination(BanditAlgorithm):
    """Successive Elimination algorithm for unknown number of arms"""
    def __init__(self, n_arms, delta=0.1):
        super().__init__(n_arms)
        self.delta = delta
        self.active_arms = set(range(n_arms))
        self.round_number = 1
        self.arms_per_round = n_arms
        self.pulls_per_arm = 1
    
    def select_arm(self):
        if len(self.active_arms) == 1:
            return list(self.active_arms)[0]
        
        # Check if we need to move to next round
        total_pulls_needed = self.pulls_per_arm * len(self.active_arms)
        current_pulls = sum(self.counts[i] for i in self.active_arms)
        
        if current_pulls >= total_pulls_needed:
            self._eliminate_arms()
            self.round_number += 1
            self.pulls_per_arm = int(np.ceil((2 * np.log(3 * self.n_arms / self.delta)) / 
                                           (2**self.round_number - 2)))
        
        # Select arm with minimum pulls among active arms
        active_counts = {i: self.counts[i] for i in self.active_arms}
        min_pulls = min(active_counts.values())
        candidates = [i for i in self.active_arms if self.counts[i] == min_pulls]
        
        return np.random.choice(candidates)
    
    def _eliminate_arms(self):
        """Eliminate sub-optimal arms"""
        if len(self.active_arms) <= 1:
            return
        
        # Calculate confidence intervals
        confidence_width = np.sqrt((2 * np.log(3 * self.n_arms / self.delta)) / 
                                 (2**self.round_number - 2))
        
        arm_estimates = {}
        for arm in self.active_arms:
            if self.counts[arm] > 0:
                arm_estimates[arm] = self.rewards[arm] / self.counts[arm]
        
        if not arm_estimates:
            return
        
        best_estimate = max(arm_estimates.values())
        
        # Eliminate arms whose upper confidence bound is below 
        # the lower confidence bound of the best arm
        arms_to_remove = []
        for arm in self.active_arms:
            if arm in arm_estimates:
                upper_bound = arm_estimates[arm] + confidence_width
                lower_bound_best = best_estimate - confidence_width
                
                if upper_bound < lower_bound_best:
                    arms_to_remove.append(arm)
        
        for arm in arms_to_remove:
            self.active_arms.discard(arm)