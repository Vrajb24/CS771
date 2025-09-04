import numpy as np

class BanditEnvironment:
    """Base class for bandit environments"""
    def __init__(self, n_arms, random_seed=None):
        self.n_arms = n_arms
        if random_seed is not None:
            np.random.seed(random_seed)
        self._initialize_arms()
    
    def _initialize_arms(self):
        raise NotImplementedError
    
    def pull_arm(self, arm):
        raise NotImplementedError
    
    def get_optimal_arm(self):
        raise NotImplementedError
    
    def get_expected_rewards(self):
        raise NotImplementedError

class BernoulliBandit(BanditEnvironment):
    """Bernoulli bandit environment"""
    def __init__(self, n_arms, probabilities=None, random_seed=None):
        self.probabilities = probabilities
        super().__init__(n_arms, random_seed)
    
    def _initialize_arms(self):
        if self.probabilities is None:
            # Generate random probabilities
            self.probabilities = np.random.uniform(0.1, 0.9, self.n_arms)
        else:
            assert len(self.probabilities) == self.n_arms
            self.probabilities = np.array(self.probabilities)
    
    def pull_arm(self, arm):
        return np.random.bernoulli(self.probabilities[arm])
    
    def get_optimal_arm(self):
        return np.argmax(self.probabilities)
    
    def get_expected_rewards(self):
        return self.probabilities.copy()
    
    def get_regret(self, arm):
        optimal_reward = np.max(self.probabilities)
        return optimal_reward - self.probabilities[arm]

class GaussianBandit(BanditEnvironment):
    """Gaussian bandit environment"""
    def __init__(self, n_arms, means=None, variances=None, random_seed=None):
        self.means = means
        self.variances = variances
        super().__init__(n_arms, random_seed)
    
    def _initialize_arms(self):
        if self.means is None:
            # Generate random means
            self.means = np.random.uniform(-2, 2, self.n_arms)
        else:
            assert len(self.means) == self.n_arms
            self.means = np.array(self.means)
        
        if self.variances is None:
            # Default variance of 1 for all arms
            self.variances = np.ones(self.n_arms)
        else:
            assert len(self.variances) == self.n_arms
            self.variances = np.array(self.variances)
    
    def pull_arm(self, arm):
        return np.random.normal(self.means[arm], np.sqrt(self.variances[arm]))
    
    def get_optimal_arm(self):
        return np.argmax(self.means)
    
    def get_expected_rewards(self):
        return self.means.copy()
    
    def get_regret(self, arm):
        optimal_reward = np.max(self.means)
        return optimal_reward - self.means[arm]

class InstanceGenerator:
    """Generate bandit instances for experiments"""
    
    @staticmethod
    def generate_bernoulli_instance(n_arms, difficulty='medium', random_seed=None):
        """Generate Bernoulli bandit instance with specified difficulty"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if difficulty == 'easy':
            # Large gaps between arms
            optimal_prob = 0.8
            suboptimal_probs = np.random.uniform(0.2, 0.4, n_arms - 1)
        elif difficulty == 'medium':
            # Medium gaps
            optimal_prob = 0.7
            suboptimal_probs = np.random.uniform(0.4, 0.6, n_arms - 1)
        elif difficulty == 'hard':
            # Small gaps
            optimal_prob = 0.6
            suboptimal_probs = np.random.uniform(0.5, 0.58, n_arms - 1)
        else:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
        
        # Randomly place the optimal arm
        probabilities = np.append(suboptimal_probs, optimal_prob)
        np.random.shuffle(probabilities)
        
        return BernoulliBandit(n_arms, probabilities)
    
    @staticmethod
    def generate_gaussian_instance(n_arms, difficulty='medium', random_seed=None):
        """Generate Gaussian bandit instance with specified difficulty"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if difficulty == 'easy':
            # Large gaps between arms
            optimal_mean = 2.0
            suboptimal_means = np.random.uniform(-1.0, 0.5, n_arms - 1)
        elif difficulty == 'medium':
            # Medium gaps
            optimal_mean = 1.0
            suboptimal_means = np.random.uniform(-0.5, 0.5, n_arms - 1)
        elif difficulty == 'hard':
            # Small gaps
            optimal_mean = 0.5
            suboptimal_means = np.random.uniform(0.0, 0.4, n_arms - 1)
        else:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
        
        # Randomly place the optimal arm
        means = np.append(suboptimal_means, optimal_mean)
        np.random.shuffle(means)
        
        # Unit variance for all arms
        variances = np.ones(n_arms)
        
        return GaussianBandit(n_arms, means, variances)
    
    @staticmethod
    def generate_test_instances():
        """Generate standard test instances for reproducible experiments"""
        instances = {}
        
        # Bernoulli instances
        instances['bernoulli_easy'] = BernoulliBandit(5, [0.2, 0.3, 0.8, 0.4, 0.1])
        instances['bernoulli_medium'] = BernoulliBandit(10, [0.4, 0.7, 0.5, 0.6, 0.3, 0.45, 0.55, 0.35, 0.65, 0.25])
        instances['bernoulli_hard'] = BernoulliBandit(20, np.linspace(0.5, 0.6, 20))
        
        # Gaussian instances  
        instances['gaussian_easy'] = GaussianBandit(5, [-1, 0, 2, -0.5, 0.5])
        instances['gaussian_medium'] = GaussianBandit(10, [-0.5, 1.0, 0.0, 0.5, -0.2, 0.3, 0.7, -0.3, 0.8, -0.1])
        instances['gaussian_hard'] = GaussianBandit(20, np.linspace(0.0, 0.5, 20))
        
        return instances