import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithms import *
from bandit_environments import *
import time

class ExperimentRunner:
    """Run bandit experiments and collect results"""
    
    def __init__(self):
        self.results = {}
    
    def run_single_experiment(self, algorithm, environment, T, seed=None):
        """Run a single experiment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset algorithm and environment
        algorithm.reset()
        
        # Track metrics
        regret_history = []
        reward_history = []
        arm_history = []
        
        cumulative_regret = 0
        
        for t in range(T):
            # Select arm
            chosen_arm = algorithm.select_arm()
            
            # Pull arm and get reward
            reward = environment.pull_arm(chosen_arm)
            
            # Update algorithm
            algorithm.update(chosen_arm, reward)
            
            # Calculate regret
            instantaneous_regret = environment.get_regret(chosen_arm)
            cumulative_regret += instantaneous_regret
            
            # Store metrics
            regret_history.append(cumulative_regret)
            reward_history.append(reward)
            arm_history.append(chosen_arm)
        
        return {
            'regret_history': regret_history,
            'reward_history': reward_history,
            'arm_history': arm_history,
            'final_regret': cumulative_regret,
            'final_counts': algorithm.counts.copy()
        }
    
    def run_multiple_experiments(self, algorithm_class, algorithm_params, 
                                environment, T, n_runs=10, seeds=None):
        """Run multiple experiments and average results"""
        if seeds is None:
            seeds = range(n_runs)
        
        all_results = []
        
        for i, seed in enumerate(seeds[:n_runs]):
            # Create fresh algorithm instance
            algorithm = algorithm_class(**algorithm_params)
            
            # Run experiment
            result = self.run_single_experiment(algorithm, environment, T, seed)
            all_results.append(result)
        
        # Average results
        regret_histories = np.array([r['regret_history'] for r in all_results])
        mean_regret = np.mean(regret_histories, axis=0)
        std_regret = np.std(regret_histories, axis=0)
        
        final_regrets = [r['final_regret'] for r in all_results]
        
        return {
            'mean_regret': mean_regret,
            'std_regret': std_regret,
            'final_regrets': final_regrets,
            'all_results': all_results
        }
    
    def compare_algorithms(self, algorithms_config, environment, T, n_runs=10):
        """Compare multiple algorithms on the same environment"""
        comparison_results = {}
        
        print(f"Running comparison on {environment.__class__.__name__} with {environment.n_arms} arms")
        print(f"Time horizon: {T}, Number of runs: {n_runs}")
        print("-" * 60)
        
        for alg_name, (alg_class, alg_params) in algorithms_config.items():
            print(f"Running {alg_name}...")
            start_time = time.time()
            
            results = self.run_multiple_experiments(
                alg_class, alg_params, environment, T, n_runs
            )
            
            comparison_results[alg_name] = results
            
            end_time = time.time()
            mean_final_regret = np.mean(results['final_regrets'])
            print(f"  Completed in {end_time - start_time:.2f}s")
            print(f"  Mean final regret: {mean_final_regret:.4f}")
        
        return comparison_results

class TaskExperiments:
    """Specific experiments for each task"""
    
    def __init__(self):
        self.runner = ExperimentRunner()
    
    def task1_bernoulli_experiments(self, T=1000, n_runs=10):
        """Task 1: Bernoulli bandits with known number of arms"""
        print("=" * 60)
        print("TASK 1: BERNOULLI BANDITS")
        print("=" * 60)
        
        # Define algorithms
        algorithms = {
            'Epsilon-Greedy (ε=0.01)': (EpsilonGreedy, {'n_arms': 10, 'epsilon': 0.01}),
            'Epsilon-Greedy (ε=0.1)': (EpsilonGreedy, {'n_arms': 10, 'epsilon': 0.1}),
            'UCB': (UCB, {'n_arms': 10}),
            'Thompson Sampling': (ThompsonSamplingBernoulli, {'n_arms': 10}),
            'KL-UCB': (KL_UCB, {'n_arms': 10})
        }
        
        # Test on different instances
        instances = {
            'Easy Instance': InstanceGenerator.generate_bernoulli_instance(10, 'easy', 42),
            'Medium Instance': InstanceGenerator.generate_bernoulli_instance(10, 'medium', 42),
            'Hard Instance': InstanceGenerator.generate_bernoulli_instance(10, 'hard', 42)
        }
        
        task1_results = {}
        
        for instance_name, environment in instances.items():
            print(f"\n{instance_name}:")
            print(f"Arm probabilities: {environment.get_expected_rewards()}")
            
            results = self.runner.compare_algorithms(algorithms, environment, T, n_runs)
            task1_results[instance_name] = results
        
        return task1_results
    
    def task2_gaussian_experiments(self, T=1000, n_runs=10):
        """Task 2: Gaussian bandits with known number of arms"""
        print("\n" + "=" * 60)
        print("TASK 2: GAUSSIAN BANDITS")
        print("=" * 60)
        
        # Define algorithms
        algorithms = {
            'Epsilon-Greedy (ε=0.01)': (EpsilonGreedy, {'n_arms': 10, 'epsilon': 0.01}),
            'Epsilon-Greedy (ε=0.1)': (EpsilonGreedy, {'n_arms': 10, 'epsilon': 0.1}),
            'UCB': (UCB, {'n_arms': 10}),
            'Thompson Sampling': (ThompsonSamplingGaussian, {'n_arms': 10, 'sigma': 1.0})
        }
        
        # Test on different instances
        instances = {
            'Easy Instance': InstanceGenerator.generate_gaussian_instance(10, 'easy', 42),
            'Medium Instance': InstanceGenerator.generate_gaussian_instance(10, 'medium', 42),
            'Hard Instance': InstanceGenerator.generate_gaussian_instance(10, 'hard', 42)
        }
        
        task2_results = {}
        
        for instance_name, environment in instances.items():
            print(f"\n{instance_name}:")
            print(f"Arm means: {environment.get_expected_rewards()}")
            
            results = self.runner.compare_algorithms(algorithms, environment, T, n_runs)
            task2_results[instance_name] = results
        
        return task2_results
    
    def task3_elimination_experiments(self, T=5000, n_runs=5):
        """Task 3: Successive elimination for unknown number of arms"""
        print("\n" + "=" * 60)
        print("TASK 3: SUCCESSIVE ELIMINATION (UNKNOWN ARMS)")
        print("=" * 60)
        
        # Test with different number of arms
        arm_counts = [5, 10, 20, 50]
        task3_results = {}
        
        for n_arms in arm_counts:
            print(f"\nTesting with {n_arms} arms:")
            
            # Create environment
            environment = InstanceGenerator.generate_bernoulli_instance(n_arms, 'medium', 42)
            
            # Define algorithms
            algorithms = {
                'Successive Elimination': (SuccessiveElimination, {'n_arms': n_arms, 'delta': 0.1}),
                'UCB (baseline)': (UCB, {'n_arms': n_arms}),
                'Thompson Sampling (baseline)': (ThompsonSamplingBernoulli, {'n_arms': n_arms})
            }
            
            results = self.runner.compare_algorithms(algorithms, environment, T, n_runs)
            task3_results[f'{n_arms}_arms'] = results
        
        return task3_results
    
    def run_all_tasks(self, T_short=1000, T_long=5000, n_runs=10):
        """Run all tasks"""
        all_results = {}
        
        # Task 1: Bernoulli bandits
        all_results['task1'] = self.task1_bernoulli_experiments(T_short, n_runs)
        
        # Task 2: Gaussian bandits
        all_results['task2'] = self.task2_gaussian_experiments(T_short, n_runs)
        
        # Task 3: Successive elimination
        all_results['task3'] = self.task3_elimination_experiments(T_long, max(5, n_runs//2))
        
        return all_results

class ResultsAnalyzer:
    """Analyze and visualize experimental results"""
    
    @staticmethod
    def plot_regret_curves(results, title, save_path=None):
        """Plot regret curves for algorithm comparison"""
        plt.figure(figsize=(12, 8))
        
        for alg_name, alg_results in results.items():
            mean_regret = alg_results['mean_regret']
            std_regret = alg_results['std_regret']
            T = len(mean_regret)
            time_steps = range(1, T + 1)
            
            plt.plot(time_steps, mean_regret, label=alg_name, linewidth=2)
            plt.fill_between(time_steps, 
                           mean_regret - std_regret, 
                           mean_regret + std_regret, 
                           alpha=0.3)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Regret')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def print_summary_table(results, title):
        """Print summary table of final regrets"""
        print(f"\n{title}")
        print("=" * len(title))
        print(f"{'Algorithm':<25} {'Mean Regret':<12} {'Std Regret':<12}")
        print("-" * 50)
        
        for alg_name, alg_results in results.items():
            final_regrets = alg_results['final_regrets']
            mean_regret = np.mean(final_regrets)
            std_regret = np.std(final_regrets)
            print(f"{alg_name:<25} {mean_regret:<12.4f} {std_regret:<12.4f}")
    
    @staticmethod
    def analyze_task_results(task_results, task_name):
        """Analyze results for a specific task"""
        print(f"\n{'='*60}")
        print(f"{task_name.upper()} ANALYSIS")
        print(f"{'='*60}")
        
        for instance_name, results in task_results.items():
            print(f"\n{instance_name}:")
            ResultsAnalyzer.print_summary_table(results, f"{instance_name} - Final Regret Summary")
            
            # Plot regret curves
            title = f"{task_name} - {instance_name} - Regret vs Time"
            ResultsAnalyzer.plot_regret_curves(results, title)

def run_quick_test():
    """Quick test to verify implementations"""
    print("Running quick verification test...")
    
    # Create simple environment
    env = BernoulliBandit(5, [0.2, 0.4, 0.8, 0.3, 0.1])
    
    # Test each algorithm briefly
    algorithms = [
        ('Epsilon-Greedy', EpsilonGreedy(5, 0.1)),
        ('UCB', UCB(5)),
        ('Thompson Sampling', ThompsonSamplingBernoulli(5)),
        ('KL-UCB', KL_UCB(5))
    ]
    
    T = 100
    for alg_name, alg in algorithms:
        alg.reset()
        total_reward = 0
        
        for t in range(T):
            arm = alg.select_arm()
            reward = env.pull_arm(arm)
            alg.update(arm, reward)
            total_reward += reward
        
        print(f"{alg_name}: Average reward = {total_reward/T:.3f}")
    
    print("Quick test completed successfully!")

if __name__ == "__main__":
    # Run quick verification
    run_quick_test()