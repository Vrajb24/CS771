#!/usr/bin/env python3
"""
Standalone script to run specific experiments
Usage examples:
  python run_experiments.py --task 1 --runs 10 --horizon 1000
  python run_experiments.py --task all --runs 20 --horizon 2000
  python run_experiments.py --quick-test
"""

import argparse
import numpy as np
from bandit_algorithms import *
from bandit_environments import *
from experiments import TaskExperiments, ExperimentRunner, ResultsAnalyzer
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Multi-Armed Bandit Experiments')
    
    parser.add_argument('--task', type=str, choices=['1', '2', '3', 'all'], default='all',
                      help='Which task to run (1, 2, 3, or all)')
    
    parser.add_argument('--runs', type=int, default=10,
                      help='Number of independent runs')
    
    parser.add_argument('--horizon', type=int, default=1000,
                      help='Time horizon for experiments')
    
    parser.add_argument('--quick-test', action='store_true',
                      help='Run quick verification test')
    
    parser.add_argument('--algorithms', nargs='+', 
                      choices=['epsilon_greedy', 'ucb', 'thompson', 'kl_ucb', 'elimination'],
                      help='Specific algorithms to test')
    
    parser.add_argument('--arms', type=int, default=10,
                      help='Number of arms for the bandit')
    
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium',
                      help='Difficulty level of the bandit instance')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--save-plots', action='store_true',
                      help='Save plots to files')
    
    return parser.parse_args()

def run_custom_experiment(args):
    """Run custom experiment based on arguments"""
    np.random.seed(args.seed)
    
    print(f"Running custom experiment:")
    print(f"- Task: {args.task}")
    print(f"- Arms: {args.arms}")
    print(f"- Difficulty: {args.difficulty}")
    print(f"- Horizon: {args.horizon}")
    print(f"- Runs: {args.runs}")
    print(f"- Seed: {args.seed}")
    print("-" * 50)
    
    # Create environment based on task
    if args.task == '1':
        environment = InstanceGenerator.generate_bernoulli_instance(
            args.arms, args.difficulty, args.seed
        )
        print(f"Bernoulli Bandit - Arm probabilities: {environment.get_expected_rewards()}")
        
        # Define algorithms
        algorithms = {}
        if not args.algorithms or 'epsilon_greedy' in args.algorithms:
            algorithms['Epsilon-Greedy (0.1)'] = (EpsilonGreedy, {'n_arms': args.arms, 'epsilon': 0.1})
        if not args.algorithms or 'ucb' in args.algorithms:
            algorithms['UCB'] = (UCB, {'n_arms': args.arms})
        if not args.algorithms or 'thompson' in args.algorithms:
            algorithms['Thompson Sampling'] = (ThompsonSamplingBernoulli, {'n_arms': args.arms})
        if not args.algorithms or 'kl_ucb' in args.algorithms:
            algorithms['KL-UCB'] = (KL_UCB, {'n_arms': args.arms})
            
    elif args.task == '2':
        environment = InstanceGenerator.generate_gaussian_instance(
            args.arms, args.difficulty, args.seed
        )
        print(f"Gaussian Bandit - Arm means: {environment.get_expected_rewards()}")
        
        # Define algorithms
        algorithms = {}
        if not args.algorithms or 'epsilon_greedy' in args.algorithms:
            algorithms['Epsilon-Greedy (0.1)'] = (EpsilonGreedy, {'n_arms': args.arms, 'epsilon': 0.1})
        if not args.algorithms or 'ucb' in args.algorithms:
            algorithms['UCB'] = (UCB, {'n_arms': args.arms})
        if not args.algorithms or 'thompson' in args.algorithms:
            algorithms['Thompson Sampling'] = (ThompsonSamplingGaussian, {'n_arms': args.arms})
            
    elif args.task == '3':
        environment = InstanceGenerator.generate_bernoulli_instance(
            args.arms, args.difficulty, args.seed
        )
        print(f"Successive Elimination - Arm probabilities: {environment.get_expected_rewards()}")
        
        # Define algorithms
        algorithms = {}
        if not args.algorithms or 'elimination' in args.algorithms:
            algorithms['Successive Elimination'] = (SuccessiveElimination, {'n_arms': args.arms})
        if not args.algorithms or 'ucb' in args.algorithms:
            algorithms['UCB (baseline)'] = (UCB, {'n_arms': args.arms})
        if not args.algorithms or 'thompson' in args.algorithms:
            algorithms['Thompson Sampling (baseline)'] = (ThompsonSamplingBernoulli, {'n_arms': args.arms})
    
    # Run experiments
    runner = ExperimentRunner()
    results = runner.compare_algorithms(algorithms, environment, args.horizon, args.runs)
    
    # Print results
    ResultsAnalyzer.print_summary_table(results, f"Task {args.task} Results")
    
    # Plot results
    title = f"Task {args.task} - {args.difficulty.title()} Instance ({args.arms} arms)"
    ResultsAnalyzer.plot_regret_curves(results, title, 
                                     f"task_{args.task}_results.png" if args.save_plots else None)
    
    return results

def algorithm_comparison(args):
    """Compare specific algorithms on multiple instances"""
    print("Algorithm Comparison Across Multiple Instances")
    print("=" * 50)
    
    difficulties = ['easy', 'medium', 'hard']
    all_results = {}
    
    for difficulty in difficulties:
        print(f"\nTesting {difficulty} instance...")
        
        if args.task == '1':
            env = InstanceGenerator.generate_bernoulli_instance(args.arms, difficulty, args.seed)
        else:
            env = InstanceGenerator.generate_gaussian_instance(args.arms, difficulty, args.seed)
        
        # Define algorithms based on task
        if args.task == '1':
            algorithms = {
                'UCB': (UCB, {'n_arms': args.arms}),
                'Thompson Sampling': (ThompsonSamplingBernoulli, {'n_arms': args.arms}),
                'KL-UCB': (KL_UCB, {'n_arms': args.arms})
            }
        else:
            algorithms = {
                'UCB': (UCB, {'n_arms': args.arms}),
                'Thompson Sampling': (ThompsonSamplingGaussian, {'n_arms': args.arms})
            }
        
        runner = ExperimentRunner()
        results = runner.compare_algorithms(algorithms, env, args.horizon, args.runs)
        all_results[difficulty] = results
        
        # Print summary for this difficulty
        ResultsAnalyzer.print_summary_table(results, f"{difficulty.title()} Instance Results")
    
    # Overall comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Algorithm Comparison - Task {args.task}', fontsize=16)
    
    for i, (difficulty, results) in enumerate(all_results.items()):
        ax = axes[i]
        
        for alg_name, alg_results in results.items():
            mean_regret = alg_results['mean_regret']
            T = len(mean_regret)
            time_steps = range(1, T + 1)
            
            ax.plot(time_steps, mean_regret, label=alg_name, linewidth=2)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title(f'{difficulty.title()} Instance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if args.save_plots:
        plt.savefig(f'algorithm_comparison_task_{args.task}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return all_results

def run_quick_test_extended():
    """Extended quick test with more detailed output"""
    print("Running Extended Quick Test")
    print("=" * 30)
    
    # Test environments
    environments = {
        'Simple Bernoulli': BernoulliBandit(3, [0.2, 0.8, 0.4]),
        'Simple Gaussian': GaussianBandit(3, [-1, 1, 0])
    }
    
    # Test algorithms
    for env_name, env in environments.items():
        print(f"\nTesting {env_name}:")
        print(f"Expected rewards: {env.get_expected_rewards()}")
        print(f"Optimal arm: {env.get_optimal_arm()}")
        
        # Test each algorithm
        if isinstance(env, BernoulliBandit):
            algorithms = [
                ('Epsilon-Greedy', EpsilonGreedy(3, 0.1)),
                ('UCB', UCB(3)),
                ('Thompson Sampling', ThompsonSamplingBernoulli(3)),
                ('KL-UCB', KL_UCB(3))
            ]
        else:
            algorithms = [
                ('Epsilon-Greedy', EpsilonGreedy(3, 0.1)),
                ('UCB', UCB(3)),
                ('Thompson Sampling', ThompsonSamplingGaussian(3))
            ]
        
        T = 200
        results_summary = []
        
        for alg_name, alg in algorithms:
            alg.reset()
            cumulative_regret = 0
            
            for t in range(T):
                arm = alg.select_arm()
                reward = env.pull_arm(arm)
                alg.update(arm, reward)
                cumulative_regret += env.get_regret(arm)
            
            results_summary.append((alg_name, cumulative_regret, alg.counts))
        
        # Print results
        print(f"\nResults after {T} steps:")
        print(f"{'Algorithm':<20} {'Total Regret':<12} {'Arm Counts'}")
        print("-" * 50)
        for alg_name, regret, counts in results_summary:
            counts_str = str(counts.astype(int))
            print(f"{alg_name:<20} {regret:<12.2f} {counts_str}")

def main():
    args = parse_arguments()
    
    if args.quick_test:
        run_quick_test_extended()
        return
    
    if args.task == 'all':
        # Run comprehensive experiments
        experiment_runner = TaskExperiments()
        
        print("Running all tasks...")
        all_results = experiment_runner.run_all_tasks(
            T_short=args.horizon, 
            T_long=min(args.horizon * 2, 5000), 
            n_runs=args.runs
        )
        
        # Analyze results
        for task_name, task_results in all_results.items():
            ResultsAnalyzer.analyze_task_results(task_results, task_name)
    
    else:
        # Run specific task
        if args.algorithms and len(args.algorithms) > 1:
            algorithm_comparison(args)
        else:
            run_custom_experiment(args)

if __name__ == "__main__":
    main()