#!/usr/bin/env python3
"""
CS771 Mini Project 2: Multi-Armed Bandits
Main execution script for all tasks
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from bandit_algorithms import *
from bandit_environments import *
from experiments import TaskExperiments, ResultsAnalyzer
import warnings
warnings.filterwarnings('ignore')

def create_output_directory():
    """Create output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    return output_dir

def save_results(results, output_dir):
    """Save experimental results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(results)
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_dir}/results.json")

def generate_detailed_plots(all_results, output_dir):
    """Generate and save detailed plots for each task"""
    
    # Task 1: Bernoulli Bandits
    task1_results = all_results['task1']
    for instance_name, results in task1_results.items():
        plt.figure(figsize=(12, 8))
        
        for alg_name, alg_results in results.items():
            mean_regret = alg_results['mean_regret']
            std_regret = alg_results['std_regret']
            T = len(mean_regret)
            time_steps = range(1, T + 1)
            
            plt.plot(time_steps, mean_regret, label=alg_name, linewidth=2)
            plt.fill_between(time_steps, 
                           np.array(mean_regret) - np.array(std_regret), 
                           np.array(mean_regret) + np.array(std_regret), 
                           alpha=0.2)
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Cumulative Regret', fontsize=12)
        plt.title(f'Task 1: Bernoulli Bandits - {instance_name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        filename = f"{output_dir}/plots/task1_{instance_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Task 2: Gaussian Bandits
    task2_results = all_results['task2']
    for instance_name, results in task2_results.items():
        plt.figure(figsize=(12, 8))
        
        for alg_name, alg_results in results.items():
            mean_regret = alg_results['mean_regret']
            std_regret = alg_results['std_regret']
            T = len(mean_regret)
            time_steps = range(1, T + 1)
            
            plt.plot(time_steps, mean_regret, label=alg_name, linewidth=2)
            plt.fill_between(time_steps, 
                           np.array(mean_regret) - np.array(std_regret), 
                           np.array(mean_regret) + np.array(std_regret), 
                           alpha=0.2)
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Cumulative Regret', fontsize=12)
        plt.title(f'Task 2: Gaussian Bandits - {instance_name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        filename = f"{output_dir}/plots/task2_{instance_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Task 3: Successive Elimination
    task3_results = all_results['task3']
    for config_name, results in task3_results.items():
        plt.figure(figsize=(12, 8))
        
        for alg_name, alg_results in results.items():
            mean_regret = alg_results['mean_regret']
            std_regret = alg_results['std_regret']
            T = len(mean_regret)
            time_steps = range(1, T + 1)
            
            plt.plot(time_steps, mean_regret, label=alg_name, linewidth=2)
            plt.fill_between(time_steps, 
                           np.array(mean_regret) - np.array(std_regret), 
                           np.array(mean_regret) + np.array(std_regret), 
                           alpha=0.2)
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Cumulative Regret', fontsize=12)
        plt.title(f'Task 3: Successive Elimination - {config_name.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        filename = f"{output_dir}/plots/task3_{config_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}/plots/")

def generate_summary_report(all_results, output_dir):
    """Generate summary report"""
    report_lines = []
    report_lines.append("CS771 Mini Project 2: Multi-Armed Bandits")
    report_lines.append("=" * 50)
    report_lines.append(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Task 1 Summary
    report_lines.append("TASK 1: BERNOULLI BANDITS")
    report_lines.append("-" * 30)
    
    task1_results = all_results['task1']
    for instance_name, results in task1_results.items():
        report_lines.append(f"\n{instance_name}:")
        report_lines.append(f"{'Algorithm':<25} {'Mean Regret':<12} {'Std Regret':<12}")
        report_lines.append("-" * 50)
        
        for alg_name, alg_results in results.items():
            final_regrets = alg_results['final_regrets']
            mean_regret = np.mean(final_regrets)
            std_regret = np.std(final_regrets)
            report_lines.append(f"{alg_name:<25} {mean_regret:<12.4f} {std_regret:<12.4f}")
    
    # Task 2 Summary
    report_lines.append("\n\nTASK 2: GAUSSIAN BANDITS")
    report_lines.append("-" * 30)
    
    task2_results = all_results['task2']
    for instance_name, results in task2_results.items():
        report_lines.append(f"\n{instance_name}:")
        report_lines.append(f"{'Algorithm':<25} {'Mean Regret':<12} {'Std Regret':<12}")
        report_lines.append("-" * 50)
        
        for alg_name, alg_results in results.items():
            final_regrets = alg_results['final_regrets']
            mean_regret = np.mean(final_regrets)
            std_regret = np.std(final_regrets)
            report_lines.append(f"{alg_name:<25} {mean_regret:<12.4f} {std_regret:<12.4f}")
    
    # Task 3 Summary
    report_lines.append("\n\nTASK 3: SUCCESSIVE ELIMINATION")
    report_lines.append("-" * 30)
    
    task3_results = all_results['task3']
    for config_name, results in task3_results.items():
        report_lines.append(f"\n{config_name.replace('_', ' ').title()}:")
        report_lines.append(f"{'Algorithm':<25} {'Mean Regret':<12} {'Std Regret':<12}")
        report_lines.append("-" * 50)
        
        for alg_name, alg_results in results.items():
            final_regrets = alg_results['final_regrets']
            mean_regret = np.mean(final_regrets)
            std_regret = np.std(final_regrets)
            report_lines.append(f"{alg_name:<25} {mean_regret:<12.4f} {std_regret:<12.4f}")
    
    # Key Insights
    report_lines.append("\n\nKEY INSIGHTS:")
    report_lines.append("-" * 15)
    report_lines.append("1. Thompson Sampling generally shows good performance across different settings")
    report_lines.append("2. UCB performs well for both Bernoulli and Gaussian bandits")
    report_lines.append("3. KL-UCB is specifically designed for Bernoulli rewards and shows competitive performance")
    report_lines.append("4. Successive Elimination can effectively identify the best arm with theoretical guarantees")
    report_lines.append("5. Epsilon-greedy performance depends heavily on the choice of epsilon parameter")
    
    # Save report
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print report
    for line in report_lines:
        print(line)
    
    print(f"\nSummary report saved to {output_dir}/summary_report.txt")

def main():
    """Main execution function"""
    print("CS771 Mini Project 2: Multi-Armed Bandits")
    print("=" * 50)
    print("Starting comprehensive experiments...")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Initialize experiment runner
    experiment_runner = TaskExperiments()
    
    # Configuration
    T_short = 1000  # Time horizon for Tasks 1 and 2
    T_long = 5000   # Time horizon for Task 3
    n_runs = 20     # Number of independent runs
    
    print(f"\nExperiment Configuration:")
    print(f"- Time horizon (Tasks 1&2): {T_short}")
    print(f"- Time horizon (Task 3): {T_long}")
    print(f"- Number of runs: {n_runs}")
    
    # Run all experiments
    print("\nStarting experiments...")
    all_results = experiment_runner.run_all_tasks(T_short, T_long, n_runs)
    
    # Save results
    save_results(all_results, output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    generate_detailed_plots(all_results, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(all_results, output_dir)
    
    print(f"\nAll experiments completed!")
    print(f"Results available in: {output_dir}/")
    print("Files generated:")
    print(f"- results.json: Raw experimental data")
    print(f"- summary_report.txt: Summary of all results")
    print(f"- plots/: Regret curves for all tasks")

def run_quick_demo():
    """Run a quick demonstration"""
    print("Running Quick Demo...")
    print("-" * 30)
    
    # Quick test with fewer runs
    experiment_runner = TaskExperiments()
    
    # Just run Task 1 with fewer parameters
    T = 500
    n_runs = 5
    
    print(f"Demo: Bernoulli Bandits (T={T}, runs={n_runs})")
    
    # Create simple environment
    env = BernoulliBandit(5, [0.2, 0.4, 0.8, 0.3, 0.1])
    print(f"Arm probabilities: {env.get_expected_rewards()}")
    
    algorithms = {
        'UCB': (UCB, {'n_arms': 5}),
        'Thompson Sampling': (ThompsonSamplingBernoulli, {'n_arms': 5}),
        'Epsilon-Greedy (0.1)': (EpsilonGreedy, {'n_arms': 5, 'epsilon': 0.1})
    }
    
    runner = ExperimentRunner()
    results = runner.compare_algorithms(algorithms, env, T, n_runs)
    
    # Show results
    ResultsAnalyzer.print_summary_table(results, "Demo Results")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        run_quick_demo()
    else:
        main()