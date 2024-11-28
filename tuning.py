from typing import List

import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import studentnumber1_studentnumber2_GA, create_problem


# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(42)

FINAL_BUDGET = 5000
TUNING_BUDGET = 1000000

def evaluate_params(pop_size, mut_rate, cross_rate, decay, n_trials=5):
    f18_scores = []
    f23_scores = []
    
    for _ in range(n_trials):
        # F18 - LABS
        F18, _ = create_problem(dimension=50, fid=18)
        f18_result, _ = studentnumber1_studentnumber2_GA(
            F18, pop_size, mut_rate, cross_rate, decay
        )
        f18_scores.append(f18_result)
        
        # F23 - N-Queens
        F23, _ = create_problem(dimension=49, fid=23)
        f23_result, _ = studentnumber1_studentnumber2_GA(
            F23, pop_size, mut_rate, cross_rate, decay
        )
        f23_scores.append(f23_result)
    
    return np.mean(f18_scores), np.std(f18_scores), np.mean(f23_scores), np.std(f23_scores)

def tune_hyperparameters():
    param_ranges = {
        'population_size': np.linspace(20, 100, 5, dtype=int),
        'mutation_rate': np.linspace(0.01, 0.1, 5),
        'crossover_rate': np.linspace(0.5, 0.9, 5),
        'decay': np.linspace(0.85, 0.99, 5)
    }
    
    best_score = float('-inf')
    best_params = None
    evaluations_used = 0
    
    # phase 1: broad search 
    for pop_size in param_ranges['population_size']:
        for mut_rate in param_ranges['mutation_rate']:
            for cross_rate in param_ranges['crossover_rate']:
                for decay in param_ranges['decay']:
                    if evaluations_used + (FINAL_BUDGET * 2 * 3) > TUNING_BUDGET:
                        break
                        
                    f18_mean, f18_std, f23_mean, f23_std = evaluate_params(
                        pop_size, mut_rate, cross_rate, decay, n_trials=3
                    )
                    
                    # normalize scores
                    normalized_score = (f18_mean / 10 - f18_std/20) + (f23_mean / 7 - f23_std/14)
                    evaluations_used += FINAL_BUDGET * 2 * 3
                    
                    if normalized_score > best_score:
                        best_score = normalized_score
                        best_params = (pop_size, mut_rate, cross_rate, decay)
    
    # phase 2: local search around best parameters
    if best_params:
        pop_size, mut_rate, cross_rate, decay = best_params
        local_ranges = {
            'population_size': np.array([max(20, pop_size-10), pop_size, min(100, pop_size+10)]),
            'mutation_rate': np.array([max(0.01, mut_rate-0.01), mut_rate, min(0.1, mut_rate+0.01)]),
            'crossover_rate': np.array([max(0.5, cross_rate-0.05), cross_rate, min(0.9, cross_rate+0.05)]),
            'decay': np.array([max(0.85, decay-0.02), decay, min(0.99, decay+0.02)])
        }
        
        for pop_size in local_ranges['population_size']:
            for mut_rate in local_ranges['mutation_rate']:
                for cross_rate in local_ranges['crossover_rate']:
                    for decay in local_ranges['decay']:
                        if evaluations_used + (FINAL_BUDGET * 2 * 5) > TUNING_BUDGET:
                            break
                            
                        f18_mean, f18_std, f23_mean, f23_std = evaluate_params(
                            pop_size, mut_rate, cross_rate, decay, n_trials=5
                        )
                        
                        normalized_score = (f18_mean / 10 - f18_std/20) + (f23_mean / 7 - f23_std/14)
                        evaluations_used += FINAL_BUDGET * 2 * 5
                        
                        if normalized_score > best_score:
                            best_score = normalized_score
                            best_params = (pop_size, mut_rate, cross_rate, decay)
    
    print(f"Total evaluations used: {evaluations_used}")
    return best_params

if __name__ == "__main__":
    print("Starting improved hyperparameter tuning...")
    pop_size, mut_rate, cross_rate, decay = tune_hyperparameters()
    print("\nBest parameters found:")
    print(f"Population size: {pop_size}")
    print(f"Mutation rate: {mut_rate}")
    print(f"Crossover rate: {cross_rate}")
    print(f"Decay rate: {decay}")