from typing import List

import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import studentnumber1_studentnumber2_GA, create_problem


# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(42)

FINAL_BUDGET = 3000
TUNING_BUDGET = 5000000

def normalize_nqueens_score(score):
    # map from [-1043, 7] to [0, 1]
    min_score = -1043  # all 1s results in this
    max_score = 7
    return (score - min_score) / (max_score - min_score)

def normalize_labs_score(score):
    # map from [0, 8] to [0, 1]
    return score / 8

def evaluate_params(pop_size, mut_rate, cross_rate, decay, budget, n_trials=3):
    f18_scores = []
    f23_scores = []
    
    for _ in range(n_trials):
        # F18 - LABS
        F18, _ = create_problem(dimension=50, fid=18)
        f18_result, _ = studentnumber1_studentnumber2_GA(
            F18, pop_size, mut_rate, cross_rate, decay, budget
        )
        f18_scores.append(f18_result)
        
        # F23 - N-Queens
        F23, _ = create_problem(dimension=49, fid=23)
        f23_result, _ = studentnumber1_studentnumber2_GA(
            F23, pop_size, mut_rate, cross_rate, decay, budget
        )
        f23_scores.append(f23_result)
    
    # normalize
    f18_norm = np.mean([normalize_labs_score(s) for s in f18_scores])
    f23_norm = np.mean([normalize_nqueens_score(s) for s in f23_scores])
    
    #normalized stdevs
    f18_std_norm = np.std([normalize_labs_score(s) for s in f18_scores])
    f23_std_norm = np.std([normalize_nqueens_score(s) for s in f23_scores])
    
    return f18_norm, f18_std_norm, f23_norm, f23_std_norm

def tune_hyperparameters():
    param_ranges = {
        'population_size': np.linspace(20, 110, 4, dtype=int),
        'mutation_rate': np.linspace(0.01, 0.1, 4),
        'crossover_rate': np.linspace(0.5, 0.9, 4),
        'decay': np.linspace(0.85, 0.99, 3)
    }
    
    all_scores = []
    best_score = float('-inf')
    best_params = None
    evaluations_used = 0
    print("Start of phase 1.")
    # phase 1: broad search 
    for pop_size in param_ranges['population_size']:
        for mut_rate in param_ranges['mutation_rate']:
            for cross_rate in param_ranges['crossover_rate']:
                for decay in param_ranges['decay']:
                    if evaluations_used + (FINAL_BUDGET * 2 * 3) > TUNING_BUDGET:
                        break
                        
                    f18_norm, f18_std_norm, f23_norm, f23_std_norm = evaluate_params(
                        pop_size, mut_rate, cross_rate, decay, FINAL_BUDGET ,n_trials=3
                    )
                    
                    # normalize scores
                    normalized_score = (f18_norm - f18_std_norm/2) + (f23_norm - f23_std_norm/2)
                    all_scores.append(normalized_score)
                    evaluations_used += FINAL_BUDGET * 2 * 3
                    
                    if normalized_score > best_score:
                        best_score = normalized_score
                        best_params = (pop_size, mut_rate, cross_rate, decay)
    print("Start of phase 2.")
    # phase 2: local search around best parameters
    if best_params:
        pop_size, mut_rate, cross_rate, decay = best_params
        local_ranges = {
            'population_size': np.array([max(20, pop_size-10), pop_size, min(110, pop_size+10)]),
            'mutation_rate': np.array([max(0.01, mut_rate-0.01), mut_rate, min(0.1, mut_rate+0.01)]),
            'crossover_rate': np.array([max(0.5, cross_rate-0.05), cross_rate, min(0.9, cross_rate+0.05)]),
            'decay': np.array([max(0.85, decay-0.02), decay, min(0.99, decay+0.02)])
        }
        
        for pop_size in local_ranges['population_size']:
            for mut_rate in local_ranges['mutation_rate']:
                for cross_rate in local_ranges['crossover_rate']:
                    for decay in local_ranges['decay']:
                        if evaluations_used + (FINAL_BUDGET * 2 * 3) > TUNING_BUDGET:
                            break
                            
                        f18_norm, f18_std_norm, f23_norm, f23_std_norm = evaluate_params(
                            pop_size, mut_rate, cross_rate, decay, FINAL_BUDGET ,n_trials=3
                        )
                        
                        # normalize scores
                        normalized_score = (f18_norm - f18_std_norm/2) + (f23_norm - f23_std_norm/2)
                        all_scores.append(normalized_score)
                        evaluations_used += FINAL_BUDGET * 2 * 3
                        
                        if normalized_score > best_score:
                            best_score = normalized_score
                            best_params = (pop_size, mut_rate, cross_rate, decay)
    
    all_scores = np.array(all_scores)
    
    std_improvement = (best_score - np.mean(all_scores)) / np.std(all_scores)
    
    
    print(f"Best score: {best_score} , scores stdev {np.std(all_scores)} , scores mean {np.mean(all_scores)}")
    print("Stdevs above the mean: " ,std_improvement)
    print(f"Total evaluations used: {evaluations_used}")
    return best_params

if __name__ == "__main__":
    print("Starting hyperparameter tuning:")
    pop_size, mut_rate, cross_rate, decay = tune_hyperparameters()
    print("\nBest parameters found:")
    print(f"Population size: {pop_size}")
    print(f"Mutation rate: {mut_rate}")
    print(f"Crossover rate: {cross_rate}")
    print(f"Decay rate: {decay}")