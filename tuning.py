from typing import List

import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import studentnumber1_studentnumber2_GA, create_problem

budget = 1000000

# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(42)

def tune_hyperparameters(n_trials: int = 2, evaluations_per_trial: int = 1000) -> tuple[int, float, float, float]:

    hyperparameter_space = {
        "population_size": [20, 40, 80],
        "mutation_rate": [0.02, 0.05, 0.08],
        "crossover_rate": [0.5, 0.7, 0.9],
        "decay": [0.85, 0.9, 0.95]
    }
    
    best_score = float('-inf')
    best_params = None
    
    total_configs = (len(hyperparameter_space['population_size']) * 
                    len(hyperparameter_space['mutation_rate']) * 
                    len(hyperparameter_space['crossover_rate']) *
                    len(hyperparameter_space['decay']))
                    
    print(f"Total configurations to test: {total_configs}")
    print(f"Trials per configuration: {n_trials}")
    
    config_counter = 0
    
    for pop_size in hyperparameter_space['population_size']:
        for mut_rate in hyperparameter_space['mutation_rate']:
            for cross_rate in hyperparameter_space['crossover_rate']:
                for decay in hyperparameter_space['decay']:
                    config_counter += 1
                    print(f"\nTesting configuration {config_counter}/{total_configs}")
                    print(f"Parameters: pop={pop_size}, mut={mut_rate}, cross={cross_rate}, decay={decay}")
                    
                    f18_scores = []
                    f23_scores = []
                    
                    for trial in range(n_trials):
                        print(f"  Trial {trial + 1}/{n_trials}")
                        
                        #F18
                        F18, _ = create_problem(dimension=50, fid=18)
                        f18_result, _ = studentnumber1_studentnumber2_GA(
                            F18, pop_size, mut_rate, cross_rate, decay
                        )
                        f18_scores.append(f18_result)
                        
                        #F23
                        F23, _ = create_problem(dimension=49, fid=23)
                        f23_result, _ = studentnumber1_studentnumber2_GA(
                            F23, pop_size, mut_rate, cross_rate, decay
                        )
                        f23_scores.append(f23_result)
                    
                    avg_f18 = np.mean(f18_scores)
                    avg_f23 = np.mean(f23_scores)
                    
                    # combined
                    normalized_score = (avg_f18 / 10) + (avg_f23 / 7)
                    
                    print(f"  Results - F18 avg: {avg_f18:.2f}, F23 avg: {avg_f23:.2f}")
                    print(f"  Combined score: {normalized_score:.4f}")
                    
                    if normalized_score > best_score:
                        best_score = normalized_score
                        best_params = (pop_size, mut_rate, cross_rate, decay)
                        print(f"  New best parameters found! Score: {best_score:.4f}")
    
    if best_params is None:
        return (100, 0.05, 0.7, 0.975)          
    return best_params

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    pop_size, mut_rate, cross_rate, decay = tune_hyperparameters()
    print("\nBest parameters found:")
    print(f"Population size: {pop_size}")
    print(f"Mutation rate: {mut_rate}")
    print(f"Crossover rate: {cross_rate}")
    print(f"Decay rate: {decay}")