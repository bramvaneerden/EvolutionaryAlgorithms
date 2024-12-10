from ioh import get_problem
from ioh import logger
import ioh
import sys
import numpy as np
import time

dimension = 5
def sphere(x: np.ndarray) -> float:
    return np.sum(np.power(x, 2))

ioh.problem.wrap_real_problem(
    sphere,
    name="Sphere",
    optimization_type=ioh.OptimizationType.MIN,
    lb=-5,
    ub=5,
)
sphere = get_problem("Sphere", dimension=dimension)
optimum = 0

l = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="evolution strategy", 
    algorithm_info="The lab session of the evolutionary algorithm course in LIACS")

sphere.attach_logger(l)

def initialization(mu, dimension, lowerbound=-5.0, upperbound=5.0):
    parent = []
    parent_sigma = []
    initial_sigma = 0.05 * (upperbound - lowerbound)
    
    for i in range(mu):
        parent.append(np.random.uniform(lowerbound, upperbound, dimension))
        parent_sigma.append(np.ones(dimension) * initial_sigma)
    
    return parent, parent_sigma

def mutation(parent, parent_sigma, tau, tau_prime):
    """
    Mutation according to the slide:
    g ~ N(0,1)
    σ'ᵢ = σᵢ exp(τ'g + τN(0,1)) ∀i∈[n]
    x'ᵢ = xᵢ + σ'ᵢN(0,1) ∀i∈[n]
    """
    for i in range(len(parent)):
        # Step 1: Sample global perturbation
        g = np.random.normal(0, 1)
        
        # Step 2: Mutate individual step sizes
        for j in range(len(parent[i])):
            parent_sigma[i][j] = parent_sigma[i][j] * np.exp(tau_prime * g + tau * np.random.normal(0, 1))
        
        # Step 3: Mutate search variables using new step sizes
        for j in range(len(parent[i])):
            parent[i][j] = parent[i][j] + parent_sigma[i][j] * np.random.normal(0, 1)

def recombination(parent, parent_sigma, parent_f):
    if len(parent) < 3:
        raise ValueError("Parent population must have at least 3 individuals for tournament selection")
    
    # Select two parents through tournament selection
    selected = []
    sigma_selected = []
    for _ in range(2):
        tournament_idx = np.random.choice(len(parent), 3, replace=False)
        tournament_fitness = [parent_f[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        selected.append(parent[winner_idx].copy())
        sigma_selected.append(parent_sigma[winner_idx].copy())
    
    # Intermediate recombination for both x and σ vectors
    offspring = np.mean(selected, axis=0)
    offspring_sigma = np.mean(sigma_selected, axis=0)
    
    return offspring, offspring_sigma

def evolution_strategy(func, budget=None):
    if budget is None:
        budget = 50000
    
    f_opt = sys.float_info.max
    x_opt = None

    # Parameters setting
    mu_ = 10
    lambda_ = 100
    # Parameters from the slides
    tau = 1 / np.sqrt(2 * dimension)  # τ for individual sigma updates
    tau_prime = 1 / np.sqrt(2 * np.sqrt(dimension))  # τ' for global update
    
    # Initialization and Evaluation
    parent, parent_sigma = initialization(mu_, func.meta_data.n_variables)
    parent_f = []
    for i in range(mu_):
        parent_f.append(func(parent[i]))
        budget = budget - 1
        if parent_f[i] < f_opt:
            f_opt = parent_f[i]
            x_opt = parent[i].copy()

    # Optimization Loop
    while (f_opt > optimum and budget > 0):
        offspring = []
        offspring_sigma = []
        offspring_f = []

        # Generate offspring
        for i in range(lambda_):
            o, s = recombination(parent, parent_sigma, parent_f)
            offspring.append(o)
            offspring_sigma.append(s)

        # Mutation
        mutation(offspring, offspring_sigma, tau, tau_prime)

        # Evaluate offspring
        for i in range(lambda_):
            if budget <= 0:
                break
            offspring_f.append(func(offspring[i]))
            budget -= 1
            if offspring_f[i] < f_opt:
                f_opt = offspring_f[i]
                x_opt = offspring[i].copy()

        # Selection (μ + λ)
        combined_solutions = parent + offspring
        combined_sigma = parent_sigma + offspring_sigma
        combined_f = parent_f + offspring_f
        
        # Sort by fitness
        sorted_indices = np.argsort(combined_f)
        
        # Select the best μ individuals
        parent = [combined_solutions[i] for i in sorted_indices[:mu_]]
        parent_sigma = [combined_sigma[i] for i in sorted_indices[:mu_]]
        parent_f = [combined_f[i] for i in sorted_indices[:mu_]]

    func.reset()
    print(f_opt,x_opt)

    return f_opt, x_opt

def main():
    for run in range(20):
        f_opt, x_opt = evolution_strategy(sphere)
        print(f"Run {run + 1}: Found optimum: {f_opt:.6f}")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("The program takes %s seconds" % (end-start))