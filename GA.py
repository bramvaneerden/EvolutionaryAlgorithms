from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass



# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(42)

def tournament_selection(parent, parent_f, tournament_size=3):
    population_size = len(parent)
    selected = []
    for _ in range(population_size):

        tournament_idx = np.random.choice(population_size, tournament_size, replace=False)
        tournament_fitness = [parent_f[i] for i in tournament_idx]

        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        selected.append(parent[winner_idx].copy())
    return selected

def binary_crossover(p1, p2, crossover_rate):
    if np.random.uniform(0, 1) < crossover_rate:
        n = len(p1)
        mask = np.random.randint(2, size=n)
        
        child1 = np.where(mask, p1, p2)
        child2 = np.where(mask, p2, p1)
        
        return child1, child2
    
    return p1.copy(), p2.copy()

def get_mutation_rate(current_rate, current_iter, decay):
    min_rate = 0.001
    new_rate = current_rate * ( decay ** current_iter)
    return max(new_rate, min_rate)

def mutation(p, mutation_rate):
    for i in range(len(p)) :
        if np.random.uniform(0,1) < mutation_rate:
            p[i] = 1 - p[i]

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO, pop_size, mutation_rate, crossover_rate, decay, budget) -> None:

    f_opt = -np.inf
    x_opt = None
    parent = []
    parent_f = []
    current_iter = 0

    if problem.meta_data.name == 'NQueens':
        for i in range(pop_size):
            # init population
            parent.append(np.random.randint(2, size=problem.meta_data.n_variables))
            parent_f.append(problem(parent[i]))
            if parent_f[i] > f_opt:
                f_opt = parent_f[i]
                x_opt = parent[i].copy()

        while problem.state.evaluations < budget:
            mutation_rate = get_mutation_rate(mutation_rate, current_iter, decay)
            offspring = tournament_selection(parent, parent_f, tournament_size=3)

            for i in range(0, pop_size - (pop_size % 2), 2):
                offspring[i], offspring[i+1] = binary_crossover(offspring[i], offspring[i+1], crossover_rate) 

            for i in range(pop_size):
                mutation(offspring[i], mutation_rate)  

            for i in range(pop_size):
                offspring_f = problem(offspring[i])
                if offspring_f > parent_f[i]:
                    parent[i] = offspring[i]
                    parent_f[i] = offspring_f
                    
                    if offspring_f > f_opt:
                        f_opt = offspring_f
                        x_opt = offspring[i].copy()
            
    
    elif problem.meta_data.name == 'LABS':
        for i in range(pop_size):  
            # init population 
            parent.append(np.random.randint(2, size=problem.meta_data.n_variables))
            parent_f.append(problem(parent[i]))
            
        while problem.state.evaluations < budget:
            mutation_rate = get_mutation_rate(mutation_rate, current_iter, decay)
            offspring = tournament_selection(parent, parent_f, tournament_size=3)

            for i in range(0, pop_size - (pop_size % 2), 2):
                offspring[i], offspring[i+1] = binary_crossover(offspring[i], offspring[i+1], crossover_rate)

            for i in range(pop_size):
                mutation(offspring[i], mutation_rate)

            #evaluate
            for i in range(pop_size):
                offspring_f = problem(offspring[i])
                if offspring_f > parent_f[i]:
                    parent[i] = offspring[i]
                    parent_f[i] = offspring_f
                    
                    if offspring_f > f_opt:
                        f_opt = offspring_f
                        x_opt = offspring[i].copy()
    # print(f_opt)
    # print(x_opt)
    # print(problem.state.evaluations)
    return f_opt, x_opt


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    
    pop_size, mutation_rate, crossover_rate = 30, 0.060000000000000005, 0.5
    decay = 0.94
    budget=5000
    
    F18, _logger = create_problem(dimension=50, fid=18)
    print("F18")
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18, pop_size, mutation_rate, crossover_rate, decay, budget)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    print("F23")
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23, pop_size, mutation_rate, crossover_rate, decay, budget)
        F23.reset()
    _logger.close()