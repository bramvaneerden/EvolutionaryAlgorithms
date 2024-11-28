from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass



# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(42)

def mating_seletion(parent, parent_f):

    f_min = min(parent_f)
    f_sum = sum(parent_f) - (f_min - 0.001) * len(parent_f)
    
    rw = [(parent_f[0] - f_min + 0.001)/f_sum]
    for i in range(1,len(parent_f)):
        rw.append(rw[i-1] + (parent_f[i] - f_min + 0.001) / f_sum)
    
    select_parent = []
    for i in range(len(parent)) :
        r = np.random.uniform(0,1)
        index = 0
        # print(rw,r)
        while(r > rw[index]) :
            index = index + 1
        
        select_parent.append(parent[index].copy())
    return select_parent

def swap_mutation(p, mutation_rate):
    n = len(p)
    mutated = p.copy()

    for _ in range(n):  # randomly attempt to swap any two indices, over all n
        if np.random.uniform(0, 1) < mutation_rate:
            i, j = np.random.choice(n, 2, replace=False) 
            mutated[i], mutated[j] = mutated[j], mutated[i]  

    return mutated

def repair(child, p1, p2, point1, point2):
    n = len(child)
    mapping1 = {p1[i]: p2[i] for i in range(point1, point2)}
    mapping2 = {p2[i]: p1[i] for i in range(point1, point2)}

    seen = set()
    for i in range(n):
        if i < point1 or i >= point2:
            # Avoid infinite loops
            while child[i] in mapping1 and child[i] not in seen:
                seen.add(child[i])
                child[i] = mapping1[child[i]]
            while child[i] in mapping2 and child[i] not in seen:
                seen.add(child[i])
                child[i] = mapping2[child[i]]
    
    return child

def pmx_crossover(p1, p2, crossover_rate):
    if np.random.uniform(0, 1) < crossover_rate:
        n = len(p1)
        point1, point2 = sorted(np.random.choice(n, 2, replace=False))
        # print(n, point1, point2)
        
        child1 = np.concatenate([p1[:point1] , p2[point1:point2] , p1[point2:]])
        child2 = np.concatenate([p2[:point1] , p1[point1:point2] , p2[point2:]])
        
        child1 = repair(child1, p1, p2, point1, point2)
        child2 = repair(child2, p1, p2, point1, point2)
        
        return child1, child2
    
    return p1, p2

def get_mutation_rate(current_rate, current_iter, decay):
    min_rate = 0.001
    new_rate = current_rate * ( decay ** current_iter)
    return max(new_rate, min_rate)

def mutation(p, mutation_rate):
    for i in range(len(p)) :
        if np.random.uniform(0,1) < mutation_rate:
            p[i] = 1 - p[i]

def crossover(p1, p2, crossover_probability):
    if(np.random.uniform(0,1) < crossover_probability):
        for i in range(len(p1)) :
            if np.random.uniform(0,1) < 0.5:
                t = p1[i]
                p1[i] = p2[i]
                p2[i] = t

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO, pop_size, mutation_rate, crossover_rate, decay) -> None:
    # initial_pop = ... make sure you randomly create the first population
    # print(problem)
    budget = 5000
    
    f_opt = -np.inf
    x_opt = None
    parent = []
    parent_f = []
    current_iter = 0

    if problem.meta_data.name == 'NQueens':
        for i in range(pop_size):
            parent.append(np.random.permutation(49))
            parent_f.append(problem(parent[i]))

        while problem.state.evaluations < budget:

            mutation_rate = get_mutation_rate(mutation_rate, current_iter, decay)

            offspring = mating_seletion(parent,parent_f)
            # print ( "budget ", budget)
            # print ( "problem.state.evaluations ", problem.state.evaluations)
            for i in range(0,pop_size - (pop_size%2),2) :
                # print("pre-crossover: " , offspring[i])
                offspring[i], offspring[i+1] = pmx_crossover(offspring[i], offspring[i+1], crossover_rate)
                # print("post-crossover: " , offspring[i])
                
            for i in range(pop_size):
                offspring[i] = swap_mutation(offspring[i],  mutation_rate)
            
            # evaluate
            offspring_f = [problem(x) for x in offspring]
            
            for i in range(pop_size):
                if offspring_f[i] >= parent_f[i]:
                    parent[i] = offspring[i]
                    parent_f[i] = offspring_f[i]
            
            f_opt = max(parent_f)
            x_opt = parent[np.argmax(parent_f)] 
            
    
    elif problem.meta_data.name == 'LABS':
        for i in range(pop_size):   
            parent.append(np.random.randint(2, size=problem.meta_data.n_variables))
            parent_f.append(problem(parent[i]))
            
        while problem.state.evaluations < budget:
            mutation_rate = get_mutation_rate(mutation_rate, current_iter, decay)
            offspring = mating_seletion(parent, parent_f)

            for i in range(0, pop_size - (pop_size % 2), 2):
                crossover(offspring[i], offspring[i+1], crossover_rate)

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
    print(f_opt)
    return f_opt, x_opt
        # pass


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
    
    pop_size, mutation_rate, crossover_rate = 40, 0.08, 0.7
    decay = 0.95
    
    F18, _logger = create_problem(dimension=50, fid=18)
    print("F18")
    for run in range(10): 
        studentnumber1_studentnumber2_GA(F18, pop_size, mutation_rate, crossover_rate, decay)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    print("F23")
    for run in range(10): 
        studentnumber1_studentnumber2_GA(F23, pop_size, mutation_rate, crossover_rate, decay)
        F23.reset()
    _logger.close()