import importlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count

from utils.tree_search import Node, Tree
import shelfsolver
        
####################################

# Collect search depths
depths = []

use_multiprocessing = False

def thread_job(i):
    print('Position', i)
    solver = ShelfSolver()
    shelf = solver.new_random_shelf()
    base_ids = solver.assign_bases(shelf)
    results = solver.solve(shelf, base_ids)
    return results

n_tries = 10000
iterator = [i for i in range(n_tries)]  

@timer
def run_script():
    if use_multiprocessing:
        pool = Pool(processes=cpu_count())
        collected_depths = pool.map(thread_job, iterator)
        pool.close()

    else:
        depths = []
        for i in range(n_tries):
            if i%200==0:
                print('Step {} '.format(i))
            solver = ShelfSolver()
            shelf = solver.new_random_shelf()
            base_ids = solver.assign_bases(shelf)
            results = solver.solve(shelf, base_ids)
            if results:
                depths.append(results)
                with open('collected_depths/one_color_per_row.pickle', 'wb') as handle:
                    pickle.dump(depths, handle, protocol=pickle.HIGHEST_PROTOCOL)

run_script()