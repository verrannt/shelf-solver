import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count

from utils.timer_decorator import timer
import shelfsolver
        
####################################

# Collect search depths
depths = []
solver = shelfsolver.ShelfSolver()

use_multiprocessing = True
n_tries_per_core = 100
task = 3

def thread_job(coreID):
    core_results = []
    shelfgen = shelfsolver.ShelfGenerator()
    for i in range(n_tries_per_core+1):
        if i%4==0:
            print('\nCore {}, Step {}'.format(coreID, i))
            print('Successful runs so far: {}'.format(len(core_results)))
            with open('collected_depths/TEMP_task{}_core{}.pickle'
                      .format(task, coreID), 'wb') as handle:
                pickle.dump(core_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        shelf = shelfgen.new_random_shelf(seed=coreID+i*10)
        core_result = solver.solve_task(shelf, task=3, search_type="stack", collect=True)
        if core_result:
            core_results.append(core_result)
    print('\nCore {} finished.'.format(coreID))
    return core_results

@timer
def run_script():
    if use_multiprocessing:
        n_cores = cpu_count()-1
    else:
        n_cores = 1
    print('Running solver script with {} core(s).'.format(n_cores))
    iterator = [coreID for coreID in range(1,n_cores+1)] 
    pool = Pool(processes=n_cores)
    collected_results = pool.map(thread_job, iterator)
    pool.close()
    # Merge results
    overall_results = []
    for result_from_core in collected_results:
        for entry in result_from_core:
            overall_results.append(entry)
    with open('collected_depths/task3_multiproc.pickle', 'wb') as handle:
        pickle.dump(overall_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Remove temporary files
    print('\nRemoving temporary files:')
    for filename in os.listdir('collected_depths'):
        if 'TEMP' in filename:
            print(filename)
            os.remove('collected_depths/'+filename)

    print('Finished with {} results from {} core(s).'.format(
        len(overall_results), n_cores))

run_script()
