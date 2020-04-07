import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count

from utils.timer_decorator import timer
import shelfsolver
        
    
def getArgs(args = sys.argv[1:]):
    """
    Parses arguments from the command line when file
    is run as main script. Run with --help flag to
    see available arguments at the command line.
    """
    parser = argparse.ArgumentParser(
        description='Parses command line arguments')

    # Add arguments to be parsed here ####
    parser.add_argument('--task', type=int,
        help='Which task to run')
    parser.add_argument('--multiprocessing', nargs='?', const=True,
        default=False, help='Whether to use all cpu threads or single one.')
    parser.add_argument('--tries_per_core', type=int,
        default=100, help='How many random shelves to be run on each core.')
    
    return parser.parse_args(args)

    
####################################
    
depths = []
solver = shelfsolver.ShelfSolver()
    

def thread_job(coreID):
    core_results = []
    shelfgen = shelfsolver.ShelfGenerator()
    for i in range(n_tries_per_core+1):
        if i%20==0:
            print('\nCore {}, Step {}'.format(coreID, i))
            print('Successful runs so far: {}'.format(len(core_results)))
            with open('collected_depths/TEMP_task{}_core{}.pickle'
                      .format(task, coreID), 'wb') as handle:
                pickle.dump(core_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        shelf = shelfgen.new_random_shelf(seed=coreID+i*10)
        core_result = solver.solve_task(shelf, task, search_type="stack", collect=True)
        if core_result:
            core_results.append(core_result)
    print('\nCore {} finished.'.format(coreID))
    return core_results

@timer
def run_script(use_multiprocessing, n_tries_per_core, task):

    if use_multiprocessing:
        n_cores = cpu_count()
    else:
        n_cores = 1
    
    print('Running solver script with {} core(s) on {} instances per core.'
          .format(n_cores, n_tries_per_core))

    iterator = [coreID for coreID in range(1,n_cores+1)] 
    pool = Pool(processes=n_cores)
    collected_results = pool.map(thread_job, iterator)
    pool.close()
    # Merge results
    overall_results = []
    for result_from_core in collected_results:
        for entry in result_from_core:
            overall_results.append(entry)
    with open('collected_depths/task{}_latestrun.pickle'.format(task), 'wb') as handle:
        pickle.dump(overall_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Remove temporary files
    print('\nRemoving temporary files:')
    for filename in os.listdir('collected_depths'):
        if 'TEMP' in filename:
            print(filename)
            os.remove('collected_depths/'+filename)

    print('Finished with {} results from {} core(s).'.format(
        len(overall_results), n_cores))

if __name__ == '__main__':
    args = getArgs()
    
    use_multiprocessing = args.multiprocessing
    n_tries_per_core = args.tries_per_core
    task = args.task
    
    if task not in [1,2,3]:
        raise ValueError('task must be 1, 2 or 3.')
    
    run_script(use_multiprocessing, n_tries_per_core, task)
