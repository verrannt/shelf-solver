###########################################################################
#
# Run the shelf solver script a specified number of times on one of three
# tasks and collect the depths (i.e. minimal number of moves to find a 
# solution):
#
# - Task 1: Sort object so that each row has only objects of one color
# - Task 2: Sort objects so that each row has all the unique colors and all 
#   the unique shapes once 
# - Task 3: Sort objects so that each row and column has each of the four 
#   colors once
#
# The task can be specified by passing 1, 2 or 3 to the --task flag when
# calling this script.
# Furthermore, multiprocessing can be enabled, which puts the amount of 
# instances specified using the --tries_per_core flag (default 100)on each
# available CPU thread. If using only one thread, tries_per_core is the 
# total amount of randomly generated shelves that will be solved for, else
# it is the max amount of threads * tries_per_core.
#
# The script saves the collected depths to a directory 'collected_depths/'
# at the location it is called from. If this dir does not exist, it will 
# be created. 
# !!!CAREFUL!!!: If this dir exists and has contents 'task1.pickle', 
# 'task2.pickle' etc. those will be overridden by the new files.
#
# Example usage:
# $ python run_solver.py --task 1 --tries_per_core 250 --multiprocessing 
#
# -------------------------------------------------------------------------
#
# @author Pascal Schroeder
# @github verrannt
# @date   2020-04-20
#
###########################################################################

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

    parser.add_argument('--task', type=int,
        help='Which task to run')
    parser.add_argument('--multiprocessing', nargs='?', const=True,
        default=False, help='Whether to use all cpu threads or single one.')
    parser.add_argument('--tries_per_core', type=int,
        default=100, help='How many random shelves to be run on each core.')

    return parser.parse_args(args)


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
