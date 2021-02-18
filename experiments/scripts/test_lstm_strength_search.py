import sys
sys.path.append("./")
import os
import subprocess
from queue import Queue
import threading
from glob import glob
from experiments.execution import TaskExecutor, TaskParam, Scheduler

def main():
 
    ROOT_PATH="./results/"
    task_list = []

    # Attack param examples

    task_names = ["test_lstm"]
    
    for pi in [True, False]:
        for bl in task_names:
            print("TaskName: ", bl)
            for proportion in [0.05]:
                for strength in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]:
                    task_args = TaskParam(proportion, strength, poison_imputed=pi)
                    bex = TaskExecutor(bl, task_args, os.path.join(ROOT_PATH, "test_lstm_strength_search_"+{True:"all", False:"notimputed"}[pi]))
                    task_list.append(bex)

            task_args = TaskParam(0.0, 0.0, pi)
            bex = TaskExecutor(bl, task_args, os.path.join(ROOT_PATH, "test_lstm_strength_search_"+{True:"all", False:"notimputed"}[pi]))
            task_list.append(bex)
    
    Scheduler([0,1,2,3]*3, task_list)


if __name__ == "__main__":
    main()