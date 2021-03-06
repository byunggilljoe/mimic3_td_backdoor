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

    task_names = ["test_lr"]
    
    for pi in [True, False]:
        for bl in task_names:
            print("TaskName: ", bl)
            for proportion in [0.01, 0.02, 0.03, 0.04, 0.05]:
                for strength in [0.5, 1.0, 1.5, 2.0]:
                    task_args = TaskParam(proportion, strength, poison_imputed=pi)
                    bex = TaskExecutor(bl, task_args, os.path.join(ROOT_PATH, "test_lr_"+{True:"all", False:"notimputed"}[pi]))
                    task_list.append(bex)
            task_args = TaskParam(0.0, 0.0, pi)
            bex = TaskExecutor(bl, task_args, os.path.join(ROOT_PATH, "test_lr_"+{True:"all", False:"notimputed"}[pi]))
            task_list.append(bex)
    
    
    Scheduler([0,1,2,3]*3, task_list)


if __name__ == "__main__":
    main()