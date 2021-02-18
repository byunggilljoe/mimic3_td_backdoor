import sys
import os
import subprocess
from queue import Queue
import threading
from glob import glob
from time import time 


class Scheduler(object):
    def __init__(self, gpu_list, executors_list):
        #self.error_cmd_list = []
        self.waiting_queue = Queue()
        self.gpu_queue = Queue()
        # threads notify Scheduler they finish their job by queing ready_queue
        self.ready_queue = Queue()

        for i in gpu_list:
            self.gpu_queue.put(i)

        for e in executors_list:
            self.waiting_queue.put(e)     

        xs =[]
        for i in range(min(len(gpu_list), self.waiting_queue.qsize())):
            e = self.waiting_queue.get()
            x = threading.Thread(target=self.start_executor, args=(e,))
            xs.append(x)

        for x in xs:
            x.start()

        self.schedule()
        
    def schedule(self):
        while self.ready_queue.empty() == False or self.waiting_queue.empty() == False:
            ready_e = self.ready_queue.get(block=True)
            
            x = threading.Thread(target=self.start_executor, args=(ready_e,))
            x.start()
        #print("error cmd list:", self.error_cmd_list)
                

    def start_executor(self, e):
        print("----start executor", self.gpu_queue.qsize())
        e.set_gpu(self.gpu_queue.get())

        try:
            e.execute()
        except Exception:
            #self.error_cmd_list.append(e.cmd)
            print("execution exception occured!")

        self.gpu_queue.put(e.get_gpu())
        if self.waiting_queue.empty() == False:
            self.ready_queue.put(self.waiting_queue.get())

class Executor():
    def __init__(self, cmd):
        self.cmd = cmd
        self.gpu_num = 0

    def set_gpu(self, gpu_num):
        self.gpu_num = gpu_num
    def get_gpu(self):
        return self.gpu_num

    def execute(self):
        self.before_script()
        print("Execute:", self.cmd)
        self.run_script()
        self.after_script()

    def before_script(self):
        pass

    def run_script(self):
        os.system(self.cmd)

    def after_script(self):
        pass


class TaskExecutor(Executor):
    def __init__(self, task_name, task_args, result_dir):
        self.gpu_num = 0
        self.FILE_DIR = os.path.dirname(__file__)
        self.task_name = task_name
        self.task_args = task_args.copy()

        self.BASE_CMDS = {"train_lr":"python -um mimic3models.in_hospital_mortality.torch.poisoning_train_raw_714 --model lr --network aaa",
                        "train_mlp": "python -um mimic3models.in_hospital_mortality.torch.poisoning_train_raw_714 --model mlp --network aaa",
                        "train_lstm": "python -um mimic3models.in_hospital_mortality.torch.poisoning_train_raw_48_76 --network aaa",
                        "test_lr": "python -um mimic3models.in_hospital_mortality.torch.poisoning_attack_raw_714 --model lr --network aaa",
                        "test_mlp": "python -um mimic3models.in_hospital_mortality.torch.poisoning_attack_raw_714 --model mlp --network aaa",
                        "test_lstm": "python -um mimic3models.in_hospital_mortality.torch.poisoning_attack_raw_48_76 --network aaa",
                        }


        start_dir = os.path.abspath(os.getcwd())
        self.result_full_path = os.path.join(
            start_dir, result_dir, task_name, str(task_args["poisoning_proportion"]) +"_" + str(task_args["poisoning_strength"]) +"_"+ str(int(time())))

        self.cmd = self.make_cmd()
 
    def before_script(self):
        if not os.path.exists(self.result_full_path):
            os.makedirs(self.result_full_path, exist_ok=True)

    def set_gpu(self, gpu_num):
        self.gpu_num = gpu_num
        self.cmd = self.make_cmd()

    def make_cmd(self):
        gpu_script = "CUDA_VISIBLE_DEVICES={}".format(self.gpu_num)

        task_param_str = " ".join(
            ["--{} {}".format(k, v) for k, v in self.task_args.items()])

        return gpu_script + " " + self.BASE_CMDS[self.task_name] + " " + task_param_str 

    def run_script(self):
        #super(TaskExecutor, self).run_script()
        out = str(subprocess.check_output(self.cmd, shell=True))
        
        result_txt = out.split("accuracy = ")[1].split("\\n")
        result_txt = "accuracy = "+"\n".join(result_txt)
        result_txt = result_txt.strip("\"")
        txt_file = open(os.path.join(self.result_full_path, "result.txt") , "w+")
        txt_file.write(result_txt)
        txt_file.close()
        #print(result_txt)
        
    def after_script(self):
        pass
        

def TaskParam(poisoning_proportion, poisoning_strength, poison_imputed):
    attack_args = {}
    attack_args["poisoning_proportion"] = poisoning_proportion
    attack_args["poisoning_strength"] = poisoning_strength
    attack_args["poison_imputed"] = {True:"all", False:"notimputed"}[poison_imputed]
    
    
    return attack_args


def main():
 
    ROOT_PATH="./results/"
    task_list = []

    # Attack param examples

    task_names = ["test_lstm"]
    
    for bl in task_names:
        print("TaskName: ", bl)
        task_args = TaskParam(0.01, 1.0)
        bex = TaskExecutor(bl, task_args, os.path.join(ROOT_PATH, "attack_test"))
        task_list.append(bex)

    
    
    Scheduler([0,1,2,3], task_list)


if __name__ == "__main__":
    main()