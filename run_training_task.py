#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse 
import subprocess
from multiprocessing import Process, Queue 

task_list = [
#        ['ResNet18', False, 128], 
#        ['SubResNet18', True, 128],
#        ['ResNet34', False, 128],
#        ['SubResNet34', True, 128],
#        ['ResNet50', False, 128],
#        ['SubResNet50', True, 128],
        ['VGG16', False, 128],
        ['SubVGG16', True, 128],
        ]

training_stages = [
        [150, 0.1],
        [100, 0.01],
        [100, 0.001],
        ]

def train_exec_(task_queue, gpu_id):
    while True:
        train_task = task_queue.get() 
        if train_task is None:
            break
        for i in range(len(training_stages)):
            current_stage = training_stages[i]
            cmd_list = ['./train_model.py'] + [train_task[0]]
            cmd_list += ['--batchsize={}'.format(train_task[2])]
            cmd_list += ['--gpu_id={}'.format(gpu_id)]
            if train_task[1]:
                cmd_list += ['--sublinear']
            cmd_list += ['--num_epoches={}'.format(current_stage[0])]
            cmd_list += ['--lr={}'.format(current_stage[1])] 
            if i != 0:
                cmd_list += ['--resume']
           
            p = subprocess.run(cmd_list, check=True,
                    cwd=os.path.dirname(os.path.realpath(__file__)))
            p.check_returncode()

def main():
    parser = argparse.ArgumentParser(description="dispatch training task to GPUs",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('num_gpus', default=1, type=int, 
            help="the number of GPUs to run training tasks")

    args = parser.parse_args() 

    task_queue = Queue()
    process_list = []
    for i in range(args.num_gpus):
        p = Process(target=train_exec_, args=((task_queue, i,)))
        p.start()
        process_list.append(p)

    for task_item in task_list:
        task_queue.put(task_item)
    for i in range(args.num_gpus):
        task_queue.put(None)

    for p in process_list:
        p.join() 

if __name__ == '__main__':
    main()
