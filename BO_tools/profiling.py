import torch
import subprocess as sp
import os

byte_to_GiB = 1073741824


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    
    # return memory_free_values
    print(memory_free_values)


def get_GPU_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)/byte_to_GiB
    a = torch.cuda.memory_allocated(0)/byte_to_GiB
    f = r-a  # free inside reserved

    print('memory allocated:', a, ' GiB; free reserved memory: ', r, ' GiB; free inside reserved', f, ' GiB')



def get_CPU_memory_info():
    pass