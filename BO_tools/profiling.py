import torch


def get_GPU_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print('memory allocated:', a, '; free reserved memory: ', r, '; free inside reserved', f)



def get_CPU_memory_info():
    