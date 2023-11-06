#!/usr/bin/env python
"""
author: Ken Chen
date: 2020-01-21
"""
import logging
import argparse
logger = logging.getLogger(__name__)


def idle_gpu(n: int=1, min_memory: int=4096, time_step: int=60, time_out: int=3600 * 16, skip: set=set()):
    import random, time, os
    from subprocess import Popen, PIPE
    if type(skip) is int:
        skip = {str(skip)}
    elaspsed_time = 0
    p = Popen(['/bin/bash', '-c', "nvidia-smi | grep GeForce | wc -l"], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    n_GPUs = int(out)
    random.seed(int(time.time()) % os.getpid())
    rand_priority = [random.random() for x in range(n_GPUs)]
    while elaspsed_time < time_out:
        cmd = "nvidia-smi | grep Default | awk '{print NR-1,$9,$11,$13,$3}' | sed 's/MiB//g;s/%//g;s/C//g'"
        p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        query_result, err = out.decode('utf8'), err.decode('utf8')
        rc = p.returncode
        query_result = query_result.strip().split('\n')

        gpu_list = list()
        for i, gpu_info in enumerate(query_result):
            gpu, memory_usage, memory_total, gpu_usage, temp = gpu_info.split(' ')
            if gpu in skip:
                continue
            memory_usage, memory_total, gpu_usage, temp = int(memory_usage), int(memory_total), int(gpu_usage), int(temp)
            memory = memory_total - memory_usage
            gpu_list.append((gpu, 500 * int(round(memory_usage / 500)) + rand_priority[i], int(round(gpu_usage/10)), int(round(temp / 10)), memory)) # reverse use
        ans = sorted(gpu_list, key=lambda x:(x[1], x[2], x[3]))
        if ans[0][-1] < min_memory:
            print("Waiting for available GPU... (%s)" % (time.asctime()))
            # time.sleep(60 * 10)
            time.sleep(time_step)
            elaspsed_time += time_step
            if elaspsed_time > time_out:
                raise MemoryError("Error: No available GPU with memory > %d MiB" % (min_memory))
        else:
            break

    #return ','.join(ans[0][0])
    return ','.join([ans[i][0] for i in range(n)])

def select_device(device_id=None):
    import os, torch
    ## device_id: None: auto / -1: cpu /
    if device_id is not None:
        try:
            d = int(device_id)
            device_id = d
        except ValueError:
            pass
    if device_id == -1 or str(device_id).lower() == "cpu":
        device_id = 'cpu'
        device = torch.device('cpu')
    else:
        if device_id is None:
            device_id = str(idle_gpu())
        else:
            device_id = str(device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device('cuda')
    return device_id, device

def set_gpu():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idle_gpu())


if __name__ == "__main__":
    ans = idle_gpu(n=1)
    print(ans)
