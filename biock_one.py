#!/usr/bin/env python3

import os
import sys
import warnings
import shutil
import time
import numpy as np
import logging
from typing import Literal, Optional, Tuple, List, Any
from argparse import Namespace
from subprocess import Popen, PIPE

### =========================== BASIC FUNCTIONS ======================= ###
def get_run_info(argv: List[str], args: Namespace=None, **kwargs) -> str:
    s = list()
    s.append("")
    s.append("##time: {}".format(time.asctime()))
    s.append("##cwd: {}".format(os.getcwd()))
    s.append("##cmd: {}".format(' '.join(argv)))
    if args is not None:
        s.append("##args: {}".format(args))
    for k, v in kwargs.items():
        s.append("##{}: {}".format(k, v))
    return '\n'.join(s)


def run_bash(cmd) -> Tuple[int, str, str]:
    r"""
    Return
    -------
    rc : return code
    out : output
    err : error
    """
    p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out, err = out.decode('utf8'), err.decode('utf8')
    rc = p.returncode
    return (rc, out, err)

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

def make_logger(
        title: Optional[str]="", 
        filename: Optional[str]=None, 
        level: Literal["INFO", "DEBUG"]="INFO", 
        mode: Literal['w', 'a']='w',
        trace: bool=True, 
        **kwargs):
    if isinstance(level, str):
        level = getattr(logging, level)
    logger = logging.getLogger(title)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)

    if trace is True or ("show_line" in kwargs and kwargs["show_line"] is True):
        formatter = logging.Formatter(
                '%(levelname)s(%(asctime)s) [%(filename)s:%(lineno)d]:%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s(%(asctime)s):%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    # formatter = logging.Formatter(
    #     '%(message)s\t%(levelname)s(%(asctime)s)', datefmt='%Y%m%d %H:%M:%S'
    # )

    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if filename is not None:
        if os.path.exists(filename):
            suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(filename)))
            while os.path.exists("{}.conflict_{}".format(filename, suffix)):
                suffix = "{}_1".format(suffix)
            shutil.move(filename, "{}.conflict_{}".format(filename, suffix))
            warnings.warn("log {} exists, moved to to {}.conflict_{}.log".format(filename, filename, suffix))
        fh = logging.FileHandler(filename=filename, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


### =========================== MACHINE LEARNING TOOLS ========================== ## 
def split_train_val_test_by_group(groups: List[Any], n_splits: int, val_folds: int, test_folds: int) -> Tuple[List, List, List]:
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=n_splits)
    train_inds, val_inds, test_inds = list(), list(), list()
    for i, (_, inds) in enumerate(splitter.split(groups, groups=groups)):
        if i < val_folds:
            val_inds.append(inds)
        elif i >= val_folds and i < test_folds + val_folds:
            test_inds.append(inds)
        else:
            train_inds.append(inds)
    train_inds = np.concatenate(train_inds)
    if val_folds > 0:
        val_inds = np.concatenate(val_inds)
    if test_folds:
        test_inds = np.concatenate(test_inds)
    return train_inds, val_inds, test_inds


### ============================= PYTORCH ===========================- 
try:
    import torch
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

except ImportError as err:
    warnings.warn("{}".format(err))
# END import torch

