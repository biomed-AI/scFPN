#!/usr/bin/env python3

import argparse, os, sys, time, gzip, pickle, warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Dict, List, Union
from functools import partial
print = partial(print, flush=True)
from collections import defaultdict, OrderedDict
from scipy.stats import spearmanr, pearsonr
import scipy.sparse as ssp
from scipy.sparse import csr_matrix, csc_matrix

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import Dataset, DataLoader
except ImportError as err:
    warnings.warn(err)

try:
    import scanpy as sc
except ImportError as err:
    warnings.warn(err)

