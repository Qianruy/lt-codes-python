import os
import sys
import math
import time
import numpy as np
import random
from random import choices
from numpy.random import Generator

import numpy as np
from random import random
from abc import *
from dataclasses import dataclass
from typing import *

# for alignement, index=0 corresponds to no input. 
# actual packet indices start from 1. 

@dataclass
class CodewordBatch:
    index   : np.ndarray
    data    : np.ndarray
    degree  : np.ndarray

@dataclass
class Codeword:
    index   : np.ndarray 
    data    : np.ndarray
    degree  : int

SYSTEMATIC = False
VERBOSE = False
# PACKET_SIZE = 65536
# PACKET_SIZE = 32768
# PACKET_SIZE = 16384
# PACKET_SIZE = 4096
# PACKET_SIZE = 1024
PACKET_SIZE = 512
# PACKET_SIZE = 128
ROBUST_FAILURE_PROBABILITY = 0.01
NUMPY_TYPE = np.uint64
# NUMPY_TYPE = np.uint32
# NUMPY_TYPE = np.uint16
# NUMPY_TYPE = np.uint8
EPSILON = 0.0001

class Symbol:
    __slots__ = ["index", "degree", "data", "neighbors"] # fixing attributes may reduce memory usage

    def __init__(self, index, degree=0, data=0, neighbors=None):
        self.index = index
        self.degree = degree
        self.data = data
        self.neighbors = neighbors if neighbors is not None else set()

    def log(self, blocks_quantity):
        neighbors, _ = generate_indexes(self.index, self.degree, 0, blocks_quantity)
        print("symbol_{} degree={}\t {}".format(self.index, self.degree, neighbors))

def generate_indexes(symbol_index, degree, start, blocks_quantity):
    """Randomly get `degree` indexes, given the symbol index as a seed

    Generating with a seed allows saving only the seed (and the amount of degrees) 
    and not the whole array of indexes. That saves memory, but also bandwidth when paquets are sent.

    The random indexes need to be unique because the decoding process uses dictionnaries for performance enhancements.
    Additionnally, even if XORing one block with itself among with other is not a problem for the algorithm, 
    it is better to avoid uneffective operations like that.

    To be sure to get the same random indexes, we need to pass 
    """
    if SYSTEMATIC and symbol_index < blocks_quantity:
        indexes = [symbol_index]               
        degree = 1     
    else:
        random.seed(symbol_index)
        indexes = random.sample(range(start, blocks_quantity), degree)

    return indexes, degree

def generate_indexes_plow(i, redundancy, encode_range, deg):
    """
    Generate selection indexes with a random value for each k in the range (1 - 1/(k-1), 1 - 1/k)
    
    Args:
        i (int): The index for the current input symbol.
        redundancy (int): The redundancy factor used for the encoding.
        encode_range (int): The range used for encoding.
        deg (int): The degree (number of encoded symbols to map to).
    
    Returns:
        list: A list of selected indexes.
    """
    selection_indexes = [math.ceil(i * redundancy)]  # Add the first index
    
    alpha = 1
    for k in range(2, deg + 1):
        # Generate a random value between (1 - 1/(k-1)) and (1 - 1/k)
        lower_bound = 1 - 1/(k-1)*alpha
        upper_bound = 1 - 1/k*alpha
        # random_point = random.uniform(lower_bound, upper_bound) 
        rng = np.random.default_rng()
        random_point = rng.poisson(upper_bound*encode_range-1) 
        while random_point > encode_range: random_point = rng.poisson(upper_bound*encode_range-1)  
        
        # Calculate the index based on the random point
        selected_index = int(i * redundancy + random_point - 1)
        selection_indexes.append(selected_index)

    return selection_indexes

def log(process, iteration, total, start_time):
    """Log the processing in a gentle way, each seconds"""
    global log_actual_time
    
    if "log_actual_time" not in globals():
        log_actual_time = time.time()

    if time.time() - log_actual_time > 1 or iteration == total - 1:
        
        log_actual_time = time.time()
        elapsed = log_actual_time - start_time + EPSILON
        speed = (iteration + 1) / elapsed * PACKET_SIZE / (1024 * 1024)

        print("-- {}: {}/{} - {:.2%} symbols at {:.2f} MB/s       ~{:.2f}s".format(
            process, iteration + 1, total, (iteration + 1) / total, speed, elapsed), end="\r", flush=True)