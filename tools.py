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
    index : np.ndarray
    data  : np.ndarray
    degree: np.ndarray

@dataclass
class Codeword:
    index : np.ndarray 
    data  : np.ndarray
    degree: int

class RingBuff:
    def __init__(self, size: int, block: int):
        self.data = np.zeros((size, block), dtype=np.int8)
        self.tail = 0
        self.head = 0
    def push(self, data: np.ndarray):
        assert self.tail - self.head < self.data.shape[0]
        self.data[self.tail % self.data.shape[0]] = data
        self.tail += 1
    def pop_one(self) -> Tuple[int, np.array]:
        assert self.tail > self.head
        head = self.head
        data = self.data[head % self.data.shape[0]]
        self.head += 1
        return head, data
    def pop_bat(self, batch: int) -> Tuple[int, np.array]:
        assert self.tail >= self.head + batch
        assert batch >= 1
        assert batch <= self.data.shape[0]
        head = self.head
        s = (self.head) % self.data.shape[0]
        e = (self.head + batch) % self.data.shape[0]
        self.head += batch
        data = self.data[s:e] if s < e else np.concatenate([self.data[s:], self.data[:e]], axis=-1)
        return head, data

def file_read(name: str, block: int) -> Tuple[np.ndarray, int]:
    """
    @param(name): the name of file
    @param(block): the block size of file
    @return(data, size): 
        data: the bytes from file, padded and reshaped to [..., block]
        size: the bytes size before padding
    """
    with open(name, 'rb') as f: 
        buffer = f.read()
    data = np.frombuffer(buffer, dtype=np.uint8)
    size = (data.shape[-1] + block - 1) // block
    size = size * block
    pad  = np.zeros(size - data.shape[-1], dtype=np.uint8)
    size = data.shape[-1]
    data = np.concatenate([data, pad]).reshape(-1, block)
    return data, size

if __name__ == '__main__':
    print(file_read("benchmarks/benchmark.log", 1024).shape)