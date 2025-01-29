import numpy as np
import string
import argparse
import random
from abc import *
from dataclasses import dataclass
from typing import *
from numba import jit

config = {
    "SYSTEMATIC": False,
    "VERBOSE": False,
    "MAX_DEGREE": 5,
    "WINDOWSIZE": 500,
    "LOSS_PROBABILITY": 0.01,
    # PACKET_SIZE = 65536
    # PACKET_SIZE = 32768
    # PACKET_SIZE = 16384
    # PACKET_SIZE = 4096
    # PACKET_SIZE = 1024
    # PACKET_SIZE = 512
    "PACKET_SIZE": 128,
    "ROBUST_FAILURE_PROBABILITY": 0.01,
    "NUMPY_TYPE": np.uint64,
    # NUMPY_TYPE = np.uint32
    # NUMPY_TYPE = np.uint16
    # NUMPY_TYPE = np.uint8
    "EPSILON": 0.0001
}

# for alignement, index=0 corresponds to no input. 
# actual packet indices start from 1. 

@dataclass
class Codeword:
    index : np.ndarray # Array of indices, indicating positions or identifiers for the source symbols
    data  : np.ndarray # Array holding the actual data bits of the codeword
    degree: int # Number of connections a codeword has

@dataclass
class CodewordBatch:
    index : np.ndarray # 2D array for indices of all codewords in the batch
    data  : np.ndarray  # 2D array for data of all codewords in the batch
    degree: np.ndarray # 1D array for degrees of all codewords in the batch
    used: int = 0 # Tracks the number of used elements

    def add(self, code: Codeword):
        if self.used + 1 > self.index.shape[0]:
            self.index.resize((self.index.shape[0] * 2, self.index.shape[1]))
            self.data.resize((self.data.shape[0] * 2, self.data.shape[1]))
            self.degree.resize((self.degree.shape[0] * 2))

        self.index[self.used] = code.index
        self.data[self.used] = code.data
        self.degree[self.used] = code.degree
        self.used += 1

    def join(self, code: 'CodewordBatch'):
        self.index.resize((self.index.shape[0] + code.index.shape[0], self.index.shape[1]))
        self.index[-code.index.shape[0]:] = code.index
        self.data.resize((self.data.shape[0] + code.data.shape[0], self.data.shape[1]))
        self.data[-code.data.shape[0]:] = code.data
        self.degree.resize((self.degree.shape[0] + code.degree.shape[0]))
        self.degree[-code.degree.shape[0]:] = code.degree

    def apply_loss(self, lossrate: float):
        """
        Randomly drop codewords with probability `lossrate`.
        """
        if lossrate <= 0.0: return
        # add function may cause shape unalignment
        drop_mask = (np.random.rand(self.data.shape[0]) < lossrate)
        self.index = self.index[~drop_mask]
        self.data = self.data[~drop_mask]
        self.degree = self.degree[~drop_mask]
        self.used = self.index.shape[0]

        print(f"Applied loss: {np.count_nonzero(drop_mask)} codewords dropped out of {len(drop_mask)}")

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

def generate_random_text_file(filename, filesize):
    chars = string.ascii_letters + string.digits + string.punctuation + ' '

    with open(filename, 'w') as f:
        size_written = 0
        while size_written < filesize:
            chunk_size = min(1024, filesize - size_written)
            random_text = ''.join(random.choice(chars) for _ in range(chunk_size))
            f.write(random_text)
            size_written += chunk_size

    print(f"{filename} successfully generated, size = {filesize} bytes.")

if __name__ == '__main__':
    # Test function 'file_read()'
    print(file_read("benchmarks/benchmark.log", 1024)[0].shape)

    # Test function 'generate_random_text_file'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filename', type=str, help='file name')
    # parser.add_argument('filesize', type=int, help='file size')

    # args = parser.parse_args()

    # generate_random_text_file(args.filename, args.filesize)
