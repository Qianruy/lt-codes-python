import numpy as np
from random import random
from abc import *
from dataclasses import dataclass
from typing import *
from tools import *
from distributions import *
from joblib import *

class Encoder(ABC):
    @abstractmethod
    def get_one(self) -> Codeword:
        """
        get codeword from encoder
        """
        pass

    @abstractmethod
    def get_bat(self, batch: int) -> CodewordBatch:
        """
        get codeword from encoder
        """
        pass

    @abstractmethod
    def put_one(self, data: np.ndarray):
        """
        put input into encoder
        """
        pass

    @abstractmethod
    def put_bat(self, data: np.ndarray):
        """
        put input into encoder
        """
        pass

class LubyEncoder(Encoder):
    """
    Luby Transform Encoder
    @field(data): all the inputs, array of shape [l]
    @field(prob): cummulative sum of degree distribution probability
    """
    def __init__(self, dd: np.ndarray, block: int, seed: int = 42):
        """
        @param(dd): degree distribution array of shape [d]
        @param(block): the input code word size
        """
        super().__init__()
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.prob = dd.cumsum(0)
        self.prob[-1] = 1
        self.rng = np.random.default_rng(seed=seed)

    def get_one(self) -> Codeword:
        """
        @return sample a degree d. then xor d inputs into a codeword
        """
        degree  = (self.rng.random() > self.prob).sum() + 1
        index   = self.rng.choice(np.arange(1, self.data.shape[0]), (self.prob.shape[0],), replace=False)
        index   = (np.arange(1, self.prob.shape[0] + 1) <= degree) * index
        data    = np.bitwise_xor.reduce(self.data[index])
        return Codeword(index, data, degree)

    def get_bat(self, batch: int) -> CodewordBatch:
        """
        @param(batch) the size of the batch
        @return sample multiple degrees [..d], for each [..d], xor d inputs into a codeword
        """
        degree  = (self.rng.random(size=(batch, 1)) > self.prob).sum(axis=-1) + 1
        index   = np.zeros((batch, self.prob.shape[0]), dtype=np.int_)
        seed = self.rng.random(size=batch)
        @delayed
        def fill(b):
            rng = np.random.default_rng(b + 42 + int(10 * seed[b]))
            index[b] = rng.choice(np.arange(1, self.data.shape[0]), (self.prob.shape[0],), replace=False)
        Parallel(n_jobs=8, require='sharedmem')(fill(b) for b in range(batch))
        index   = (np.arange(1, self.prob.shape[0] + 1) <= degree.reshape(-1, 1)) * index
        print(index)
        data    = np.bitwise_xor.reduce(self.data[index], axis=1)
        return CodewordBatch(index, data, degree)

    def put_one(self, data: np.ndarray):
        """
        @param(data) one input packet
        """
        assert data.dtype == np.uint8
        assert len(data.shape) == 1
        assert data.shape[-1] == self.data.shape[-1]
        self.data = np.concatenate([self.data, data.reshape(1, -1)], axis=0)

    def put_bat(self, data: np.ndarray):
        """
        @param(data) a batch of input packets 
        """
        assert data.dtype == np.uint8
        assert len(data.shape) == 2
        assert data.shape[-1] == self.data.shape[-1]
        self.data = np.concatenate([self.data, data], axis=0)

class PlowEncoder(Encoder):
    """
    Plow Encoder for real time streaming
    @field(ring) the ring buffer for all inputs
    """
    def __init__(self, rsize: int):
        """
        @param(rsize): the maximum size of ring buffer
        """
        self.ring = RingBuff()
        self.buff = CodewordBatch()

    def put_one(self, data: np.ndarray):
        pass

    def get_one(self) -> Codeword:
        pass

if __name__ == '__main__':
    encoder = LubyEncoder(np.array([0.5, 0.25, 0.25]), 1024)
    encoder.put_one(np.zeros(1024, dtype=np.uint8))
    print(encoder.get_one())
    print(encoder.get_one())
