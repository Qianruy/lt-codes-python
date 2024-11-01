import numpy as np
from random import random
from abc import *
from dataclasses import dataclass
from typing import *
from tools import *
from collections import *
from joblib import *

# for alignement, index=0 corresponds to no input. 
# actual packet indices start from 1. 

class Decoder(ABC):
    @abstractmethod
    def put_one(self, code: Codeword):
        """
        put codeword into decoder
        """
        pass

    @abstractmethod
    def put_bat(self, code: CodewordBatch):
        """
        put batch codeword into decoder
        """
        pass

    @abstractmethod
    def get_one(self) -> Optional[bytes]:
        """
        get one input from decoder
        """
        pass

    @abstractmethod
    def get_all(self) -> Optional[bytes]:
        """
        get all input from decoder
        """
        pass

class LubyDecoder(Decoder):
    """
    Luby Transform Decoder
    @field(buff): the buffer of received codewords
    @field(data): decoded data
    @field(collected): 
    """
    def __init__(self, d: int, block: int):
        """
        @param(d): maximum degree
        @param(block): the block size
        """
        self.buff = CodewordBatch(
            index=np.zeros((0, d), dtype=np.int64), 
            data=np.zeros((0, block), dtype=np.uint8), 
            degree=np.zeros((0, ), dtype=np.int64)
        )
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.collected = np.zeros((1, ), dtype=np.bool_)
        self.collected[0] = True
        self.block = block
        pass

    def put_one(self, code: Codeword):
        self.buff.add(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), self.block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))

    def put_bat(self, code: CodewordBatch):
        self.buff.join(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), self.block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))
    
    def get_one(self) -> Optional[bytes]: 
        raise NotImplementedError("Only support full decoding")
    
    def get_all(self) -> Optional[bytes]:
        print(self.collected.shape)
        while True:
            ripple = self.buff.degree == 1
            if np.all(~ripple): break
            # put data from degree=1 codewords to inputs
            index = np.bitwise_or.reduce(self.buff.index[ripple, :], axis=-1)
            self.data[index, :] = self.buff.data[ripple, :]
            # remove existing index from codewords
            index = np.unique(index)
            index = index * ~self.collected[index]
            self.collected[index] = True
            @delayed
            def fill(i):
                links = index == self.buff.index[i].reshape(-1, 1)
                # exist z: links[y][z] = true => clip buff.index[y] = 0
                # exist y: links[y][z] = true <=> index[z] is inside i
                if links.sum() >= 1:
                    self.buff.index[i] *= links.sum(-1, dtype=np.int64) < 1
                    self.buff.degree[i] -= links.sum()
                    remove = np.bitwise_xor.reduce(self.data[index * links.sum(-2)], axis=0)
                    self.buff.data[i] = np.bitwise_xor(self.buff.data[i], remove)
            Parallel(n_jobs=4, require='sharedmem')(fill(i) for i in range(self.buff.data.shape[0]))
        if not np.all(self.collected):
            print(np.arange(self.collected.shape[0])[~self.collected])
            return None
        else:
            return self.data[1:]

class PlowDecoder(Decoder):
    """
    """
    def __init__(self, d: int):
        pass
    
    def put_one(self, code: Codeword):
        pass

    def put_bat(self, code: CodewordBatch):
        pass
    
    def get_one(self) -> Optional[bytes]: 
        pass
    
    def get_all(self) -> Optional[bytes]: 
        pass

