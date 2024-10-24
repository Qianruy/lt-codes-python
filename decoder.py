import numpy as np
from random import random
from abc import *
from dataclasses import dataclass
from typing import *
from tools import *
from collections import *

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
    def get(self) -> Optional[bytes]:
        """
        get input from decoder
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
            index=np.zeros((0, d), dtype=np.uint8), 
            data=np.zeros((0, block), dtype=np.uint8), 
            degree=np.zeros((0, ), dtype=np.uint8)
        )
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.collected = np.zeros((1, ), dtype=np.bool_)
        self.collected[0] = True
        pass

    def put_one(self, code: Codeword):
        self.buff.add(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))

    def put_bat(self, code: CodewordBatch):
        self.buff.join(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))
    
    def get_one(self) -> Optional[bytes]: 
        raise NotImplementedError("Only support full decoding")
    
    def get_all(self) -> Optional[bytes]:
        while True:
            ripple = self.buff.degree == 1
            if np.all(~ripple): break
            # put data from degree=1 codewords to inputs
            index = self.buff.index[ripple].flatten()
            self.data[index] = self.buff.data[ripple]
            self.collected[index] = True
            # remove index from codewords
            links = index == self.buff.index.reshape(*self.buff.shape, 1)
            
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
