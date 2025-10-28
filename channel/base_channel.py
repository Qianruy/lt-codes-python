import numpy as np
from abc import ABC, abstractmethod
from tools import CodewordBatch

class Channel(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass
    
    @abstractmethod
    def model(self, num_packets):
        pass

    @abstractmethod
    def apply(self, codeBatch: 'CodewordBatch'):
        """
        data: CodewordBatch of symbols or bits
        returns: corrupted version of data
        """
        drop_mask = self.model(codeBatch.num_codewords)
        codeBatch.drop_rows(drop_mask)
        if np.any(drop_mask):
            print(f"Applied loss: {np.count_nonzero(drop_mask)} codewords dropped out of {len(drop_mask)}")
        return codeBatch

class BEC_Channel(Channel):
    def __init__(self, erasure_rate: float, fixed=False):
        self.p = erasure_rate
        self.fixed = fixed

    def model(self, num_packets):
        if self.fixed: # apply fixed rate of loss
            # choose packet without replacement
            k = int(num_packets * self.p)  
            drop_mask = np.zeros(num_packets, dtype=bool)
            drop_mask[np.random.choice(np.arange(num_packets), size=k, replace=False)] = True
        else:
            drop_mask = (np.random.rand(num_packets) < self.p)
        return drop_mask

    def apply(self, data: 'CodewordBatch'):
        return super().apply(data)
