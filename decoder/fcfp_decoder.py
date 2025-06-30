from .base_decoder import *
from numba import njit, prange
from logging import *
import csv
from datetime import datetime

class FIFODecoder(Decoder):
    """
    First Come First Peel Decoder
    @field(buff): the buffer of received codewords
    @field(data): decoded data
    @field(collected): 
    """
    def __init__(self, d: int, block: int, lossrate: float = 0.0):
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
        self.collected_at = np.zeros((1, ), dtype=np.int64)
        self.block = block
        self.lossrate = lossrate
    
    def put_one(self, code: Codeword):
        self.buff.add(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), self.block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))
    
    def put_bat(self, code: CodewordBatch):
        code.apply_loss(self.lossrate)
        self.buff.join(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), self.block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))
        self.collected_at.resize(max(1+code.index.max(), self.collected.shape[0]))

    def get_one(self) -> Optional[bytes]: 
        raise NotImplementedError("Only support full decoding")    

    def get_all(self) -> Optional[int]:
        snapshots = []
        for i in range(1, self.buff.data.shape[0] + 1):
            if i % 100 == 0 and i < self.buff.data.shape[0] - 4*600:  
                snapshots.append(self.buff.degree[i: i+3*600].copy())
            self.peel(i + 1)
        snapshots = np.asarray(snapshots) 
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%m")
        filename = f'./experiments/fcfp_{timestamp}_{self.lossrate}.csv'
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter'] + list(range(snapshots.shape[1])))
            for snap_idx, row in enumerate(snapshots):
                iter_no = snap_idx * 100
                writer.writerow([iter_no] + row.tolist())
        num_of_solved = np.count_nonzero(self.collected)
        num_of_source = self.collected.shape[0]
        print(f"Solved symbols: {num_of_solved}/{num_of_source}")
        if not np.all(self.collected):
            print("Blocks are not all recovered, we cannot proceed the file writing.")
            # print unsolved indices of source symbols
            print(np.arange(num_of_source)[~self.collected])
            return num_of_solved
        else:
            return num_of_solved

    def peel(self, cut: int) -> Optional[int]:
        peeled = set()
        while True:
            # get ripple index 
            ripple = (self.buff.degree[:cut] == 1)
            print("iter", cut, "nonzero in window:",
                np.count_nonzero(self.buff.degree[cut:cut+3*600]), np.count_nonzero(self.buff.degree))
            # loop runs until no codewords of degree 1 are left 
            if not np.any(ripple): break
            # put data from degree=1 codewords to inputs
            index = np.bitwise_or.reduce(self.buff.index[:cut][ripple, :], axis=-1)
            self.data[index, :] = self.buff.data[:cut][ripple, :]
            # remove existing index from codewords
            index = np.unique(index)
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d%H")
            filename = f'./experiments/sf_{timestamp}_{self.lossrate}.csv'
            with open(filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([cut, ", ".join(map(str,index))])
            # create index
            index = index * ~self.collected[index]
            self.collected[index] = True
            self.collected_at[index] = cut
            # peel decoded index from buffer
            update_buffer(self.buff.index, self.buff.degree, self.buff.data[:cut+900], index, self.data[:cut+900])