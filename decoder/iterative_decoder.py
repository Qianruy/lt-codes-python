from .base_decoder import *
from numba import njit, prange
from logging import *

@njit(parallel = True, nogil=True)
def update_buffer(buff_index, buff_degree, buff_data, indices_set, data):
    """ 
    Optimized update function using Numba to process buffer modifications. 
    """
    num_codewords = buff_data.shape[0]
    num_source = data.shape[0]

    # Precompute indices as a set for fast lookups
    indices = set(indices_set)

    for i in prange(num_codewords):  # prange enables parallelism
        links = np.zeros(buff_index[i].size, dtype=np.bool_)
        for j in range(buff_index[i].size):
            if buff_index[i, j] in indices:
                links[j] = True

        if np.any(links):  
            buff_index[i] *= ~links  # remove matched indices
            buff_degree[i] -= np.count_nonzero(links)  # reduce degree

            # Perform XOR reduction (manual for speed)
            removal = np.zeros(num_source, dtype=np.bool_)
            for j in prange(1, num_source+1):
                if j in buff_index[i][links]:
                    removal[j] = True
            xor_value = np.uint8(0)
            for value in data[removal][0]:
                xor_value ^= value
            buff_data[i] = xor_value

class IterativeDecoder(Decoder):
    """
    Luby Transform Decoder
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
    
    def get_one(self) -> Optional[bytes]: 
        raise NotImplementedError("Only support full decoding")
    
    def get_all(self) -> Optional[int]:
        round = 0
        buff = self.buff

        while True:
            round += 1
            ripple = (buff.degree == 1)
            # loop runs until no codewords of degree 1 are left 
            if not np.any(ripple): break

            # put data from degree=1 codewords to inputs
            index = np.bitwise_or.reduce(buff.index[ripple, :], axis=-1)
            self.data[index, :] = buff.data[ripple, :]

            # remove existing index from codewords
            index = np.unique(index)
            # log index of decoded symbols for each round
            # log.log_decoded_symbols(round, index)

            print("ripple size: {}".format(np.count_nonzero(index)))
            index = index * ~self.collected[index]
            self.collected[index] = True

            update_buffer(self.buff.index, self.buff.degree, self.buff.data, index, self.data)
            
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
            # return self.data[1:]
