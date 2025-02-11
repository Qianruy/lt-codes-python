from .base_decoder import *
from numba import njit
from logging import *

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
        while True:
            round += 1
            ripple = self.buff.degree == 1
            # loop runs until no codewords of degree 1 are left 
            if np.all(~ripple): break

            # put data from degree=1 codewords to inputs
            index = np.bitwise_or.reduce(self.buff.index[ripple, :], axis=-1)
            self.data[index, :] = self.buff.data[ripple, :]

            # remove existing index from codewords
            index = np.unique(index)
            # log index of decoded symbols for each round
            # log.log_decoded_symbols(round, index)

            print("ripple size: {}".format(np.count_nonzero(index)))
            index = index * ~self.collected[index]
            self.collected[index] = True
            # indices = self.buff.index[ripple, :].flatten()
            # unique_indices = np.unique(indices)
            # unique_indices = unique_indices[~self.collected[unique_indices]]
            # self.data[unique_indices, :] = self.buff.data[ripple, :]
            # self.collected[unique_indices] = True
            # # Compute updates for all codewords in one step
            # links = self.buff.index[:, None, :] == index[:, None]
            # valid_links = links.sum() > 0
            # Update buffer using optimized numba function
            
            def update_buffer(buff_index, buff_degree, buff_data, indices, data):
                """ Optimized update function using Numba to process buffer modifications. """
                num_codewords = buff_data.shape[0]
                num_source = data.shape[0]

                for i in range(num_codewords):
                    links = np.isin(buff_index[i], indices)  # mask for linked indices
                    
                    if np.any(links):  
                        buff_index[i] *= ~links  # remove matched indices
                        buff_degree[i] -= np.count_nonzero(links)  # reduce degree

                        # Perform XOR reduction to update data
                        removal = np.isin(range(1, num_source+1), buff_index[i][links])
                        buff_data[i] ^= np.bitwise_xor.reduce(data[removal], axis=0)

            update_buffer(self.buff.index, self.buff.degree, self.buff.data, index, self.data)
            # @delayed
            # def fill(i):
            #     # check if unique indices resolved in the current iteration
            #     # could update the codewords in the buffer
            #     links = index == self.buff.index[i].reshape(-1, 1)
            #     # exist z: links[y][z] = true => clip buff.index[y] = 0
            #     # exist y: links[y][z] = true <=> index[z] is inside codeword i
            #     # links[y][z] = true <=> self.buff.index[i][y] == unique_index[z]
            #     if links.sum() >= 1: 
            #         self.buff.index[i] *= links.sum(-1, dtype=np.int64) < 1
            #         self.buff.degree[i] -= links.sum()
            #         removal = np.bitwise_xor.reduce(self.data[index * links.sum(-2)], axis=0)
            #         self.buff.data[i] = np.bitwise_xor(self.buff.data[i], removal)
            # Parallel(n_jobs=4, require='sharedmem')(fill(i) for i in range(self.buff.data.shape[0]))
        if not np.all(self.collected):
            print("Blocks are not all recovered, we cannot proceed the file writing.")
            # print unsolved indices of source symbols
            print(np.count_nonzero(self.collected))
            print(np.arange(self.collected.shape[0])[~self.collected])
            return np.count_nonzero(self.collected)
        else:
            return np.count_nonzero(self.collected)
            # return self.data[1:]
