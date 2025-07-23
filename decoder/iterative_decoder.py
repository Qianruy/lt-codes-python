from .base_decoder import *
from decode_logging import *
from datetime import datetime

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
        self.log = decoderState()

    def put_one(self, code: Codeword):
        self.buff.add(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), self.block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))

    def put_bat(self, code: CodewordBatch):
        # code.apply_loss(self.lossrate, fixed=True)
        # code.apply_burst_loss(1,5,100)
        self.buff.join(code)
        self.data.resize((max(1+code.index.max(), self.data.shape[0]), self.block))
        self.collected.resize(max(1+code.index.max(), self.collected.shape[0]))
    
    def get_one(self) -> Optional[bytes]: 
        raise NotImplementedError("Only support full decoding")
    
    def get_all(self) -> Optional[int]:
        round = 0
        ripple_stats = []
        decoded_stats = []
        while True:
            round += 1
            # if round > 1: break # test for only the first several iterations
            ripple = (self.buff.degree == 1)
            # loop runs until no codewords of degree 1 are left 
            if not np.any(ripple): break

            # put data from degree=1 codewords to inputs
            index = np.bitwise_or.reduce(self.buff.index[ripple, :], axis=-1)
            self.data[index, :] = self.buff.data[ripple, :]

            # add logs
            # codeword_ids = np.where(ripple)[0]
            # for idx, cw_id in zip(index,codeword_ids):
            #     self.log.log_codeword_degree_removal(idx, cw_id, round)

            # remove existing index from codewords
            print("codewords with degree 1: {}".format(np.count_nonzero(index)))
            decoded_stats.append(np.count_nonzero(index))
            index = np.unique(index)
            # log index of decoded symbols for each round
            self.log.log_decoded_symbols(round, index)

            print("ripple size: {}".format(np.count_nonzero(index)))
            ripple_stats.append(np.count_nonzero(index))
            index = index * ~self.collected[index]
            self.collected[index] = True

            update_buffer(self.buff.index, self.buff.degree, self.buff.data, index, self.data)
            
        num_of_solved = np.count_nonzero(self.collected)
        num_of_source = self.collected.shape[0]
        
        ripple_sum = 0; decoded_sum = 0
        rows = []

        for i, (r, d) in enumerate(zip(ripple_stats, decoded_stats), start=1):
            ripple_sum += r; decoded_sum += d
            rows.append([i, ripple_sum, decoded_sum, 1 - ripple_sum / num_of_source, 1 - decoded_sum / self.buff.data.size])
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        self.log._write_to_csv(f"./experiments/decoding_stats_{timestamp}.csv", rows)

        print(f"Solved symbols: {num_of_solved}/{num_of_source}")
        # write log to the file
        # self.log._write_to_json("./experiments/decoding_log.json")
        if not np.all(self.collected):
            print("Blocks are not all recovered, we cannot proceed the file writing.")
            # print unsolved indices of source symbols
            print(np.arange(num_of_source)[~self.collected])
            return num_of_solved
        else:
            return num_of_solved
            # return self.data[1:]
