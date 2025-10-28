from .base_decoder import *
from decode_logging import *
from datetime import datetime
from collections import deque

class IterativeDecoder(Decoder):
    """
    Beleif Propagation Decoder over BEC
    @field(buff): the buffer of received codewords
    @field(data): decoded data
    @field(collected): 
    """
    def __init__(self, d: int, block: int, lossrate: float = 0.0):
        """
        @param(d): maximum degree
        @param(block): the block size
        """
        self.buff = CodewordBatch.empty(block)
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.collected = np.zeros((1, ), dtype=np.bool_)
        self.collected[0] = True
        self.block = block
        self.lossrate = lossrate
        self.log = decoderState()

    def put_one(self, code: Codeword):
        self.buff.add(code)
        max_idx = int(code.index.max()) if code.index.size > 0 else 0
        self._ensure_capacity(max_idx)

    def put_bat(self, code: CodewordBatch):
        # code.apply_loss(self.lossrate, fixed=True)
        # code.apply_burst_loss(1,5,100)
        self.buff.join(code)
        self._ensure_capacity(self.buff.max_index)

    def _ensure_capacity(self, max_index: int):
        target = max(max_index + 1, 1)
        if target > self.data.shape[0]:
            pad = target - self.data.shape[0]
            self.data = np.concatenate((self.data, np.zeros((pad, self.block), dtype=np.uint8)), axis=0)
            self.collected = np.concatenate((self.collected, np.zeros((pad,), dtype=np.bool_)), axis=0)
    
    def get_one(self) -> Optional[bytes]: 
        raise NotImplementedError("Only support full decoding")
    
    # def get_all(self) -> Optional[int]:
    #     # pack to CSR
    #     row_ptr, src_idx_flat = batch_to_csr(self.buff.index, self.buff.degree)
    #     M = row_ptr.size - 1
    #     N = 0 if src_idx_flat.size == 0 else int(src_idx_flat.max()) + 1
    #     N = max(N, self.data.shape[0])
    #     # ensure outputs sized
    #     if self.data.shape[0] < N:
    #         self.data = np.resize(self.data, (N, self.block))
    #         self.collected = np.resize(self.collected, (N,))

    #     # build CSC
    #     # print("Construct fast backlinks...")
    #     col_ptr, code_idx_flat = csr2csc(M, N, row_ptr, src_idx_flat)

    #     # work copies
    #     cw_data = self.buff.data[:M, :].copy().astype(np.uint8)
    #     src_known = np.full(N, -1, np.int8)
    #     src_known[self.collected.astype(np.bool_)] = 1

    #     # print("Peeling begins...")
    #     solved, degree_left = peeling(
    #         row_ptr, src_idx_flat, col_ptr, code_idx_flat,
    #         cw_data, self.data, src_known
    #     )

    #     self.collected = (src_known == 1)
    #     num_of_solved = int(self.collected.sum())
    #     num_of_source = self.collected.size

    #     if not self.collected.all():
    #         # print indices still unknown
    #         remaining = np.where(~self.collected)[0]
    #         print(f"Solved symbols: {num_of_solved}/{num_of_source}")
    #     return num_of_solved
    
    def get_all(self) -> Optional[int]:
        round = 0
        # ripple_stats = []
        # decoded_stats = []
        ripple_queue = deque(np.nonzero(self.buff.degree == 1)[0])
        col_ptr, code_idx = csr2csc(
            self.buff.num_codewords,
            self.data.shape[0],
            self.buff.row_ptr,
            self.buff.src_idx
        )

        while ripple_queue:
            round += 1
            rows = np.array([r for r in ripple_queue if self.buff.degree[r] == 1], dtype=np.int32)
            ripple_queue.clear()
            if rows.size == 0:
                continue

            indices = self.buff.degree_one_indices(rows)
            valid = indices >= 0
            if np.any(valid):
                self._ensure_capacity(int(indices[valid].max()))
                self.data[indices[valid], :] = self.buff.data[rows[valid], :]

            indices = np.unique(indices[valid])
            # print("codewords with degree 1: {}".format(indices.size))
            # print("ripple size: {}".format(indices.size))
            # ripple_stats.append(indices.size)

            new_indices_mask = ~self.collected[indices]
            new_indices = indices[new_indices_mask]
            self.collected[new_indices] = True
            # decoded_stats.append(new_indices.size)
            # self.log.log_decoded_symbols(round, indices)

            if new_indices.size > 0:
                next_rows = update_buffer(
                    self.buff.row_ptr,
                    self.buff.src_idx,
                    self.buff.degree,
                    self.buff.data,
                    new_indices.astype(np.int32),
                    self.data,
                    col_ptr,
                    code_idx
                )
                if next_rows:
                    ripple_queue.extend(next_rows)

        num_of_solved = np.count_nonzero(self.collected)
        num_of_source = self.collected.shape[0]
        
        # ripple_sum = 0; decoded_sum = 0
        # rows = []

        # for i, (r, d) in enumerate(zip(ripple_stats, decoded_stats), start=1):
        #     ripple_sum += r; decoded_sum += d
        #     rows.append([i, ripple_sum, decoded_sum, 1 - ripple_sum / num_of_source, 1 - decoded_sum / self.buff.data.size])
        # now = datetime.now()
        # timestamp = now.strftime("%Y%m%d%H%M%S")
        # self.log._write_to_csv(f"./experiments/decoding_stats_{timestamp}.csv", rows)

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
