from .base_decoder import *
from numba import njit, prange
from logging import *
import csv
from datetime import datetime
from collections import deque
import os

class FIFODecoder(Decoder):
    """
    First Come First Peel Decoder
    @field(buff): the buffer of received codewords
    @field(data): decoded data
    @field(collected): 
    """
    def __init__(self, d: int, block: int, seed = 42, wdn = 200, rdn=1, loss=0):
        """
        @param(d): maximum degree
        @param(block): the block size
        """
        self.buff = CodewordBatch.empty(block)
        self.data = np.zeros((1, block), dtype=np.uint8)
        self.collected = np.zeros((1, ), dtype=np.bool_)
        self.collected[0] = True
        self.block = block
        self.seed = seed
        self.wdn = wdn; self.rdn = rdn; self.loss = loss
        self.ripple_queue = set() # store degree-one candidates
        self._col_ptr = np.zeros(1, dtype=np.int32)
        self._code_idx = np.zeros(0, dtype=np.int32)
        self._csc_dirty = True
        self._cut_limit = 0
    
    def put_one(self, code: Codeword):
        self.buff.add(code)
        max_idx = int(code.index.max()) if code.index.size > 0 else 0
        self._ensure_capacity(max_idx)
        last_row = self.buff.num_codewords - 1
        if last_row >= 0 and self.buff.degree[last_row] == 1:
            self.ripple_queue.add(last_row)
        self._csc_dirty = True

    def put_bat(self, code: CodewordBatch):
        prev_rows = self.buff.num_codewords
        self.buff.join(code)
        self._ensure_capacity(self.buff.max_index)
        total_rows = self.buff.num_codewords
        for row in range(prev_rows, total_rows):
            if self.buff.degree[row] == 1:
                self.ripple_queue.add(row)
        self._csc_dirty = True

    def get_one(self) -> Optional[bytes]: 
        raise NotImplementedError("Only support full decoding")    

    def _ensure_capacity(self, max_index: int):
        target = max(max_index + 1, 1)
        if target > self.data.shape[0]:
            pad = target - self.data.shape[0]
            self.data = np.concatenate((self.data, np.zeros((pad, self.block), dtype=np.uint8)), axis=0)
            self.collected = np.concatenate((self.collected, np.zeros((pad,), dtype=np.bool_)), axis=0)
            self._csc_dirty = True

    def _refresh_backlinks(self):
        if not self._csc_dirty:
            return
        N = max(self.data.shape[0], self.buff.max_index + 1)
        self._col_ptr, self._code_idx = csr2csc(
            self.buff.num_codewords,
            N,
            self.buff.row_ptr,
            self.buff.src_idx
        )
        self._csc_dirty = False
        
    # def get_all(self) -> Optional[int]:
    #     for cut in range(1, self.buff.data.shape[0] + 1):
    #         # pack to CSR
    #         row_ptr, src_idx_flat = batch_to_csr(self.buff.index[:cut+1], self.buff.degree[:cut+1])
    #         M = row_ptr.size - 1
    #         N = 0 if src_idx_flat.size == 0 else int(src_idx_flat.max()) + 1
    #         N = max(N, self.data.shape[0])
    #         # ensure outputs sized
    #         if self.data.shape[0] < N:
    #             self.data = np.resize(self.data, (N, self.block))
    #             self.collected = np.resize(self.collected, (N,))

    #         # build CSC
    #         # print("Construct fast backlinks...")
    #         col_ptr, code_idx_flat = csr2csc(M, N, row_ptr, src_idx_flat)

    #         # work copies
    #         cw_data = self.buff.data[:M, :].copy().astype(np.uint8)
    #         src_known = np.full(N, -1, np.int8)
    #         src_known[self.collected.astype(np.bool_)] = 1

    #         # print("Peeling begins...")
    #         solved, degree_left = peeling(
    #             row_ptr, src_idx_flat, col_ptr, code_idx_flat,
    #             cw_data, self.data, src_known
    #         )

    #     self.collected = (src_known == 1)
    #     num_of_solved = int(self.collected.sum())
    #     num_of_source = self.collected.size

    #     if not self.collected.all():
    #         # print indices still unknown
    #         remaining = np.where(~self.collected)[0]
    #         print(f"Solved symbols: {num_of_solved}/{num_of_source}")
    #     return num_of_solved

    def get_all(self) -> Optional[int]:
        # snapshots = []
        self._refresh_backlinks()
        for i in range(1, self.buff.num_codewords + 1):
        #     if i < self.buff.data.shape[0] - 4*600:  
        #         snapshots.append(self.buff.degree[i: i+3*600].copy())
            self.peel(i)
        # snapshots = np.asarray(snapshots) 
        # now = datetime.now()
        # timestamp = now.strftime("%Y%m%d%H%m")
        # filename = f'./experiments/plow_RDD_{timestamp}_{self.seed}.csv'
        # with open(filename, mode='a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['iter'] + list(range(snapshots.shape[1])))
        #     for snap_idx, row in enumerate(snapshots):
        #         writer.writerow([snap_idx] + row.tolist())
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
        deferred = []
        logfile = f'./experiments/fcfp_release_plow_{self.wdn}/plow_seqno_{self.seed}_{self.wdn}_{self.loss}_{self.rdn}.csv'

        # if self._csc_dirty: self._refresh_backlinks()
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        
        ripple = []
        if self.ripple_queue:
            if cut-1 in self.ripple_queue and self.buff.degree[cut-1] == 1:
                self.ripple_queue.remove(cut-1)
                ripple.append(cut-1)

        while ripple:
            rows = np.asarray(ripple, dtype=np.int32)
            ripple.clear()
            if rows.size == 0:
                continue

            indices = self.buff.degree_one_indices(rows)
            valid = indices >= 0
            if np.any(valid):
                self._ensure_capacity(int(indices[valid].max()))
                self.data[indices[valid], :] = self.buff.data[rows[valid], :]

            indices = np.unique(indices[valid])
            # indices_list = ",".join(str(int(idx)) for idx in indices)
            # with open(logfile, mode='a', newline='') as f:
            #     writer = csv.writer(f)
            #     seq_idx = min(max(row, 0), self.buff.seqno.size - 1)
            #     writer.writerow([self.buff.seqno[seq_idx], indices_list])

            new_indices_mask = ~self.collected[indices]
            new_indices = indices[new_indices_mask]
            self.collected[new_indices] = True

            if new_indices.size > 0:
                next_rows = update_buffer(
                    self.buff.row_ptr,
                    self.buff.src_idx,
                    self.buff.degree,
                    self.buff.data,
                    new_indices.astype(np.int32),
                    self.data,
                    self._col_ptr,
                    self._code_idx
                )
                for nr in next_rows:
                    if nr < cut:
                       ripple.append(nr)
                    else:
                        deferred.append(nr)

        for row in deferred:
            self.ripple_queue.add(row)
