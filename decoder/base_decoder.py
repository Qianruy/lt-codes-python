import numpy as np
from random import random
from abc import *
from dataclasses import dataclass
from typing import *
from tools import *
from symbols import *
from collections import *
from joblib import Parallel, delayed
from numba import njit, prange

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

@njit
def csr2csc(M, N, row_ptr, src_idx_flat):
    """
    Convert a codeword adjacency stored in CSR form into CSC.

    Parameters
    ----------
    M : int
        Number of codeword rows.
    N : int
        Number of source symbols (columns). This should cover the maximum
        source index contained in `src_idx_flat`.
    row_ptr : np.ndarray
        CSR row pointer array of length M+1 where row_ptr[i+1] - row_ptr[i]
        gives the number of neighbours (degree) for codeword i.
    src_idx_flat : np.ndarray
        Flattened list of neighbour source indices for each codeword row.
        Entries marked with -1 are treated as removed edges and ignored.

    Returns
    -------
    col_ptr : np.ndarray
        CSC column pointer array of length N+1. 
    code_idx_flat : np.ndarray
        Flattened list of codeword indices, grouped by source symbol.
    """
    nnz = src_idx_flat.size
    col_ptr = np.zeros(N+1, np.int32)
    code_idx_flat = np.empty(nnz, np.int32)
    # Count valid edges for each source.
    for e in range(nnz):
        s = src_idx_flat[e]
        if s >= 0:
            col_ptr[s+1] += 1
    # Prefix sum to build column offsets.
    for i in range(1, N+1):
        col_ptr[i] += col_ptr[i-1]
    # Fill the codeword indices.
    fill = col_ptr.copy()
    for m in range(M):
        start, end = row_ptr[m], row_ptr[m+1]
        for e in range(start, end):
            s = src_idx_flat[e]
            if s < 0: continue
            pos = fill[s]
            code_idx_flat[pos] = m
            fill[s] += 1
    return col_ptr, code_idx_flat

def update_buffer(row_ptr, src_idx_flat, buff_degree, buff_data,
                  new_indices, src_data, col_ptr, code_idx_flat):
    """
    Peel newly solved source symbols from all connected codewords.

    Parameters
    ----------
    row_ptr, src_idx_flat : CSR representation of the bipartite graph.
    buff_degree : array of current degrees per codeword (updated in place).
    buff_data : codeword payload blocks (updated in place).
    new_indices : np.ndarray of source indices resolved in this round.
    src_data : matrix of decoded source blocks.
    col_ptr, code_idx_flat : CSC view giving backlinks from source to codewords.

    Returns
    -------
    List[int]
        Codeword indices whose degree just dropped to 1 and should be
        re-queued for the next decoding ripple.
    """
    degree_one_rows = []
    seen_rows = set()
    for idx in new_indices:
        if idx < 0 or idx + 1 >= col_ptr.size:
            continue
        start, end = col_ptr[idx], col_ptr[idx + 1]
        for pos in range(start, end):
            row = code_idx_flat[pos]
            if row < 0 or row >= buff_degree.size:
                continue
            row_start, row_end = row_ptr[row], row_ptr[row + 1]
            for edge_pos in range(row_start, row_end):
                if src_idx_flat[edge_pos] == idx:
                    prev_deg = buff_degree[row]
                    if prev_deg <= 0:
                        break
                    buff_data[row, :] ^= src_data[idx, :]
                    src_idx_flat[edge_pos] = -1
                    new_deg = prev_deg - 1
                    buff_degree[row] = new_deg if new_deg > 0 else 0
                    if prev_deg > 1 and new_deg == 1 and row not in seen_rows:
                        degree_one_rows.append(row)
                        seen_rows.add(row)
                    break
    return degree_one_rows
