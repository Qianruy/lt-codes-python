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

def batch_to_csr(indices: np.ndarray, degree: np.ndarray):
    """
    row_ptr: np.ndarray, shape (M+1,)
        Cumulative sum of degrees for each codeword
    src_idx_flat : np.ndarray, shape (nnz,)
        Flattened list of all source indices, concatenated row by row.
    """
    M = indices.shape[0]
    assert(np.all(degree >= 0))
    row_ptr = np.zeros(M+1, np.int32)
    row_ptr[1:] = np.cumsum(degree)
    nnz = int(row_ptr[-1])
    src_idx_flat = np.empty(nnz, np.int32)
    pos = 0
    for m in range(M):
        dm = int(degree[m]) # number of valid indices 
        vi = indices[m, :dm]
        # if vi.dtype.kind == 'i':
        #     dm = int(np.sum(vi >= 0)) # drop any -1 pads if present
        #     vi = vi[:dm]
        src_idx_flat[pos:pos+dm] = vi
        pos += dm
    return row_ptr, src_idx_flat

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

@njit(parallel=True, nogil=True)
def peeling(row_ptr, src_idx_flat, col_ptr, code_idx_flat,
            cw_data, src_data, src_known):

    M = row_ptr.size - 1
    B = cw_data.shape[1]

    # compute unresolved degrees
    degree = np.zeros(M, np.int32)
    for m in range(M):
        s, t = row_ptr[m], row_ptr[m+1]
        cnt = 0
        for e in range(s, t):
            v = src_idx_flat[e]
            if src_known[v] < 0:
                cnt += 1
        degree[m] = cnt

    # ring queue of degree-1 checks
    q = np.empty(max(1, 2*M), np.int64)
    qh = 0; qt = 0
    in_q = np.zeros(M, np.uint8)

    def qpush(x):
        nonlocal qt
        nxt = (qt + 1) % q.size
        if nxt == qh:  # full -> simple grow (rare)
            newq = np.empty(q.size*2, np.int64)
            # linearize
            k = 0
            i = qh
            while i != qt:
                newq[k] = q[i]
                k += 1
                i = (i + 1) % q.size
            q[:] = newq[:q.size]  # Numba needs shapes fixed
        if in_q[x] == 0:
            q[qt] = x
            qt = (qt + 1) % q.size
            in_q[x] = 1

    for m in range(M):
        if degree[m] == 1:
            qpush(m)

    solved_total = 0

    while qh != qt:
        m = q[qh]; qh = (qh + 1) % q.size
        in_q[m] = 0
        if degree[m] != 1:
            continue

        # find the lone unknown var in row m
        lone = -1
        s, t = row_ptr[m], row_ptr[m+1]
        for e in range(s, t):
            v = src_idx_flat[e]
            if src_known[v] < 0:
                lone = v
                break
        if lone == -1:
            continue

        # resolve: x[lone] = current check block
        for k in range(B):
            src_data[lone, k] = cw_data[m, k]
        src_known[lone] = 1
        solved_total += 1

        # peel from all checks containing this var
        s2, t2 = col_ptr[lone], col_ptr[lone+1]
        for ee in range(s2, t2):
            m2 = code_idx_flat[ee]
            if degree[m2] == 0:
                continue
            # XOR block
            for k in range(B):
                cw_data[m2, k] ^= src_data[lone, k]
            # dec degree
            dnew = degree[m2] - 1
            if dnew < 0: dnew = 0
            degree[m2] = dnew
            if dnew == 1 and in_q[m2] == 0:
                qpush(m2)

    return solved_total, degree

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
    block = buff_data.shape[1]
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
                    for k in range(block):
                        buff_data[row, k] ^= src_data[idx, k]
                    src_idx_flat[edge_pos] = -1
                    new_deg = prev_deg - 1
                    buff_degree[row] = new_deg if new_deg > 0 else 0
                    if prev_deg > 1 and new_deg == 1 and row not in seen_rows:
                        degree_one_rows.append(row)
                        seen_rows.add(row)
                    break
    return degree_one_rows
