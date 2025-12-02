import numpy as np
import string
import argparse
import random
from abc import *
from dataclasses import dataclass
from typing import *

PACKET_SIZE = 128, # select from 65536, 32768, 16384, 4096, 1024, 512, 128
ROBUST_FAILURE_PROBABILITY = 0.01
NUMPY_TYPE = np.uint16, # select from np.uint64, np.uint32, np.uint16, np.uint8
EPSILON = 0.0001

# for alignement, index=0 corresponds to no input. 
# actual packet indices start from 1. 

@dataclass
class Codeword:
    data  : np.ndarray
    index : np.ndarray
    degree: int
    seqno : int = -1

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.uint8)
        self.index = np.asarray(self.index, dtype=np.int32)
        self.degree = int(self.degree)
        self.seqno = int(self.seqno)

@dataclass
class CodewordBatch:
    data  : np.ndarray           # shape (M, block), codeword payloads
    row_ptr : np.ndarray         # shape (M+1,), CSR row pointers
    src_idx : np.ndarray         # shape (nnz,), flattened source indices
    degree: np.ndarray           # shape (M,), current degrees per 
    seqno : np.ndarray           # shape (M,)
    used: int = 0

    def __post_init__(self):
        self.seqno = np.asarray(self.seqno, dtype=np.int32)
        self.row_ptr = np.asarray(self.row_ptr, dtype=np.int32)
        self.src_idx = np.asarray(self.src_idx, dtype=np.int32)
        self.data = np.asarray(self.data, dtype=np.uint8)
        self.degree = np.asarray(self.degree, dtype=np.int32)
        if self.used == 0:
            self.used = self.seqno.size

    @classmethod
    def empty(cls, block: int) -> 'CodewordBatch':
        return cls(
            seqno=np.zeros((0,), dtype=np.int32),
            row_ptr=np.zeros((1,), dtype=np.int32),
            src_idx=np.zeros((0,), dtype=np.int32),
            data=np.zeros((0, block), dtype=np.uint8),
            degree=np.zeros((0,), dtype=np.int32),
            used=0
        )

    @classmethod
    def from_dense(cls,
                   seqno: np.ndarray,
                   dense_idx: np.ndarray,
                   degree: np.ndarray,
                   data: np.ndarray) -> 'CodewordBatch':
        seqno = np.asarray(seqno, dtype=np.int32)
        dense_idx = np.asarray(dense_idx, dtype=np.int32)
        degree = np.asarray(degree, dtype=np.int32)
        data = np.asarray(data, dtype=np.uint8)
        assert dense_idx.shape[0] == seqno.shape[0] == degree.shape[0] == data.shape[0]
        row_ptr = np.zeros((degree.size + 1,), dtype=np.int32)
        row_ptr[1:] = np.cumsum(degree, dtype=np.int32)
        nnz = int(row_ptr[-1])
        src_idx = np.empty((nnz,), dtype=np.int32)
        pos = 0
        for i in range(degree.size):
            d = int(degree[i])
            src_idx[pos:pos+d] = dense_idx[i, :d]
            pos += d
        return cls(seqno=seqno, row_ptr=row_ptr, src_idx=src_idx, data=data, degree=degree, used=degree.size)

    def copy(self) -> 'CodewordBatch':
        return CodewordBatch(
            seqno=self.seqno.copy(),
            row_ptr=self.row_ptr.copy(),
            src_idx=self.src_idx.copy(),
            data=self.data.copy(),
            degree=self.degree.copy(),
            used=self.used
        )

    @property
    def num_codewords(self) -> int:
        return self.degree.size

    @property
    def nnz(self) -> int:
        return self.src_idx.size

    @property
    def max_index(self) -> int:
        if self.src_idx.size == 0:
            return 0
        valid = self.src_idx >= 0
        if not np.any(valid):
            return 0
        return int(self.src_idx[valid].max())

    def _append_row(self, seqno: int, indices: np.ndarray, payload: np.ndarray):
        indices = np.asarray(indices, dtype=np.int32)
        payload = np.asarray(payload, dtype=np.uint8).reshape(1, -1)
        degree = indices.size
        self.seqno = np.concatenate((self.seqno, np.array([seqno], dtype=np.int32)))
        self.row_ptr = np.concatenate((self.row_ptr, np.array([self.row_ptr[-1] + degree], dtype=np.int32)))
        self.src_idx = np.concatenate((self.src_idx, indices))
        self.data = np.concatenate((self.data, payload), axis=0)
        self.degree = np.concatenate((self.degree, np.array([degree], dtype=np.int32)))
        self.used = self.seqno.size

    def add(self, code: Codeword):
        self._append_row(code.seqno, code.index[:code.degree], code.data)

    def join(self, other: 'CodewordBatch'):
        if other.num_codewords == 0:
            return
        if self.num_codewords == 0:
            self.seqno = other.seqno.copy()
            self.row_ptr = other.row_ptr.copy()
            self.src_idx = other.src_idx.copy()
            self.data = other.data.copy()
            self.degree = other.degree.copy()
            self.used = other.used
            return
        offset = int(self.row_ptr[-1])
        self.seqno = np.concatenate((self.seqno, other.seqno))
        tail_ptr = other.row_ptr[1:] + offset
        self.row_ptr = np.concatenate((self.row_ptr, tail_ptr))
        self.src_idx = np.concatenate((self.src_idx, other.src_idx))
        self.data = np.concatenate((self.data, other.data), axis=0)
        self.degree = np.concatenate((self.degree, other.degree))
        self.used = self.seqno.size

    def _filter_rows(self, keep_mask: np.ndarray):
        keep_mask = np.asarray(keep_mask, dtype=bool)
        idx = np.nonzero(keep_mask)[0]
        new_seqno = self.seqno[keep_mask]
        new_data = self.data[keep_mask]
        new_row_ptr = np.zeros((idx.size + 1,), dtype=np.int32)
        new_degree = np.zeros((idx.size,), dtype=np.int32)
        kept_entries = []
        for out_pos, row in enumerate(idx):
            start, end = self.row_ptr[row], self.row_ptr[row + 1]
            row_entries = self.src_idx[start:end]
            valid = row_entries[row_entries >= 0]
            kept_entries.extend(valid.tolist())
            new_row_ptr[out_pos + 1] = len(kept_entries)
            new_degree[out_pos] = valid.size
        new_src_idx = np.asarray(kept_entries, dtype=np.int32)
        self.seqno = new_seqno
        self.data = new_data
        self.row_ptr = new_row_ptr
        self.src_idx = new_src_idx
        self.degree = new_degree
        self.used = self.seqno.size

    def drop_rows(self, drop_mask: np.ndarray):
        if self.num_codewords == 0:
            return
        drop_mask = np.asarray(drop_mask, dtype=bool)
        if drop_mask.shape[0] != self.num_codewords:
            raise ValueError("drop mask size mismatch")
        if not np.any(drop_mask):
            return
        keep_mask = ~drop_mask
        self._filter_rows(keep_mask)

    def degree_one_indices(self, mask: np.ndarray, limit: int = None) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.dtype == bool:
            rows = np.nonzero(mask)[0]
        else:
            rows = mask.astype(np.int32)
        if limit is not None:
            rows = rows[rows < limit]
        out = np.full(rows.size, -1, dtype=np.int32)
        for i, row in enumerate(rows):
            start, end = self.row_ptr[row], self.row_ptr[row + 1]
            for pos in range(start, end):
                src = self.src_idx[pos]
                if src >= 0:
                    out[i] = src
                    break
        return out

    def slice_prefix(self, n: int) -> 'CodewordBatch':
        n = min(n, self.num_codewords)
        row_ptr = self.row_ptr[:n + 1].copy()
        src_idx = self.src_idx[:row_ptr[-1]].copy()
        return CodewordBatch(
            seqno=self.seqno[:n].copy(),
            row_ptr=row_ptr,
            src_idx=src_idx,
            data=self.data[:n].copy(),
            degree=self.degree[:n].copy(),
            used=n
        )

class RingBuff:
    def __init__(self, size: int, block: int):
        self.data = np.zeros((size, block), dtype=np.int8)
        self.tail = 0
        self.head = 0
    def push(self, data: np.ndarray):
        assert self.tail - self.head < self.data.shape[0]
        self.data[self.tail % self.data.shape[0]] = data
        self.tail += 1
    def pop_one(self) -> Tuple[int, np.array]:
        assert self.tail > self.head
        head = self.head
        data = self.data[head % self.data.shape[0]]
        self.head += 1
        return head, data
    def pop_bat(self, batch: int) -> Tuple[int, np.array]:
        assert self.tail >= self.head + batch
        assert batch >= 1
        assert batch <= self.data.shape[0]
        head = self.head
        s = (self.head) % self.data.shape[0]
        e = (self.head + batch) % self.data.shape[0]
        self.head += batch
        data = self.data[s:e] if s < e else np.concatenate([self.data[s:], self.data[:e]], axis=-1)
        return head, data

if __name__ == '__main__':
    pass
