import numpy as np
from random import random
from abc import *
from dataclasses import dataclass
from typing import *
from tools import *
from symbols import *
from distributions import ideal_distribution, robust_distribution
from numpy.random import Generator
from collections import deque
from joblib import Parallel, delayed

class Encoder(ABC):
    @abstractmethod
    def get_one(self) -> Codeword:
        """
        get codeword from encoder
        """
        pass

    @abstractmethod
    def get_bat(self, batch: int) -> CodewordBatch:
        """
        get codeword from encoder
        """
        pass

    @abstractmethod
    def get_all(self) -> CodewordBatch:
        """
        get all codeword from encoder
        """
        pass

    @abstractmethod
    def put_one(self, data: np.ndarray):
        """
        put input into encoder
        """
        pass

    @abstractmethod
    def put_bat(self, data: np.ndarray):
        """
        put input into encoder
        """
        pass