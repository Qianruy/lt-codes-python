import numpy as np
from random import random
from abc import *
from dataclasses import dataclass
from typing import *
from tools import *
from collections import *

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
        pass

    @abstractmethod
    def get(self) -> Optional[bytes]:
        """
        get input from decoder
        """
        pass

class LTDecoder(Decoder):
    """
    Luby Transform Decoder
    @field(buff): the buffer of received codewords
    @field(data): decoded data
    """
    def __init__(self, d: int):
        """
        @param(d): 
        """
        pass
    
    def put_one(self, code: Codeword):
        pass

    def put_bat(self, code: CodewordBatch):
        pass
    
    def get_one(self) -> Optional[bytes]: 
        pass
    
    def get_all(self) -> Optional[bytes]: 
        pass

def recover_graph(symbols, blocks_quantity):
    """ Get back the same random indexes (or neighbors), thanks to the symbol id as seed.
    For an easy implementation purpose, we register the indexes as property of the Symbols objects.
    """

    for symbol in symbols:
        
        neighbors, deg = generate_indexes(symbol.index, symbol.degree, 0, blocks_quantity)
        symbol.neighbors = {x for x in neighbors}
        symbol.degree = deg

        if VERBOSE:
            symbol.log(blocks_quantity)

    return symbols

def reduce_neighbors(block_index, blocks, symbols):
    """ Loop over the remaining symbols to find for a common link between 
    each symbol and the last solved block `block`

    To avoid increasing complexity and another for loop, the neighbors are stored as dictionnary
    which enable to directly delete the entry after XORing back.
    """
    
    for other_symbol in symbols:
        if other_symbol.degree > 1 and block_index in other_symbol.neighbors:
        
            # XOR the data and remove the index from the neighbors
            other_symbol.data = np.bitwise_xor(blocks[block_index], other_symbol.data)
            other_symbol.neighbors.remove(block_index)

            other_symbol.degree -= 1
            
            if VERBOSE:
                print("XOR block_{} with symbol_{} :".format(block_index, other_symbol.index), list(other_symbol.neighbors.keys())) 


def decode(symbols, blocks_quantity):
    """ Iterative decoding - Decodes all the passed symbols to build back the data as blocks. 
    The function returns the data at the end of the process.
    
    1. Search for an output symbol of degree one
        (a) If such an output symbol y exists move to step 2.
        (b) If no output symbols of degree one exist, iterative decoding exits and decoding fails.
    
    2. Output symbol y has degree one. Thus, denoting its only neighbour as v, the
        value of v is recovered by setting v = y.

    3. Update.

    4. If all k input symbols have been recovered, decoding is successful and iterative
        decoding ends. Otherwise, go to step 1.
    """

    symbols_n = len(symbols)
    print(f"\n#symbols: {symbols_n}")
    assert symbols_n > 0, "There are no symbols to decode."

    # We keep `blocks_n` notation and create the empty list
    blocks_n = blocks_quantity
    blocks = [None] * blocks_n

    # Recover the degrees and associated neighbors using the seed (the index, cf. encoding).
    # symbols = recover_graph(symbols, blocks_n)
    # print("Graph built back. Ready for decoding.", flush=True)
    
    empty_symbol = 0
    solved_blocks_count = 0
    iteration_solved_count = 0
    start_time = time.time()
    
    while iteration_solved_count > 0 or solved_blocks_count == 0:
    
        iteration_solved_count = 0
        

        # Search for solvable symbols
        for i, symbol in enumerate(symbols):

            # Check the current degree. If it's 1 then we can recover data
            if symbol.degree == 0: continue
            if symbol.degree == 1: 

                iteration_solved_count += 1 
                block_index = next(iter(symbol.neighbors), None) 
                symbol.degree -= 1
                # symbols.pop(i)

                # This symbol is redundant: another already helped decoding the same block
                if block_index is None: 
                    empty_symbol += 1
                if block_index is None or blocks[block_index] is not None:
                    continue

                blocks[block_index] = symbol.data

                if VERBOSE:
                    print("Solved block_{} with symbol_{}".format(block_index, symbol.index))
              
                # Update the count and log the processing
                solved_blocks_count += 1
                log("Decoding", solved_blocks_count, blocks_n, start_time)

                # Reduce the degrees of other symbols that contains the solved block as neighbor             
                reduce_neighbors(block_index, blocks, symbols)
    # DEBUG 
    degrees = {}         
    for i, symbol in enumerate(symbols):
        if symbol.degree == 0: continue
        else: degrees[symbol.degree] = degrees.get(symbol.degree, 0) + 1
    degrees_sorted = OrderedDict(sorted(degrees.items()))

    print("\n----- Solved Blocks {:2}/{:2} ---".format(solved_blocks_count, blocks_n))
    print(f"----- Empty Symbol: {empty_symbol} ---")
    for deg, cnt in degrees_sorted.items():
        print(f"{cnt} symbols with degree {deg}")

    return np.asarray(blocks), solved_blocks_count
