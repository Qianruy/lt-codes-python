from core import *
from collections import *

def recover_graph(symbols, blocks_quantity):
    """ Get back the same random indexes (or neighbors), thanks to the symbol id as seed.
    For an easy implementation purpose, we register the indexes as property of the Symbols objects.
    """

    for symbol in symbols:
        
        neighbors, deg = generate_indexes(symbol.index, symbol.degree, 0, blocks_quantity)
        symbol.neighbors = {x for x in neighbors}
        symbol.degree = deg

        if config["VERBOSE"]:
            symbol.log(blocks_quantity)

    return symbols

def reduce_neighbors(block_index, blocks, symbols, redundancy, code_type):
    """ Loop over the remaining symbols to find for a common link between 
    each symbol and the last solved block `block`

    To avoid increasing complexity and another for loop, the neighbors are stored as dictionnary
    which enable to directly delete the entry after XORing back.
    """
    if code_type in ["PLOW", "WALZER"]:
        # start = max(0,len(symbols)-int(WINDOWSIZE*redundancy)-10)
        # end = min(len(symbols),start + int(WINDOWSIZE*redundancy)+10)
        start = max(0, block_index)
        end = min(len(symbols),int(block_index*redundancy) + int(config["WINDOWSIZE"]*redundancy))
    elif code_type == "LT":
        start, end = 0, len(symbols)

    for other_symbol in symbols[start:end]:
        if other_symbol.degree > 1 and block_index in other_symbol.neighbors:
        
            # XOR the data and remove the index from the neighbors
            other_symbol.data = np.bitwise_xor(blocks[block_index], other_symbol.data)
            other_symbol.neighbors.remove(block_index)

            other_symbol.degree -= 1
            
            if config["VERBOSE"]:
                print("XOR block_{} with symbol_{} :".format(block_index, other_symbol.index), list(other_symbol.neighbors)) 

def decode_incremental(symbols, recovered_blocks, solved_blocks_count, redundancy, code_type):
    """
    Incremental decoding function that takes in a symbol and updates the decoding state.
    
    Args:
        symbols: Current received symbols.
        recovered_blocks: Current list of recovered blocks.
        solved_blocks_count: Number of successfully recovered blocks.
    
    Returns:
        updated recovered_blocks and recovered_n.
    """
    symbols_n = len(symbols)
    print(f"\nsymbol index: {symbols_n-1}")
    assert symbols_n > 0, "There are no symbols to decode."

    iteration_solved_count = 0
    start = 1
    start_time = time.time()
    blocks_n = len(recovered_blocks)
    total_delay = 0

    # Pre-process the most recent symbol
    symbol = symbols[-1]
    for idx in list(symbol.neighbors):
        if recovered_blocks[idx] is not None:
            if config["VERBOSE"]: print("XOR block_{} with previous info".format(idx))
            symbol.data = np.bitwise_xor(recovered_blocks[idx], symbol.data)
            symbol.neighbors.remove(idx)
            symbol.degree -= 1

    while iteration_solved_count > 0 or start:

        iteration_solved_count = 0
        start = 0

        if symbols[-1].degree != 1: 
            if config["VERBOSE"]: print(symbols[-1].degree, symbols[-1].neighbors)
            break
        for i, symbol in enumerate(symbols):
            
            # if symbol.degree and VERBOSE: 
            #     print(symbol.index, symbol.degree)

            if symbol.degree == 0: continue
            if symbol.degree == 1:

                iteration_solved_count += 1
                block_index = next(iter(symbol.neighbors), None)
                symbol.degree -= 1

                if block_index is None or recovered_blocks[block_index] is not None:
                    continue

                recovered_blocks[block_index] = symbol.data
                
                print("Solved block_{} with symbol_{}".format(block_index, symbol.index))
                print("Delayed Timeframe: {}".format(symbols[-1].index-block_index))
                total_delay += symbols[-1].index-block_index

                # Update the count and log the processing
                solved_blocks_count += 1
                log("Decoding", solved_blocks_count, blocks_n, start_time)
        
                # Reduce the degrees of other symbols that contains the solved block as neighbor
                reduce_neighbors(block_index, recovered_blocks, symbols, redundancy, code_type) 

    return solved_blocks_count, total_delay

def decode(symbols, blocks_quantity, code_type):
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
    redundancy = len(symbols)/blocks_quantity

    # Recover the degrees and associated neighbors using the seed (the index, cf. encoding).
    # symbols = recover_graph(symbols, blocks_n)
    # print("Graph built back. Ready for decoding.", flush=True)
    
    empty_symbol = 0
    solved_blocks_count = 0
    iteration_solved_count = 0
    start_time = time.time()
    total_delay = 0
    
    while iteration_solved_count > 0 or solved_blocks_count == 0:
    
        iteration_solved_count = 0
        # Defined in LT process: the set of covered input symbols that have not yet been processed
        ripple = set()
        
        print("Iteration begins:")
        # Search for solvable symbols
        for i, symbol in enumerate(symbols):

            if symbol.degree: 
                print(symbol.index, symbol.degree)

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
                ripple.add(block_index)

                print("Solved block_{} with symbol_{}".format(block_index, symbol.index))
                print("Delayed Timeframe: {}".format(symbol.index/redundancy-block_index))
                total_delay += symbol.index/redundancy-block_index
              
                # Update the count and log the processing
                solved_blocks_count += 1
                log("Decoding", solved_blocks_count, blocks_n, start_time)

                # Reduce the degrees of other symbols that contains the solved block as neighbor
                reduce_neighbors(block_index, blocks, symbols, redundancy, code_type)

        print("Size of current ripple: {}".format(len(ripple)))                       

    # DEBUG 
    degrees = {}         
    for i, symbol in enumerate(symbols):
        if symbol.degree == 0: continue
        else: degrees[symbol.degree] = degrees.get(symbol.degree, 0) + 1
    degrees_sorted = OrderedDict(sorted(degrees.items()))

    print("\n----- Solved Blocks {:2}/{:2} ---".format(solved_blocks_count, blocks_n))
    print(f"----- Empty Symbol: {empty_symbol} ---")
    print(f"----- Avg Delayed Timeframe: {total_delay/solved_blocks_count} ---")
    for deg, cnt in degrees_sorted.items():
        print(f"{cnt} symbols with degree {deg}")

    return np.asarray(blocks), solved_blocks_count