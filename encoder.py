from core import *
from distributions import *
import numpy as np
from numpy.random import Generator
from collections import deque

def get_degrees_from(distribution_name, N, k):
    """ Returns the random degrees from a given distribution of probabilities.
    The degrees distribution must look like a Poisson distribution and the 
    degree of the first drop is 1 to ensure the start of decoding.
    """

    if distribution_name == "ideal":
        probabilities = ideal_distribution(N)
    elif distribution_name == "robust":
        probabilities = robust_distribution(N)
    else:
        probabilities = None
    
    population = list(range(0, N+1))
    return [1] + choices(population, probabilities, k=k-1)
   
def encode(blocks, redundancy, codetype):
    """ Iterative encoding - Encodes new symbols and yield them.
    Encoding one symbol is described as follow:

    1.  Randomly choose a degree according to the degree distribution, save it into "deg"
        Note: below we prefer to randomly choose all the degrees at once for our symbols.

    2.  Choose uniformly at random 'deg' distinct input blocs. 
        These blocs are also called "neighbors" in graph theory.
    
    3.  Compute the output symbol as the combination of the neighbors.
        In other means, we XOR the chosen blocs to produce the symbol.
    """

    # Display statistics
    blocks_n = len(blocks)
    drops_quantity = int(blocks_n * redundancy)
    assert blocks_n <= drops_quantity, "Because of the unicity in the random neighbors, it is need to drop at least the same amount of blocks"

    print(f"WINDOWSIZE = {config['WINDOWSIZE']}")
    print("Generating graph...")
    start_time = time.time()

    print("Ready for encoding.", flush=True)

    if codetype == "LT":
        # Generate random indexes associated to random degrees, seeded with the symbol id
        random_degrees = get_degrees_from("robust", blocks_n, k=drops_quantity)

        for i in range(drops_quantity):
            
            # Get the random selection, generated precedently (for performance)
            selection_indexes, deg = generate_indexes(i, random_degrees[i], 0, blocks_n)
            # Xor each selected array within each other gives the drop (or just take one block if there is only one selected)
            drop = blocks[selection_indexes[0]]
            for n in range(1, deg):
                drop = np.bitwise_xor(drop, blocks[selection_indexes[n]])
                # drop = drop ^ blocks[selection_indexes[n]] # according to my tests, this has the same performance

            # Create symbol, then log the process
            symbol = Symbol(index=i, degree=deg, data=drop, neighbors=selection_indexes)
            
            if config["VERBOSE"]:
                symbol.log(blocks_n)

            log("Encoding", i, drops_quantity, start_time)

            yield symbol

    elif codetype == "PLOW":

        # Preparation before start encoding
        encode_range = int(config["WINDOWSIZE"] * redundancy)
        symbols = [Symbol(index=i) for i in range(config["WINDOWSIZE"])]
        # symbols = deque(); symbols.append(Symbol(index=0))
        sent_symbol_count = 0

        for i in range(blocks_n):
            min_degree = max_degree = config["MAX_DEGREE"]
            deg = random.randint(min_degree, max_degree) # same degree for current setting
            
            # Option1: Determinist selection of indexes
            # selection_indexes = [int(i*redundancy)] + [int(i*redundancy + encode_range*(k-1)/k)-1 for k in range(2,deg+1)]
            
            # NEW: select deg from [k1, k2]
            # if random.random() >= 0.86752: deg = 12 # avg_deg = 4.08
            # if random.random() >= 0.88743: deg = 21 # avg_deg = 5.03
            # deg = 12 if random.random() >= 0.87 else 4  # avg_deg = 5.04
            # deg = 21 if random.random() >= 0.88235 else 4  # avg_deg = 6
            
            # Option2: Random generation
            selection_indexes = generate_indexes_plow(i, redundancy, deg)

            # if config["VERBOSE"]: print(i, selection_indexes)

            # XOR the input block to each selected encoded symbol
            for idx in selection_indexes:
                
                while idx >= len(symbols): 
                    symbol = Symbol(index=len(symbols))
                    symbols.append(symbol)
                # while idx > symbols[-1].index: 
                #     symbol = Symbol(index=symbols[-1].index+1)
                #     symbols.append(symbol)

                # Update data for selected encoded symbol
                # idx = (idx - sent_symbol_count) % encode_range
                if symbols[idx].degree == 0: symbols[idx].data = blocks[i]
                else: symbols[idx].data = np.bitwise_xor(symbols[idx].data, blocks[i])

                # Update symbol attributes
                if i in symbols[idx].neighbors: # collision
                    symbols[idx].degree -= 1
                    symbols[idx].neighbors.remove(i)
                else:
                    symbols[idx].degree += 1
                    symbols[idx].neighbors.add(i)
                assert(symbols[idx].degree == len(symbols[idx].neighbors))

            # print(len(symbols), sent_symbol_count)
            # FIXME: Remove the symbols in the front and send to decoder
            # while len(symbols) > encode_range:
            #     symbol = symbols.popleft()
            #     sent_symbol_count += 1
            #     yield symbol
            
            log("Encoding", i, blocks_n, start_time)
        
        # TODO: inserting 1-degree symbols periodicly: low efficiency for now
        # upper_bound = int(blocks_n * redundancy)
        # insert_counter = 0  # Track iterations for periodic insertion
        # insert_point = interval = 1000
        # rng = np.random.default_rng()
        # while i < insert_point and insert_counter > 0:
        #     insert_point = rng.poisson(interval*insert_counter) 
        #     while insert_point > upper_bound: insert_point = rng.poisson(interval)  
        
        #     batch_size = int((redundancy - 1)*insert_point)
        
        #     for id in range(batch_size):
        #         symbol = Symbol(index=len(symbols), data=blocks[insert_point+id])
        #         symbols.append(symbol)

        #     if config["VERBOSE"]:
        #         print(f"Inserted {batch_size} degree-1 symbols")
            
        # insert_counter += 1 
            
        for symbol in symbols: yield symbol

    elif codetype == "WALZER":
        deg = config["MAX_DEGREE"]
        encode_range = int(config["WINDOWSIZE"]*redundancy)
        symbols = []
        
        for i in range(blocks_n):
            
            end_index = int(i*redundancy) + encode_range
            selection_indexes, _ = generate_indexes(i, deg, 0, encode_range)
            selection_indexes.sort()

            # Stretch the randomly and uniformly selected indexes to 
            # make the last index equal to the encode range
            scaling =  encode_range / selection_indexes[-1]
            selection_indexes = np.array(selection_indexes)
            selection_indexes_stretched = (end_index - (selection_indexes * scaling)).tolist()
            selection_indexes_stretched.pop()
            selection_indexes_stretched.append(math.ceil(i * redundancy)) # add 1st determinist connection

            if config["VERBOSE"]: print(i, [int(x) for x in selection_indexes_stretched])

            for idx in selection_indexes_stretched[:-1] + [int(i*redundancy)]:
                
                idx = int(idx)
                while idx >= len(symbols): 
                    symbol = Symbol(index=len(symbols))
                    symbols.append(symbol)
                
                # Update data for selected encoded symbol
                if symbols[idx].degree == 0: symbols[idx].data = blocks[i]
                else: symbols[idx].data = np.bitwise_xor(symbols[idx].data, blocks[i])

                # Update symbol attributes
                if i in symbols[idx].neighbors: # collision
                    symbols[idx].degree -= 1
                    symbols[idx].neighbors.remove(i)
                else:
                    symbols[idx].degree += 1
                    symbols[idx].neighbors.add(i)
                assert(symbols[idx].degree == len(symbols[idx].neighbors))
            
            log("Encoding", i, blocks_n, start_time)

        for symbol in symbols: yield symbol

    print("\n----- Correctly dropped {} symbols (packet size={})".format(drops_quantity, PACKET_SIZE))
