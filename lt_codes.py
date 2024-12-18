#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import core
import stats
from experiments.rank import *
from encoder import encode
from decoder import decode, decode_incremental



def blocks_read(file, filesize):
    """ Read the given file by blocks of `core.PACKET_SIZE` and use np.frombuffer() improvement.

    Byt default, we store each octet into a np.uint8 array space, but it is also possible
    to store up to 8 octets together in a np.uint64 array space.  
    
    This process is not saving memory but it helps reduce dimensionnality, especially for the 
    XOR operation in the encoding. Example:
    * np.frombuffer(b'\x01\x02', dtype=np.uint8) => array([1, 2], dtype=uint8)
    * np.frombuffer(b'\x01\x02', dtype=np.uint16) => array([513], dtype=uint16)
    """

    blocks_n = math.ceil(filesize / core.PACKET_SIZE)
    blocks = []

    # Read data by blocks of size core.PACKET_SIZE
    for i in range(blocks_n):
            
        data = bytearray(file.read(core.PACKET_SIZE))

        if not data:
            raise "stop"

        # The last read bytes needs a right padding to be XORed in the future
        if len(data) != core.PACKET_SIZE:
            data = data + bytearray(core.PACKET_SIZE - len(data))
            assert i == blocks_n-1, "Packet #{} has a not handled size of {} bytes".format(i, len(blocks[i]))

        # Paquets are condensed in the right array type
        blocks.append(np.frombuffer(data, dtype=core.NUMPY_TYPE))

    return blocks

def blocks_write(blocks, file, filesize):
    """ Write the given blocks into a file
    """

    count = 0
    for data in recovered_blocks[:-1]:
        file_copy.write(data)
        count += len(data)

    # Convert back the bytearray to bytes and shrink back 
    last_bytes = bytes(recovered_blocks[-1])
    shrinked_data = last_bytes[:filesize % core.PACKET_SIZE]
    file_copy.write(shrinked_data)

#########################################################
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Robust implementation of LT Codes encoding/decoding process.")
    parser.add_argument("filename", help="file path of the file to split in blocks")
    parser.add_argument("-r", "--redundancy", help="the wanted redundancy.", default=2.0, type=float)
    parser.add_argument("--systematic", help="ensure that the k first drops are exactaly the k first blocks (systematic LT Codes)", action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--x86", help="avoid using np.uint64 for x86-32bits systems", action="store_true")
    parser.add_argument("-t", "--codetype", help="select wanted code type.", default="LT")
    parser.add_argument("-w", "--windowsize", help="set appropriate range for encoding.", default=500, type=int)
    parser.add_argument("-k", "--numofdegree", help="set number of encoded symbols connected to single input symbol", default=3, type=int)
    parser.add_argument("-l", "--lossrate", help="loss probability for encoded symbols", default=0.01, type=float)
    args = parser.parse_args()

    core.NUMPY_TYPE = np.uint32 if args.x86 else core.NUMPY_TYPE
    core.config["SYSTEMATIC"] = True if args.systematic else core.config["SYSTEMATIC"]
    core.config["VERBOSE"] = True if args.verbose else core.config["VERBOSE"]    
    core.config["WINDOWSIZE"] = args.windowsize 
    core.config["LOSS_PROBABILITY"] = args.lossrate  
    core.config["MAX_DEGREE"] = args.numofdegree 

    # Initialize stats
    results_stats = stats.Stats(args)

    with open(args.filename, "rb") as file:

        print("Redundancy: {}".format(args.redundancy))
        print("Code Type: {}".format(args.codetype))
        # print("Systematic: {}".format(core.config["SYSTEMATIC"]))
        print("Loss Rate: {}".format(core.config["LOSS_PROBABILITY"]))
        print("Max Degree: {}".format(core.config["MAX_DEGREE"]))

        filesize = os.path.getsize(args.filename)
        print("Filesize: {} bytes".format(filesize))

        # Splitting the file in blocks & compute drops
        file_blocks = blocks_read(file, filesize)
        file_blocks_n = len(file_blocks)
        drops_quantity = int(file_blocks_n * args.redundancy)

        print("Blocks: {}".format(file_blocks_n))
        print("Drops: {}\n".format(drops_quantity))

        # Generating symbols (or drops) from the blocks
        file_symbols = []
        recovered_blocks, recovered_n = [None] * file_blocks_n, 0
        total_delay = 0

        for i, curr_symbol in enumerate(encode(file_blocks, args.redundancy, args.codetype)):

            # # Calculate current rank
            # encode_range = int(core.config["WINDOWSIZE"]*args.redundancy)
            # if i == 0: A = sp.csr_matrix(([1]), dtype=int)
            # else: 
            #     A = add_encoded_symbol(A, curr_symbol, 2*encode_range)
            #     A = expand_and_trim(A, 2*encode_range)
            #     rank = sparse_matrix_rank(A)
            #     print("current rank: {}".format(rank))

            if random.random() >= core.config["LOSS_PROBABILITY"]: # Simulating the loss of packets
                file_symbols.append(curr_symbol)

            if args.codetype in ["PLOW", "WALZER"]:
                recovered_n, curr_delay = decode_incremental(file_symbols, recovered_blocks, recovered_n, 
                                                                               args.redundancy, args.codetype)
                total_delay += curr_delay

        if args.codetype == "LT":
            # Recovering the blocks from symbols
            recovered_blocks, recovered_n = decode(file_symbols, blocks_quantity=file_blocks_n, code_type=args.codetype)
        
        # if core.VERBOSE:
        #     print(recovered_blocks)
        #     print("------ Blocks :  \t-----------")
        #     print(file_blocks)

        print("\n----- Solved Blocks {:2}/{:2} ---".format(recovered_n, file_blocks_n))
        print(f"----- Avg Delayed Timeframe: {total_delay/recovered_n} ---")

        # Record results
        results_stats.add_result("solved percentage", round(recovered_n/file_blocks_n, 3))
        if file_blocks_n * 0.8 <= recovered_n:
            results_stats.add_result("success", True)

        if recovered_n != file_blocks_n:
            print("Blocks are not all recovered, we cannot proceed the file writing")
            exit()

        splitted = args.filename.split(".")
        if len(splitted) > 1:
            filename_copy = "".join(splitted[:-1]) + "-copy." + splitted[-1] 
        else:
            filename_copy = args.filename + "-copy"

        # Write down the recovered blocks in a copy 
        with open(filename_copy, "wb") as file_copy:
            blocks_write(recovered_blocks, file_copy, filesize)

        results_stats.save_to_json("./results/summary/out_w{}_{}.json".format(args.windowsize, results_stats.data["timestamp"]))
        print("Wrote {} bytes in {}".format(os.path.getsize(filename_copy), filename_copy))


