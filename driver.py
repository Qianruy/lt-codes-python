import json
import os
import numpy as np
import random
import time
import math
from encoder import *
from decoder import *
from distributions import ideal_distribution, robust_distribution

# Load configuration from JSON
CONFIG_FILE_PATH = "experiments/config_example.json"
NUM_OF_TRIALS = 1

def load_config(config_path):
    """ Load JSON configuration file. """
    with open(config_path, "r") as f:
        return json.load(f)

def run_experiment(config):
    """ Runs a single encoding-decoding experiment based on config parameters. """
    
    begin = time.time()
    
    N = config["number_of_source"]  # Number of source symbols
    redundancy = config["redundancy"]
    windowsize = config["windowsize"]
    numofdegree = config["numofdegree"]
    encodertype = config["encodertype"]
    decodertype = config["decodertype"]
    lossrate = config["lossrate"]
    if "seed" in config: seed = config["seed"]
    overlap = config["overlap"] if "overlap" in config else 0.5
    
    print(f"Running experiment: {encodertype} | Redundancy: {redundancy}")

    # Initialize encoder
    if encodertype == "PLOW":
        print(f"Degree: {numofdegree} | Loss: {lossrate} | WinSize: {windowsize}")
        encoder = PlowEncoder(1, wdn_size=windowsize, redundancy=redundancy, maxdegree=numofdegree)
    elif encodertype == "WALZER":
        print(f"Degree: {numofdegree} | Loss: {lossrate} | WinSize: {windowsize}")
        encoder = WalzerEncoder(1, wdn_size=windowsize, redundancy=redundancy, maxdegree=numofdegree)
    elif encodertype == "LT":
        print(f"WinSize: {N}")
        encoder = LubyEncoder(np.array(robust_distribution(N-1)), 1, redundancy=redundancy, seed=10000)
    elif encodertype == "SF":
        print(f"WinSize: {windowsize} | Overlap: {overlap} ")
        encoder = SlidingFountainEncoder(np.array(robust_distribution(windowsize-1)), 1, wdn_size=windowsize, overlap=overlap)
    elif encodertype == "NOSC":
        print(f"WinSize: {N} | Loss: {lossrate}")
        encoder = NoscEncoder(1, redundancy=redundancy, maxdegree=numofdegree, seed=seed)
    else:
        raise ValueError(f"Unsupported code type: {encodertype}")

    encoder.put_bat(np.ones((N, 1), dtype=np.uint8))
    print("Pass codewords to the decoder.")

    # Initialize decoder and insert encoded data into the decoder
    if encodertype in ["PLOW", "WALZER", "NOSC"]: degree = numofdegree*5
    elif  encodertype in ["LT", "SF"]: degree = windowsize
    if decodertype == "iterative":
        decoder = IterativeDecoder(degree, 1, lossrate=lossrate)
    elif decodertype == "fcfp":
        decoder = FIFODecoder(degree, 1, lossrate=lossrate) 
    else:
        raise ValueError(f"Unsupported code type: {encodertype}")
    decoder.put_bat(encoder.get_all())
    print("Decoding process starts...")
    
    # Check if decoding was successful
    # decoded_success = (decoder.get_all() == np.ones((N, 1024), dtype=np.uint8)).all()
    decoded_success = (decoder.get_all() >= N * 0.99)
    
    end = time.time()
    
    print(f"Experiment result: {'SUCCESS' if decoded_success else 'FAILED'}")
    print(f"Runtime of the program: {end - begin:.4f} seconds\n")

    return decoded_success

def binary_search(config, start, end):
    """ Performs a binary search on redundancy to find an optimal setting. """
    
    while (end - start) >= 0.01:
        r = (start + end) / 2.0
        config["redundancy"] = round(r, 4)
        
        res = 0
        seeds = random.sample(range(1, 1000000), NUM_OF_TRIALS)
        for i in range(NUM_OF_TRIALS):
            config['seed'] = seeds[i]
            print("seeds: ", seeds[i])
            success = run_experiment(config)
            if success: res += 1

        success = (res >= NUM_OF_TRIALS * 0.9)
        if success:
            end = r  # Try a lower redundancy
        else:
            start = r  # Increase redundancy
        
    print(f"Optimized redundancy: {round((start + end) / 2.0, 4)}")

if __name__ == "__main__":
    config = load_config(CONFIG_FILE_PATH)

    # Run experiments for each combination of parameters
    random.seed(42)
    seeds = random.sample(range(1, 1000000), 100)
    print("seeds: ", seeds)
    for t in config["encodertype"]:
        for deg in config["numofdegree"]:
            for wsize in config["windowsize"]:
                for loss in config["lossrate"]:
                    for seed in seeds:
                        cur_config = config.copy()
                        cur_config.update({
                            "encodertype": t,
                            "numofdegree": deg,
                            "windowsize": wsize,
                            "lossrate": loss,
                            "seed": seed
                        })
                        
                        if config.get("binary_search", False):
                            start, end = 1.0, 2.5
                            binary_search(cur_config, start, end)
                        else:
                            run_experiment(cur_config)
