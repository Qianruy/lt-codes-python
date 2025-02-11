import json
import os
import numpy as np
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
    codetype = config["codetype"]
    lossrate = config["lossrate"]
    if "seed" in config: seed = config["seed"]
    overlap = config["overlap"] if "overlap" in config else 0.5
    
    print(f"Running experiment: {codetype} | Redundancy: {redundancy} | WinSize: {windowsize}")

    # Initialize encoder
    if codetype == "PLOW":
        print(f"Degree: {numofdegree} | Loss: {lossrate}")
        encoder = PlowEncoder(1024, wdn_size=windowsize, redundancy=redundancy, maxdegree=numofdegree)
    elif codetype == "WALZER":
        print(f"Degree: {numofdegree} | Loss: {lossrate}")
        encoder = WalzerEncoder(1024, wdn_size=windowsize, redundancy=redundancy, maxdegree=numofdegree)
    elif codetype == "LT":
        encoder = LubyEncoder(np.array(robust_distribution(N-1)), 1024, 10000)
    elif codetype == "SF":
        print(f"Overlap: {overlap} ")
        encoder = SlidingFountainEncoder(np.array(robust_distribution(windowsize-1)), 1024, wdn_size=windowsize, overlap=overlap)
    else:
        raise ValueError(f"Unsupported code type: {codetype}")

    encoder.put_bat(np.ones((N, 1024), dtype=np.uint8))
    print("Pass codewords to the decoder.")

    # Initialize decoder and insert encoded data into the decoder
    if codetype == "PLOW":
        decoder = IterativeDecoder(numofdegree*5, 1024, lossrate=lossrate)
        decoder.put_bat(encoder.get_all())
    elif codetype == "WALZER":
        decoder = IterativeDecoder(numofdegree*5, 1024, lossrate=lossrate)
        decoder.put_bat(encoder.get_all())
    elif codetype == "LT":
        decoder = IterativeDecoder(windowsize, 1024)
        decoder.put_bat(encoder.get_bat(int(N*redundancy)))
    elif codetype == "SF":
        decoder = IterativeDecoder(windowsize, 1024)
        shift = int((1-overlap) * windowsize)
        for i in range((N-windowsize)//shift + 1):
            decoder.put_bat(encoder.get_bat(int(windowsize*redundancy)))
            encoder.shift_window()
            encoder.remove_bat(shift)
    else:
        raise ValueError(f"Unsupported code type: {codetype}")
    
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
        for i in range(NUM_OF_TRIALS):
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
    for t in config["codetype"]:
        for deg in config["numofdegree"]:
            for wsize in config["windowsize"]:
                for loss in config["lossrate"]:
                    cur_config = config.copy()
                    cur_config.update({
                        "codetype": t,
                        "numofdegree": deg,
                        "windowsize": wsize,
                        "lossrate": loss
                    })
                    
                    if config.get("binary_search", False):
                        start, end = 1.0, 2.5
                        binary_search(cur_config, start, end)
                    else:
                        run_experiment(cur_config)
