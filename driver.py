import json
import csv
import os
import numpy as np
import random
import time
import math
from encoder import *
from decoder import *
from channel import *
from tools import CodewordBatch
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
    channeltype = config["channeltype"]
    
    if "seed" in config: seed = config["seed"]
    mode = config["mode"] if "mode" in config else 1
    overlap = config["overlap"] if "overlap" in config else 0.5
    extra = config["extra"] if "extra" in config else 1

    tail_reduction = config["tail_reduction"] if "tail_reduction" in config else False
    
    print(f"Running experiment: {encodertype} | Redundancy: {redundancy}")

    # Initialize channel
    if channeltype == "BEC":
        lossrate = config["lossrate"]
        fixed = config["fixed"] if "fixed" in config else False
        print(f"Channel: {channeltype} | Loss: {lossrate} | Fixed: {fixed}")
        channel = BEC_Channel(lossrate, fixed)
    elif channeltype == "GE":
        alpha = config["alpha"]
        beta = config["beta"]
        eps = config["epsilon"]
        delta = config["delta"]
        print(f"Channel: {channeltype} | alpha: {alpha} | beta: {beta} | epsilon: {eps} | delta: {delta}")
        channel = GE_Channel(alpha, beta, eps, delta)
    else: 
        raise ValueError(f"Unsuppported channel type: {channeltype}")

    # Initialize encoder
    if encodertype == "PLOW":
        print(f"Degree: {numofdegree} | WinSize: {windowsize}")
        encoder = PlowEncoder(1, wdn_size=windowsize, redundancy=redundancy, maxdegree=numofdegree, seed=seed, tail_reduction = tail_reduction)
    elif encodertype == "WALZER":
        print(f"Degree: {numofdegree} | WinSize: {windowsize}")
        encoder = WalzerEncoder(1, wdn_size=windowsize, redundancy=redundancy, maxdegree=numofdegree, seed=seed, has_tail = tail_reduction)
    elif encodertype == "LT":
        print(f"WinSize: {N}")
        encoder = LubyEncoder(np.array(robust_distribution(N-1)), 1, redundancy=redundancy, seed=seed)
    elif encodertype == "SF":
        print(f"WinSize: {windowsize} | Overlap: {overlap} ")
        encoder = SlidingFountainEncoder(np.array(robust_distribution(windowsize-1)), 1, wdn_size=windowsize, overlap=overlap, seed=seed)
    elif encodertype == "NOSC":
        print(f"WinSize: {N}")
        encoder = NoscEncoder(1, redundancy=redundancy, maxdegree=numofdegree, seed=seed, mode=mode)
    elif encodertype == "MIX":
        print(f"Degree: {numofdegree} | WinSize: {windowsize} | LT encode range: {N} | Extra LT codewords percentage: {extra}")
        encoder_plow = PlowEncoder(1, wdn_size=windowsize, redundancy=redundancy, maxdegree=numofdegree)
        encoder_lt = LubyEncoder(np.array(robust_distribution(N-1)), 1, redundancy=extra, seed=seed)
    else:
        raise ValueError(f"Unsupported code type: {encodertype}")

    if encodertype == "MIX":
        encoder_plow.put_bat(np.ones((N, 1), dtype=np.uint8))
        encoder_lt.put_bat(np.ones((N, 1), dtype=np.uint8))
    else:
        encoder.put_bat(np.ones((N, 1), dtype=np.uint8))
    print("Pass codewords to the decoder.")

    # Initialize decoder and insert encoded data into the decoder
    if encodertype in ["PLOW", "WALZER", "NOSC"]: degree = numofdegree*5
    elif encodertype in ["LT", "SF", "MIX"]: degree = N
    if decodertype == "iterative":
        decoder = IterativeDecoder(degree, 1)
    elif decodertype == "fcfp":
        decoder = FIFODecoder(degree, 1, seed=seed, wdn=windowsize, rdn=redundancy, loss=lossrate) 
    else:
        raise ValueError(f"Unsupported code type: {decodertype}")
    if encodertype == "MIX":
        sent_bat = encoder_plow.get_all()
        sent_bat.join(encoder_lt.get_all())
    else: sent_bat = encoder.get_all()
    received_bat = channel.apply(sent_bat)
    decoder.put_bat(received_bat)
    print("Decoding process starts...")
    
    # Check if decoding was successful
    # decoded_success = (decoder.get_all() == np.ones((N, 1024), dtype=np.uint8)).all()
    decode_begin = time.time()
    num_of_solved = decoder.get_all()
    decoded_success = (num_of_solved >= N * 0.99 + 300)
    
    end = time.time()
    
    print(f"Experiment result: {'SUCCESS' if decoded_success else 'FAILED'}")
    print(f"Runtime of the program: {end - begin:.4f} seconds\n")
    print(f"Decoding time: {end - decode_begin:.4f} seconds\n")

    return decoded_success, num_of_solved

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
    rdm_seed = int(time.time() * 1000) + os.getpid()
    np.random.seed(rdm_seed % 1000000)
    seeds = random.sample(range(1, 1000000), NUM_OF_TRIALS)
    for t in config["encodertype"]:
        for deg in config["numofdegree"]:
            for wsize in config["windowsize"]:
                for loss in config["lossrate"]:
                    for oh in config["redundancy"]:
                        for seed in seeds:
                            cur_config = config.copy()
                            cur_config.update({
                                "encodertype": t,
                                "numofdegree": deg,
                                "windowsize": wsize,
                                "lossrate": loss,
                                "redundancy": oh,
                                "seed": seed
                            })
                            
                            if config.get("binary_search", False):
                                start, end = 1.0, 2.5
                                binary_search(cur_config, start, end)
                            else:
                                success, num_of_solved = run_experiment(cur_config)
                                channel = config["channeltype"]
                                decoder = config["decodertype"]
                                encoder = t + f'_mode{config["mode"]}' if t in ["WALZER", "NOSC"] else t
                                tail = "has_tail" if config["tail_reduction"] else "no_tail"
                                filename = f'experiments/{tail}_{encoder}_{decoder}_{deg}_{wsize}_{channel}'
                                if config["channeltype"] == "BEC":
                                    filename += f'_{loss}_{oh}.csv'
                                if config["channeltype"] == "GE":
                                    filename += f'_{config["beta"]}_{config["delta"]}_{oh}.csv'
                                with open(filename, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([success, num_of_solved, seed])



