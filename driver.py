import json
import csv
import os
import random
import time
from pathlib import Path
import numpy as np
from encoder import *
from decoder import *
from channel import *
from distributions import robust_distribution

# Load configuration from JSON
CONFIG_FILE_PATH = "experiments/config_example.json"

def ensure_list(value, default=None):
    """
    Normalise a value into a list, using default if value is None.
    """
    if value is None:
        return default if default is not None else []
    if isinstance(value, list):
        return value
    return [value]


def prepare_payload(experiment_cfg):
    """
    Load or generate the payload for the encoder input.
    Creates a dummy dataset if the configured file is missing.
    """
    symbol_size = int(experiment_cfg.get("symbol_size", 1))
    requested_sources = int(experiment_cfg.get("num_sources", 0))
    dataset_path = experiment_cfg.get("dataset")

    data_bytes = b""
    dataset_file = Path(dataset_path)
    total_bytes = requested_sources * symbol_size

    print(f"Generating {total_bytes} bytes of dummy data.")
    data_bytes = os.urandom(total_bytes)
    dataset_file.write_bytes(data_bytes)

    payload = np.frombuffer(data_bytes, dtype=np.uint8).reshape(requested_sources, symbol_size)
    print(f"Using dummy dataset: {dataset_file} ({requested_sources} symbols, symbol size {symbol_size})")
    return payload, requested_sources, symbol_size

def load_config(config_path):
    """ Load JSON configuration file. """
    with open(config_path, "r") as f:
        return json.load(f)

def run_experiment(experiment_cfg, encoder_cfg, decoder_cfg, channel_cfg, seed):
    """Run a single encoding-decoding experiment based on configuration parameters."""
    begin = time.time()

    payload, num_sources, symbol_size = prepare_payload(experiment_cfg)
    encoder_type = encoder_cfg["type"]
    decoder_type = decoder_cfg["type"]
    channel_type = channel_cfg["type"]

    redundancy = encoder_cfg["redundancy"]
    window_size = encoder_cfg.get("window_size")
    if window_size is None or (isinstance(window_size, (int, float)) and window_size <= 0):
        window_size = num_sources
    degree_setting = encoder_cfg.get("degree", 0)
    overlap = encoder_cfg.get("overlap", 0.5)
    extra_ratio = encoder_cfg.get("extra_lt_ratio", 1.0)
    mode = encoder_cfg.get("mode", 1)

    tail_reduction = experiment_cfg.get("tail_reduction", False)

    print(f"Running experiment: {encoder_type} | Decoder: {decoder_type} | Redundancy: {redundancy}")

    # Initialize channel
    if channel_type == "BEC":
        loss_rate = channel_cfg["loss_rate"]
        fixed = channel_cfg.get("fixed", False)
        print(f"Channel: {channel_type} | Loss: {loss_rate} | Fixed: {fixed}")
        channel = BEC_Channel(loss_rate, fixed)
    elif channel_type == "GE":
        alpha = channel_cfg["alpha"]
        beta = channel_cfg["beta"]
        eps = channel_cfg["epsilon"]
        delta = channel_cfg["delta"]
        print(f"Channel: {channel_type} | alpha: {alpha} | beta: {beta} | epsilon: {eps} | delta: {delta}")
        channel = GE_Channel(alpha, beta, eps, delta)
        loss_rate = alpha*delta/(alpha+beta) + beta*eps/(alpha+beta)
    else:
        raise ValueError(f"Unsupported channel type: {channel_type}")

    # Initialize encoder
    if encoder_type == "PLOW":
        print(f"Degree: {degree_setting} | WinSize: {window_size}")
        encoder = PlowEncoder(symbol_size, wdn_size=window_size, redundancy=redundancy, maxdegree=degree_setting, seed=seed, tail_reduction=tail_reduction)
    elif encoder_type == "WALZER":
        print(f"Degree: {degree_setting} | WinSize: {window_size}")
        encoder = WalzerEncoder(symbol_size, wdn_size=window_size, redundancy=redundancy, maxdegree=degree_setting, seed=seed, has_tail=tail_reduction)
    elif encoder_type == "LT":
        print(f"WinSize: {num_sources}")
        encoder = LubyEncoder(np.array(robust_distribution(num_sources - 1)), symbol_size, redundancy=redundancy, seed=seed)
    elif encoder_type == "SF":
        print(f"WinSize: {window_size} | Overlap: {overlap}")
        encoder = SlidingFountainEncoder(np.array(robust_distribution(window_size - 1)), symbol_size, wdn_size=window_size, overlap=overlap, seed=seed)
    elif encoder_type == "NOSC":
        print(f"WinSize: {num_sources}")
        encoder = NoscEncoder(symbol_size, redundancy=redundancy, maxdegree=degree_setting, seed=seed, mode=mode)
    elif encoder_type == "MIX":
        print(f"Degree: {degree_setting} | WinSize: {window_size} | LT encode range: {num_sources} | Extra LT codewords percentage: {extra_ratio}")
        encoder = None  # Placeholder, MIX uses two encoders
        encoder_plow = PlowEncoder(symbol_size, wdn_size=window_size, redundancy=redundancy, maxdegree=degree_setting, seed=seed, tail_reduction=tail_reduction)
        encoder_lt = LubyEncoder(np.array(robust_distribution(num_sources - 1)), symbol_size, redundancy=extra_ratio, seed=seed)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    if encoder_type == "MIX":
        encoder_plow.put_bat(payload)
        encoder_lt.put_bat(payload)
    else:
        encoder.put_bat(payload)

    # Initialize decoder and insert encoded data into the decoder
    if encoder_type in ["PLOW", "WALZER", "NOSC"]:
        decoder_degree = degree_setting * 5
    else:
        decoder_degree = num_sources

    if decoder_type == "iterative":
        decoder = IterativeDecoder(decoder_degree, symbol_size)
    elif decoder_type == "fcfp":
        decoder = FIFODecoder(decoder_degree, symbol_size, seed=seed, wdn=window_size, rdn=redundancy, loss=loss_rate)
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")

    print("Encoding process starts...")
    encode_begin = time.time()
    if encoder_type == "MIX":
        sent_batch = encoder_plow.get_all()
        sent_batch.join(encoder_lt.get_all())
    else:
        sent_batch = encoder.get_all()

    received_batch = channel.apply(sent_batch)
    decoder.put_bat(received_batch)

    print("Decoding process starts...")
    decode_begin = time.time()
    num_of_solved = decoder.get_all()
    

    end = time.time()
    total = num_sources * symbol_size / (1024 * 1024)
    encode_time, decode_time = decode_begin-encode_begin, end - decode_begin

    decoded_success = (num_of_solved >= num_sources * 0.99 + 300)
    print(f"Experiment result: {'SUCCESS' if decoded_success else 'FAILED'}")
    print(f"Runtime of the program: {end - begin:.4f} seconds\n")
    print(f"Encoding: {total/encode_time:.2f}MB/s ~{encode_time:.4f}s\n")
    print(f"Decoding: {total/decode_time:.2f}MB/s ~{decode_time:.4f}s\n")

    return decoded_success, num_of_solved

def binary_search(experiment_cfg, encoder_cfg, decoder_cfg, channel_cfg, num_trials):
    """Perform a binary search on redundancy to find an optimal setting."""
    start, end = 1.0, 2.5
    while (end - start) >= 0.01:
        trial_redundancy = round((start + end) / 2.0, 4)
        trial_encoder = dict(encoder_cfg)
        trial_encoder["redundancy"] = trial_redundancy

        successes = 0
        seeds = random.sample(range(1, 1_000_000), num_trials)
        for seed in seeds:
            print("seeds: ", seed)
            success, _ = run_experiment(experiment_cfg, trial_encoder, decoder_cfg, channel_cfg, seed)
            if success:
                successes += 1

        if successes >= num_trials * 0.9:
            end = trial_redundancy  # Try a lower redundancy
        else:
            start = trial_redundancy  # Increase redundancy

    optimized = round((start + end) / 2.0, 4)
    print(f"Optimized redundancy: {optimized}")

if __name__ == "__main__":
    config = load_config(CONFIG_FILE_PATH)

    experiment_cfg = config.setdefault("experiment", {})
    encoder_settings = config.get("encoder", {})
    decoder_cfg = config.get("decoder", {})
    channel_settings = config.get("channel", {})
    output_settings = config.get("output", {})

    results_dir = Path(output_settings.get("results_dir", "experiments/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    num_trials = max(1, int(experiment_cfg.get("num_trials", 1)))

    # Prepare sweep parameters
    encoder_types = ensure_list(encoder_settings.get("types"), default=["PLOW"])
    encoder_degrees = ensure_list(encoder_settings.get("degrees"), default=[4])
    default_window = experiment_cfg.get("num_sources")
    encoder_windows = ensure_list(encoder_settings.get("window_sizes"), default=[default_window])
    encoder_redundancies = ensure_list(encoder_settings.get("redundancies"), default=[1.0])

    channel_type = channel_settings.get("type", "BEC")
    if channel_type == "BEC":
        loss_rates = ensure_list(channel_settings.get("loss_rates"), default=[0.0])
    else:
        loss_rates = [None]

    # Global seed management for reproducibility
    rdm_seed = int(time.time() * 1000) + os.getpid()
    np.random.seed(rdm_seed % 1_000_000)
    seeds = random.sample(range(1, 1_000_000), num_trials)

    for enc_type in encoder_types:
        for degree in encoder_degrees:
            for window_size in encoder_windows:
                for redundancy in encoder_redundancies:
                    for loss_rate in loss_rates:
                        encoder_combo = {
                            "type": enc_type,
                            "degree": degree,
                            "window_size": window_size,
                            "redundancy": redundancy,
                            "overlap": encoder_settings.get("overlap", 0.5),
                            "extra_lt_ratio": encoder_settings.get("extra_lt_ratio", 1.0),
                        }

                        if channel_type == "BEC":
                            channel_combo = {
                                "type": "BEC",
                                "loss_rate": loss_rate,
                                "fixed": channel_settings.get("fixed", False),
                            }
                        elif channel_type == "GE":
                            channel_combo = {
                                "type": "GE",
                                "alpha": channel_settings["alpha"],
                                "beta": channel_settings["beta"],
                                "epsilon": channel_settings["epsilon"],
                                "delta": channel_settings["delta"],
                            }
                        else:
                            raise ValueError(f"Unsupported channel type: {channel_type}")

                        if experiment_cfg.get("binary_search", False):
                            binary_search(experiment_cfg, encoder_combo, decoder_cfg, channel_combo, num_trials)
                            continue

                        for seed in seeds:
                            success, num_of_solved = run_experiment(experiment_cfg, encoder_combo, decoder_cfg, channel_combo, seed)

                            encoder_label = f"{enc_type}_mode{experiment_cfg.get('mode', 1)}" if enc_type in ["WALZER", "NOSC"] else enc_type
                            tail_label = "has_tail" if experiment_cfg.get("tail_reduction", False) else "no_tail"
                            if window_size is None or (isinstance(window_size, (int, float)) and window_size <= 0):
                                window_component = "NA"
                            else:
                                window_component = window_size
                            filename_parts = [tail_label, encoder_label, decoder_cfg["type"], str(degree), str(window_component), channel_type]

                            if channel_type == "BEC":
                                filename_parts.extend([str(loss_rate), str(redundancy)])
                            elif channel_type == "GE":
                                filename_parts.extend([
                                    str(channel_settings["alpha"]),
                                    str(channel_settings["beta"]),
                                    str(channel_settings["epsilon"]),
                                    str(channel_settings["delta"]),
                                    str(redundancy),
                                ])

                            result_path = results_dir / f'{"_".join(filename_parts)}.csv'
                            with result_path.open("a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([success, num_of_solved, seed])
