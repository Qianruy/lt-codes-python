#!/usr/bin/env python3
# /// script
# dependencies = ["numpy"]
# ///
import argparse
import json
import numpy as np

from channel.ge_channel import GE_Channel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation for the Gilbert-Elliott channel model."
    )
    parser.add_argument("--alpha", type=float, required=True, help="Transition rate from good to bad state.")
    parser.add_argument("--beta", type=float, required=True, help="Transition rate from bad to good state.")
    parser.add_argument("--epsilon", type=float, required=True, help="Isolated loss probability in the good state.")
    parser.add_argument("--delta", type=float, required=True, help="Drop probability while in the bad state.")
    parser.add_argument(
        "--num-packets", type=int, default=600,
        help="Number of packets transmit6ted per trial (default: 600)."
    )
    parser.add_argument(
        "--num-trials", type=int, default=1_000_000,
        help="Number of Monte Carlo trials to run (default: 1,000,000)."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Number of trials to simulate per chunk to balance speed and memory."
    )
    parser.add_argument(
        "--save-loss-rates",
        type=str,
        default=None,
        help="Optional path to save the per-trial average loss rates as a .npy file."
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the summary statistics as JSON instead of plain text."
    )
    return parser.parse_args()


def summarize(loss_rates: np.ndarray):
    percentiles = [50, 90, 95, 99, 99.9, 99.99, 99.999]
    summary = {
        "mean": float(np.mean(loss_rates)),
        "std": float(np.std(loss_rates)),
        "min": float(np.min(loss_rates)),
        "max": float(np.max(loss_rates)),
        "percentiles": {f"p{p}": float(np.percentile(loss_rates, p)) for p in percentiles},
    }
    return summary


def main():
    args = parse_args()

    channel = GE_Channel(args.alpha, args.beta, args.epsilon, args.delta)
    loss_rates = channel.simulate_loss_rates(args.num_packets, args.num_trials, args.chunk_size)
    summary = summarize(loss_rates)
    summary["num_trials"] = args.num_trials
    summary["num_packets"] = args.num_packets

    if args.save_loss_rates:
        np.save(args.save_loss_rates, loss_rates)
        summary["saved_loss_rates"] = args.save_loss_rates

    if args.show_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Trials: {summary['num_trials']:,} | Packets per trial: {summary['num_packets']}")
        print(f"Mean loss rate: {summary['mean']:.6f}")
        print(f"Std dev: {summary['std']:.6f}")
        print(f"Min/Max: {summary['min']:.6f} / {summary['max']:.6f}")
        for label, value in summary["percentiles"].items():
            print(f"{label.upper()}: {value:.6f}")
        if "saved_loss_rates" in summary:
            print(f"Loss rates saved to: {summary['saved_loss_rates']}")


if __name__ == "__main__":
    main()
