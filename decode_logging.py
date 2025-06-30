import csv
import json
import logging
import numpy as np
from collections import defaultdict

# logging.basicConfig(filename='./experiments/decoding.log', level=logging.INFO,
#                     format='%(asctime)s-%(levelname)s-%(message)s')

class decoderState:
    def __init__(self):
        self.codeword_log = defaultdict(list) # store the symbol id removed its degree with round number
        self.decoded_log = defaultdict(list)
        self.decode_graph = defaultdict(list)
        self.decoding_order = [] # index: round, store the symbol id being decoded in each round
        self.released_by = defaultdict(set)
        self.recovered_source = {}

        self.cvs_file = "decoding_log.csv"
        self.json_file = "decoding_details.json"

    def log_decoded_symbols(self, round, index):
        # Store with step number
        self.decoded_log[round] = list(index)

    def log_codeword_degree_removal(self, source_symbol, codeword, round):
        """
        Record the round the degree removed by a certain source symbol
        The end of the list is the final one to set the codeword free 
        Codeword with degree k will have k-1 tuples if it contributes to decoding
        """
        self.codeword_log[codeword].append((source_symbol, round))
        self.decode_graph[source_symbol] = codeword

    def add_dependency_code2source(self, sources_symbol, codeword):
        self.decode_graph[codeword] = sources_symbol

    def release_codeword(self, source_symbol, codewords):
        """
        Record the codewords that have degree 1 immediately after source symbol decoded
        """
        self.released_by[source_symbol].extend(codewords)

    def save_logs(self):
        """ Save final logs to CSV and JSON """
        self._write_degree_log("degree_reduction_log.csv")
        self._write_json("decoding_details.json", {
            "decoding_order": self.decoding_order,
            "release_tracking": {k: list(v) for k, v in self.released_by.items()},
            "source_recovery": self.source_recovery
        })

    def _write_degree_log(self, filename):
        """ Save decoding process logs to CSV """
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Source Symbol", "Codeword that release it"])
            for round_num, changes in enumerate(self.degree_reduction_log):
                for source_symbol, codewords in changes.items():
                    for codeword in codewords:
                        writer.writerow([round_num, source_symbol, codeword])

    def _write_to_csv(self, filename, data):
        """ Helper function to write decoding logs to CSV """
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def _write_to_json(self, filename, data):
        """ Save structured data to JSON """
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=self._convert_numpy)

    def _convert_numpy(self, o):
        """ Convert NumPy types to native Python types """
        if isinstance(o, np.integer):
            return int(o)  # Convert NumPy int64 to Python int
        elif isinstance(o, np.floating):
            return float(o)  # Convert NumPy float to Python float
        elif isinstance(o, np.ndarray):
            return o.tolist()  # Convert NumPy arrays to lists
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
