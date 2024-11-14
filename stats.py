import json
import sys
import os
from datetime import datetime

class Stats:
    def __init__(self, args):
        self.data = {
            "args": vars(args),
            "results": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d")
        }

    def add_result(self, key, value):
        """Add results under the certain key."""
        if key not in self.data["results"]:
            self.data["results"][key] = []
        self.data["results"][key].append(value)

    def save_to_json(self, filename):
        """Append the recorded data to a JSON file as a list"""
        data_to_append = self.data

        if not os.path.exists(filename):
            # Create a new file and initialize a list
            with open(filename, 'w') as f:
                f.write('[\n')
                json.dump(data_to_append, f, indent=4)
                f.write('\n]')
        else:
            # if the file exists, open the file using append mode
            with open(filename, 'r+') as f:
                f.seek(0, os.SEEK_END)  
                pos = f.tell()
                
                # Examine if the end is emmpy
                if pos > 2:
                    f.seek(pos - 2)
                    f.truncate()  # delet last character "]"
                    f.write(',\n')
                
                # Append new data
                json.dump(data_to_append, f, indent=4)
                f.write('\n]')

        print(f"Data saved to {filename}")

    

    