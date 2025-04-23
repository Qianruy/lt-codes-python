import csv
import numpy as np
import glob

# Function to calculate statistics
def calculate_statistics(differences):
    avg = np.mean(differences)
    median = np.median(differences)
    percentile_95 = np.percentile(differences, 95)
    return avg, median, percentile_95

# Read the CSV file
file_pattern = "experiments/sf_0320_2025032*_0.csv"  
file_list = glob.glob(file_pattern)
# print(file_list)
differences = []
redundancy = 1; loss = 0

if not file_list:
    print("No matched files!")
    exit()

for filename in file_list:
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                timestamp = int(row[0])
                id_list_str = row[1].strip('"')  
                id_list = [int(id.strip()) for id in id_list_str.split(",")] 
                for id_val in id_list:
                    if (timestamp < id_val+1): print(timestamp, id_list)
                    # assert(timestamp>=id_val+1)
                    differences.append((timestamp-1)*(1+loss)-(int(id_val*redundancy)))
            except ValueError:
                print(timestamp)
                continue

# Calculate statistics
avg, median, percentile_95 = calculate_statistics(differences)

# Print results
print(f"Average: {avg}")
print(f"Median: {median}")
print(f"95th Percentile: {percentile_95}")