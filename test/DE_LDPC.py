import numpy as np
from scipy.stats import poisson

# Parameters
d_l = 5
d_r = 100
num_iterations = 60
delta_list = np.linspace(0.01, 0.1, 1000) 

threshold = 0

for delta in delta_list:
    X = 1  # Start from full erasure
    for _ in range(num_iterations):
        X = (1 - (1 - X) ** (d_r-1)) ** (d_l-1) * delta
    if X < 1e-6:  
        threshold = delta

print(f"Threshold for LDPC ({d_l}, {d_r}) is about {threshold:.5f}")
