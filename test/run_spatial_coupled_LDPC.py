import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time

def generate_sc_ldpc_optimize(d_v=3, d_c=6, M=30, n_blk=20, coupling_width=3, seed=42):
    """
    Generate a spatial coupled LDPC matrix using socket matching and edge permutation.

    @param n_blk: number of variable blocks (total variable nodes = M * n_blk)
    @param M: number of variable nodes per block (lifting factor)
    @param d_v: degree of each variable node
    @param d_c: degree of each check node
    @param coupling_width: coupling window
    @param seed: random seed 

    @return: np.ndarray
    """
    if seed is not None: np.random.seed(seed)

    n_var = n_blk * M 
    n_check_per_blk = M * d_v // d_c
    n_check = (n_blk + coupling_width - 1) * n_check_per_blk
    H = np.zeros((n_check, n_var), dtype=int)

    # Prepare sockets for check nodes
    check_sockets = [list(np.repeat(np.arange(i * n_check_per_blk, (i+1) * n_check_per_blk), d_c))
                     for i in range(n_blk + coupling_width - 1)] 
    check_degs = [0] * n_check

    for blk in range(n_blk):

        var_sockets_per_blk = np.repeat(np.arange(blk * M, (blk+1) * M), d_v)
        # Prevent the scenario that mutiple edges in one check 
        # matched to the same variable nodes  
        np.random.shuffle(var_sockets_per_blk) 

        check_candidates = []
        if blk < coupling_width - 1:
            for w in range(coupling_width):
                check_candidates.extend(check_sockets[blk+w])
            np.random.shuffle(check_candidates)
            
        else: 
            # print("blk_id: ", blk)
            for w in range(coupling_width):
                check_blk = check_sockets[blk + w]
                np.random.shuffle(check_blk)
                check_candidates.extend(check_blk)
 
        assert len(check_candidates) >= len(var_sockets_per_blk), "Not enough check sockets to match variable sockets."
        edge_pairs = zip(check_candidates[:len(var_sockets_per_blk)], var_sockets_per_blk)
        used_pairs = set()
        
        for c, v in edge_pairs:
            while (c, v) in used_pairs or check_degs[c] >= d_c:
                c = np.random.choice(check_candidates)
            H[c, v] += 1
            used_pairs.add((c, v))
            check_degs[c] += 1
            assert H[c, v] <= 1

        # Remove connected check sockets
        for c, _ in used_pairs:
            blk_id = c // n_check_per_blk
            check_sockets[blk_id].remove(c)
    
    # print(check_degs)
    return H

def bec_channel(x, erasure_prob=0.2):
    received = x.copy()
    erasures = np.random.rand(len(x)) < erasure_prob
    received[erasures] = -1  
    return received

def peeling_decoder(H, received):
    n_check, n_var = H.shape
    decoded = received.copy()
    assert len(received) == n_var, "Received length mismatch"
    
    while True:
        progress = False
        for i in range(n_check):
            idxs = np.where(H[i, :] == 1)[0]
            known = idxs[decoded[idxs] != -1]
            unknown = idxs[decoded[idxs] == -1]
            if len(unknown) == 1:
                j = unknown[0]
                parity = np.sum(decoded[known]) % 2
                decoded[j] = parity
                progress = True
        if not progress:
            break
    return decoded

def run_one_trial(eps, d_v, d_c, n, n_blk, coupling_width):
    np.random.seed(42)
    H = generate_sc_ldpc_optimize(d_v=d_v, d_c=d_c, M=n, n_blk=n_blk, coupling_width=coupling_width)
    codeword = np.zeros(n * n_blk, dtype=int)
    received = bec_channel(codeword, eps)
    decoded = peeling_decoder(H, received)
    return int(-1 not in decoded and np.all(np.mod(H @ decoded, 2) == 0))

def run_experiments(n, d_v, d_c, n_blk,
    coupling_width, start=0.075, end=0.09, step=3):
    num_trials = 10
    epsilons = np.linspace(start, end, int(step))  
    n = int(n); n_blk = int(n_blk)
    d_v = int(d_v); d_c = int(d_c)
    coupling_width = int(coupling_width)
    success_rates = []

    seed = int(time.time() * 1000) + os.getpid()
    seed = seed % 1000000
    np.random.seed(seed)
    # H = generate_regular_ldpc(n_var=n, d_v=d_v, d_c=d_c)
    H = generate_sc_ldpc_optimize(d_v=d_v, d_c=d_c, M=n, n_blk=n_blk, coupling_width=coupling_width)
    print(f"variable degree: {d_v}, check degree: {d_c}, blk_size: {n}, blk_number: {n_blk}, coupling width: {coupling_width}")
    for eps in epsilons:
        success = 0
        for _ in range(num_trials):
            if run_one_trial(eps, d_v, d_c, n, n_blk, coupling_width):
                success += 1
        success_rate = success / num_trials
        success_rates.append(success_rate)
        print(f"Epsilon={eps:.3f}, Success Rate={success_rate:.3f}")
    
    # Save results to a CSV file
    with open(f'test/results_{n}_{d_v}_{d_c}_{n_blk}_{coupling_width}_{seed}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epsilon', 'success_rate'])
        for eps, s, num in zip(epsilons, success, num_trials):
            writer.writerow([eps, s, num])
    
    # Plotting the results
    # plt.plot(epsilons, success_rates, marker='o')
    # plt.xlabel('Erasure Probability (epsilon)')
    # plt.ylabel('Success Rate')
    # plt.title(f'(dv={d_v}, dc={d_c}) LDPC over BEC')
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    # Use csv file to load the parameters
    params = []
    with open('test/params.csv', newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            params.append([float(x) for x in row])
    # Run experiments with the parameters
    for param in params:
        print(f"Running with parameters: {param}")
        run_experiments(*param)