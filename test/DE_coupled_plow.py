import numpy as np
from scipy.stats import poisson, binom
import matplotlib.pyplot as plt
import os 
import psutil

# Parameters
alpha, delta = 0.891, 0.1
L_show, L_full = 80000, 160000
w, max_iter, r = 600, 3000, 0
k = 4
# Create averaging window
# w = int(w/alpha)
# window = np.ones(w) / w
p2 = binom.pmf(np.arange(w), w-1, 1/2)
# p2 /= sum(p2)
# print(1-sum(p2))
p3 = binom.pmf(np.arange(w), w-1, 3/4)
# print(1-sum(p3))
# p3 /= sum(p3)
if k == 4:
    p4 = binom.pmf(np.arange(w), w-1, 7/8)
    # print(1-sum(p4))
    # p4 /= sum(p4)

print(f"alpha (load): {alpha}, delta: {delta}, window_size: {w}, k: {k}")

# Initialize S1, S2 arrays
S1 = np.ones((max_iter + 1, L_full))
S2 = np.ones((max_iter + 1, L_full))
S3 = np.ones((max_iter + 1, L_full))
S4 = np.ones((max_iter + 1, L_full))

# Boundary Condition
S2_alt_pad = np.empty(L_full + w - 1)
S3_alt_pad = np.empty(L_full + w - 1)
S4_alt_pad = np.empty(L_full + w - 1)
S2_alt_pad[:w-1] = 0
S3_alt_pad[:w-1] = 0
S4_alt_pad[:w-1] = 0
C2_raw_pad = np.empty(L_full + w - 1)
C2_raw_pad[L_full:] = 1

# Iteration loop
conv_metric = np.ones((max_iter + 1, L_full))
while conv_metric[r][:L_show].max() > 1e-3 and r < max_iter:
    
    # Vectorized sliding‐window average via convolution
    S2_alt_pad[w-1:] = S2[r]
    S3_alt_pad[w-1:] = S3[r]
    S4_alt_pad[w-1:] = S4[r]
    avg_S2 = np.convolve(S2_alt_pad, p2, mode='valid')
    avg_S3 = np.convolve(S3_alt_pad, p3, mode='valid')
    
    assert avg_S2.shape[-1] == L_full
    assert avg_S3.shape[-1] == L_full
    if k == 4:
        avg_S4 = np.convolve(S4_alt_pad, p4, mode='valid')
        assert avg_S4.shape[-1] == L_full
    
    # Update C
    miss_from_1 = 1-alpha*S1[r]
    miss_from_2 = np.exp(-alpha*avg_S2)
    miss_from_3 = np.exp(-alpha*avg_S3)
    C1_raw = 1 -  miss_from_2 * miss_from_3 * (1 - delta)
    C2_raw = 1 -  miss_from_2 * miss_from_3 * miss_from_1 * (1 - delta)
    if k == 4 : 
        miss_from_4 = np.exp(-alpha*avg_S4)
        C1_raw = 1 + (C1_raw-1) * miss_from_4
        C2_raw = 1 + (C2_raw-1) * miss_from_4

    #Padding + Correlate: L_full
    C2_raw_pad[:L_full] = C2_raw
    avg_C2 = np.correlate(C2_raw_pad, p2, mode='valid')
    avg_C3 = np.correlate(C2_raw_pad, p3, mode='valid')
    if k == 4:
        avg_C4 = np.correlate(C2_raw_pad, p4, mode='valid')
    
    # Update S1, S2
    S1[r + 1] = avg_C2 * avg_C3 
    S2[r + 1] = C1_raw * avg_C3 
    S3[r + 1] = C1_raw * avg_C2 

    conv_metric[r + 1] = C1_raw * avg_C2 * avg_C3 

    if k == 4:
        S1[r + 1] *= avg_C4
        S2[r + 1] *= avg_C4
        S3[r + 1] *= avg_C4
        S4[r + 1] = C1_raw * avg_C2 * avg_C3
        
        conv_metric[r + 1] *= avg_C4

    print(f"t={r}: S={np.sum(conv_metric[r][:L_show])/L_show}")
    if r % 100 == 0:
        print(f"Memory at iter {r}: {psutil.Process(os.getpid()).memory_info().rss/1e9:.2f} GB")
    r += 1
    
print(f"iteration number: {r}")

# Sample w positions evenly from the 10000
sample_indices = np.linspace(0, L_show, 600, dtype=int)

# Extract the sampled evolution
S2_sampled = S2[:r, :][:, sample_indices]  # shape (iterations+1, 500)
conv_sampled = conv_metric[:r, :][:, sample_indices]

# for t in range(min(r, 300)):
#     print(f"t={t}: conv={conv_metric[t][sample_indices]}")

# Plot the evolution as a heatmap
plt.figure(figsize=(12, 6))
plt.imshow(
    conv_sampled,
    aspect='auto',
    origin='lower',
    extent=[0, 600-1, 0, r]
)
plt.colorbar(label='$Convolution\ result$')
plt.xlabel(f'Sampled position index (0-{w-1})')
plt.ylabel('Iteration')
plt.title(f'Evolution of $Convolution\ result$ ({600} sampled positions from {L_show})')
plt.tight_layout()
plt.show()