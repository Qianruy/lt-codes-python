import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import os 
import psutil

# Parameters
alpha, delta = 0.8665, 0.1
L_show, L_full = 80000, 160000
w, max_iter, r = 600, 2000, 0
print(f"alpha (load): {alpha}, delta: {delta}, window_size: {w}")

# Create averaging window
# w = int(w/alpha)
window = np.ones(w) / w

# Initialize S1, S2 arrays
S2 = np.ones((max_iter + 1, L_full))
S1 = np.ones((max_iter + 1, L_full)) # length: L_full

# Boundary Condition 
S2_alt_pad = np.empty(L_full + w - 1)
S2_alt_pad[:w-1] = 0
C2_raw_pad = np.empty(L_full + w - 1)
C2_raw_pad[L_full:] = 1

# Iteration loop
conv_metric = np.ones((max_iter + 1, L_full))   
while conv_metric[r][:L_show].max() > 1e-3 and r < max_iter:
    # Vectorized sliding‐window average via convolution
    # S2_alt = np.pad(S2[r], (w-1, 0), constant_values=0)
    S2_alt_pad[w-1:] = S2[r]
    avg_S2 = np.convolve(S2_alt_pad, window, mode='valid') # length: L_full 
    assert avg_S2.shape[-1] == L_full

    # Update C
    lmbda = 2 * alpha * avg_S2
    C1_raw = 1 - np.exp(-lmbda) * (1 - delta)
    C2_raw = 1 - np.exp(-lmbda) * (1 - alpha*S1[r]) * (1 - delta)
    
    # Padding + Conv: L_full
    # C2_raw = np.pad(C2_raw, (0, w-1), constant_values=1)
    C2_raw_pad[:L_full] = C2_raw
    avg_C2 = np.convolve(C2_raw_pad, window, mode='valid') 
    
    # Update S1, S2
    S1[r + 1] = avg_C2**2
    S2[r + 1] = C1_raw * avg_C2
    
    conv_metric[r + 1] = C1_raw * (avg_C2 ** 2)

    print(f"t={r}: S={np.sum(conv_metric[r][:L_show])/L_show}")
    if r % 100 == 0:
        print(f"Memory at iter {r}: {psutil.Process(os.getpid()).memory_info().rss/1e9:.2f} GB")
    r += 1

print(f"iteration number: {r}")

# Sample w positions evenly from the 10000
sample_indices = np.linspace(0, L_show, w, dtype=int)

# Extract the sampled evolution
S2_sampled = S2[:r, :][:, sample_indices]  # shape (iterations+1, 500)
# for t in range(min(r, 500)):
#     print(f"t={t}: conv={conv_metric[t][sample_indices]}")

# Plot the evolution as a heatmap
plt.figure(figsize=(12, 6))
plt.imshow(
    S2_sampled,
    aspect='auto',
    origin='lower',
    extent=[0, w-1, 0, r]
)
plt.colorbar(label='$S_2$')
plt.xlabel(f'Sampled position index (0-{w-1})')
plt.ylabel('Iteration')
plt.title(f'Evolution of $S_2$ ({w} sampled positions from 80000)')
plt.tight_layout()
plt.show()