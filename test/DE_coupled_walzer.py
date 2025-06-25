import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Parameters
alpha, delta = 1, 0.01
lmbda = 3 * alpha
L_show, L_full = 80000, 160000
w, max_iter, r = 600, 3000, 0
print(f"alpha (load): {alpha}, delta: {delta}")

# Create averaging window
window = np.ones(w) / w

# Initialize S1, S2 arrays
S = np.ones((max_iter + 1, L_full))
C = []

# Iteration loop
conv_metric = np.ones((max_iter + 1, L_full))   
while conv_metric[r][:L_show].max() > 1e-3 and r < max_iter:
    S_alt = np.concat([np.zeros(w-1), S[r]], axis=-1)
    # Vectorized sliding‐window average via convolution
    avg_S = np.convolve(S_alt, window, mode='valid')
    
    # Update C
    C_raw = 1 - np.exp(-lmbda * avg_S) * (1 - delta)

    C_raw = np.concat([C_raw, np.ones(w-1)], axis=-1)
    avg_C = np.convolve(C_raw, window, mode='valid')
    
    # Update S1, S2
    S[r + 1] = avg_C ** 2
    
    conv_metric[r + 1] = (avg_C ** 3)
    
    print(f"t={r}: S={np.sum(conv_metric[r][:L_show])/L_show}")
    r += 1
print(f"iteration number: {r}")

# Sample 600 positions evenly from the 80000
sample_indices = np.linspace(0, L_show, w, dtype=int)

# Extract the sampled evolution
S_sampled = S[:r, :][:, sample_indices]  # shape (iterations+1, 600)

for t in range(min(r, 500)):
    print(f"t={t+1}: S={S[t][sample_indices]}")

# Plot the evolution as a heatmap
plt.figure(figsize=(12, 6))
plt.imshow(
    S_sampled,
    aspect='auto',
    origin='lower',
    extent=[0, 599, 0, r]
)
plt.colorbar(label='$S$')
plt.xlabel('Sampled position index (0-599)')
plt.ylabel('Iteration')
plt.title('Evolution of $S$ (600 sampled positions from 80000)')
plt.tight_layout()
plt.show()