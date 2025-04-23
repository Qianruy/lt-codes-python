import numpy as np
from scipy.stats import poisson

# Parameters
alpha = 0.8
n = 10000
lmbda = 3 * alpha
num_iterations = 36
max_i = 11
delta = 0.03
print(f"alpha (load): {alpha}, delta: {delta}")

# Precompute beta_i
beta = np.array([i * poisson.pmf(i, lmbda) / lmbda for i in range(max_i + 1)])

# Initialize
S = [1]
C = []

# Iteration loop
for t in range(1, num_iterations + 1):
    
    # Update C
    # c_next = 1.0
    # for i in range(1, max_i):
    #     c_next -= beta[i] * ((1 - S[t - 1]) ** (i - 1)) * (1 - delta)
    c_next = 1 - np.exp(-lmbda * S[-1]) * (1 - delta)
    C.append(c_next)

    # Update S
    s = C[-1] ** 2
    S.append(s)

for t in range(num_iterations):
    print(f"t={t+1}: C={C[t]:.4f}, S={S[t]:.4f}, conv: {C[t]**3:.4f}")
