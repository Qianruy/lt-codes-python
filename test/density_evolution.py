import numpy as np
from scipy.stats import poisson

# Parameters
alpha = 0.8
n = 10000
lmbda = 2 * alpha
num_iterations = 20
max_i = 11
delta = 0.03
print(f"alpha (load): {alpha}, delta: {delta}")

# Precompute beta_i
beta = np.array([i * poisson.pmf(i, lmbda) / lmbda for i in range(max_i + 1)])

# Initialize
S1 = [1.0]
S2 = [1.0]
C1 = []
C2 = []

# Iteration loop
for t in range(1, num_iterations + 1):
    
    # Update C2
    sum_beta_terms = sum([
        beta[i] * ((1 - S2[t - 1]) ** (i - 1)) * (1 - delta)
        for i in range(1, max_i)
    ])
    
    # c2_next = alpha * (S1[t - 1] + (1 - S1[t - 1]) * (1 - sum_beta_terms)) + \
    #           (1 - alpha) * (1 - sum_beta_terms)
    c2_next = 1 - np.exp(-lmbda * S2[-1]) * (1-alpha*S1[-1]) * (1 - delta)
    C2.append(c2_next)

    # Update C1
    # c1_next = 1.0
    # for i in range(max_i):
    #     c1_next -= poisson.pmf(i, lmbda) * ((1 - S2[t - 1]) ** i) * (1 - delta)
    c1_next = 1 - np.exp(-lmbda * S2[-1]) * (1 - delta)
    C1.append(c1_next)

    # Update S1, S2
    s1 = C2[-1] ** 2
    s2 = C1[-1] * C2[-1]
    S1.append(s1)
    S2.append(s2)

for t in range(num_iterations):
    print(f"t={t+1}: C1={C1[t]:.4f}, C2={C2[t]:.4f}, S1={S1[t]:.4f}, S2={S2[t]:.4f}, conv: {C1[t]*C2[t]**2:.4f}")
