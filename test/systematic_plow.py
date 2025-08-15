import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from sage.all import Matrix, GF, block_diagonal_matrix, identity_matrix
from encoder import *

k = 300 # number of source symbols
redundancy = 1.05
encoder = PlowEncoder(1, wdn_size=30, redundancy=redundancy, maxdegree=3, seed = 234761)
encoder.put_bat(np.ones((k, 1), dtype=np.uint8))

# Get the encoding for a batch of codewords
idx_list = encoder.get_all().index
n, _ = idx_list.shape # number of codewords
mask = np.zeros((n, k+1), dtype=int)
rows = np.arange(n)[:, None]
mask[rows, idx_list] = 1

M = Matrix(GF(2), mask.tolist())
print("Original M:")
# print(M)

# Take k rows as pivot block
pivot_rows = [math.ceil(i*redundancy) for i in range(1,k+1)]
assert len(pivot_rows) == k and len(set(pivot_rows)) == k
rest_rows = [i for i in range(n) if i not in pivot_rows]
perm = pivot_rows + rest_rows
M_perm = M[perm, :]
A = M_perm[:k, 1:]
print(A.dimensions())
if A.rank() < k:
    print(f"rank: {A.rank()}")
    raise ValueError("A isn't full-rank; regenerate or pick different rows")
A_inv = A.inverse()

# Build the block-diagonal transform T = diag(A_inv, I_{k-n})
T = block_diagonal_matrix([A_inv, identity_matrix(GF(2), n - k)])
R = T * M_perm
print("Systematic form [I; P]: ")
print(R)
G = nx.Graph()
G.add_nodes_from([f's{i}' for i in range(k)], bipartite=0)
G.add_nodes_from([f'c{i}' for i in range(n)], bipartite=1)
R_np = R.numpy()
for c, s in zip(*np.where(R_np[:, :] == 1)):
    if s != 0 and c != 0:
        G.add_edge(f's{s-1}', f'c{c-1}')

pos = {}
pos.update({f's{i}': (0, -i) for i in range(k)})
pos.update({f'c{i}': (1, -i) for i in range(n)})

plt.figure(figsize=(6, max(5, 0.3 * (k + n))))
nx.draw(G, pos, node_size=10, with_labels=False, edge_color="gray")
plt.title("Bipartite Matching (source → codewords)")
plt.axis("off")
plt.tight_layout()
plt.savefig("test/sys_plow_example.png")