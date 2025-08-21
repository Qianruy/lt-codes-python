import sys
sys.path.append('..')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
#from sage.all import Matrix, GF, block_diagonal_matrix, identity_matrix
from encoder import *
from collections import deque
import random
import multiprocessing as mp 
from functools import partial 
max_num = 10000000

def get_node_from_partition(graph, edge, partition_id=0):
    u, v = edge
    if graph.nodes[u]['bipartite'] == partition_id:
        return u
    else:
        return v


def find_shortest_cycle_for_node(G, start_node):
    """
    In a given undirected graph G, find the shortest cycle that starts a specified start_node.

    Args:
        G (nx.Graph): A graph.
        start_node: The specified starting node in the graph.
        
    Returns:
        int : The length of the shortest cycle containing start_node.
                      If no cycle exists containing this node, return float('inf').
    """

    # Initialize the shortest cycle length to infinity
    min_cycle_len = max_num
    
    q = deque([(start_node, None)]) 

    # Distance dictionary: records the distance from start_node to all other nodes in the graph
    dist = {node: -1 for node in G.nodes()}
    dist[start_node] = 0

    # BFS core logic
    while q:
        u, parent = q.popleft()

        # Iterate over all neighbors v of the current node u
        for v in G.neighbors(u):
            # Prevent immediate backtracking
            if v == parent:
                continue

            # Case A: v is a new node that hasn't been visited
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append((v, u))
            # Case B: v is an already visited node, a cycle is found!
            else:
                # A cycle is found.
                # This cycle is formed by start_node->...->u, start_node->...->v
                # and the edge (u,v).
                cycle_len = dist[u] + dist[v] + 1
                
                # update the minimum cycle length if this cycle is shorter
                if cycle_len < min_cycle_len:
                    min_cycle_len = cycle_len
                    
    return min_cycle_len

def process_single_edge(edge, graph):
    start_node = get_node_from_partition(graph, edge, partition_id=1)
    
    new_graph = graph.copy()
    u, v = edge
    new_graph.remove_edge(u, v)
    
    girth = find_shortest_cycle_for_node(new_graph, start_node)
    
    # Print the edge and the girth found
    # print(f"Processed edge {edge}, found girth {girth}")
    
    return girth


# Parameters
k = 100000 # number of source symbols
redundancy = 1.05
num_samples = 500
used_wdn_size = 600
u_seed = 142857


encoder = PlowEncoder(1, wdn_size=300, redundancy=redundancy, maxdegree=3, seed = u_seed)
encoder.put_bat(np.ones((k, 1), dtype=np.uint8))

# Get the encoding for a batch of codewords
idx_list = encoder.get_all().index
n, _ = idx_list.shape # number of codewords
mask = np.zeros((n, k+1), dtype=int)
rows = np.arange(n)[:, None]
mask[rows, idx_list] = 1

# Create a bipartite graph
G = nx.Graph()
G.add_nodes_from([f's{i}' for i in range(k)], bipartite=0)
G.add_nodes_from([f'c{i}' for i in range(n)], bipartite=1)
for c, s in zip(*np.where(mask[:, :] == 1)):
    if s != 0 and c != 0:
        G.add_edge(f's{s-1}', f'c{c-1}')

# sample edges in the graph
random.seed(u_seed)
edges = list(G.edges())
sampled_edges = random.sample(edges, num_samples)
num_cores = mp.cpu_count()
#print(f"Number of cores available: {num_cores}")
task_func = partial(process_single_edge, graph=G)

with mp.Pool(processes=num_cores) as pool:
    results = pool.map(task_func, sampled_edges)

girth_histogram = {}
for girth in results:
    girth_histogram[girth] = girth_histogram.get(girth, 0) + 1
    
# Print the girth histogram
print("\nGirth Histogram:")
for girth, count in sorted(girth_histogram.items()):
    print(f"Girth {girth}: {count} occurrences")
