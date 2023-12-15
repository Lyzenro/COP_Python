import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from COP.Graph.GCM import check, graph_color_map

dim = 6
edges = []
neighbors = []
nodes = np.arange(dim)
nodes_value = np.zeros(shape=dim, dtype=np.int32)
color = ['red', 'green', 'blue', 'yellow', 'orange', "purple"]

for i in range(dim):
    nbs = []
    for j in range(dim):
        if i != j:
            nbs.append(j)
            edges.append((i, j))
    neighbors.append(nbs)

epochs = 10
nodes_value, energy = graph_color_map(vertex=nodes_value,
                                      weight=None,
                                      color=np.arange(len(color)),
                                      epochs=epochs,
                                      neighbor_list=neighbors
                                      )
print("Solution\n", nodes_value[-1])
print("System:", energy[-1])
print()
print(check(vertexes=nodes_value[-1], neighbor_list=neighbors))

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
position = nx.circular_layout(G)

plt.ion()
for n, value in enumerate(nodes_value):
    plt.cla()
    plt.title(f"Iteration: {n}")
    nx.draw(G, with_labels=True, node_color=[color[idx] for idx in value], pos=position)
    if n < 0.5 * epochs * len(value):
        plt.pause(0.1)
    else:
        plt.pause(0.05)
plt.ioff()

# nx.draw(G, with_labels=True, node_color=[color[idx] for idx in nodes_value[-1]], pos=nx.circular_layout(G))
plt.figure("System Energy", figsize=(8, 6))
plt.plot(np.arange(len(energy)), energy)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("System Energy", fontsize=12)
plt.show()
