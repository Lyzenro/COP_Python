import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from COP.Graph.GCM import check, graph_color_map

dim = 12
nodes = np.arange(dim)
nodes_value = np.zeros(shape=dim, dtype=np.int32)
color = ['red', 'green', 'blue', 'yellow']

edges = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 4),
         (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 11), (9, 10)]

neighbors = [[1, 3], [0, 2, 3, 4, 5], [1, 6], [0, 1, 4], [1, 3, 5], [1, 4, 6], [2, 5, 7], [6, 8], [9, 11], [8, 10], [9], [8]]

epochs = 5
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
position = nx.spring_layout(G)

plt.ion()
for n, value in enumerate(nodes_value):
    plt.cla()
    plt.title(f"Iteration: {n}")
    nx.draw(G, with_labels=True, node_color=[color[idx] for idx in value], pos=position)
    if n < 0.5 * epochs * len(value):
        plt.pause(0.1)
    else:
        plt.pause(0.01)
plt.ioff()

# nx.draw(G, with_labels=True, node_color=[color[idx] for idx in nodes_value[-1]], pos=nx.circular_layout(G))
plt.figure("System Energy", figsize=(8, 6))
plt.plot(np.arange(len(energy)), energy)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("System Energy", fontsize=12)
plt.show()
