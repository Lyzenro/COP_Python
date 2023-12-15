import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def delta_func(x, y):
    return -1 if x == y else 1


# def calculate_energy(x, weight):
#     dim = len(x)
#     sys_energy = 0
#     for i in range(dim):
#         for j in range(i + 1, dim):
#             sys_energy += weight[i, j] * delta_func(x[i], x[j])
#
#     return sys_energy * 2

def system_energy(vertexes, neighbor_list):
    sys_energy = 0
    for i, v in enumerate(vertexes):
        for vn in vertexes[neighbor_list[i]]:
            sys_energy -= delta_func(v, vn)
    return sys_energy


def local_energy(vertex, neighbor_vertex):
    loc_energy = 0
    for vn in neighbor_vertex:
        loc_energy -= delta_func(vertex, vn)
    return loc_energy


def check(vertexes, neighbor_list):
    for idx, v in enumerate(vertexes):
        index = neighbor_list[idx]
        if v in vertexes[index]:
            return False
    return True


def generate_coupling(vertexes, graph_mode="King"):
    neighbor_list = []
    size = np.sqrt(vertexes).astype(np.int32)
    edge_weight = np.zeros(shape=(vertexes, vertexes), dtype=np.int32)

    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0:                                                   # top left
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i + 1) + j + 1, size * i + j + 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i + 1) + j, size * i + j + 1])

            elif i == 0 and j == size - 1:                                          # top right
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i + 1) + j - 1, size * i + j - 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i + 1) + j, size * i + j - 1])

            elif i == size - 1 and j == 0:                                          # bottom left
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i - 1) + j, size * (i - 1) + j + 1, size * i + j + 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i - 1) + j, size * i + j + 1])

            elif i == size - 1 and j == size - 1:                                   # bottom right
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i - 1) + j, size * (i - 1) + j - 1, size * i + j - 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i - 1) + j, size * i + j - 1])

            elif i == 0 and (0 < j < size - 1):                                     # top edge
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i + 1) + j - 1, size * (i + 1) + j + 1, size * i + j - 1, size * i + j + 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i + 1) + j, size * i + j - 1, size * i + j + 1])

            elif i == size - 1 and (0 < j < size - 1):                              # bottom edge
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i - 1) + j, size * (i - 1) + j - 1, size * (i - 1) + j + 1, size * i + j - 1, size * i + j + 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i - 1) + j, size * i + j - 1, size * i + j + 1])

            elif (0 < i < size - 1) and j == 0:                                     # left edge
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i - 1) + j, size * (i + 1) + j + 1, size * (i - 1) + j + 1, size * i + j + 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i - 1) + j, size * i + j + 1])

            elif (0 < i < size - 1) and j == size - 1:                              # right edge
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i - 1) + j, size * (i + 1) + j - 1, size * (i - 1) + j - 1, size * i + j - 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i - 1) + j, size * i + j - 1])

            else:                                                                   # inside
                if graph_mode in ["k", "K", "king", "King"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i + 1) + j - 1, size * (i + 1) + j + 1, size * (i - 1) + j, size * (i - 1) + j - 1, size * (i - 1) + j + 1, size * i + j - 1, size * i + j + 1])
                elif graph_mode in ["l", "L", "lattice", "Lattice"]:
                    neighbor_list.append([size * (i + 1) + j, size * (i - 1) + j, size * i + j - 1, size * i + j + 1])

    for i in range(vertexes):
        edge_weight[i, neighbor_list[i]] = -1

    return edge_weight, neighbor_list


def graph_color_map(vertex, weight, color, epochs, neighbor_list):
    dim = len(vertex)
    vertex_log = vertex
    energy_log = [system_energy(vertex, neighbor_list)]

    for epoch in range(epochs):
        for i in range(dim):
            color_clone = color.copy().tolist()
            loc_energy_old = local_energy(vertex[i], vertex[neighbor_list[i]])

            # disturbance
            vertex_clone = vertex.copy()
            color_clone.remove(vertex_clone[i])

            # while True:
            #     color_select = np.random.choice(color_clone)
            #     if color_select not in vertex_clone[neighbor_list[i]]:
            #         break
            color_select = np.random.choice(color_clone)
            vertex_clone[i] = color_select

            loc_energy_new = local_energy(vertex_clone[i], vertex_clone[neighbor_list[i]])

            # update
            if loc_energy_new < loc_energy_old:
                vertex = vertex_clone.copy()
            energy_log.append(system_energy(vertex, neighbor_list))

            vertex_log = np.vstack((vertex_log, vertex))
    return vertex_log, energy_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter of Graph Color Map')
    parser.add_argument('--epochs', default=10, help='total iteration')
    parser.add_argument('--vertexes', default=25, help='number of vertex')
    parser.add_argument('--color_value', default=np.array([0, 1, 2, 3]), help='color value')

    args = parser.parse_args()
    spins = np.zeros(shape=args.vertexes)
    spin_coupling, neighbor = generate_coupling(args.vertexes, graph_mode="l")

    print("Initial configuration\n", spins, "\n")
    print("Edge Weight\n", spin_coupling, "\n")

    spins, energy = graph_color_map(vertex=spins,
                                    weight=spin_coupling,
                                    color=args.color_value,
                                    epochs=args.epochs,
                                    neighbor_list=neighbor
                                    )

    print("Solution\n", spins[-1])
    print("System:", energy[-1])
    print()
    print(check(vertexes=spins[-1], neighbor_list=neighbor))

    shape = np.sqrt(args.vertexes).astype(np.int32)
    fig = plt.figure("vertex color", figsize=(8, 6))
    color_map = ListedColormap(['red', 'green', 'blue', 'yellow'])
    # color_map = ListedColormap(['red', 'green', 'blue', 'yellow', 'orange'])
    # color_map = ListedColormap(['red', 'green', 'blue', 'yellow', 'orange', "purple"])
    # plt.ion()
    # for n, spin in enumerate(spins):
    #     plt.cla()
    #     plt.title(f"Iteration: {n}")
    #     plt.imshow(spin.reshape(shape, shape), cmap=color_map)
    #     ticks = np.arange(int(np.sqrt(args.vertexes)))
    #     plt.xticks(ticks, ticks)
    #     plt.yticks(ticks, ticks)
    #     if n < 3 * len(spin):
    #         plt.pause(0.01)
    #     elif n < 5 * len(spin):
    #         plt.pause(0.001)
    #     else:
    #         plt.pause(0.0001)
    # plt.ioff()
    plt.imshow(spins[-1].reshape(shape, shape), cmap=color_map)
    plt.figure("System Energy", figsize=(8, 6))
    plt.plot(np.arange(len(energy)), energy)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("System Energy", fontsize=12)
    plt.show()
