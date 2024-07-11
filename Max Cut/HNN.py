import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


def sign(x: int | float | np.integer | np.floating | list | np.ndarray):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return 1 if x == 0 else int(np.sign(x))

    elif isinstance(x, (list, np.ndarray)):
        result = np.sign(x).astype(np.int32)
        result[result == 0] = 1
        if isinstance(x, list):
            return result.tolist()
        else:
            return result

    else:
        return TypeError


def generate_noise(level, num):
    if level > 0:
        repeat_size = num // (level + 1)
        noise = np.zeros(shape=num, dtype=np.int32)
        for n, l in enumerate(range(level, 0, -1)):
            noise[n * repeat_size: (n + 1) * repeat_size] = np.random.choice(range(-l, l + 1), size=repeat_size)
        print(repeat_size)
        print(noise, "\n")
    else:
        noise = np.zeros(shape=num, dtype=np.int32)
    return noise


def calculate_energy(x, weight):
    return -0.5 * x @ weight @ x.T


def cut_value(vertex, weight):
    return 0.25 * (np.sum(weight) - vertex @ weight @ vertex.T)


class HopfieldNeuralNetwork:
    def __init__(self, neuron, weight, noise, mc_steps, plot_spin=False):
        self.neuron = neuron
        self.weight = weight
        self.mc_steps = mc_steps
        self.plot_spin = plot_spin
        self.noise = generate_noise(noise, mc_steps)
        # self.sfb_weight = np.geomspace(101, np.ones(weight.shape[0]), int(mc_steps*0.95)) - 1

        self.sfb_weight = np.linspace(80, np.zeros(weight.shape[0]), int(mc_steps * 0.8))
        # self.sfb_weight = np.vstack((np.geomspace(64, np.ones(weight.shape[0]), num=7), np.zeros(weight.shape[0])))
        # self.sfb_weight = self.sfb_weight.repeat((np.array([4/20, 4/20, 3/20, 3/20, 2/20, 2/20, 1/20, 1/20]) * mc_steps).astype(np.int32), axis=0)

        if plot_spin:
            self.fig = plt.figure("spin", figsize=(6, 6))
            plt.tight_layout()

    def run(self):
        step = 0
        spin_log = self.neuron
        value_log = [cut_value(self.neuron, -self.weight)]
        energy_log = [calculate_energy(self.neuron, self.weight)]

        fig_idx = 1
        if self.plot_spin:
            ax = self.fig.add_subplot(5, int(self.mc_steps // 5) + 1, fig_idx)
            ax.imshow(self.neuron.reshape(args.shape), cmap="binary")
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        while step < self.mc_steps:
            # update
            # Serial update
            if step < self.sfb_weight.shape[0]:
                weight = self.weight - np.diag(self.sfb_weight[step])
            else:
                weight = self.weight

            for k in range(len(self.neuron)):
                # noise
                # self.neuron[k] = sign(np.dot(self.weight[k], self.neuron) + self.noise[step])

                # chaotic simulated annealing
                self.neuron[k] = sign(np.dot(weight[k], self.neuron))

            # Parallel update
            # self.neuron = sign(np.dot(self.weight, self.neuron) + self.noise[step])

            spin_log = np.vstack((spin_log, self.neuron))
            value_log.append(cut_value(self.neuron, -self.weight))
            energy_log.append(calculate_energy(self.neuron, self.weight))

            if self.plot_spin:
                fig_idx += 1
                ax = self.fig.add_subplot(5, int(self.mc_steps // 5) + 1, fig_idx)
                ax.imshow(self.neuron.reshape(args.shape), cmap="binary")
                ax.set_xticks([], [])
                ax.set_yticks([], [])
            step += 1
        return spin_log, value_log, energy_log,


def energy_hist(path, noise, epochs, mc_steps):
    coupling = np.load(path)
    spins = np.random.choice([-1, 1], size=coupling.shape[0])
    hnn = HopfieldNeuralNetwork(neuron=spins, weight=-coupling, noise=noise, mc_steps=mc_steps)

    print("\nCoupling\n", coupling)
    print(len(coupling[coupling < 0]), len(coupling[coupling == 0]), len(coupling[coupling > 0]))
    print()

    value_list = []
    energy_list = []
    for epoch in range(epochs):
        start_time = time.time()
        spins, value, energy = hnn.run()
        stop_time = time.time()
        print(f"Trial: {epoch} \t Time slot: {stop_time - start_time: .2f}s")

        value_list.append(value[-1])
        energy_list.append(energy[-1])

        if epoch < epochs - 1:
            hnn.neuron = np.random.choice([-1, 1], size=coupling.shape[0])

    print("--------------------Result--------------------")
    print(f"Max Cut Value: {np.max(value_list)}, Mean Cut Value: {np.mean(value_list)}")
    print(f"Min Ising Energy: {np.min(energy_list)}, Mean Ising Energy: {np.mean(energy_list)}")
    print()

    plt.figure("HNN Cut Value")
    plt.hist(value_list, bins=10, edgecolor="black")
    plt.title(path.split("/")[-1].split(".")[0], fontsize=15)
    plt.xlabel("Cut value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    plt.show()


def energy_curve(path, noise, mc_steps):
    coupling = np.load(path)
    spins = np.random.choice([-1, 1], size=coupling.shape[0])
    hnn = HopfieldNeuralNetwork(neuron=spins, weight=-coupling, noise=noise, mc_steps=mc_steps)

    print("\nCoupling\n", coupling)
    print(len(coupling[coupling < 0]), len(coupling[coupling == 0]), len(coupling[coupling > 0]))
    print()

    start_time = time.time()
    spins, value, energy = hnn.run()
    stop_time = time.time()
    print(f"Time slot: {stop_time - start_time: .2f}s")
    print()

    index = energy.index(min(energy))
    print("solution:\n", spins[index])
    print(f"Cut Value: {value[index]} (best), {value[-1]} (last)")
    print(f"Ising energy: {energy[index]} (best), {energy[-1]} (last)")
    print()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    # 绘制折线图
    line1 = ax1.plot(np.arange(len(energy)), energy, label='Ising energy', color='r')
    line2 = ax2.plot(np.arange(len(value)), value, label='Cut value', color='b')

    # 设置x轴和y轴的标签，指明坐标含义
    ax1.set_xlabel('Iteration', fontdict={'size': 16})
    ax1.set_ylabel('Ising energy', fontdict={'size': 16})
    ax2.set_ylabel('Cut value', fontdict={'size': 16})

    plt.title(path.split("/")[-1].split(".")[0])
    lines = line1 + line2
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95))
    # plt.tight_layout()
    # plt.savefig("./Visualization/HNU_energy.png", dpi=300)
    plt.show()


def spin_map(path, noise, offset, mc_steps, plot_spin=False):
    coupling = np.load(path)
    alpha = np.triu(np.random.choice(np.arange(1, offset + 1), size=coupling.shape))
    alpha[np.diag_indices_from(alpha)] = 0
    alpha += alpha.T

    coupling = alpha * coupling
    spins = np.random.choice([0, 1], size=coupling.shape[0])
    hnn = HopfieldNeuralNetwork(neuron=spins, weight=-coupling, noise=noise, mc_steps=mc_steps, plot_spin=plot_spin)

    print("\nCoupling\n", coupling)
    print(len(coupling[coupling < 0]), len(coupling[coupling == 0]), len(coupling[coupling > 0]))
    print()

    start_time = time.time()
    spins, value, energy = hnn.run()
    stop_time = time.time()
    print(f"Time slot: {stop_time - start_time: .2f}s")
    print()

    print("solution:\n", spins[-1])
    print(f"Cut Value: {value[-1]}")
    print(f"Ising energy: {energy[-1]}")
    print()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    # 绘制折线图
    line1 = ax1.plot(np.arange(len(energy)), energy, label='Ising energy', color='r')
    line2 = ax2.plot(np.arange(len(value)), value, label='Cut value', color='b')

    # 设置x轴和y轴的标签，指明坐标含义
    ax1.set_xlabel('Iteration', fontdict={'size': 16})
    ax1.set_ylabel('Ising energy', fontdict={'size': 16})
    ax2.set_ylabel('Cut value', fontdict={'size': 16})

    plt.title(path.split("/")[-1].split(".")[0])
    lines = line1 + line2
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameter of Ising solver')
    parser.add_argument('--noise_level', default=2, help='level of noise')
    parser.add_argument('--offset_level', default=1, help='offset of coupling')
    parser.add_argument('--MC_steps', default=1000, help='total iteration')
    parser.add_argument('--shape', default=(16, 32), help='shape of spin image')

    args = parser.parse_args()
    # energy_hist(path="./CouplingMatrix/G39.npy",
    #             noise=args.noise_level,
    #             epochs=100,
    #             mc_steps=args.MC_steps
    #             )

    energy_curve(path="./CouplingMatrix/K2000.npy",
                 noise=args.noise_level,
                 mc_steps=args.MC_steps
                 )

    # spin_map(path="./CouplingData/Custom_IEEE.npy",
    #          noise=args.noise_level,
    #          offset=args.offset_level,
    #          mc_steps=args.MC_steps,
    #          plot_spin=True
    #          )
