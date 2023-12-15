import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


def calculate_ising_energy(x, weight):
    return -0.5 * x @ weight @ x.T


def calculate_max_cut_value(x, weight):
    return 0.25 * (np.sum(weight) - x @ weight @ x.T)


class SimulatedAnnealing:
    def __init__(self, spin, coupling, mc_steps, temperature, plot_spin_fig=True):
        self.spin = spin
        self.coupling = coupling
        self.MC_steps = mc_steps
        self.temperature = np.linspace(start=temperature, stop=1, num=len(spin) * mc_steps)

        if plot_spin_fig:
            self.fig = plt.figure("spin", figsize=(6, 6))
            plt.tight_layout()
        else:
            self.fig = None

    def run(self):
        n = 0
        fig_idx = 1
        spin_log = self.spin
        ising_energy = [calculate_max_cut_value(self.spin, self.coupling)]
        max_cut_value = [calculate_ising_energy(self.spin, -self.coupling)]

        if self.fig is not None:
            ax = self.fig.add_subplot(5, int(self.MC_steps // 5) + 1, fig_idx)
            ax.imshow(self.spin.reshape(args.shape), cmap="binary")
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        for t in self.temperature:

            local_energy = self.coupling[n, :] @ self.spin
            delta_energy = 2 * self.spin[n] * local_energy
            if delta_energy < 0 or np.exp(- delta_energy / t) >= np.random.uniform():
                self.spin[n] = - self.spin[n]

            if n == len(self.spin) - 1:
                n = 0
                spin_log = np.vstack((spin_log, self.spin))
                ising_energy.append(calculate_ising_energy(self.spin, self.coupling))
                max_cut_value.append(calculate_max_cut_value(self.spin, -self.coupling))

                if self.fig is not None:
                    fig_idx += 1
                    ax = self.fig.add_subplot(5, int(self.MC_steps // 5) + 1, fig_idx)
                    ax.imshow(self.spin.reshape(args.shape), cmap="binary")
                    ax.set_xticks([], [])
                    ax.set_yticks([], [])
            else:
                n += 1

        return spin_log, max_cut_value, ising_energy


def energy_hist(solver, trials):
    value_list = []
    energy_list = []

    for n in range(trials):
        start_time = time.time()
        spin, value, energy = solver.run()
        stop_time = time.time()
        print(f"Trial: {n} \t Time slot: {stop_time - start_time: .2f}s")
        value_list.append(value[-1])
        energy_list.append(energy[-1])

        if n < trials - 1:
            solver.spin = np.random.choice([-1, 1], size=len(solver.spin))

    print()
    print("--------------------Result--------------------")
    print(f"Max Cut Value: {np.max(value_list)}, Mean Cut Value: {np.mean(value_list)}")
    print(f"Min Ising Energy: {np.min(energy_list)}, Mean Ising Energy: {np.mean(energy_list)}")
    print()

    plt.figure("SA Cut Value")
    plt.hist(value_list, bins=10, edgecolor="black")
    plt.title(path.split("/")[-1].split(".")[0], fontsize=15)
    plt.xlabel("Cut value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    plt.show()


def energy_curve(solver):
    start_time = time.time()
    spin, value, energy = solver.run()
    stop_time = time.time()
    print(f"Time slot: {stop_time - start_time: .2f}s\n")

    print("solution:\n", spin[-1])
    print(f"Cut Value: {value[-1]}")
    print(f"Ising energy: {energy[-1]}")
    print()

    # 绘制折线图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

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
    parser.add_argument('--temperature', default=50, help='level of noise')
    parser.add_argument('--MC_steps', default=800, help='total iteration')
    parser.add_argument('--shape', default=(16, 32), help='shape of spin image')

    args = parser.parse_args()
    # path = "./CouplingMatrix/Custom_IEEE.npy"
    # path = "./CouplingMatrix/K2000.npy"
    # path = "./CouplingMatrix/G22.npy"
    path = "./CouplingMatrix/G39.npy"

    coupling_matrix = np.load(path)
    spins = np.random.choice([-1, 1], size=coupling_matrix.shape[0])

    sa = SimulatedAnnealing(spins, -coupling_matrix, args.MC_steps, args.temperature, plot_spin_fig=False)

    # energy_curve(solver=sa)
    energy_hist(solver=sa, trials=100)
