import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: int | float | np.integer | np.floating | list | np.ndarray):

    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.exp(x) / (1 + np.exp(x))

    elif isinstance(x, list):
        result = np.array(x)
        shape = result.shape
        return np.fromiter(map(lambda inx: np.exp(inx) / (1 + np.exp(inx)), result.flatten()), dtype=np.float32).reshape(shape).tolist()

    elif isinstance(x, np.ndarray):
        shape = x.shape
        return np.fromiter(map(lambda inx: np.exp(inx) / (1 + np.exp(inx)), x.flatten()), dtype=np.float32).reshape(shape)

    else:
        raise TypeError("Please enter the correct format !!!")


def calculate_ising_energy(x, weight):
    return -0.5 * x @ weight @ x.T


def calculate_max_cut_value(x, weight):
    return 0.25 * (np.sum(weight) - x @ weight @ x.T)


class StochasticCellularAutomata:
    def __init__(self, spin, coupling, mc_steps, temperature: [int, int], penalty_value: [int, int], plot_spin_fig=True):
        self.spin_num = len(spin)
        self.spin_tau = spin.copy()
        self.spin_sigma = spin.copy()
        self.coupling = coupling
        self.Steps = mc_steps
        self.T_init, self.T_fin = temperature[0], temperature[1]
        self.q_init, self.q_fin = penalty_value[0], penalty_value[1]
        self.r_q = (self.q_fin / self.q_init) ** (1 / (self.Steps - 1))
        self.r_T = (self.T_fin / self.T_init) ** (1 / (self.Steps - 1))

        if plot_spin_fig:
            self.fig = plt.figure("spin", figsize=(6, 6))
            plt.tight_layout()
        else:
            self.fig = None

    def run(self):
        fig_idx = 1
        t = self.T_init
        q = self.q_init
        spin_log = self.spin_sigma
        ising_energy = [calculate_max_cut_value(self.spin_sigma, self.coupling)]
        max_cut_value = [calculate_ising_energy(self.spin_sigma, -self.coupling)]

        if self.fig is not None:
            ax = self.fig.add_subplot(5, int(self.Steps // 5) + 1, fig_idx)
            ax.imshow(self.spin_sigma.reshape(args.shape), cmap="binary")
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        for s in range(self.Steps):
            # calculate local energy
            local_energy = self.coupling @ self.spin_sigma

            # calculate flip probability
            p = sigmoid((0.5 * local_energy * self.spin_sigma + q) / t)
            index = np.where(p < np.random.uniform())[0]
            alpha = np.ones(shape=self.spin_num, dtype=np.int32)
            alpha[index] = - alpha[index]

            # spin update
            self.spin_tau = alpha * self.spin_sigma
            self.spin_sigma = self.spin_tau.copy()

            spin_log = np.vstack((spin_log, self.spin_sigma))
            ising_energy.append(calculate_ising_energy(self.spin_sigma, self.coupling))
            max_cut_value.append(calculate_max_cut_value(self.spin_sigma, -self.coupling))

            # update hyper parameter
            q = q * self.r_q
            t = t * self.r_T

            if self.fig is not None:
                fig_idx += 1
                ax = self.fig.add_subplot(5, int(self.Steps // 5) + 1, fig_idx)
                ax.imshow(self.spin_sigma.reshape(args.shape), cmap="binary")
                ax.set_xticks([], [])
                ax.set_yticks([], [])

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
            solver.spin_tau = np.random.choice([-1, 1], size=len(solver.spin_tau))
            solver.spin_sigma = solver.spin_tau.copy()

    print()
    print("--------------------Result--------------------")
    print(f"Max Cut Value: {np.max(value_list)}, Mean Cut Value: {np.mean(value_list)}")
    print(f"Min Ising Energy: {np.min(energy_list)}, Mean Ising Energy: {np.mean(energy_list)}")
    print()

    plt.figure("SCA Cut Value")
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
    parser.add_argument('--penalty_value', default=[4, 4], help='level of noise')
    parser.add_argument('--temperature', default=[40, 5], help='temperature')
    parser.add_argument('--MC_steps', default=1000, help='total iteration')
    parser.add_argument('--shape', default=(16, 32), help='image shape')

    args = parser.parse_args()
    # path = "./CouplingMatrix/Custom_IEEE.npy"
    path = "./CouplingMatrix/K2000.npy"
    coupling_matrix = np.load(path)
    spins = np.random.choice([-1, 1], size=coupling_matrix.shape[0])

    sca = StochasticCellularAutomata(spin=spins,
                                     coupling=-coupling_matrix,
                                     mc_steps=args.MC_steps,
                                     temperature=args.temperature,
                                     penalty_value=args.penalty_value,
                                     plot_spin_fig=False
                                     )
    energy_curve(solver=sca)
    # energy_hist(solver=sca, trials=100)
