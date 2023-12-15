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


# T = 1.5*exp(-0.001*t).*(t<500)+1.5*exp(0.89)*(exp(-0.0028*t)).*(t>500&t<750)+1.5*exp(6.29)*(exp(-0.01*t)).*(t<1000&t>750)


def generate_t(iteration):
    t_array = np.arange(iteration)

    # return np.concatenate((1.5 * np.exp(-0.001 * t_array[t_array < 500]),
    #                        1.5 * np.exp(0.89) * (np.exp(-0.0028 * t_array[500 <= t_array < 750])),
    #                        1.5 * np.exp(6.29) * (np.exp(-0.01 * t_array[t_array >= 750]))
    #                        ), axis=0
    #                       )
    return np.concatenate((1.5 * np.exp(-0.001 * t_array[t_array < 500]),
                           1.5 * np.exp(0.89) * np.exp(-0.0028 * t_array[np.logical_not(np.logical_xor(t_array >= 500, t_array < 750))]),
                           1.5 * np.exp(6.29) * np.exp(-0.01 * t_array[t_array >= 750])
                           ))


def calculate_ising_energy(x, weight):
    return 0.5 * x @ weight @ x.T


def calculate_max_cut_value(x, weight):
    return -0.25 * (np.sum(weight) - x @ weight @ x.T)


class CoherentIsingMachine:
    def __init__(self, coupling, temperature, alpha, epochs, mu, sigma):
        self.coupling = coupling
        self.temperature = temperature
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.epochs = epochs
        self.h = np.zeros(shape=coupling.shape[0])
        self.s = np.zeros(shape=coupling.shape[0])

    def run(self):
        spin_log = sign(self.s)
        ising_energy = [calculate_max_cut_value(sign(self.s), self.coupling)]
        max_cut_value = [calculate_ising_energy(sign(self.s), -self.coupling)]

        for epoch in range(self.epochs):
            noise = np.random.normal(loc=self.mu, scale=self.sigma)
            phi = (self.h + self.coupling @ self.s) / np.sqrt(self.h ** 2 + np.sum(self.coupling ** 2, axis=1)) + noise
            s_ = -np.tanh(phi / self.temperature[epoch])

            # update
            self.s = self.alpha * s_ + (1 - self.alpha) * self.s
            spin_log = np.vstack((spin_log, sign(self.s)))
            ising_energy.append(calculate_ising_energy(sign(self.s), self.coupling))
            max_cut_value.append(calculate_max_cut_value(sign(self.s), -self.coupling))

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
            solver.s = np.zeros(shape=len(solver.s))

    print()
    print("--------------------Result--------------------")
    print(f"Max Cut Value: {np.max(value_list)}, Mean Cut Value: {np.mean(value_list)}")
    print(f"Min Ising Energy: {np.min(energy_list)}, Mean Ising Energy: {np.mean(energy_list)}")
    print()

    plt.figure("CIM Cut Value")
    plt.hist(value_list, bins=10, edgecolor="black")
    plt.title(args.path.split("/")[-1].split(".")[0], fontsize=15)
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

    plt.title(args.path.split("/")[-1].split(".")[0])
    lines = line1 + line2
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter of Adiabatic Simulated Bifurcation')
    parser.add_argument('--epochs', default=800, help='total iteration')
    parser.add_argument('--shape', default=(16, 32), help='shape of spin image')
    parser.add_argument('--temperature', default=1.5, help='temperature')
    parser.add_argument('--hyper_parameter', default=(0.15, 0, 0.15), help='hyper parameter')
    parser.add_argument('--path', default="./CouplingMatrix/Custom_IEEE.npy", help='path of spin coupling')

    args = parser.parse_args()
    # args.path = "./CouplingMatrix/K2000.npy"
    # args.path = "./CouplingMatrix/G22.npy"
    args.path = "./CouplingMatrix/G39.npy"

    coupling_matrix = np.load(args.path)
    t = generate_t(iteration=args.epochs)
    Noise = np.random.normal(loc=args.hyper_parameter[1], scale=args.hyper_parameter[2])

    cim = CoherentIsingMachine(coupling=coupling_matrix,
                               temperature=t,
                               alpha=args.hyper_parameter[0],
                               mu=args.hyper_parameter[0],
                               sigma=args.hyper_parameter[0],
                               epochs=args.epochs
                               )

    # energy_curve(solver=cim)
    energy_hist(solver=cim, trials=100)
