import argparse
import numpy as np
import matplotlib.pyplot as plt


def sign(x: int | list | np.ndarray):
    if isinstance(x, int):
        return 1 if x == 0 else int(np.sign(x))
    else:
        result = np.sign(x)
        result[result == 0] = 1
        if isinstance(x, list):
            return result.tolist()
        else:
            return result


def calculate_ising_energy(x, weight):
    return -0.5 * x @ weight @ x.T


def calculate_max_cut_value(x, weight):
    return 0.25 * (np.sum(weight) - x @ weight @ x.T)


class AdiabaticSimulatedBifurcation:
    def __init__(self, position, momentum, epochs, coupling, hyper_parameter, plot_spin_fig=True):
        self.x = position
        self.y = momentum
        self.epochs = epochs
        self.coupling = coupling

        # Hyper parameter
        self.a0 = hyper_parameter[0]
        self.delta_t = hyper_parameter[1]
        self.at = np.linspace(start=0, stop=self.a0, num=epochs + 1)

        n = len(position)
        sigma = np.sqrt(np.sum(coupling ** 2) / (n * n - n))
        self.c0 = 0.5 / (sigma * np.sqrt(n))

        if plot_spin_fig:
            self.fig = plt.figure("spin", figsize=(6, 6))
            plt.tight_layout()
        else:
            self.fig = None

    def run(self):
        fig_idx = 1
        spin_log = sign(self.x)
        ising_energy = [calculate_max_cut_value(spin_log, self.coupling)]
        max_cut_value = [calculate_ising_energy(spin_log, -self.coupling)]

        if self.fig is not None:
            ax = self.fig.add_subplot(5, int(len(self.at) // 5) + 1, fig_idx)
            ax.imshow(sign(self.x).reshape(args.shape), cmap="binary")
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        for t, _ in enumerate(self.at):

            self.x += self.a0 * self.y * self.delta_t
            self.y += (-(self.x ** 2 + self.a0 - self.at[t]) * self.x + self.c0 * self.coupling @ self.x) * self.delta_t

            spin = sign(self.x)
            spin_log = np.vstack((spin_log, spin))
            ising_energy.append(calculate_ising_energy(spin, self.coupling))
            max_cut_value.append(calculate_max_cut_value(spin, -self.coupling))

            if self.fig is not None:
                fig_idx += 1
                ax = self.fig.add_subplot(5, int(len(self.at) // 5) + 1, fig_idx)
                ax.imshow(sign(self.x).reshape(args.shape), cmap="binary")
                ax.set_xticks([], [])
                ax.set_yticks([], [])

        return spin_log, max_cut_value, ising_energy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter of Adiabatic Simulated Bifurcation')
    parser.add_argument('--epochs', default=1000, help='total iteration')
    parser.add_argument('--shape', default=(16, 32), help='shape of spin image')
    parser.add_argument('--hyper_parameter', default=(1, 0.5), help='hyper parameter')
    parser.add_argument('--path', default="./CouplingMatrix/Custom_IEEE.npy", help='path of spin coupling')

    args = parser.parse_args()
    args.path = "./CouplingMatrix/K2000.npy"

    coupling_matrix = np.load(args.path)
    x_position = np.random.uniform(low=-1, high=1, size=coupling_matrix.shape[0])
    y_momentum = np.random.uniform(low=-1, high=1, size=coupling_matrix.shape[0])

    aSB = AdiabaticSimulatedBifurcation(position=x_position,
                                        momentum=y_momentum,
                                        epochs=args.epochs,
                                        coupling=-coupling_matrix,
                                        hyper_parameter=args.hyper_parameter,
                                        plot_spin_fig=False
                                        )
    spins, value, energy = aSB.run()

    print("solution:\n", spins[-1])
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
