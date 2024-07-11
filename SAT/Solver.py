import time
import numpy as np


def remove_constant(a: list[str]):
    b = []
    for term in a:
        if "x" in term:
            b.append(term)
    return b


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


# def system_energy(spin, coupling):
#     energy = 0
#     num = len(spin.keys())
#     for i in range(num):
#         energy += coupling[f"x{i}"] * spin[f"x{i}"]
#         for j in range(i + 1, num):
#             energy += coupling[f"x{i}x{j}"] * spin[f"x{i}"] * spin[f"x{j}"]
#             for k in range(j + 1, num):
#                 energy += coupling[f"x{i}x{j}x{k}"] * spin[f"x{i}"] * spin[f"x{j}"] * spin[f"x{k}"]
#     return energy


# def local_energy(idx, spin, coupling):
#     num = len(spin.keys())
#     energy = -coupling[f"x{idx}"]
#     for j in range(num):
#         if j != idx:
#             a, b = tuple(np.sort([idx, j]).tolist())
#             energy -= coupling[f"x{a}x{b}"] * spin[f"x{j}"]
#             for k in range(j + 1, num):
#                 if k != idx:
#                     a, b, c = tuple(np.sort([idx, j, k]).tolist())
#                     energy -= coupling[f"x{a}x{b}x{c}"] * spin[f"x{j}"] * spin[f"x{k}"]
#     return energy


def system_energy(spin, coupling):
    energy = 0
    num = len(spin.keys())
    for i in range(num):
        energy += coupling[f"x{i}"] * spin[f"x{i}"]
        for j in range(num):
            if j != i:
                energy += coupling[f"x{i}x{j}"] * spin[f"x{i}"] * spin[f"x{j}"]
                for k in range(num):
                    if k != i and k != j:
                        energy += coupling[f"x{i}x{j}x{k}"] * spin[f"x{i}"] * spin[f"x{j}"] * spin[f"x{k}"]
    return energy


def local_energy(idx, spin, coupling):
    num = len(spin.keys())
    energy = -coupling[f"x{idx}"]
    for j in range(num):
        if j != idx:
            energy -= coupling[f"x{idx}x{j}"] * spin[f"x{j}"]
            energy -= coupling[f"x{j}x{idx}"] * spin[f"x{j}"]
            for k in range(j + 1, num):
                if k != idx and k != j:
                    energy -= coupling[f"x{idx}x{j}x{k}"] * spin[f"x{j}"] * spin[f"x{k}"]
                    energy -= coupling[f"x{idx}x{k}x{j}"] * spin[f"x{j}"] * spin[f"x{k}"]
                    energy -= coupling[f"x{j}x{idx}x{k}"] * spin[f"x{j}"] * spin[f"x{k}"]
                    energy -= coupling[f"x{j}x{k}x{idx}"] * spin[f"x{j}"] * spin[f"x{k}"]
                    energy -= coupling[f"x{k}x{idx}x{j}"] * spin[f"x{j}"] * spin[f"x{k}"]
                    energy -= coupling[f"x{k}x{j}x{idx}"] * spin[f"x{j}"] * spin[f"x{k}"]
    return energy


def generate_noise(start, stop, num):
    if num < abs(stop - start) + 1:
        raise ValueError("num < |stop - start| + 1")
    else:
        rank = abs(stop - start) + 1
        repeat_size = num // rank
        if stop < start:
            noise = np.array(stop).repeat(num)
            for n, l in enumerate(range(start, stop - 1, -1)):
                noise[n * repeat_size: (n + 1) * repeat_size] = np.random.choice(range(-l, l + 1), size=repeat_size)
        else:
            noise = np.zeros(shape=num, dtype=np.int32)
        return noise


def generate_temperature(start, stop, num):
    if num < max(start / stop, stop / start):
        raise ValueError("Set a larger parameter num!")
    else:
        rank = int(np.log2(max(start / stop, stop / start)) + 1)
        repeat_size = num // rank
        temperature = np.array([stop]).repeat(num)

        if stop >= start:
            temperature[:repeat_size * rank] = np.array([start * 2 ** r for r in range(rank)]).repeat(repeat_size)
        else:
            temperature[:repeat_size * rank] = np.array([start / 2 ** r for r in range(rank)]).repeat(repeat_size)
        return temperature


def hnn_solve(spin, coupling, iteration, noise_range):
    spin_key = list(spin.keys())
    spin_log = [list(spin.values())]
    energy_log = [system_energy(spin, coupling)]
    noise = generate_noise(start=noise_range[0], stop=noise_range[1], num=iteration)
    start_time = time.perf_counter()

    for t in range(iteration):
        for n, v in enumerate(spin.keys()):
            delta_energy = 2 * spin[spin_key[n]] * local_energy(n, spin, coupling)
            spin[v] = spin[v] * sign(delta_energy + noise[t])

        spin_log.append(list(spin.values()))
        energy_log.append(system_energy(spin, coupling))
    stop_time = time.perf_counter()
    return spin_log, energy_log, stop_time - start_time


def sa_solve(spin, coupling, iteration, temperature):
    spin_key = list(spin.keys())
    spin_log = [list(spin.values())]
    energy_log = [system_energy(spin, coupling)]
    temperature = generate_temperature(start=temperature[0], stop=temperature[1], num=iteration * len(spin.keys()))

    n = 0
    start_time = time.perf_counter()
    for t in temperature:

        delta_energy = 2 * spin[spin_key[n]] * local_energy(n, spin, coupling)
        if delta_energy < 0 or np.exp(- delta_energy / t) >= np.random.uniform():
            spin[spin_key[n]] = - spin[spin_key[n]]

        if n == len(spin.keys()) - 1:
            n = 0
            spin_log.append(list(spin.values()))
            energy_log.append(system_energy(spin, coupling))
        else:
            n += 1
    stop_time = time.perf_counter()
    return spin_log, energy_log, stop_time - start_time
