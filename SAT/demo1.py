from Solver import *
from SAT import load_cnf
from functools import reduce
from sympy import symbols, expand
from itertools import permutations
from matplotlib import pyplot as plt


def main1():
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19 = symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19")
    var_symbol = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19]
    var_string = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19"]

    top1 = 0
    top2 = 0
    top3 = 0
    top4 = 0
    for k in range(1, 1001):
        variable = {item: 1 for item in var_string}
        tot = {"".join(item): 0 for item in list(permutations(var_string, 3))}
        sot = {"".join(item): 0 for item in list(permutations(var_string, 2))}
        oot = {"".join(item): 0 for item in list(permutations(var_string, 1))}
        coupling_dict = {**tot, **sot, **oot}

        path = f"E:/Files/Code/Python/personal/COP/SAT_Dataset/UF20-91/uf20-0{k}.cnf"
        clause_matrix = load_uf2091(path)

        # 将clause matrix 转为 Ising 能量的形式
        f = 0
        for c in range(clause_matrix.shape[0]):
            g = 1
            for v in range(clause_matrix.shape[1]):
                g *= (1 - clause_matrix[c, v] * var_symbol[v])
            f += g

        # 转为字符串进行处理，去掉空格，第一项系数为正时添加+号
        p = str(expand(f)).replace(" ", "")
        if p[0] != "-" and p[0] != "+":
            p = "+" + p

        # 删除常数项，将减法转成加法
        p = p.replace("+", "+1*").replace("-", "+-1*").split("+")
        p.remove("")
        p = remove_constant(p)

        for n, item in enumerate(p):
            temp = item.split("x")[0].split("*")
            temp.remove("")
            cof = reduce(lambda x, y: int(x) * int(y), temp)  # 获取每项的系数
            factor = "".join(remove_constant(item.split("*")))  # 获取变量
            coupling_dict[factor] = int(cof)  # 给coupling字典赋值

        spin_log, energy_log, time_slot = hnn_solve(spin=variable, coupling=coupling_dict, iteration=40, noise_range=[10, 0])
        # spin_log, energy_log, time_slot = sa_solve(spin=variable, coupling=coupling_dict, iteration=40, temperature=[8, 0.125])
        # print(spin_log)
        # print()
        # print(energy_log)

        solution = np.array(spin_log[-1])
        result = np.multiply(clause_matrix, solution)
        satisfied_clauses = np.count_nonzero(np.any(result == 1, axis=1))
        print(f"UF20-"+f"{k}".zfill(4), "\t", f"Number of Satisfied Clause: {satisfied_clauses}", "\t", f"Time Slot: {time_slot: .2f}s")
        print(energy_log)
        print()
        plt.plot(range(len(energy_log)), energy_log)
        plt.show()
        if k >= 5:
            break
        if satisfied_clauses == 91:
            top1 += 1
            top2 += 1
            top3 += 1
            top4 += 1
        elif satisfied_clauses >= 90:
            top2 += 1
            top3 += 1
            top4 += 1
        elif satisfied_clauses >= 86:
            top3 += 1
            top4 += 1
        elif satisfied_clauses >= 81:
            top4 += 1
    print()
    print(f"Top-1: {top1 / 1000}")
    print(f"Top-2: {top2 / 1000}")
    print(f"Top-3: {top3 / 1000}")
    print(f"Top-4: {top4 / 1000}")


def main2():
    x0, x1, x2, x3, x4, x5, x6, x7 = symbols("x0 x1 x2 x3 x4 x5 x6 x7")
    var_symbol = [x0, x1, x2, x3, x4, x5, x6, x7]
    var_string = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]

    variable = {item: 1 for item in var_string}
    tot = {"".join(item): 0 for item in list(permutations(var_string, 3))}
    sot = {"".join(item): 0 for item in list(permutations(var_string, 2))}
    oot = {"".join(item): 0 for item in list(permutations(var_string, 1))}
    coupling_dict = {**tot, **sot, **oot}

    clause_matrix = np.array([
        [1, 1, -1, 0, 0, 0, 0, 0],
        [0, -1, 0, -1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, -1],
        [-1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, -1, 0, 0],
        [0, 0, -1, 0, 0, 0, -1, 1],
        [1, 0, 0, 0, 1, -1, 0, 0],
        [0, -1, 0, -1, 0, 0, 1, 0],
        [0, 0, 1, 0, -1, 0, 0, 1],
        [-1, 0, 0, 1, 0, 1, 0, 0],
    ])
    # 将clause matrix 转为 Ising 能量的形式
    f = 0
    for c in range(clause_matrix.shape[0]):
        g = 1
        for v in range(clause_matrix.shape[1]):
            g *= (1 - clause_matrix[c, v] * var_symbol[v])
        f += g

    # 转为字符串进行处理，去掉空格，第一项系数为正时添加+号
    p = str(expand(f)).replace(" ", "")
    if p[0] != "-" and p[0] != "+":
        p = "+" + p

    # 删除常数项，将减法转成加法
    p = p.replace("+", "+1*").replace("-", "+-1*").split("+")
    p.remove("")
    p = remove_constant(p)

    for n, item in enumerate(p):
        temp = item.split("x")[0].split("*")
        temp.remove("")
        cof = reduce(lambda x, y: int(x) * int(y), temp)  # 获取每项的系数
        factor = "".join(remove_constant(item.split("*")))  # 获取变量
        coupling_dict[factor] = int(cof)  # 给coupling字典赋值

    # spin_log, energy_log, time_slot = hnn_solve(spin=variable, coupling=coupling_dict, iteration=40, noise_range=[10, 0])
    spin_log, energy_log, time_slot = sa_solve(spin=variable, coupling=coupling_dict, iteration=10, temperature=[4, 0.25])
    solution = np.array(spin_log[-1])
    result = np.multiply(clause_matrix, solution)
    print("Solution:\n", solution)
    print("System Energy:", energy_log[-1])
    print("Number of Satisfied Clause:", np.count_nonzero(np.any(result == 1, axis=1)))
    plt.plot(range(len(energy_log)), energy_log)
    plt.show()


main2()

