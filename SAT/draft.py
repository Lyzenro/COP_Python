import numpy as np
from SAT import load_uf2091
from matplotlib import pyplot as plt


def solve(solution, problem):
    x = solution
    matrix = problem.copy()
    matrix[matrix == -1] = 0
    index = [i for i in range(problem.shape[0])]

    # for n, data in enumerate(x):
    #     if np.any(matrix[index, n]):
    #         idx = np.where(matrix[:, n] == 1)[0]
    #         for item in idx:
    #             if item in index:
    #                 index.remove(item)
    #     else:
    #         x[n] = 0

    x_index = [i for i in range(problem.shape[1])]
    while x_index:
        n = np.random.choice(x_index)
        if np.any(matrix[index, n]):
            idx = np.where(matrix[:, n] == 1)[0]
            for item in idx:
                if item in index:
                    index.remove(item)
        else:
            x[n] = 0
        x_index.remove(n)

    solution = x.copy()
    solution[solution == 0] = -1
    y = problem * solution
    result = np.any(y == 1, axis=1).astype(np.int32)
    satisfied_clause = np.count_nonzero(result)

    return solution, satisfied_clause


def search(solution, problem):
    clause_index = list(range(problem.shape[0]))

    # for n, data in enumerate(x):
    #     if np.any(matrix[index, n]):
    #         idx = np.where(matrix[:, n] == 1)[0]
    #         for item in idx:
    #             if item in index:
    #                 index.remove(item)
    #     else:
    #         x[n] = 0

    variable_index = list(range(problem.shape[1]))
    while variable_index:
        n = np.random.choice(variable_index)
        positive_idx = np.where(problem[:, n] == 1)[0]
        negative_idx = np.where(problem[:, n] == -1)[0]
        if len(positive_idx) < len(negative_idx):
            solution[n] = -1

        variable_index.remove(n)

    y = problem * solution
    result = np.any(y == 1, axis=1).astype(np.int32)
    satisfied_clause = np.count_nonzero(result)

    return solution, satisfied_clause


def main1():
    success = 0
    satisfied_clause_index = []
    for k in range(1000):

        path = f"../SAT_Dataset/UF20-91/uf20-0{k + 1}.cnf"
        x, satisfied_clause = solve(solution=np.ones(shape=20), problem=load_uf2091(path))

        if satisfied_clause == 91:
            success += 1
            satisfied_clause_index.append(k + 1)

    print(success)
    print(satisfied_clause_index)


def main2():
    for i in range(1, 11):
        path = f"../SAT_Dataset/3L-8V-8C/SAT_3L-8V-8C-{i}.txt"
        x, satisfied_clause = solve(solution=np.ones(shape=8), problem=np.loadtxt(path, dtype=np.int32))
        print(f"Solution: {x}")
        print(f"Number of satisfied clause: {satisfied_clause}")
        print()


def main3():
    path = f"../SAT_Dataset/UF20-91/uf20-01.cnf"
    satisfied_clause = []
    for i in range(10):
        satisfied_clause.append(search(solution=np.ones(shape=20), problem=load_uf2091(path))[1])
    plt.plot(range(len(satisfied_clause)), satisfied_clause)
    plt.show()


main3()
