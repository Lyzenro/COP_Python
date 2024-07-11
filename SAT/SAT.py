import os
import re
import time
import copy
import numpy as np
import matplotlib.pyplot as plt


class SAT:
    def __init__(self, literals=3, variables=8, clauses=32):
        self.literals = literals
        self.variables = variables
        self.clauses = clauses

        self.X = np.random.choice([-1, 1], size=variables)
        self.A = np.random.choice([-1, 1], size=(clauses, literals))
        self.A = np.concatenate((self.A, np.zeros((clauses, variables - literals), dtype=np.int32)), axis=1)
        for i in range(clauses):
            np.random.shuffle(self.A[i])

        print("Binary variable:", self.X, "\n")
        print("Matrix:", "\n", self.A, "\n")

    def local_search(self, epochs=10, searches=100):
        start_time = time.time()
        y = self.A * self.X
        satisfiability = np.any(y == 1, axis=1)
        success = np.array([np.count_nonzero(satisfiability)])

        for epoch in range(epochs):
            # tabu = None
            for step in range(searches):
                # idx = 0
                cnt = np.array([])

                # 领域搜索
                for n in range(self.variables):
                    v = copy.deepcopy(self.X)
                    v[n] = -v[n]
                    y = self.A * v
                    satisfiability = np.any(y == 1, axis=1)
                    cnt = np.append(cnt, len(satisfiability) - np.count_nonzero(satisfiability)).astype(np.int32)
                    # print(cnt)
                # print()

                # update
                indexes = np.where(cnt == np.min(cnt))[0]
                idx = np.random.choice(indexes)
                # if len(indexes) == 1:
                #     if tabu is None or idx != tabu:
                #         idx = indexes[0]
                #         tabu = idx
                #
                #     elif idx == tabu:
                #         idx = np.random.choice(np.arange(self.variables))
                #         tabu = idx
                #
                # else:
                #     while 1:
                #         idx = np.random.choice(indexes)
                #         # print(idx, tabu)
                #         if tabu is None or idx != tabu:
                #             # print(idx, tabu)
                #             tabu = idx
                #             break

                # print("min unsatisfied clause index:", indexes)
                # print("selected index", idx)

                self.X[idx] = -self.X[idx]

                # compute success rate
                y = self.A * self.X
                satisfiability = np.any(y == 1, axis=1)
                # print("satisfied clause:", np.count_nonzero(satisfiability))
                success = np.append(success, np.count_nonzero(satisfiability))

                # 如果clause全部满足，提前终止算法
                if np.count_nonzero(satisfiability) == self.clauses:
                    stop_time = time.time()
                    return self.X, success, stop_time - start_time
            # 跳出局部最优
            self.X = np.random.choice([-1, 1], size=self.variables)
        stop_time = time.time()
        return self.X, success, stop_time - start_time

    def simulated_annealing(self, epochs=100):
        previous_solution = copy.deepcopy(self.X)
        y = self.A * previous_solution
        satisfiability = np.any(y == 1, axis=1)
        previous_satisfied = np.count_nonzero(satisfiability)

        satisfied_log = []
        t = np.linspace(start=100, stop=1, num=epochs)
        for step in range(epochs):
            idx = np.random.choice(np.arange(self.variables))
            current_solution = copy.deepcopy(previous_solution)
            current_solution[idx] = -current_solution[idx]

            # 计算当前满足的字句数
            y = self.A * current_solution
            satisfiability = np.any(y == 1, axis=1)
            current_satisfied = np.count_nonzero(satisfiability)
            r = np.random.uniform(low=0, high=1)
            if current_satisfied >= previous_satisfied:
                previous_solution = current_solution
                previous_satisfied = current_satisfied

            elif np.exp(-current_satisfied / t[step]) >= r:
                previous_solution = current_solution
                previous_satisfied = current_satisfied

            else:
                pass

            satisfied_log.append(previous_satisfied)
            if previous_satisfied == self.clauses:
                break
        return previous_solution, satisfied_log

    def walk_solve(self, epochs):
        start_time = time.time()
        step = 1
        success_log = []
        while step <= epochs:
            step += 1
            y = self.A * self.X
            result = np.any(y == 1, axis=1)

            # index of satisfied and unsatisfied clause
            satisfied_clause = np.where(result == 1)[0]
            unsatisfied_clause = np.where(result == 0)[0]
            # print("satisfied:", len(satisfied_clause), "\tunsatisfied:", len(unsatisfied_clause), "\n")

            # termination conditions
            success_log.append(len(satisfied_clause))
            if len(satisfied_clause) == self.clauses:
                stop_time = time.time()
                return self.X, success_log, stop_time - start_time

            # get index of an unsatisfied clause randomly and variable of it
            selected_clause_index = np.random.choice(unsatisfied_clause)
            var_in_selected_clause = np.where(self.A[selected_clause_index, :] != 0)[0]

            best_break = [None, None]
            for n, idx in enumerate(var_in_selected_clause):
                # Flip variable
                try_solution = copy.deepcopy(self.X)
                try_solution[idx] = -try_solution[idx]

                # compute current satisfied clause
                y = self.A * try_solution
                result = np.any(y == 1, axis=1)

                # store the best break and corresponding variable
                break_clause = np.count_nonzero(result[satisfied_clause] == 0)
                if best_break[0] is None or best_break[0] > break_clause:
                    best_break = [break_clause, idx]

                if break_clause == 0:
                    self.X[idx] = -self.X[idx]
                    break
                else:
                    if n == self.literals - 1:
                        flag = np.random.choice([0, 1])
                        if flag == 0:
                            idx = np.random.choice(var_in_selected_clause)
                            self.X[idx] = -self.X[idx]
                        else:
                            idx = best_break[1]
                            self.X[idx] = -self.X[idx]
                    else:
                        continue
        stop_time = time.time()
        return self.X, success_log, stop_time - start_time

    def improved_local_search(self, epochs=10, searches=100):
        # set initial solution
        for n in range(self.variables):
            positive = np.count_nonzero(self.A[:, n] == 1)
            negative = np.count_nonzero(self.A[:, n] == -1)
            if positive > negative:
                self.X[n] = 1

            elif positive < negative:
                self.X[n] = -1

        success_log = []
        start_time = time.time()

        # compute satisfied clause
        y = self.A * self.X
        result = np.any(y == 1, axis=1).astype(np.int32)
        satisfied_clause = np.count_nonzero(result)
        success_log.append(satisfied_clause)

        # termination conditions
        if satisfied_clause == self.clauses:
            stop_time = time.time()
            return self.X, success_log, stop_time - start_time

        # iterative search
        for epoch in range(epochs):
            for step in range(searches):
                score = []
                # 领域搜索
                for n in range(self.variables):
                    v = copy.deepcopy(self.X)
                    v[n] = -v[n]
                    y = self.A * v
                    try_result = np.any(y == 1, axis=1).astype(np.int32)
                    # termination conditions
                    if np.count_nonzero(try_result) == self.clauses:
                        self.X[n] = -self.X[n]
                        stop_time = time.time()
                        success_log.append(self.clauses)
                        return self.X, success_log, stop_time - start_time

                    make_clause = np.count_nonzero(try_result - result == 1)
                    break_clause = np.count_nonzero(try_result - result == -1)
                    score.append(make_clause - break_clause)

                # Flip
                indexes = np.where(score == np.max(score))[0]
                idx = np.random.choice(indexes)
                self.X[idx] = -self.X[idx]

                # compute success rate
                y = self.A * self.X
                result = np.any(y == 1, axis=1).astype(np.int32)
                satisfied_clause = np.count_nonzero(result)
                success_log.append(satisfied_clause)

                # 如果clause全部满足，提前终止算法
                if satisfied_clause == self.clauses:
                    stop_time = time.time()
                    return self.X, success_log, stop_time - start_time
            # 跳出局部最优
            if epoch < epochs - 1:
                self.X = np.random.choice([-1, 1], size=self.variables)
        stop_time = time.time()
        return self.X, success_log, stop_time - start_time

    def probabilistic_search(self, epochs):
        step = 1
        success_log = []
        start_time = time.time()

        while step <= epochs:
            t = 1
            self.X = np.random.choice([-1, 1], size=self.variables)

            # compute satisfied clause
            y = self.A * self.X
            result = np.any(y == 1, axis=1).astype(np.int32)
            satisfied_clause = np.count_nonzero(result)
            success_log.append(satisfied_clause)

            # termination conditions
            if satisfied_clause == self.clauses:
                stop_time = time.time()
                return self.X, success_log, stop_time - start_time

            while t <= 3 * self.variables:
                # update
                unsatisfied_clause_index = np.random.choice(np.where(result == 0)[0])  # 随机选择一个不满足的字句，取其索引值
                unsatisfied_clause_variable_index = np.where(self.A[unsatisfied_clause_index, :] != 0)[0]  # 获得该字句内变量的索引值
                variable_index = np.random.choice(unsatisfied_clause_variable_index)  # 随机选择其中一个变量
                self.X[variable_index] = -self.X[variable_index]  # Flip

                # compute satisfied clause
                y = self.A * self.X
                result = np.any(y == 1, axis=1).astype(np.int32)
                satisfied_clause = np.count_nonzero(result)
                success_log.append(satisfied_clause)

                # termination conditions
                if satisfied_clause == self.clauses:
                    stop_time = time.time()
                    return self.X, success_log, stop_time - start_time

                t += 1
            step += 1
        stop_time = time.time()
        return self.X, success_log, stop_time - start_time

    def modified_local_search(self, epochs=10, searches=100):
        # set initial solution
        for n in range(self.variables):
            positive = np.count_nonzero(self.A[:, n] == 1)
            negative = np.count_nonzero(self.A[:, n] == -1)
            if positive > negative:
                self.X[n] = 1

            elif positive < negative:
                self.X[n] = -1

        success_log = []
        start_time = time.time()

        # compute satisfied clause
        y = self.A * self.X
        result = np.any(y == 1, axis=1).astype(np.int32)
        satisfied_clause = np.count_nonzero(result)
        success_log.append(satisfied_clause)

        # termination conditions
        if satisfied_clause == self.clauses:
            stop_time = time.time()
            return self.X, success_log, stop_time - start_time

        # iterative search
        for epoch in range(epochs):
            for step in range(searches):
                # score = []
                break_list = []
                unsatisfied_clause_index = np.random.choice(np.where(result == 0)[0])  # 随机选择一个不满足的字句，取其索引值
                unsatisfied_clause_variable_index = np.where(self.A[unsatisfied_clause_index, :] != 0)[0]  # 获得该字句内变量的索引值

                for idx in unsatisfied_clause_variable_index:
                    v = copy.deepcopy(self.X)
                    v[idx] = -v[idx]
                    y = self.A * v
                    try_result = np.any(y == 1, axis=1).astype(np.int32)
                    # make_clause = np.count_nonzero(try_result - result == 1)
                    break_clause = np.count_nonzero(try_result - result == -1)
                    # score.append(make_clause - break_clause)
                    break_list.append(break_clause)

                # Flip
                min_idx = np.random.choice(np.where(break_list == np.min(break_list))[0])
                idx = unsatisfied_clause_variable_index[min_idx]
                self.X[idx] = -self.X[idx]

                # Compute success rate
                y = self.A * self.X
                result = np.any(y == 1, axis=1).astype(np.int32)
                satisfied_clause = np.count_nonzero(result)
                success_log.append(satisfied_clause)

                # 如果clause全部满足，提前终止算法
                if satisfied_clause == self.clauses:
                    stop_time = time.time()
                    return self.X, success_log, stop_time - start_time

            # 跳出局部最优
            if epoch < epochs - 1:
                self.X = np.random.choice([-1, 1], size=self.variables)
                y = self.A * self.X
                result = np.any(y == 1, axis=1).astype(np.int32)
                satisfied_clause = np.count_nonzero(result)
                success_log.append(satisfied_clause)

                # 如果clause全部满足，提前终止算法
                if satisfied_clause == self.clauses:
                    stop_time = time.time()
                    return self.X, success_log, stop_time - start_time

        stop_time = time.time()
        return self.X, success_log, stop_time - start_time


def load_cnf(file_path, scale=(91, 20)):
    matrix = np.zeros(shape=scale, dtype=np.int32)
    with open(file_path, mode="r") as f:
        context = f.read().replace("0\n", "").replace("\n ", "")

        data = re.findall(f"{scale[0]} (.*?) %", context)[0].split(" ")
        data = np.array(list(map(int, data)))

        signs = np.sign(data).reshape(scale[0], 3)
        indexes = np.abs(data).reshape(scale[0], 3)

        for j in range(scale[0]):
            for sign, index in (zip(signs[j, :], indexes[j, :])):
                matrix[j, index - 1] = sign
    return matrix


if __name__ == '__main__':
    literal = 3
    variable = 20
    clause = 91

    sat = SAT(literals=literal, variables=variable, clauses=clause)
    """
    path = f"D:/Files/LiRenLong/Code/python_project/SAT_Dataset/UF20-91/uf20-0{1}.cnf"
    sat.A = load_uf2091(path)
    # solution, Success, tts = sat.walk_solve(epochs=100)
    # solution, Success = sat.local_search(epochs=10, searches=100)
    # solution, Success = sat.simulated_annealing(epochs=1000)
    solution, Success, tts = sat.improved_local_search_(epochs=10, searches=10)
    print(solution, Success, tts)
    """

    """
    if max(Success) == clause:
        root = f"D:/Files/LiRenLong/Code/python_project/SAT_Dataset/{literal}L-{variable}V-{clause}C"
        file_num = len(os.listdir(root))
        path = os.path.join(root, f"SAT_{literal}L-{variable}V-{clause}C-{file_num + 1}.txt")
        np.savetxt(path, sat.A, fmt="%d")
    """

    """
    plt.plot(np.arange(len(Success)), Success)
    plt.xlabel("epoch")
    plt.ylabel("Satisfiability")
    plt.title(f"{literal}L-{variable}V-{clause}C")
    plt.grid()
    plt.show()
    """

    # for m in range(10):
    tot_time = 0
    satisfied = np.array([])
    unsatisfied = np.array([])
    for k in range(1, 101):
        if k % 100 == 0:
            print(f"Solving uf{variable}-0{k}...")
        path = f"../Dataset/SAT/UF{variable}-{clause}/uf{variable}-0{k}.cnf"
        sat.A = load_cnf(path, scale=(clause, variable))
        sat.X = np.random.choice([-1, 1], size=variable)

        # solution, Success, tts = sat.walk_solve(epochs=100)
        # solution, Success, tts = sat.probabilistic_search(epochs=50)
        # solution, Success, tts = sat.local_search(epochs=30, searches=10)
        # solution, Success, tts = sat.improved_local_search(epochs=10, searches=20)
        solution, Success, tts = sat.modified_local_search(epochs=20, searches=20)

        tot_time += tts
        if max(Success) == clause:
            satisfied = np.append(satisfied, k)
        else:
            unsatisfied = np.append(unsatisfied, k)
    print(f"Accuracy: {len(satisfied) / 1}%, Average TTS: {tot_time / 100 : .5f}s")
    print(unsatisfied)
    # np.savetxt("./satisfied.txt", satisfied, fmt="%d")
    # np.savetxt("./unsatisfied.txt", unsatisfied, fmt="%d")
