import re
import time
import numpy as np


class DPLL(object):
    def __init__(self, path, scale=(91, 20)):
        self.solution = []
        self.random_solution = np.zeros(scale[1], dtype=np.int32)
        self.random_idx = np.arange(scale[1])
        np.random.shuffle(self.random_idx)
        self.problem = np.zeros(shape=scale, dtype=np.int32)
        with open(path, mode="r") as f:
            context = f.read().replace("0\n", "").replace("\n ", "")
            data = re.findall(f"{scale[1]}\s+{scale[0]} (.*?) %", context)[0].split(" ")
            data = np.array(list(map(int, data)))

            signs = np.sign(data).reshape(scale[0], 3)
            indexes = np.abs(data).reshape(scale[0], 3) - 1

            for j in range(scale[0]):
                for sign, index in (zip(signs[j, :], indexes[j, :])):
                    self.problem[j, index] = sign
        self.matrix = self.problem.copy()
        self.update_idx = np.argsort(np.sum(np.abs(self.problem), axis=0))[::-1]

    def check(self, idx, value):
        solution = self.solution + [value]
        solution = np.pad(solution, (0, self.problem.shape[1] - idx - 1))

        if np.any(np.sum(self.problem * solution, axis=1) == -3):
            return False
        else:
            return True

    def solve(self, idx=0):
        for value in [1, -1]:
            if self.check(idx, value):
                self.solution.append(value)

                if idx == self.problem.shape[1] - 1:
                    return True

                res = self.solve(idx+1)
                if res:
                    return True
                else:
                    self.solution.pop()
        return False

    def count(self):
        if len(self.solution) == self.problem.shape[1]:
            y = self.matrix * self.solution
            result = np.any(y == 1, axis=1).astype(np.int32)
            satisfied_clause = np.count_nonzero(result)
            # print(f"满足的字句数：{satisfied_clause}")
            return satisfied_clause
        else:
            return 0

    def random_check(self, idx, value):
        solution = self.random_solution.copy()
        # solution[self.random_idx[idx]] = value
        solution[self.update_idx[idx]] = value

        if np.any(np.sum(self.problem * solution, axis=1) == -3):
            return False
        else:
            return True

    def random_solve(self, idx=0):
        for value in [1, -1]:
            if self.random_check(idx, value):
                # self.random_solution[self.random_idx[idx]] = value
                self.random_solution[self.update_idx[idx]] = value

                if idx == self.problem.shape[1] - 1:
                    return True

                res = self.random_solve(idx+1)
                if res:
                    return True
                else:
                    # self.random_solution[self.random_idx[idx]] = 0
                    self.random_solution[self.update_idx[idx]] = 0

        return False

    def random_count(self):
        y = self.matrix * self.random_solution
        result = np.any(y == 1, axis=1).astype(np.int32)
        satisfied_clause = np.count_nonzero(result)
        # print(f"满足的字句数：{satisfied_clause}")
        return satisfied_clause


if __name__ == '__main__':
    # file_path = "./3L-8V-8C.txt"
    problem_scale = (91, 20)
    success = 0
    cost_time = 0
    for i in range(1, 1001):
        if i % 100 == 0:
            print(f"Solving uf{problem_scale[1]}-0{i}...")
        file_path = f"../Dataset/SAT/UF20-91/uf20-0{i}.cnf"
        dpll = DPLL(path=file_path, scale=problem_scale)
        start_time = time.time()
        # dpll.solve(idx=0)
        dpll.random_solve(idx=0)
        stop_time = time.time()
        cost_time += stop_time - start_time
        # if dpll.count() == problem_scale[0]:
        #     success += 1
        if dpll.random_count() == problem_scale[0]:
            success += 1

    print()
    print(f"solvability: {success / 10}%, TTS: {cost_time / 1000 : .5f}s")
