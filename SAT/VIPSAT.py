import re
import time
import random
import numpy as np

np.set_printoptions(threshold=100000)
# np.random.seed(1)


def load_cnf(path, scale=(91, 20)):
    matrix = np.zeros(shape=scale, dtype=np.int32)
    with open(path, mode="r") as f:
        context = f.read().replace("0\n", "").replace("\n ", "")
        data = re.findall(f"{scale[0]} (.*?) %", context)[0].split(" ")
        data = np.array(list(map(int, data)))

        Signs = np.sign(data).reshape(scale[0], 3)
        Indexes = np.abs(data).reshape(scale[0], 3) - 1

        for j in range(scale[0]):
            for sign, index in (zip(Signs[j, :], Indexes[j, :])):
                matrix[j, index] = sign
    return Signs, Indexes, matrix


def solver(path, scale=(91, 20), iteration=100):

    var_value = np.random.choice([-1, 1], size=scale[1])
    # var_value = np.ones(shape=scale[1], dtype=np.int32)
    # var_value = -np.ones(shape=scale[1], dtype=np.int32)
    signs, indexes, clause_matrix = load_cnf(path, scale=scale)
    # print(clause_matrix)
    # print()

    for epoch in range(iteration):
        log = []
        flag = False
        for n in range(scale[1]):
            row, col = np.where(indexes == n)  # 变量Vi所在的字句，row指定字句索引，col指定变量在字句中的索引
            s_signs = signs[row, col]  # 变量所在字句中Vi的极性

            c_signs = []  # 存相关变量极性
            c_indexes = []  # 存相关变量索引
            for r, c in zip(row, col):  # 变量Vi所在字句的其他两个变量Vj，Vk以及它们在字句中的形式
                c_signs.append(np.delete(signs[r, :], c))
                c_indexes.append(np.delete(indexes[r, :], c))
            c_signs = np.array(c_signs)
            c_indexes = np.array(c_indexes)

            # print(f"变量V{n}所在的字句索引为:{row}\n")
            # print(f"相关变量为:\n{c_indexes}\n")
            # print(f"相关变量极性为：\n{c_signs}\n")
            # print()

            cnt = 0
            for s_sign, c_sign, c_index in zip(s_signs, c_signs, c_indexes):
                # print(f"变量V{n}的极性：{s_sign}, 相关变量的索引：{c_index}， 相关变量的极性：{c_sign}, 相关变量的值：{var_value[c_index]}, XNOR结果：{var_value[c_index] * c_sign}")
                if 1 not in var_value[c_index] * c_sign:  # 相关变量是否能满足字句
                    cnt += s_sign

            # 变量更新
            if cnt > 0:
                var_value[n] = 1
            elif cnt < 0:
                var_value[n] = -1
            else:
                var_value[n] = -var_value[n]

            # 计算满足字句数
            y = clause_matrix * var_value
            result = np.any(y == 1, axis=1).astype(np.int32)
            satisfied_clause = np.count_nonzero(result)
            log.append(satisfied_clause)

            if satisfied_clause == scale[0]:
                flag = True
                break
        # print(log)
        if flag:
            break
        if epoch % 20 == 9:
            var_value = np.random.choice([-1, 1], size=scale[1])
    return var_value, epoch + 1, log


if __name__ == '__main__':
    problem_scale = (91, 20)
    # problem_scale = (218, 50)
    Max_iteration = 150
    success = 0
    cost = 0
    for i in range(1, 1001):
        if i % 100 == 0:
            print(f"Solving uf{problem_scale[1]}-0{i}...")
        file_path = f"../Dataset/SAT/UF{problem_scale[1]}-{problem_scale[0]}/uf{problem_scale[1]}-0{i}.cnf"
        start_time = time.time()
        solution, step, satlog = solver(path=file_path, scale=problem_scale, iteration=Max_iteration)
        cost += time.time() - start_time
        if step != 100 or np.any(satlog == problem_scale[0]):
            success += 1
        # print(solution, step)
    print()
    print(f"Accuracy: {success / 10}%, TTS: {cost / 1000 / 5 : .5f}s")
