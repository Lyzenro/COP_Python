import os
import re
import numpy as np

root = "./SAT_Dataset/UF20-91"

for i in range(1, 1001):
    file = f"uf20-0{i}.cnf"
    path = os.path.join(root, file)
    with open(path, mode="r") as f:
        context = f.read().replace("0\n", "").replace("\n ", "")

        data = re.findall("91 (.*?) %", context)[0].split(" ")
        data = np.array(list(map(int, data)))

        signs = np.sign(data).reshape(91, 3)
        indexes = np.abs(data).reshape(91, 3)

        print(signs)
        print(indexes)

        matrix = np.zeros(shape=(91, 20), dtype=np.int32)
        for j in range(91):
            for sign, index in (zip(signs[j, :], indexes[j, :])):
                matrix[j, index - 1] = sign
        print(matrix)
    break


