import os
import re
import numpy as np

root = "./SAT_Dataset/UF20-91"
np.set_printoptions(linewidth=1000, threshold=100000)


def load_uf20_91(file_path):
    matrix = np.zeros(shape=(91, 20), dtype=np.int32)
    with open(file_path, mode="r") as f:
        context = f.read().replace("0\n", "").replace("\n ", "")

        data = re.findall("91 (.*?) %", context)[0].split(" ")
        data = np.array(list(map(int, data)))

        signs = np.sign(data).reshape(91, 3)
        indexes = np.abs(data).reshape(91, 3)

        for j in range(91):
            for sign, index in (zip(signs[j, :], indexes[j, :])):
                matrix[j, index - 1] = sign
    return matrix


def load_uf50_218(file_path):
    matrix = np.zeros(shape=(218, 50), dtype=np.int32)
    with open(file_path, mode="r") as f:
        context = f.read().replace("0\n", "").replace("\n ", "")

        data = re.findall("218 (.*?) %", context)[0].split(" ")
        data = np.array(list(map(int, data)))

        signs = np.sign(data).reshape(218, 3)
        indexes = np.abs(data).reshape(218, 3)

        for j in range(218):
            for sign, index in (zip(signs[j, :], indexes[j, :])):
                matrix[j, index - 1] = sign
    return matrix


print(load_uf50_218("../Dataset/SAT/UF50-218/uf50-03.cnf"))
