import numpy as np
import random
import matplotlib.pyplot as plt


def design_matrix(function, X):
    F = []
    for i in range(X.shape[0]):
        F.append(function(X[i]))
    return np.array(F)


def pinv(F, a):
    return np.linalg.inv(F.T @ F + a * np.eye(F.shape[1])) @ F.T


def pattern(train_len, valid_len, test_len, t, X):
    train = np.zeros((train_len, 2))
    valid = np.zeros((valid_len, 2))
    test = np.zeros((test_len, 2))

    for i in range(train_len):
        index = random.randint(0, N - 1)
        train[i, 0] = X[index]
        train[i, 1] = t[index]

    for i in range(valid_len):
        index = random.randint(0, N - 1)
        valid[i, 0] = X[index]
        valid[i, 1] = t[index]

    for i in range(test_len):
        index = random.randint(0, N - 1)
        test[i, 0] = X[index]
        test[i, 1] = t[index]
    return train, valid, test


# Набор базисных функций
def basic_functions():
    func_1 = lambda x: [np.cos(x), x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10]
    func_2 = lambda x: [np.sin(x), np.sqrt(x)]
    func_3 = lambda x: [np.cos(x), np.exp(x)]
    func_4 = lambda x: [np.exp(x), np.sqrt(x)]
    func_5 = lambda x: [(lambda i: x ** i)(i) for i in range(100)]
    func_6 = lambda x: [np.sin(x), np.exp(x)]
    func_7 = lambda x: [np.cos(x), np.sqrt(x)]
    func_8 = lambda x: [np.cos(x), np.sin(x)]
    return np.array([func_1, func_2, func_3, func_4, func_5, func_6, func_7, func_8])


def best_value(funcs, train, valid, A):
    min = -1
    index = [-1, -1]
    w_best = np.empty(0)
    for i in range(funcs.shape[0]):
        for j in range(A.shape[0]):
            F_train = design_matrix(funcs[i], train[0])
            F_pinv_train = pinv(F_train, A[j])
            w = F_pinv_train @ train[1]
            F_valid = design_matrix(funcs[i], valid[0])
            y = F_valid @ w
            err = 0.5 * np.sum((y - valid[1]) ** 2)
            if min == -1:
                min = err
                index = [i, j]
                w_best = w
            elif min > err:
                min = err
                index = [i, j]
                w_best = w
    print("Функция под номером:", index[0] + 1)
    print("Лучший коэффиециент регуляризации:", A[index[1]])
    return funcs[index[0]], A[index[1]], w_best


def err(func, w, test):
    F = design_matrix(func, test[0])
    y = F @ w
    err = 0.5 * np.sum((test[1] - y) ** 2)
    print("Значение ошибки на test части:", err)
    test_x = np.sort(test[0])
    F = design_matrix(func, test_x)
    y = F @ w
    return test_x, y


# Входные данные
N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
e = 10 * np.random.randn(N)
t = z + e

# Выборки
train = int(N * 0.8)  # 80%
valid = int(N * 0.1)  # 10%
test = int(N * 0.1)  # 10%

Train, Valid, Test = pattern(train, valid, test, t, x)

# Набор значений коэффициента регуляризации
Lambda = np.array([0, 10e-10, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000])

# Лучшее значение и ошибка
func, a, w = best_value(basic_functions(), Train.T, Valid.T, Lambda)
test_x, y = err(func, w, Test.T)

# Графики
plt.plot(x, z, c="black")
plt.scatter(Test.T[0], Test.T[1])
plt.plot(test_x, y, c="red")
plt.show()
