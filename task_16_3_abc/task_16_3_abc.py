import math

import numpy as np


def solve(c: np.array, d: np.array, alpha: float, eps: float, x: np.array):
    """
    Методом последовательных приближений приближённо вычисляем решение системы A * x = b.

    :param c: Матрица c = E - (A^T * A) / л_n
    :param d: Матрица d = (A^T * b) / л_n
    :param alpha: Значение а (альфа) -- коэффициент сжатия соответствующего отображения
    :param eps: Значение э (эпсилон) -- требуемая точность приближения
    :param x: Вектор-столбец известного точного решения системы (для сравнения с найденным приближённым)
    """
    x0 = np.array(4 * [0.0])  # нулевой вектор (нулевое решение) -- берём его за начальное приближение
    x1 = np.matmul(c, x0) + d  # следующее за начальным приближение x1 = C * x0 + d
    # Априорная оценка N_apr числа итераций = э*(1-а)/р(x0,x1) (здесь э -- эпсилон, а -- альфа, р -- Евклидова метрика)
    n_apr = int(math.ceil(math.log(
        (eps * (1.0 - alpha)) / np.linalg.norm(x0 - x1),  # считаем логарифм этой величины
        alpha  # основание логарифма
    )))

    print(f"Приближённо вычисляем решение системы с точностью э = {eps} (здесь э -- эпсилон).\n"
          f"\tАприорная оценка числа итераций: {n_apr}")

    xm = x0  # решение x_(n-1) (приближение с номером n - 1 (предпоследнее))
    xn = x1  # решение x_n (приближение с номером n (последнее))
    iters = 1  # счётчик -- реальное число итераций, которые мы сделали (в цикле используется апостериорная оценка)
    while (alpha / (1.0 - alpha)) * np.linalg.norm(xm - xn) > eps:
        xm = xn
        xn = np.matmul(c, xn) + d
        iters += 1
    err = np.linalg.norm(x - xn)  # р(x,xn) -- расстояние между x и xn, т.е. между точным решением и приближённым

    print(f"\tАпостериорная оценка (сколько на самом деле понадобилось итераций): {iters}\n"
          f"\tВычисленное решение (x_n после последней итерации): {xn}\n"
          f"\tРасстояние до точного решения (итоговая погрешность): {err}\n"
          f"\t\t^^^ не превосходит э (эпсилон): {'да' if err <= eps else 'нет'}\n")


def main():
    a = np.array([  # A -- матрица коэффициентов при неизвестных системы
        [8, 2, -3, 2],
        [-6, 3, -2, 1],
        [3, 8, 4, -8],
        [2, 1, -6, 2]
    ])
    b = np.array([  # b -- вектор-столбец свободных членов системы
        102,
        -47,
        -122,
        -24
    ])

    t = np.transpose(a)  # A^T
    ta = np.matmul(t, a)  # A^T * A
    tb = np.matmul(t, b)  # A^T * b
    eigenvals_ta = np.linalg.eigvals(ta)  # собственные значения матрицы A^T * A
    lambda_n = max(eigenvals_ta)  # л_n (лямбда_n)

    print(f"Матрица A:\n{a}.\n\n"
          f"Матрица b:\n{b}\n\n"
          f"Транспонированная к A (A^T):\n{t}\n\n"
          f"Произведение A^T * A:\n{ta}\n\n"
          f"Произведение A^T * b:\n{tb}\n\n"
          f"Собственные значения матрицы A^T * A: {eigenvals_ta}\n"
          f"Наибольшее из них ^^^ (л_n, где л -- лямбда): {lambda_n}\n")

    c = np.identity(4) - ta / lambda_n  # E - (A^T * A) / л_n
    d = tb / lambda_n  # (A^T * b) / л_n
    eigenvals_c = np.linalg.eigvals(c)  # собственные значения матрицы c = E - (A^T * A) / л_n
    alpha = max(eigenvals_c)  # а (альфа)

    print(f"Матрица c = E - (A^T * A) / л_n (где E - единичная матрица 4-го порядка):\n{c}\n\n"
          f"Матрица d = (A^T * b) / л_n\n\n"
          f"Собственные значения матрицы c: {eigenvals_c}\n"
          f"Наибольшее из них ^^^ (а -- альфа): {alpha} <- оно же -- коэффициент сжатия соответствующего отображения\n")

    # https://www.wolframalpha.com/input?i=8a%2B2b-3c%2B2d%3D102%2C-6a%2B3b-2c%2Bd%3D-47%2C3a%2B8b%2B4c-8d%3D-122%2C2a%2Bb-6c%2B2d%3D-24
    x = np.array([10, 6, 20, 35])  # точное решение системы (получено в Wolfram Alpha)

    print(f"Точное решение системы: {x}\n")

    # Вычисляем два варианта приближённого решения (для двух заданных значений э)
    solve(c, d, alpha, 10 ** -2, x)
    solve(c, d, alpha, 10 ** -4, x)


if __name__ == '__main__':
    main()
