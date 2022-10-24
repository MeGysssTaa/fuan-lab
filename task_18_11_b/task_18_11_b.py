from math import ceil, log, pi, sin, cos
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def x_nt(n: int, t: float) -> float:
    """
    `n` раз применяет оператор Ф[x(t)] к функции x(t) и подставляет конкретное числовое значение `t`.
    """
    return t if n == 0 else sin(3.0 * t) + (t + 1.0) * cos(x_nt(n - 1, t) / 6.0)


def dist(a: float, b: float, x: Callable[[float], float], y: Callable[[float], float], eps: float) -> float:
    """
    Вычисляет расстояние между функциями одной вещественной переменной `x` и `y`,
    определёнными на C[a;b] (непрерывные функции на отрезке от `a` до `b`), с точностью до `eps`.
    """
    return maxval(lambda t: abs(x(t) - y(t)), a, b, eps)


def maxval(f: Callable[[float], float], left: float, right: float, eps: float) -> float:
    """
    Вычисляет максимальное значение функции `f` (из C[left;right]) на отрезке от `left`
    до `right` с точностью до `eps`. Для вычисления используется тернарный поиск.
    """
    while right - left > eps:
        a = (left * 2.0 + right) / 3.0
        b = (left + right * 2.0) / 3.0
        if f(a) > f(b):
            right = b
        else:
            left = a
    return f((left + right) / 2.0)


def plot(a: float, b: float, x_t: Callable[[float], float]):
    """
    Строит график функции `x_t`, т.е. функции одной вещественной
    переменной x(t), на отрезке [a;b] (от `a` до `b`). Для просмотра
    можно использовать, например, SciView -> Plots в PyCharm Professional.
    """
    t = np.linspace(a, b, 100)
    x = [x_t(t_i) for t_i in t]

    fig = plt.figure()
    p = fig.add_subplot()
    p.spines["left"].set_position("center")
    p.spines["bottom"].set_position("zero")
    p.spines["right"].set_color("none")
    p.spines["top"].set_color("none")
    p.xaxis.set_ticks_position("bottom")
    p.yaxis.set_ticks_position("left")

    plt.plot(t, x, "m")
    plt.grid()
    plt.show()


def main():
    a = -pi  # левый конец отрезка, на котором определена функция x(t)
    b = pi  # правый конец отрезка, на котором определена функция x(t)
    eps = 0.01  # требуемая точность (эпсилон)
    alpha = (pi + 1.0) / 6.0  # коэффициент сжатия соответствующего оператора -- вычислен в пункте A решения

    x0 = lambda t: 0.0  # x0(t) = 0 -- начальное приближение
    x1 = lambda t: x_nt(1, t)  # x1(t) = Ф[x0(t)] -- первое приближение
    n_apr = int(ceil(log(
        (eps * (1.0 - alpha)) / dist(a, b, x0, x1, eps),  # считаем логарифм этого числа
        alpha  # основание логарифма
    )))
    xn = lambda t: x_nt(n_apr, t)  # используем априорную оценку и получаем xn(t) -- искомое приближение
    print(f"Априорная оценка числа итераций: {n_apr}.\nСамо приближённое решение изображено на графике.")
    plot(a, b, xn)  # строим график приближённого решения (xn(t) -> x(t)) -- см., напр., SciView в PyCharm


if __name__ == '__main__':
    main()
