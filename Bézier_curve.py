import numpy as np
import matplotlib.pyplot as plt


def compute_bezier_curve(points, t):
    n = len(points) - 1
    x = 0
    y = 0

    for i in range(n + 1):
        coeff = np.math.comb(n, i) * (1 - t) ** (n - i) * t ** i
        x += coeff * points[i][0]
        y += coeff * points[i][1]

    return x, y


def plot_bezier_curve(points):
    t = np.arange(0, 1, 0.01)
    curve_points = np.array([compute_bezier_curve(points, ti) for ti in t])

    plt.plot(curve_points[:, 0], curve_points[:, 1])
    plt.plot(points[:, 0], points[:, 1], 'ro-')
    plt.title('Bezier Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# 示例用法
control_points = np.array([[0, 0], [1, 3], [4, 2], [5, 5]])
plot_bezier_curve(control_points)
