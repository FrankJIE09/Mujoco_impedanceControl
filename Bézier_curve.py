import numpy as np
import matplotlib.pyplot as plt


def compute_bezier_curve(points, t):
    n = len(points) - 1
    x = 0
    y = 0
    z = 0
    for i in range(n + 1):
        coeff = np.math.comb(n, i) * (1 - t) ** (n - i) * t ** i
        x += coeff * points[i][0]
        y += coeff * points[i][1]
        z += coeff * points[i][2]

    return x, y, z


def bezier_curve(control_points, ):
    t = np.linspace(0, 1, 2000)
    curve_points = np.array([compute_bezier_curve(control_points, ti) for ti in t])
    return curve_points


if __name__ == '__main__':
    control_points_ = np.array([[0, 0, 0], [1, 3, 1], [4, 2, 1.5], [5, 5, 2]])
    result = bezier_curve(control_points_)
    print()
