import numpy as np
import matplotlib.pyplot as plt


def compute_bezier_curve(points, t):
    n = len(points) - 1
    x = 0
    y = 0
    z = 0
    for i in range(n + 1):
        coefficient = np.math.comb(n, i) * (1 - t) ** (n - i) * t ** i
        x += coefficient * points[i][0]
        y += coefficient * points[i][1]
        z += coefficient * points[i][2]

    return x, y, z


def bezier_curve(control_points, seg=2000):
    t = np.linspace(0, 1, seg)
    curve_points = np.array([compute_bezier_curve(control_points, ti) for ti in t])
    return curve_points


if __name__ == '__main__':
    control_points_ = np.array([[0, 0, 0], [1, 3, 1], [4, 2, 1.5], [5, 5, 2]])
    result = bezier_curve(control_points_,seg=5)
    print()
