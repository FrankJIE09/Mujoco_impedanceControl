import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares

# 假设这些是从传感器获得的数据
# q, q_dot, q_ddot 是关节位置、速度和加速度的时间序列数据
# tau 是对应的力矩数据
q = np.random.randn(10, 1)
q_dot = np.random.randn(10, 1)
q_ddot = np.random.randn(10, 1)
tau = np.hstack((q, q_dot, q_ddot, np.ones_like(q))) @ np.array([1, 2, 3, 4])


# 定义动力学方程中的矩阵 W
def W_matrix(q, q_dot, q_ddot):
    # 这里假设 W 是一个包含 q, q_dot, q_ddot 的线性组合
    # 实际中，W 的形式取决于具体的机器人动力学
    W = np.hstack((q, q_dot, q_ddot, np.ones_like(q)))
    return W


# 计算矩阵 W
W = W_matrix(q, q_dot, q_ddot)

# QR分解
Q, R = qr(W, mode='economic')

# 提取 W_B 和 tau_B
W_B = Q
tau_B = tau


# 最小二乘法求解 P_B
def residuals(P_B, W_B, tau_B):
    return (W_B @ P_B - tau_B).flatten()


# 初始猜测
P_B_initial = np.zeros(W_B.shape[1])

# 使用最小二乘法优化
result = least_squares(residuals, P_B_initial, args=(W_B, tau_B))
P_B_estimated = result.x

# 最终的参数估计
print("Estimated Parameters (P_B):", P_B_estimated)
print("Parameters:", np.linalg.inv(R) @ P_B_estimated)

# 用估计的参数重新计算力矩
tau_estimated = W_B @ P_B_estimated

# 比较实际力矩和估计力矩
print("Actual Torque (tau):", tau.flatten())
print("Estimated Torque (tau_estimated):", tau_estimated.flatten())
