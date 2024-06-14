import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares

# 假设这些是从传感器获得的数据
# q 是关节位置的时间序列数据
# q_dot 是关节速度的时间序列数据
# q_ddot 是关节加速度的时间序列数据
# tau 是对应的力矩数据
num_samples = 100
num_joints = 2

q = np.random.randn(num_samples, num_joints)
q_dot = np.random.randn(num_samples, num_joints)
q_ddot = np.random.randn(num_samples, num_joints)
tau = np.random.randn(num_samples, num_joints)

# 定义动力学方程中的矩阵 W
def W_matrix(q, q_dot, q_ddot):
    # 这里假设 W 是一个包含 q, q_dot, q_ddot 的线性组合
    # 实际中，W 的形式取决于具体的机器人动力学
    W = []
    for i in range(q.shape[0]):
        q_i = q[i]
        q_dot_i = q_dot[i]
        q_ddot_i = q_ddot[i]
        W_i = np.vstack([
            q_ddot_i,  # 惯性项
            q_dot_i,  # 科氏力和离心力项
            q_i  # 阻尼和摩擦项
        ])
        W.append(W_i)
    return np.array(W)

# 计算矩阵 W
W = W_matrix(q, q_dot, q_ddot)

# QR分解
Q, R = qr(W, mode='economic')

# 提取 W_B 和 tau_B
W_B = Q

# 最小二乘法求解 P_B
def residuals(P_B, W_B, tau_B):
    return W_B @ P_B - tau_B

# 初始猜测
P_B_initial = np.zeros(W_B.shape[1])

# 使用最小二乘法优化
result = least_squares(residuals, P_B_initial, args=(W_B, tau))
P_B_estimated = result.x

# 从 P_B 计算 P
P_estimated = np.linalg.inv(R) @ P_B_estimated

# 最终的参数估计
print("Estimated Parameters (P_B):", P_B_estimated)
print("Estimated Original Parameters (P):", P_estimated)

# 用估计的参数重新计算力矩
tau_estimated = W @ P_estimated

# 比较实际力矩和估计力矩
print("Actual Torque (tau):", tau.flatten())
print("Estimated Torque (tau_estimated):", tau_estimated.flatten())
