import numpy as np

# 假设动力学参数
m = 1.5  # 质量
I = 0.5  # 惯性

# 时间参数
t = np.linspace(0, 10, 100)  # 时间从0到10秒，共100个数据点

# 生成模拟的位置、速度和加速度数据
theta = np.sin(t)  # 位置数据
theta_dot = np.cos(t)  # 速度数据
theta_double_dot = -np.sin(t)  # 加速度数据

# 根据动力学方程生成模拟扭矩数据
# 扭矩 τ = I*α + m*g*l*sin(θ)
# 这里简化处理：g = 9.81, l = 1
g = 9.81
l = 1
torque = I * theta_double_dot + m * g * l * np.sin(theta)

# 堆叠特征矩阵
X = np.vstack([theta_double_dot, np.sin(theta)]).T

# 使用普通最小二乘法估计参数
params = np.linalg.lstsq(X, torque, rcond=None)[0]
estimated_I, estimated_mgl = params

print("Estimated Inertia (I):", estimated_I)
print("Estimated m*g*l:", estimated_mgl)
