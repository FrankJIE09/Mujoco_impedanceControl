import numpy as np

# 假设机械臂参数和初始状态
num_joints = 6
joint_angles = np.random.rand(num_joints) * np.pi  # 随机初始化关节角度
joint_velocities = np.zeros(num_joints)  # 初始速度为零
joint_accelerations = np.zeros(num_joints)  # 初始加速度为零
desired_torque = np.zeros(num_joints)  # 目标扭矩


# 动力学模型函数（简化模型，仅示例）
def compute_torque(joint_angles, joint_velocities, joint_accelerations, link_lengths, link_masses, gravity=9.81):
    num_joints = len(joint_angles)
    torques = np.zeros(num_joints)

    # 假设每个关节后面的连杆质量集中影响该关节
    for i in range(num_joints):
        # 求和所有后续连杆对当前关节产生的力矩
        for j in range(i, num_joints):
            # 链接末端到关节i的距离
            distance_to_joint = np.sum(link_lengths[i:j + 1])
            # 链接的重力产生的力矩
            torque_due_to_gravity = link_masses[j] * gravity * distance_to_joint * np.sin(joint_angles[i])
            # 链接的加速度产生的力矩
            torque_due_to_acceleration = link_masses[j] * distance_to_joint * joint_accelerations[i]

            # 累加计算总扭矩
            torques[i] += torque_due_to_gravity + torque_due_to_acceleration

    # 加上由于摩擦产生的扭矩（简化处理为与速度成正比）
    friction_coefficient = 0.1  # 摩擦系数
    torques -= friction_coefficient * joint_velocities

    return torques


# 牛顿法更新
def newton_update(current_angles, target_torque, tolerance=1e-6, max_iterations=100):
    for i in range(max_iterations):
        current_torque = compute_torque(current_angles, joint_velocities, joint_accelerations, np.ones(6)*0.1,  np.ones(6)*10)
        torque_error = current_torque - target_torque
        if np.linalg.norm(torque_error) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            return current_angles
        # 这里的Jacobian矩阵是假设的简化形式
        J = np.eye(num_joints)  # 假设每个关节独立影响扭矩
        # 牛顿法更新步骤
        current_angles -= np.linalg.solve(J, torque_error)
    print("Failed to converge.")
    return current_angles


# 运行牛顿法求解
final_angles = newton_update(joint_angles, desired_torque)
print("Final joint angles:", final_angles)
