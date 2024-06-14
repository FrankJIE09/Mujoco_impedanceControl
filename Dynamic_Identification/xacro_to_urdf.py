import xacro

def xacro_to_urdf(xacro_file, urdf_file):
    try:
        # 读取 Xacro 文件并处理
        doc = xacro.process_file(xacro_file)
        # 转换为 URDF XML 字符串
        urdf_content = doc.toprettyxml(indent='  ')
        # 将 URDF 内容写入文件
        with open(urdf_file, 'w') as f:
            f.write(urdf_content)
        print(f"Successfully generated {urdf_file} from {xacro_file}")
    except Exception as e:
        print(f"Error generating URDF: {e}")

# 示例使用
xacro_file = './universal_robot/ur_description/urdf/ur5e.xacro'
urdf_file = 'robot.urdf'
xacro_to_urdf(xacro_file, urdf_file)
