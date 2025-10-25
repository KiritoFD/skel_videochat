import pandas as pd
import numpy as np
import os

def scale_keypoints(input_csv, output_csv, target_width=1920, target_height=1080):
    """
    将原始 CSV 文件中的关键点横坐标乘以 target_width，纵坐标乘以 target_height。
    :param input_csv: 输入 CSV 文件路径
    :param output_csv: 输出 CSV 文件路径
    :param target_width: 目标宽度（默认 1920）
    :param target_height: 目标高度（默认 1080）
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_csv)
    
    # 遍历所有列，找到关键点的 x 和 y 坐标列
    for col in df.columns:
        if col.endswith('_x'):
            # 横坐标乘以目标宽度
            df[col] = df[col] * target_width
        elif col.endswith('_y'):
            # 纵坐标乘以目标高度
            df[col] = df[col] * target_height

    # 保存修改后的 CSV 文件
    df.to_csv(output_csv, index=False)
    print(f"关键点已缩放并保存到 {output_csv}")

if __name__ == "__main__":
    # 示例用法
    input_csv = "1.csv"  # 替换为实际输入文件路径
    output_csv = "s.csv"  # 替换为实际输出文件路径

    # 调用函数进行关键点缩放
    scale_keypoints(input_csv, output_csv)
