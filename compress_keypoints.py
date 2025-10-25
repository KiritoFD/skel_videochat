import argparse
import pandas as pd
import numpy as np
import os

def compress_keypoints(input_csv, output_csv=None, target_width=1920, target_height=1080):
    """
    压缩关键点数据：
    1. 只保留 face 部分（468 个点）
    2. 转换为 uint16 像素坐标
    3. 保存为新的 CSV 文件
    """
    print(f"读取 CSV 文件: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 检测 face 关键点数量
    max_face_idx = -1
    for col in df.columns:
        if col.startswith('face_') and col.endswith('_x'):
            try:
                idx = int(col.split('_')[1])
                if idx > max_face_idx:
                    max_face_idx = idx
            except Exception:
                pass
    
    face_count = max_face_idx + 1
    print(f"检测到 {face_count} 个 face 关键点")
    
    # 提取 frame, time 和 face 列
    cols_to_keep = ['frame', 'time']
    for i in range(face_count):
        cols_to_keep.append(f'face_{i}_x')
        cols_to_keep.append(f'face_{i}_y')
    
    # 检查列是否存在
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    
    if missing_cols:
        print(f"警告：以下列不存在: {missing_cols}")
    
    # 提取子集
    df_face = df[existing_cols].copy()
    
    # 转换坐标为 uint16
    for i in range(face_count):
        x_col = f'face_{i}_x'
        y_col = f'face_{i}_y'
        
        if x_col in df_face.columns:
            # 填充 NaN 值为 0
            df_face[x_col] = df_face[x_col].fillna(0)
            # 转换为 uint16（限制在 0-65535 范围内）
            df_face[x_col] = np.clip(df_face[x_col], 0, 65535).astype(np.uint16)
        
        if y_col in df_face.columns:
            # 填充 NaN 值为 0
            df_face[y_col] = df_face[y_col].fillna(0)
            # 转换为 uint16（限制在 0-65535 范围内）
            df_face[y_col] = np.clip(df_face[y_col], 0, 65535).astype(np.uint16)
    
    # 生成输出文件名
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(os.path.dirname(input_csv), f"{base_name}_face_compressed.csv")
    
    # 保存为 CSV
    df_face.to_csv(output_csv, index=False)
    
    # 计算压缩率
    original_size = os.path.getsize(input_csv)
    compressed_size = os.path.getsize(output_csv)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"\n压缩完成:")
    print(f"  原始文件大小: {original_size / 1024 / 1024:.2f} MB")
    print(f"  压缩后大小: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"  压缩率: {compression_ratio:.1f}%")
    print(f"  帧数: {len(df_face)}")
    print(f"  关键点数: {face_count}")
    print(f"  输出文件: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='压缩关键点数据，只保留 face 部分并转换为 uint16')
    parser.add_argument('-i', '--input', required=True, help='输入 CSV 文件路径')
    parser.add_argument('-o', '--output', default=None, help='输出 CSV 文件路径（默认自动生成）')
    parser.add_argument('--target-width', type=int, default=1920, help='目标宽度（默认 1920）')
    parser.add_argument('--target-height', type=int, default=1080, help='目标高度（默认 1080）')
    
    args = parser.parse_args()
    compress_keypoints(args.input, args.output, args.target_width, args.target_height)

if __name__ == '__main__':
    main()
