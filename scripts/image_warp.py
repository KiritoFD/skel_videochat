import argparse
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# OpenPose 18点骨架连接定义
OPENPOSE_18_EDGES = [
    (1, 0),   # neck → nose
    (1, 2),   # neck → r_shoulder
    (2, 3),   # r_shoulder → r_elbow
    (3, 4),   # r_elbow → r_wrist
    (1, 5),   # neck → l_shoulder
    (5, 6),   # l_shoulder → l_elbow
    (6, 7),   # l_elbow → l_wrist
    (1, 8),   # neck → r_hip
    (8, 9),   # r_hip → r_knee
    (9, 10),  # r_knee → r_ankle
    (1, 11),  # neck → l_hip
    (11, 12), # l_hip → l_knee
    (12, 13), # l_knee → l_ankle
    (0, 14),  # nose → r_eye (可选)
    (0, 15),  # nose → l_eye (可选)
    (14, 16), # r_eye → r_ear (可选)
    (15, 17), # l_eye → l_ear (可选)
]

# MediaPipe Pose 33点的部分连接（与pose相关的主要骨架）
MEDIAPIPE_POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # 面部-颈部
    (0, 4), (4, 5), (5, 6), (6, 8),       # 面部-颈部另一侧
    (9, 10),                               # 嘴部
    (11, 12),                              # 肩膀连线
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # 左臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # 右臂
    (11, 23), (12, 24), (23, 24),         # 躯干
    (23, 25), (25, 27), (27, 29), (27, 31),  # 左腿
    (24, 26), (26, 28), (28, 30), (28, 32),  # 右腿
]

def bilinear_interpolate(img, x, y):
    """双线性插值采样，处理边界情况"""
    h, w = img.shape[:2]
    
    # 边界处理：clamp到有效范围
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)
    
    # 计算权重
    wx = x - x0
    wy = y - y0
    
    # 双线性插值
    if len(img.shape) == 3:  # 彩色图像
        result = (1 - wy) * ((1 - wx) * img[y0, x0] + wx * img[y0, x1]) + \
                 wy * ((1 - wx) * img[y1, x0] + wx * img[y1, x1])
    else:  # 灰度图像
        result = (1 - wy) * ((1 - wx) * img[y0, x0] + wx * img[y0, x1]) + \
                 wy * ((1 - wx) * img[y1, x0] + wx * img[y1, x1])
    
    return result.astype(img.dtype)

def beier_neely_warp(src_img, src_kps, tgt_kps, edges, a=0.1, b=1.0, mask=None):
    """
    基于Beier & Neely算法的图像变形
    
    参数:
    - src_img: 源图像 (H, W, C)
    - src_kps: 源关键点 (N, 2) 像素坐标
    - tgt_kps: 目标关键点 (N, 2) 像素坐标  
    - edges: 骨架连接列表 [(i,j), ...]
    - a, b: 权重参数
    - mask: 可选的人体mask (H, W) bool数组
    
    返回: 变形后的图像
    """
    h, w = src_img.shape[:2]
    warped = np.zeros_like(src_img)
    
    # 构建线段对
    line_pairs = []
    for i, j in edges:
        if i < len(src_kps) and j < len(src_kps) and i < len(tgt_kps) and j < len(tgt_kps):
            P_prime = src_kps[i]  # 源线段起点
            Q_prime = src_kps[j]  # 源线段终点
            P = tgt_kps[i]        # 目标线段起点
            Q = tgt_kps[j]        # 目标线段终点
            
            # 跳过包含NaN的关键点
            if not (np.isnan(P_prime).any() or np.isnan(Q_prime).any() or 
                   np.isnan(P).any() or np.isnan(Q).any()):
                line_pairs.append((P, Q, P_prime, Q_prime))
    
    if not line_pairs:
        return src_img.copy()
    
    # 对每个输出像素进行反向映射
    for y in range(h):
        for x in range(w):
            # 如果有mask且当前像素不在mask内，直接复制源像素
            if mask is not None and not mask[y, x]:
                warped[y, x] = src_img[y, x]
                continue
                
            X = np.array([x, y], dtype=float)
            DSUM = np.zeros(2)
            weight_sum = 0.0
            
            for P, Q, P_prime, Q_prime in line_pairs:
                # 跳过退化线段
                vec = Q - P
                len_sq = np.dot(vec, vec)
                if len_sq < 1e-6:
                    continue
                
                vec_prime = Q_prime - P_prime
                len_prime = np.linalg.norm(vec_prime)
                if len_prime < 1e-6:
                    continue
                
                # 计算局部坐标 (u, v)
                u = np.dot(X - P, vec) / len_sq
                v = np.cross(X - P, vec) / np.sqrt(len_sq)
                
                # 计算到线段的距离
                if u < 0:
                    dist = np.linalg.norm(X - P)
                elif u > 1:
                    dist = np.linalg.norm(X - Q)
                else:
                    dist = abs(v)
                
                # 计算源图像中的对应点
                perp_vec_p = np.array([-vec_prime[1], vec_prime[0]]) / len_prime
                X_prime = P_prime + u * vec_prime + v * perp_vec_p
                
                # 位移向量
                D = X_prime - X
                
                # 权重计算
                length = np.sqrt(len_sq)
                weight = (length / (a + dist)) ** b
                
                DSUM += weight * D
                weight_sum += weight
            
            # 计算最终采样坐标
            if weight_sum > 0:
                X_src = X + DSUM / weight_sum
            else:
                X_src = X
            
            # 双线性插值采样
            warped[y, x] = bilinear_interpolate(src_img, X_src[0], X_src[1])
    
    return warped

def load_source_keypoints(src_kps_path, src_img_shape):
    """加载源关键点，支持.npy文件或交互式标注"""
    if src_kps_path and os.path.exists(src_kps_path):
        if src_kps_path.endswith('.npy'):
            return np.load(src_kps_path)
        else:
            # CSV格式
            df = pd.read_csv(src_kps_path)
            if 'x' in df.columns and 'y' in df.columns:
                return df[['x', 'y']].values
            else:
                return df.iloc[:, :2].values
    else:
        print(f"源关键点文件不存在: {src_kps_path}")
        print("请先手动标注源图像关键点并保存为 .npy 或 CSV 文件")
        return None

def load_pose_sequence(csv_path, pose_format='mediapipe'):
    """从CSV加载姿态序列"""
    df = pd.read_csv(csv_path)
    
    if pose_format == 'mediapipe':
        # 检测MediaPipe pose关键点数量
        pose_cols = [col for col in df.columns if col.startswith('pose_') and col.endswith('_x')]
        if not pose_cols:
            raise ValueError("CSV中未找到MediaPipe pose关键点数据")
        
        # 提取pose关键点 (归一化坐标)
        max_idx = max([int(col.split('_')[1]) for col in pose_cols])
        pose_count = max_idx + 1
        
        poses = []
        for idx, row in df.iterrows():
            pose = np.full((pose_count, 2), np.nan)
            for i in range(pose_count):
                x_col = f'pose_{i}_x'
                y_col = f'pose_{i}_y'
                if x_col in df.columns and y_col in df.columns:
                    pose[i, 0] = row[x_col]
                    pose[i, 1] = row[y_col]
            poses.append(pose)
        
        return np.array(poses)
    
    else:  # openpose格式：假设每行是 frame_id,x0,y0,x1,y1,...
        poses = []
        for idx, row in df.iterrows():
            coords = row.values[1:] if 'frame' in df.columns else row.values
            pose = coords.reshape(-1, 2)
            poses.append(pose)
        return np.array(poses)

def smooth_poses(poses, window_size=5):
    """对姿态序列进行平滑滤波"""
    from scipy.signal import savgol_filter
    smoothed = poses.copy()
    for i in range(poses.shape[1]):  # 对每个关键点
        for j in range(2):  # x, y坐标
            valid_mask = ~np.isnan(poses[:, i, j])
            if np.sum(valid_mask) > window_size:
                smoothed[valid_mask, i, j] = savgol_filter(
                    poses[valid_mask, i, j], 
                    min(window_size, np.sum(valid_mask)), 
                    3
                )
    return smoothed

def main():
    parser = argparse.ArgumentParser(description='基于关键点变化的图像变形视频生成')
    parser.add_argument('-s', '--source', required=True, help='源图像路径')
    parser.add_argument('-k', '--source-kps', required=True, help='源图像关键点文件(.npy或CSV)')
    parser.add_argument('-c', '--csv', required=True, help='姿态序列CSV文件(extract_keypoints输出)')
    parser.add_argument('-o', '--output', default='warped_video.mp4', help='输出视频路径')
    parser.add_argument('--width', type=int, help='输出视频宽度(默认使用源图像尺寸)')
    parser.add_argument('--height', type=int, help='输出视频高度(默认使用源图像尺寸)')
    parser.add_argument('--fps', type=float, default=30, help='输出视频帧率')
    parser.add_argument('--pose-format', choices=['mediapipe', 'openpose'], default='mediapipe', 
                       help='姿态数据格式')
    parser.add_argument('--edges', choices=['mediapipe', 'openpose18'], default='mediapipe',
                       help='使用的骨架连接定义')
    parser.add_argument('--smooth', action='store_true', help='对姿态序列进行平滑滤波')
    parser.add_argument('--mask', help='可选的人体mask图像路径')
    parser.add_argument('-a', type=float, default=0.1, help='Beier-Neely权重参数a')
    parser.add_argument('-b', type=float, default=1.0, help='Beier-Neely权重参数b')
    
    args = parser.parse_args()
    
    # 加载源图像
    src_img = cv2.imread(args.source)
    if src_img is None:
        raise ValueError(f"无法读取源图像: {args.source}")
    
    h, w = src_img.shape[:2]
    output_w = args.width or w
    output_h = args.height or h
    
    # 调整源图像尺寸（如需要）
    if output_w != w or output_h != h:
        src_img = cv2.resize(src_img, (output_w, output_h))
        scale_x = output_w / w
        scale_y = output_h / h
    else:
        scale_x = scale_y = 1.0
    
    # 加载源关键点
    src_kps = load_source_keypoints(args.source_kps, src_img.shape)
    if src_kps is None:
        return
    
    # 调整源关键点坐标（如果图像被缩放）
    if scale_x != 1.0 or scale_y != 1.0:
        # 假设源关键点是像素坐标
        src_kps[:, 0] *= scale_x
        src_kps[:, 1] *= scale_y
    
    # 加载姿态序列
    poses = load_pose_sequence(args.csv, args.pose_format)
    
    # 转换归一化坐标到像素坐标
    poses[:, :, 0] *= output_w  # x坐标
    poses[:, :, 1] *= output_h  # y坐标
    
    # 平滑处理（可选）
    if args.smooth:
        try:
            poses = smooth_poses(poses)
            print("已应用姿态平滑滤波")
        except ImportError:
            print("警告: 需要scipy库才能使用平滑功能，跳过平滑处理")
    
    # 选择骨架连接
    if args.edges == 'openpose18':
        edges = OPENPOSE_18_EDGES
    else:
        edges = MEDIAPIPE_POSE_EDGES
    
    # 加载mask（可选）
    mask = None
    if args.mask:
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            mask = cv2.resize(mask_img, (output_w, output_h)) > 127
            print("已加载人体mask")
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (output_w, output_h))
    
    # 逐帧生成
    num_frames = len(poses)
    pbar = tqdm(total=num_frames, desc='生成变形视频', unit='frame')
    
    try:
        for i in range(num_frames):
            tgt_kps = poses[i]
            
            # 执行变形
            warped_frame = beier_neely_warp(
                src_img, src_kps, tgt_kps, edges, 
                a=args.a, b=args.b, mask=mask
            )
            
            out.write(warped_frame)
            pbar.update(1)
    
    finally:
        pbar.close()
        out.release()
    
    print(f"变形视频已保存: {args.output}")
    print(f"总帧数: {num_frames}, FPS: {args.fps}")

if __name__ == '__main__':
    main()
