import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from collections import deque
from args import parse_args
from debug_utils import (
    print_mediapipe_keypoint_diff,
    save_debug_frame,
    save_first_last_comparison,
    save_first_frame_keypoints,
    save_debug_log,
)

def infer_face_count(df):
    """检测CSV中face关键点数量（返回点数）"""
    max_i = -1
    for col in df.columns:
        if col.startswith('face_') and col.endswith('_x'):
            try:
                idx = int(col.split('_')[1])
                if idx > max_i:
                    max_i = idx
            except Exception:
                pass
    return max_i + 1 if max_i >= 0 else 0

def load_face_keypoints_matrix(df, face_count):
    """从CSV加载面部关键点矩阵 (n_frames, face_count, 2)，CSV 中存的是像素坐标"""
    n = len(df)
    pts = np.full((n, face_count, 2), np.nan, dtype=np.float64)
    for i in range(face_count):
        cx = f'face_{i}_x'
        cy = f'face_{i}_y'
        if cx in df.columns and cy in df.columns:
            pts[:, i, 0] = df[cx].astype(np.float64).values
            pts[:, i, 1] = df[cy].astype(np.float64).values
    return pts

def build_face_triangles(points):
    """使用 Delaunay 三角剖分构建面部三角形分割"""
    from scipy.spatial import Delaunay
    # 过滤 NaN 点
    valid_mask = ~np.isnan(points).any(axis=1)
    valid_points = points[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    delaunay = Delaunay(valid_points)
    triangles = delaunay.simplices
    
    # 将三角形索引映射回原始索引
    triangles_mapped = np.zeros_like(triangles)
    for i in range(len(triangles)):
        for j in range(3):
            triangles_mapped[i, j] = valid_indices[triangles[i, j]]
    
    return triangles_mapped

def warp_face_triangles(src_img, src_pts, tgt_pts, triangles):
    """
    使用三角形变形：对每个三角形内的所有像素做仿射变换。
    """
    # CPU版本（原有代码）- 保证颜色正确
    h, w = src_img.shape[:2]
    warped = src_img.copy()
    mask_accum = np.zeros((h, w), dtype=np.uint8)

    for tri in triangles:
        src_tri = np.asarray(src_pts[tri], dtype=np.float32)
        tgt_tri = np.asarray(tgt_pts[tri], dtype=np.float32)

        if np.isnan(src_tri).any() or np.isnan(tgt_tri).any():
            continue

        M = cv2.getAffineTransform(src_tri, tgt_tri)
        x, y, ww, hh = cv2.boundingRect(tgt_tri)
        if ww <= 0 or hh <= 0:
            continue

        tgt_tri_roi = (tgt_tri - [x, y]).astype(np.int32)
        mask = np.zeros((hh, ww), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tgt_tri_roi, 255)

        warped_full = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        roi_src = warped_full[y:y+hh, x:x+ww]
        roi_dst = warped[y:y+hh, x:x+ww]
        roi_dst[mask > 0] = roi_src[mask > 0]
        warped[y:y+hh, x:x+ww] = roi_dst
        mask_accum[y:y+hh, x:x+ww][mask > 0] = 255

    warped[mask_accum == 0] = src_img[mask_accum == 0]
    
    return warped

def warp_with_scale(image, src_kps, tgt_kps, warp_target='face', triangles=None):
    """
    简化接口：使用三角形变形。
    """
    if warp_target == 'face':
        if triangles is None:
            triangles = build_face_triangles(src_kps)
        return warp_face_triangles(image, src_kps, tgt_kps, triangles)
    return image.copy()

def fill_missing_keypoints(kps, keypoint_count):
    """使用线性插值填充缺失的关键点"""
    n_frames, n_kps, _ = kps.shape
    kps_flat = kps.reshape(n_frames, -1)
    df = pd.DataFrame(kps_flat)
    df.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
    filled_flat = df.to_numpy()
    filled = filled_flat.reshape(n_frames, n_kps, 2)
    if np.isnan(filled).any():
        filled = np.nan_to_num(filled, nan=0.0)
        print("警告：插值后仍有 NaN 值，已用 0 填充")
    return filled

def compute_frame_diff(src_img, warped_img):
    """计算原图与变形后图像的差异"""
    diff = cv2.absdiff(src_img, warped_img)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    return {"mean_diff": float(mean_diff), "max_diff": float(max_diff)}

def generate_frame(idx, tgt_kps, src_img, src_kps, args, triangles=None, log_data=None, log_lock=None):
    """生成变形帧（线程安全版本）"""
    if idx == 0:
        if log_data is not None and log_lock is not None:
            with log_lock:
                log_data[str(idx)] = {
                    "frame_id": idx,
                    "diff": {"mean_diff": 0.0, "max_diff": 0.0},
                    "keypoint_points": []
                }
        return idx, src_img.copy()

    frame = warp_with_scale(src_img, src_kps, tgt_kps, args.warp_target, triangles)

    # 计算diff（线程安全的日志更新）
    if log_data is not None and log_lock is not None:
        diff_info = compute_frame_diff(src_img, frame)
        with log_lock:
            log_data[str(idx)] = {
                "frame_id": idx,
                "diff": diff_info
            }
            
            # 只在每 10 帧时调用 MediaPipe 检测关键点差异
            if idx % 10 == 0:
                print(f"\n=== 帧 {idx} ===")
                print(f"图像变形差异 - 平均: {diff_info['mean_diff']:.2f}, 最大: {diff_info['max_diff']:.2f}")
                print_mediapipe_keypoint_diff(frame, tgt_kps, args.warp_target, log_data, idx)

    return idx, frame

def generate_frame_batch(indices, batch_tgt_kps, src_img, src_kps, args, triangles, log_data_list):
    """批量生成帧 - 直接使用原有的单帧处理逻辑"""
    result = []
    
    for idx, tgt_kps in zip(indices, batch_tgt_kps):
        if idx == 0:
            log_data_list.append((str(idx), {
                "frame_id": idx,
                "diff": {"mean_diff": 0.0, "max_diff": 0.0}
            }))
            result.append((idx, src_img.copy()))
        else:
            # 直接使用原有的单帧处理函数，保证颜色处理完全一致
            frame = warp_with_scale(src_img, src_kps, tgt_kps, args.warp_target, triangles)
            diff_info = compute_frame_diff(src_img, frame)
            log_data_list.append((str(idx), {
                "frame_id": idx,
                "diff": diff_info
            }))
            result.append((idx, frame))
    
    return result

def evaluate_generated_frames(output_dir, n_frames):
    """评估生成的帧，统计生成的帧数量和文件大小"""
    frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.png')])
    if len(frame_files) != n_frames:
        print(f"警告：生成的帧数量 ({len(frame_files)}) 与预期帧数 ({n_frames}) 不一致")
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in frame_files)
    print(f"评估结果：生成了 {len(frame_files)} 帧，总大小 {total_size / 1024:.2f} KB")
    return len(frame_files), total_size

def main():
    args = parse_args()
    src_img = cv2.imread(args.source)
    if src_img is None:
        raise ValueError(f'无法读取源图像: {args.source}')
    width, height = args.width or src_img.shape[1], args.height or src_img.shape[0]
    src_img = cv2.resize(src_img, (width, height))
    df = pd.read_csv(args.csv)
    keypoint_count = infer_face_count(df)
    kps = load_face_keypoints_matrix(df, keypoint_count)
    src_kps = kps[0]
    filled_kps = fill_missing_keypoints(kps, keypoint_count)
    triangles = build_face_triangles(src_kps)

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(args.output), "generated_frames")
    os.makedirs(output_dir, exist_ok=True)
    print(f"生成的帧将保存在目录: {output_dir}")

    # 初始化调试日志
    log_data = {}
    exceptions = []
    exception_lock = threading.Lock()
    
    total_frames = len(filled_kps)
    batch_size = 10  # 一次生成 4 帧（利用 4GB 显存）
    
    # 第一步：批量生成帧
    print("\n🚀 阶段 1: 批量生成帧到内存（批大小=%d）..." % batch_size)
    frames_data = {}
    log_data_batch = []
    
    gen_pbar = tqdm(total=total_frames, desc="生成帧")
    for batch_start in range(0, total_frames, batch_size):
        batch_end = min(batch_start + batch_size, total_frames)
        batch_indices = list(range(batch_start, batch_end))
        batch_kps = [filled_kps[i] for i in batch_indices]
        
        try:
            batch_result = generate_frame_batch(
                batch_indices, batch_kps, src_img, src_kps, args, triangles, log_data_batch
            )
            for idx, frame in batch_result:
                frames_data[idx] = frame
        except Exception as e:
            with exception_lock:
                exceptions.append(f"批生成 {batch_start}-{batch_end} 异常: {str(e)}")
        
        gen_pbar.update(batch_end - batch_start)
    gen_pbar.close()
    
    # 合并日志
    for key, val in log_data_batch:
        log_data[key] = val
    
    # 第二步：激进并行保存（最大 CPU 核心）
    print("\n⚡ 阶段 2: 激进并行保存到磁盘...")
    
    def save_frame_optimized(idx):
        if idx not in frames_data:
            return
        try:
            frame_path = os.path.join(output_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(frame_path, frames_data[idx], [cv2.IMWRITE_PNG_COMPRESSION, 0])
        except Exception as e:
            with exception_lock:
                exceptions.append(f"保存帧 {idx} 异常: {str(e)}")
    
    num_workers = os.cpu_count() or 16
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(save_frame_optimized, range(total_frames)), 
                 total=total_frames, desc="保存帧"))
    
    # 第三步：清理内存
    print("\n🧹 清理内存...")
    frames_data.clear()
    log_data_batch.clear()
    if torch is not None and CUDA_AVAILABLE:
        torch.cuda.empty_cache()
    print("✅ 内存和显存已释放")

    if exceptions:
        print("\n⚠️ 异常:")n⚠️ 异常:")
        for exc in exceptions::
            print(f"  - {exc}")

    save_debug_log(log_data, args)
    evaluate_generated_frames(output_dir, len(filled_kps))rated_frames(output_dir, len(filled_kps))

if __name__ == '__main__':__main__':
    main()