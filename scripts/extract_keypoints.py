import argparse
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

# 新增：CSV 数据格式说明
"""
CSV 数据格式说明
- 列按顺序：
  1) frame: 帧索引（从0开始，整数）
  2) time: 时间，单位为秒，计算方式 time = frame / fps
  3) pose_0_x, pose_0_y, pose_0_z, pose_0_v, ..., pose_32_v
  4) face_0_x, face_0_y, face_0_z, ..., face_467_z
  5) lhand_0_x, lhand_0_y, lhand_0_z, ..., lhand_20_z
  6) rhand_0_x, rhand_0_y, rhand_0_z, ..., rhand_20_z

- 字段含义与取值：
  - x, y: 像素坐标，单位为图像的实际宽高（以输入帧尺寸换算，示例 1920×1080）。
  - z: 相对深度值，单位与 x、y 的归一化尺度一致；对于部分模型 z 值为相对距离，负值通常表示靠近相机。
  - pose_*_v: visibility，可见性分数，范围约为 [0,1]（越大表示该关键点越可靠）。
  - 缺失关键点：若某帧未检测到对应关键点，CSV 对应单元格为 NaN（pandas 导出默认行为）。

- 列数：
  - 总列数 = 2（frame,time） + 33*4（pose） + 468*3（face） + 21*3（left hand） + 21*3（right hand） = 1664 列

- 额外说明：
  - CSV 由 pandas.DataFrame.to_csv 保存，index=False。
  - 若只需部分关键点（例如仅 pose），可在后处理时选择相应列或修改脚本的 build_columns / 填充逻辑。
  - time 列依赖于读取到的 fps，脚本提供覆盖参数以应对视频报告 fps 为 0 的情况。
"""

def build_columns():
    cols = []
    # pose: 33 landmarks, each x,y,z,visibility
    for i in range(33):
        cols += [f'pose_{i}_x', f'pose_{i}_y', f'pose_{i}_z', f'pose_{i}_v']
    # face: 468 landmarks, each x,y,z
    for i in range(468):
        cols += [f'face_{i}_x', f'face_{i}_y', f'face_{i}_z']
    # left hand: 21 landmarks, each x,y,z
    for i in range(21):
        cols += [f'lhand_{i}_x', f'lhand_{i}_y', f'lhand_{i}_z']
    # right hand: 21 landmarks, each x,y,z
    for i in range(21):
        cols += [f'rhand_{i}_x', f'rhand_{i}_y', f'rhand_{i}_z']
    return cols

def extract(args):
    mp_holistic = mp.solutions.holistic
    cols = ['frame', 'time'] + build_columns()
    rows = []

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError('无法打开视频: ' + args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 进度条总帧数（如不可得则为 None）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    # 是否启用 OpenCV CUDA 预处理（不保证 MediaPipe 使用 GPU）
    use_cuda_cv = bool(args.use_cuda and hasattr(cv2, 'cuda'))
    if args.use_cuda and not use_cuda_cv:
        print('未检测到 OpenCV CUDA 支持，回退到 CPU。')

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        pbar = tqdm(total=total_frames, desc='处理帧', unit='frame', ncols=80)
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                height, width = frame.shape[:2]
                # MediaPipe 需要 RGB 格式
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 尝试使用 OpenCV CUDA 做颜色转换以减轻 CPU 负担（如果可用）
                if use_cuda_cv:
                    try:
                        gpu = cv2.cuda_GpuMat()
                        gpu.upload(frame)
                        gpu_rgb = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2RGB)
                        image = gpu_rgb.download()
                    except Exception:
                        # 若 CUDA 路径失败，回退到 CPU
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # 初始化一行为 NaN
                row = {c: np.nan for c in cols}
                row['frame'] = frame_idx
                row['time'] = frame_idx / fps

                # pose
                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        row[f'pose_{i}_x'] = lm.x * width
                        row[f'pose_{i}_y'] = lm.y * height
                        row[f'pose_{i}_z'] = lm.z
                        # visibility 有时不存在，保守读取
                        v = getattr(lm, 'visibility', np.nan)
                        row[f'pose_{i}_v'] = v

                # face
                if results.face_landmarks:
                    for i, lm in enumerate(results.face_landmarks.landmark):
                        row[f'face_{i}_x'] = lm.x * width
                        row[f'face_{i}_y'] = lm.y * height
                        row[f'face_{i}_z'] = lm.z

                # left hand
                if results.left_hand_landmarks:
                    for i, lm in enumerate(results.left_hand_landmarks.landmark):
                        row[f'lhand_{i}_x'] = lm.x * width
                        row[f'lhand_{i}_y'] = lm.y * height
                        row[f'lhand_{i}_z'] = lm.z

                # right hand
                if results.right_hand_landmarks:
                    for i, lm in enumerate(results.right_hand_landmarks.landmark):
                        row[f'rhand_{i}_x'] = lm.x * width
                        row[f'rhand_{i}_y'] = lm.y * height
                        row[f'rhand_{i}_z'] = lm.z

                rows.append(row)
                frame_idx += 1
                pbar.update(1)
        finally:
            pbar.close()

    cap.release()
    df = pd.DataFrame.from_records(rows, columns=cols)
    # 将输出保存到带时间戳的目录中，方便区分多次运行结果
    out_path = args.output
    base = os.path.splitext(out_path)[0]
    name = os.path.basename(out_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"{base}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, name)
    df.to_csv(out_file, index=False)
    print(f'保存完成: {out_file}，帧数: {len(df)}')

def main():
    parser = argparse.ArgumentParser(description='从视频提取 MediaPipe 关键点时间序列并保存为 CSV')
    parser.add_argument('-i', '--input', default='1.mp4', help='输入视频文件路径，默认: 1.mp4')
    parser.add_argument('-o', '--output', default='output.csv', help='输出 CSV 文件路径，默认: output.csv')
    parser.add_argument('--use-cuda', action='store_true', help='尝试使用 OpenCV CUDA 做预处理（若支持）')
    args = parser.parse_args()
    extract(args)

if __name__ == '__main__':
    main()