import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from sklearn.metrics import mean_squared_error

def load_csv(csv_path):
    """加载 CSV 文件并返回关键点矩阵 (n_frames, n_kps, 2)"""
    df = pd.read_csv(csv_path)
    n_frames = len(df)
    # 只加载 pose 关键点 (0-32)
    n_kps = 33
    kps = np.full((n_frames, n_kps, 2), np.nan, dtype=np.float32)
    for i in range(n_kps):
        kps[:, i, 0] = df[f'pose_{i}_x'].values
        kps[:, i, 1] = df[f'pose_{i}_y'].values
    return kps

def save_csv(kps, output_csv_path):
    """保存关键点矩阵为 CSV 文件"""
    n_frames, n_kps, _ = kps.shape
    columns = [f'pose_{i}_{axis}' for i in range(n_kps) for axis in ['x', 'y']]
    flattened_kps = kps.reshape(n_frames, -1)
    pd.DataFrame(flattened_kps, columns=columns).to_csv(output_csv_path, index=False)

def extract_keypoints_from_image(image_path):
    """使用 Mediapipe 从图像中提取关键点"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1, model_complexity=2)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        raise ValueError("未检测到任何关键点")
    landmarks = results.pose_landmarks.landmark
    keypoints = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    return keypoints

def auto_encode(csv_path, image_path, output_image_path, output_csv_path):
    """实现自动编码器行为"""
    # 加载原始 CSV
    original_kps = load_csv(csv_path)

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 在图像上绘制关键点 (只绘制第一个帧)
    frame_kps = original_kps[0]
    for x, y in frame_kps:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(image, (int(x), int(y)), 10, (0, 255, 0), -1)

    # 保存生成的图像
    cv2.imwrite(output_image_path, image)

    # 从生成的图像中提取关键点
    extracted_kps = extract_keypoints_from_image(output_image_path)

    # 保存提取的关键点为 CSV
    save_csv(extracted_kps[None, :, :], output_csv_path)  # 添加帧维度

    # 计算与原始关键点的误差 (只比较第一个帧)
    extracted_kps_pixel = extracted_kps * np.array([image.shape[1], image.shape[0]])
    mse = mean_squared_error(original_kps[0].reshape(-1, 2), extracted_kps_pixel.reshape(-1, 2))
    print(f"关键点重建误差 (MSE): {mse}")
    print(f"原始关键点样本: {original_kps[0][:5]}")  # 前5个
    print(f"提取关键点样本 (像素): {extracted_kps_pixel[:5]}")  # 前5个

if __name__ == "__main__":
    auto_encode(
        csv_path="s.csv",
        image_path="1.jpg",
        output_image_path="generated.jpg",
        output_csv_path="generated_keypoints.csv"
    )
