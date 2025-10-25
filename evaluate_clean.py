import argparse
import os
import cv2
import numpy as np
import pandas as pd
from math import sqrt
import mediapipe as mp
from tqdm import tqdm

def infer_count_from_csv(df, prefix):
    max_i = -1
    for col in df.columns:
        if col.startswith(prefix) and col.endswith('_x'):
            try:
                idx = int(col.split('_')[1])
                if idx > max_i:
                    max_i = idx
            except Exception:
                pass
    return max_i + 1 if max_i >= 0 else 0

def load_kps_xy_from_csv(df, prefix, count):
    # 返回 (n_frames, count, 2) 的 numpy 数组，x,y 像素或 NaN
    n = len(df)
    pts = np.full((n, count, 2), np.nan, dtype=float)
    for i in range(count):
        cx = f'{prefix}{i}_x'
        cy = f'{prefix}{i}_y'
        if cx in df.columns and cy in df.columns:
            pts[:, i, 0] = pd.to_numeric(df[cx], errors='coerce').values
            pts[:, i, 1] = pd.to_numeric(df[cy], errors='coerce').values
    return pts

def extract_from_video(video_path, target='pose', max_frames=None):
    """使用 MediaPipe 从视频提取关键点(x,y)。返回 frames, kps (n, count,2), frame_indices"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    mp_holistic = mp.solutions.holistic
    mp_face = mp.solutions.face_mesh
    frames = []
    kps_list = []
    frame_indices = []
    with (mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) if target=='pose' else mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)) as proc:
        pbar = tqdm(total=total, desc='提取关键点', unit='frame', ncols=80)
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h,w = frame.shape[:2]
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if target == 'pose':
                    res = proc.process(image)
                    lm_list = res.pose_landmarks.landmark if res.pose_landmarks else None
                    count = 33
                else:
                    res = proc.process(image)
                    lm_list = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
                    count = 468
                pts = np.full((count,2), np.nan, dtype=float)
                if lm_list:
                    for i, lm in enumerate(lm_list):
                        if i >= count:
                            break
                        # MediaPipe face/pose uses normalized coords
                        pts[i,0] = lm.x * w
                        pts[i,1] = lm.y * h
                kps_list.append(pts)
                frame_indices.append(idx)
                idx += 1
                pbar.update(1)
                if max_frames and idx >= max_frames:
                    break
        finally:
            pbar.close()
            cap.release()
    kps = np.stack(kps_list, axis=0) if kps_list else np.zeros((0, count, 2))
    return frame_indices, kps

def align_and_compute(gt_df, pred_frames, pred_kps, target, normalize_by=None):
    """
    gt_df: ground-truth dataframe (must contain 'frame' column)
    pred_frames: list of frame indices (from video) or [] if pred_kps aligns by row index
    pred_kps: (n_pred, count,2)
    target: 'pose' or 'face'
    normalize_by: None or scalar to divide errors (e.g., image diagonal)
    """
    prefix = 'pose_' if target=='pose' else 'face_'
    count = infer_count_from_csv(gt_df, prefix)
    gt_kps = load_kps_xy_from_csv(gt_df, prefix, count)
    # build mapping: if pred_frames provided align by frame number, else assume same ordering and same length
    if pred_frames:
        # create dict frame->index
        frame_to_pred = {f:i for i,f in enumerate(pred_frames)}
        paired = []
        gt_frames = gt_df['frame'].astype(int).values if 'frame' in gt_df.columns else np.arange(len(gt_df))
        for gi, gframe in enumerate(gt_frames):
            if gframe in frame_to_pred:
                pi = frame_to_pred[gframe]
                paired.append((gi, pi))
    else:
        # align by index range
        n = min(gt_kps.shape[0], pred_kps.shape[0])
        paired = [(i,i) for i in range(n)]
    if not paired:
        raise RuntimeError('没有重叠的帧用于比较')
    # accumulate errors
    per_keypoint_errors = []  # list of (count, ) arrays of squared errors mean per kp per frame
    per_frame_stats = []
    for gi, pi in paired:
        gt_pts = gt_kps[gi]
        pr_pts = pred_kps[pi]
        # valid mask where both present
        valid = ~np.isnan(gt_pts[:,0]) & ~np.isnan(gt_pts[:,1]) & ~np.isnan(pr_pts[:,0]) & ~np.isnan(pr_pts[:,1])
        if not np.any(valid):
            per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,
                                    'n_valid':0, 'mse':np.nan, 'mae':np.nan, 'rmse':np.nan})
            per_keypoint_errors.append(np.full((count,), np.nan))
            continue
        dif = pr_pts[valid] - gt_pts[valid]
        se = np.sum(dif**2, axis=1)  # per valid kp squared error (x^2+y^2)
        ae = np.sqrt(se)           # per-kp euclidean error
        if normalize_by:
            se = se / (normalize_by**2)
            ae = ae / normalize_by
        mse = float(np.mean(se))
        mae = float(np.mean(ae))
        rmse = float(np.sqrt(np.mean(se)))
        # build per-keypoint: set NaN where invalid
        pk = np.full((count,), np.nan)
        pk[valid] = se  # store squared error per kp
        per_keypoint_errors.append(pk)
        per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,
                                'n_valid': int(np.sum(valid)), 'mse':mse, 'mae':mae, 'rmse':rmse})
    # aggregate across frames
    per_keypoint_errors = np.stack(per_keypoint_errors, axis=0)  # (n_pairs, count)
    mean_se_per_kp = np.nanmean(per_keypoint_errors, axis=0)  # per-kp mean squared error
    overall_mse = np.nanmean(per_keypoint_errors)
    overall_rmse = sqrt(float(overall_mse)) if not np.isnan(overall_mse) else np.nan
    return {
        'per_frame': pd.DataFrame(per_frame_stats),
        'per_kp_mse': mean_se_per_kp,
        'overall_mse': float(overall_mse),
        'overall_rmse': float(overall_rmse),
        'n_pairs': len(paired)
    }

def main():
    parser = argparse.ArgumentParser(description='评估关键点误差：支持从视频实时提取或用预测 CSV 对比 ground-truth CSV')
    parser.add_argument('--gt-csv', required=True, help='ground-truth CSV（extract_keypoints.py 输出）')
    parser.add_argument('--pred-csv', help='预测 CSV（同样格式）；若未提供则使用 --video 提取关键点')
    parser.add_argument('--video', help='视频文件路径（当未提供 pred-csv 时使用）')
    parser.add_argument('--target', choices=['pose','face'], default='pose', help='评估目标关键点类型')
    parser.add_argument('--normalize', action='store_true', help='按图像对角线进行归一化（便于不同分辨率比较）')
    parser.add_argument('--max-frames', type=int, help='限制从视频提取的帧数（可选）')
    parser.add_argument('--output', help='保存评估结果 CSV 的路径（可选）')
    args = parser.parse_args()

    gt_df = pd.read_csv(args.gt_csv)
    # load predictions
    if args.pred_csv:
        pred_df = pd.read_csv(args.pred_csv)
        prefix = 'pose_' if args.target=='pose' else 'face_'
        count = infer_count_from_csv(gt_df, prefix)
        pred_kps = load_kps_xy_from_csv(pred_df, prefix, count)
        pred_frames = None
    elif args.video:
        pred_frames, pred_kps = extract_from_video(args.video, target=args.target, max_frames=args.max_frames)
        # pred_kps shape (n, count,2)
    else:
        raise RuntimeError('需要提供 --pred-csv 或 --video 作为预测来源')

    # Determine normalization scalar if requested
    normalize_by = None
    if args.normalize:
        # try get image size from video or gt info: prefer video
        if args.video:
            cap = cv2.VideoCapture(args.video)
            if cap.isOpened():
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap.release()
                normalize_by = np.hypot(w, h)
        if normalize_by is None:
            # fallback: try to infer from first frame values in gt_df
            if 'frame' in gt_df.columns:
                # attempt to parse any pose_x values to find max width/height (not reliable)
                normalize_by = 1.0

    res = align_and_compute(gt_df, pred_frames if not args.pred_csv else None, pred_kps, args.target, normalize_by=normalize_by)
    print("评估结果摘要:")
    print(f"配对帧数: {res['n_pairs']}")
    print(f"Overall MSE: {res['overall_mse']:.4f}")
    print(f"Overall RMSE: {res['overall_rmse']:.4f}")
    # per-frame stats head
    print("\n每帧统计（前10）:")
    print(res['per_frame'].head(10).to_string(index=False))

    if args.output:
        out_dir = os.path.dirname(args.output) or '.'
        os.makedirs(out_dir, exist_ok=True)
        # save per-frame and per-keypoint summary
        res['per_frame'].to_csv(args.output, index=False)
        print(f"已保存每帧统计到: {args.output}")
        # save per-keypoint mse as npy for convenience
        kp_file = os.path.splitext(args.output)[0] + f'_{args.target}_per_kp_mse.npy'
        np.save(kp_file, res['per_kp_mse'])
        print(f"已保存每关键点 MSE 到: {kp_file}")

if __name__ == '__main__':
    main()