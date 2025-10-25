import argparseimport argparseimport argparseimport argparse

import os

import cv2import cv2

import numpy as np

import pandas as pdimport numpy as npimport cv2import cv2

from math import sqrt

import mediapipe as mpimport pandas as pd

from tqdm import tqdm

import mediapipe as mpimport numpy as npimport numpy as np

def infer_count_from_csv(df, prefix):

    max_i = -1import os

    for col in df.columns:

        if col.startswith(prefix) and col.endswith('_x'):from tqdm import tqdmimport pandas as pdimport pandas as pd

            try:

                idx = int(col.split('_')[1])from math import sqrt

                if idx > max_i:

                    max_i = idximport mediapipe as mpimport mediapipe as mp

            except Exception:

                passdef infer_count_from_csv(df, prefix):

    return max_i + 1 if max_i >= 0 else 0

    max_i = -1import osimport os

def load_kps_xy_from_csv(df, prefix, count):

    # 返回 (n_frames, count, 2) 的 numpy 数组，x,y 像素或 NaN    for col in df.columns:

    n = len(df)

    pts = np.full((n, count, 2), np.nan, dtype=float)        if col.startswith(prefix) and col.endswith('_x'):from tqdm import tqdmfrom tqdm import tqdm

    for i in range(count):

        cx = f'{prefix}{i}_x'            try:

        cy = f'{prefix}{i}_y'

        if cx in df.columns and cy in df.columns:                idx = int(col.split('_')[1])from math import sqrtfrom math import sqrt

            pts[:, i, 0] = pd.to_numeric(df[cx], errors='coerce').values

            pts[:, i, 1] = pd.to_numeric(df[cy], errors='coerce').values                if idx > max_i:

    return pts

                    max_i = idx

def extract_from_video(video_path, target='pose', max_frames=None):

    """使用 MediaPipe 从视频提取关键点(x,y)。返回 frames, kps (n, count,2), frame_indices"""            except Exception:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():                passdef infer_count_from_csv(df, prefix):def infer_count_from_csv(df, prefix):

        raise RuntimeError(f'无法打开视频: {video_path}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None    return max_i + 1 if max_i >= 0 else 0

    mp_holistic = mp.solutions.holistic

    mp_face = mp.solutions.face_mesh    max_i = -1    max_i = -1

    frames = []

    kps_list = []def load_kps_xy_from_csv(df, prefix, count):

    frame_indices = []

    with (mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) if target=='pose' else mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)) as proc:    # 返回 (n_frames, count, 2) 的 numpy 数组，x,y 像素或 NaN    for col in df.columns:    for col in df.columns:

        pbar = tqdm(total=total, desc='提取关键点', unit='frame', ncols=80)

        idx = 0    n = len(df)

        try:

            while True:    pts = np.full((n, count, 2), np.nan, dtype=float)        if col.startswith(prefix) and col.endswith('_x'):        import argparse

                ret, frame = cap.read()

                if not ret:    for i in range(count):

                    break

                h,w = frame.shape[:2]        cx = f'{prefix}{i}_x'            try:import cv2

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if target == 'pose':        cy = f'{prefix}{i}_y'

                    res = proc.process(image)

                    lm_list = res.pose_landmarks.landmark if res.pose_landmarks else None        if cx in df.columns and cy in df.columns:                idx = int(col.split('_')[1])import numpy as np

                    count = 33

                else:            pts[:, i, 0] = pd.to_numeric(df[cx], errors='coerce').values

                    res = proc.process(image)

                    lm_list = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None            pts[:, i, 1] = pd.to_numeric(df[cy], errors='coerce').values                if idx > max_i:import pandas as pd

                    count = 468

                pts = np.full((count,2), np.nan, dtype=float)    return pts

                if lm_list:

                    for i, lm in enumerate(lm_list):                    max_i = idximport mediapipe as mp

                        if i >= count:

                            breakdef extract_from_video(video_path, target='pose', max_frames=None):

                        # MediaPipe face/pose uses normalized coords

                        pts[i,0] = lm.x * w    """使用 MediaPipe 从视频提取关键点(x,y)。返回 frames, kps (n, count,2), frame_indices"""            except Exception:import os

                        pts[i,1] = lm.y * h

                kps_list.append(pts)    cap = cv2.VideoCapture(video_path)

                frame_indices.append(idx)

                idx += 1    if not cap.isOpened():                passfrom tqdm import tqdm

                pbar.update(1)

                if max_frames and idx >= max_frames:        raise RuntimeError(f'无法打开视频: {video_path}')

                    break

        finally:    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None    return max_i + 1 if max_i >= 0 else 0from math import sqrt

            pbar.close()

            cap.release()    mp_holistic = mp.solutions.holistic

    kps = np.stack(kps_list, axis=0) if kps_list else np.zeros((0, count, 2))

    return frame_indices, kps    mp_face = mp.solutions.face_mesh



def align_and_compute(gt_df, pred_frames, pred_kps, target, normalize_by=None):    frames = []

    """

    gt_df: ground-truth dataframe (must contain 'frame' column)    kps_list = []def load_kps_xy_from_csv(df, prefix, count):def infer_count_from_csv(df, prefix):

    pred_frames: list of frame indices (from video) or [] if pred_kps aligns by row index

    pred_kps: (n_pred, count,2)    frame_indices = []

    target: 'pose' or 'face'

    normalize_by: None or scalar to divide errors (e.g., image diagonal)    with (mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) if target=='pose' else mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)) as proc:    # 返回 (n_frames, count, 2) 的 numpy 数组，x,y 像素或 NaN    max_i = -1

    """

    prefix = 'pose_' if target=='pose' else 'face_'        pbar = tqdm(total=total, desc='提取关键点', unit='frame', ncols=80)

    count = infer_count_from_csv(gt_df, prefix)

    gt_kps = load_kps_xy_from_csv(gt_df, prefix, count)        idx = 0    n = len(df)    for col in df.columns:

    # build mapping: if pred_frames provided align by frame number, else assume same ordering and same length

    if pred_frames:        try:

        # create dict frame->index

        frame_to_pred = {f:i for i,f in enumerate(pred_frames)}            while True:    pts = np.full((n, count, 2), np.nan, dtype=float)        if col.startswith(prefix) and col.endswith('_x'):

        paired = []

        gt_frames = gt_df['frame'].astype(int).values if 'frame' in gt_df.columns else np.arange(len(gt_df))                ret, frame = cap.read()

        for gi, gframe in enumerate(gt_frames):

            if gframe in frame_to_pred:                if not ret:    for i in range(count):            try:

                pi = frame_to_pred[gframe]

                paired.append((gi, pi))                    break

    else:

        # align by index range                h,w = frame.shape[:2]        cx = f'{prefix}{i}_x'                idx = int(col.split('_')[1])

        n = min(gt_kps.shape[0], pred_kps.shape[0])

        paired = [(i,i) for i in range(n)]                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not paired:

        raise RuntimeError('没有重叠的帧用于比较')                if target == 'pose':        cy = f'{prefix}{i}_y'                if idx > max_i:

    # accumulate errors

    per_keypoint_errors = []  # list of (count, ) arrays of squared errors mean per kp per frame                    res = proc.process(image)

    per_frame_stats = []

    for gi, pi in paired:                    lm_list = res.pose_landmarks.landmark if res.pose_landmarks else None        if cx in df.columns and cy in df.columns:                    max_i = idx

        gt_pts = gt_kps[gi]

        pr_pts = pred_kps[pi]                    count = 33

        # valid mask where both present

        valid = ~np.isnan(gt_pts[:,0]) & ~np.isnan(gt_pts[:,1]) & ~np.isnan(pr_pts[:,0]) & ~np.isnan(pr_pts[:,1])                else:            pts[:, i, 0] = pd.to_numeric(df[cx], errors='coerce').values            except Exception:

        if not np.any(valid):

            per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,                    res = proc.process(image)

                                    'n_valid':0, 'mse':np.nan, 'mae':np.nan, 'rmse':np.nan})

            per_keypoint_errors.append(np.full((count,), np.nan))                    lm_list = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None            pts[:, i, 1] = pd.to_numeric(df[cy], errors='coerce').values                pass

            continue

        dif = pr_pts[valid] - gt_pts[valid]                    count = 468

        se = np.sum(dif**2, axis=1)  # per valid kp squared error (x^2+y^2)

        ae = np.sqrt(se)           # per-kp euclidean error                pts = np.full((count,2), np.nan, dtype=float)    return pts    return max_i + 1 if max_i >= 0 else 0

        if normalize_by:

            se = se / (normalize_by**2)                if lm_list:

            ae = ae / normalize_by

        mse = float(np.mean(se))                    for i, lm in enumerate(lm_list):

        mae = float(np.mean(ae))

        rmse = float(np.sqrt(np.mean(se)))                        if i >= count:

        # build per-keypoint: set NaN where invalid

        pk = np.full((count,), np.nan)                            breakdef extract_from_video(video_path, target='pose', max_frames=None):def load_kps_xy_from_csv(df, prefix, count):

        pk[valid] = se  # store squared error per kp

        per_keypoint_errors.append(pk)                        # MediaPipe face/pose uses normalized coords

        per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,

                                'n_valid': int(np.sum(valid)), 'mse':mse, 'mae':mae, 'rmse':rmse})                        pts[i,0] = lm.x * w    """使用 MediaPipe 从视频提取关键点(x,y)。返回 frames, kps (n, count,2), frame_indices"""    # 返回 (n_frames, count, 2) 的 numpy 数组，x,y 像素或 NaN

    # aggregate across frames

    per_keypoint_errors = np.stack(per_keypoint_errors, axis=0)  # (n_pairs, count)                        pts[i,1] = lm.y * h

    mean_se_per_kp = np.nanmean(per_keypoint_errors, axis=0)  # per-kp mean squared error

    overall_mse = np.nanmean(per_keypoint_errors)                kps_list.append(pts)    cap = cv2.VideoCapture(video_path)    n = len(df)

    overall_rmse = sqrt(float(overall_mse)) if not np.isnan(overall_mse) else np.nan

    return {                frame_indices.append(idx)

        'per_frame': pd.DataFrame(per_frame_stats),

        'per_kp_mse': mean_se_per_kp,                idx += 1    if not cap.isOpened():    pts = np.full((n, count, 2), np.nan, dtype=float)

        'overall_mse': float(overall_mse),

        'overall_rmse': float(overall_rmse),                pbar.update(1)

        'n_pairs': len(paired)

    }                if max_frames and idx >= max_frames:        raise RuntimeError(f'无法打开视频: {video_path}')    for i in range(count):



def compute_ssim(img1, img2, data_range=255):                    break

    """简单 SSIM 计算"""

    C1 = (0.01 * data_range) ** 2        finally:    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None        cx = f'{prefix}{i}_x'

    C2 = (0.03 * data_range) ** 2

                pbar.close()

    img1 = img1.astype(float)

    img2 = img2.astype(float)            cap.release()    mp_holistic = mp.solutions.holistic        cy = f'{prefix}{i}_y'

    

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)    kps = np.stack(kps_list, axis=0) if kps_list else np.zeros((0, count, 2))

    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        return frame_indices, kps    mp_face = mp.solutions.face_mesh        if cx in df.columns and cy in df.columns:

    mu1_sq = mu1 ** 2

    mu2_sq = mu2 ** 2

    mu1_mu2 = mu1 * mu2

    def align_and_compute(gt_df, pred_frames, pred_kps, target, normalize_by=None):    frames = []            pts[:, i, 0] = pd.to_numeric(df[cx], errors='coerce').values

    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq

    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq    """

    sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2

        gt_df: ground-truth dataframe (must contain 'frame' column)    kps_list = []            pts[:, i, 1] = pd.to_numeric(df[cy], errors='coerce').values

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)

    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)    pred_frames: list of frame indices (from video) or [] if pred_kps aligns by row index

    

    ssim_map = numerator / denominator    pred_kps: (n_pred, count,2)    frame_indices = []    return pts

    return np.mean(ssim_map)

    target: 'pose' or 'face'

def compute_image_quality(original_video, warped_video, max_frames=None):

    """    normalize_by: None or scalar to divide errors (e.g., image diagonal)    with (mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) if target=='pose' else mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)) as proc:

    计算两个视频之间的图像质量指标：SSIM, PSNR, MSE

    返回每帧的指标和平均值    """

    """

    cap_orig = cv2.VideoCapture(original_video)    prefix = 'pose_' if target=='pose' else 'face_'        pbar = tqdm(total=total, desc='提取关键点', unit='frame', ncols=80)def extract_from_video(video_path, target='pose', max_frames=None):

    cap_warp = cv2.VideoCapture(warped_video)

        count = infer_count_from_csv(gt_df, prefix)

    if not cap_orig.isOpened() or not cap_warp.isOpened():

        raise RuntimeError('无法打开视频文件')    gt_kps = load_kps_xy_from_csv(gt_df, prefix, count)        idx = 0    """使用 MediaPipe 从视频提取关键点(x,y)。返回 frames, kps (n, count,2), frame_indices"""

    

    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))    # build mapping: if pred_frames provided align by frame number, else assume same ordering and same length

    total_warp = int(cap_warp.get(cv2.CAP_PROP_FRAME_COUNT))

    total = min(total_orig, total_warp)    if pred_frames:        try:    cap = cv2.VideoCapture(video_path)

    if max_frames:

        total = min(total, max_frames)        # create dict frame->index

    

    ssim_scores = []        frame_to_pred = {f:i for i,f in enumerate(pred_frames)}            while True:    if not cap.isOpened():

    psnr_scores = []

    mse_scores = []        paired = []

    frame_indices = []

            gt_frames = gt_df['frame'].astype(int).values if 'frame' in gt_df.columns else np.arange(len(gt_df))                ret, frame = cap.read()        raise RuntimeError(f'无法打开视频: {video_path}')

    for idx in tqdm(range(total), desc='计算图像质量', unit='frame', ncols=80):

        ret_orig, frame_orig = cap_orig.read()        for gi, gframe in enumerate(gt_frames):

        ret_warp, frame_warp = cap_warp.read()

                    if gframe in frame_to_pred:                if not ret:    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

        if not ret_orig or not ret_warp:

            break                pi = frame_to_pred[gframe]

        

        # 确保帧尺寸相同                paired.append((gi, pi))                    break    mp_holistic = mp.solutions.holistic

        if frame_orig.shape != frame_warp.shape:

            h, w = min(frame_orig.shape[0], frame_warp.shape[0]), min(frame_orig.shape[1], frame_warp.shape[1])    else:

            frame_orig = cv2.resize(frame_orig, (w, h))

            frame_warp = cv2.resize(frame_warp, (w, h))        # align by index range                h,w = frame.shape[:2]    mp_face = mp.solutions.face_mesh

        

        # 计算指标        n = min(gt_kps.shape[0], pred_kps.shape[0])

        ssim_val = compute_ssim(frame_orig, frame_warp)

        psnr_val = cv2.PSNR(frame_orig, frame_warp)        paired = [(i,i) for i in range(n)]                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    frames = []

        mse_val = np.mean((frame_orig.astype(float) - frame_warp.astype(float)) ** 2)

            if not paired:

        ssim_scores.append(ssim_val)

        psnr_scores.append(psnr_val)        raise RuntimeError('没有重叠的帧用于比较')                if target == 'pose':    kps_list = []

        mse_scores.append(mse_val)

        frame_indices.append(idx)    # accumulate errors

    

    cap_orig.release()    per_keypoint_errors = []  # list of (count, ) arrays of squared errors mean per kp per frame                    res = proc.process(image)    frame_indices = []

    cap_warp.release()

        per_frame_stats = []

    if not ssim_scores:

        return None    for gi, pi in paired:                    lm_list = res.pose_landmarks.landmark if res.pose_landmarks else None    with (mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) if target=='pose' else mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)) as proc:

    

    return {        gt_pts = gt_kps[gi]

        'per_frame': pd.DataFrame({

            'frame': frame_indices,        pr_pts = pred_kps[pi]                    count = 33        pbar = tqdm(total=total, desc='提取关键点', unit='frame', ncols=80)

            'ssim': ssim_scores,

            'psnr': psnr_scores,        # valid mask where both present

            'mse': mse_scores

        }),        valid = ~np.isnan(gt_pts[:,0]) & ~np.isnan(gt_pts[:,1]) & ~np.isnan(pr_pts[:,0]) & ~np.isnan(pr_pts[:,1])                else:        idx = 0

        'avg_ssim': np.mean(ssim_scores),

        'avg_psnr': np.mean(psnr_scores),        if not np.any(valid):

        'avg_mse': np.mean(mse_scores)

    }            per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,                    res = proc.process(image)        try:



def main():                                    'n_valid':0, 'mse':np.nan, 'mae':np.nan, 'rmse':np.nan})

    parser = argparse.ArgumentParser(description='评估关键点误差：支持从视频实时提取或用预测 CSV 对比 ground-truth CSV')

    parser.add_argument('--gt-csv', required=True, help='ground-truth CSV（extract_keypoints.py 输出）')            per_keypoint_errors.append(np.full((count,), np.nan))                    lm_list = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None            while True:

    parser.add_argument('--pred-csv', help='预测 CSV（同样格式）；若未提供则使用 --video 提取关键点')

    parser.add_argument('--video', help='视频文件路径（当未提供 pred-csv 时使用）')            continue

    parser.add_argument('--target', choices=['pose','face'], default='pose', help='评估目标关键点类型')

    parser.add_argument('--normalize', action='store_true', help='按图像对角线进行归一化（便于不同分辨率比较）')        dif = pr_pts[valid] - gt_pts[valid]                    count = 468                ret, frame = cap.read()

    parser.add_argument('--max-frames', type=int, help='限制从视频提取的帧数（可选）')

    parser.add_argument('--output', help='保存评估结果 CSV 的路径（可选）')        se = np.sum(dif**2, axis=1)  # per valid kp squared error (x^2+y^2)

    parser.add_argument('--original-video', help='原始视频路径，用于计算图像质量指标（SSIM, PSNR, MSE）与变形视频比较')

    parser.add_argument('--warped-video', help='变形视频路径，用于计算图像质量指标（与 --original-video 一起使用）')        ae = np.sqrt(se)           # per-kp euclidean error                pts = np.full((count,2), np.nan, dtype=float)                if not ret:

    args = parser.parse_args()

        if normalize_by:

    gt_df = pd.read_csv(args.gt_csv)

    # load predictions            se = se / (normalize_by**2)                if lm_list:                    break

    if args.pred_csv:

        pred_df = pd.read_csv(args.pred_csv)            ae = ae / normalize_by

        prefix = 'pose_' if args.target=='pose' else 'face_'

        count = infer_count_from_csv(gt_df, prefix)        mse = float(np.mean(se))                    for i, lm in enumerate(lm_list):                h,w = frame.shape[:2]

        pred_kps = load_kps_xy_from_csv(pred_df, prefix, count)

        pred_frames = None        mae = float(np.mean(ae))

    elif args.video:

        pred_frames, pred_kps = extract_from_video(args.video, target=args.target, max_frames=args.max_frames)        rmse = float(np.sqrt(np.mean(se)))                        if i >= count:                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # pred_kps shape (n, count,2)

    else:        # build per-keypoint: set NaN where invalid

        raise RuntimeError('需要提供 --pred-csv 或 --video 作为预测来源')

        pk = np.full((count,), np.nan)                            break                if target == 'pose':

    # Determine normalization scalar if requested

    normalize_by = None        pk[valid] = se  # store squared error per kp

    if args.normalize:

        # try get image size from video or gt info: prefer video        per_keypoint_errors.append(pk)                        # MediaPipe face/pose uses normalized coords                    res = proc.process(image)

        if args.video:

            cap = cv2.VideoCapture(args.video)        per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,

            if cap.isOpened():

                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                                'n_valid': int(np.sum(valid)), 'mse':mse, 'mae':mae, 'rmse':rmse})                        pts[i,0] = lm.x * w                    lm_list = res.pose_landmarks.landmark if res.pose_landmarks else None

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                cap.release()    # aggregate across frames

                normalize_by = np.hypot(w, h)

        if normalize_by is None:    per_keypoint_errors = np.stack(per_keypoint_errors, axis=0)  # (n_pairs, count)                        pts[i,1] = lm.y * h                    count = 33

            # fallback: try to infer from first frame values in gt_df

            if 'frame' in gt_df.columns:    mean_se_per_kp = np.nanmean(per_keypoint_errors, axis=0)  # per-kp mean squared error

                # attempt to parse any pose_x values to find max width/height (not reliable)

                normalize_by = 1.0    overall_mse = np.nanmean(per_keypoint_errors)                kps_list.append(pts)                else:



    res = align_and_compute(gt_df, pred_frames if not args.pred_csv else None, pred_kps, args.target, normalize_by=normalize_by)    overall_rmse = sqrt(float(overall_mse)) if not np.isnan(overall_mse) else np.nan

    print("评估结果摘要:")

    print(f"配对帧数: {res['n_pairs']}")    return {                frame_indices.append(idx)                    res = proc.process(image)

    print(f"Overall MSE: {res['overall_mse']:.4f}")

    print(f"Overall RMSE: {res['overall_rmse']:.4f}")        'per_frame': pd.DataFrame(per_frame_stats),

    # per-frame stats head

    print("\n每帧统计（前10）:")        'per_kp_mse': mean_se_per_kp,                idx += 1                    lm_list = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None

    print(res['per_frame'].head(10).to_string(index=False))

        'overall_mse': float(overall_mse),

    if args.output:

        out_dir = os.path.dirname(args.output) or '.'        'overall_rmse': float(overall_rmse),                pbar.update(1)                    count = 468

        os.makedirs(out_dir, exist_ok=True)

        # save per-frame and per-keypoint summary        'n_pairs': len(paired)

        res['per_frame'].to_csv(args.output, index=False)

        print(f"已保存每帧统计到: {args.output}")    }                if max_frames and idx >= max_frames:                pts = np.full((count,2), np.nan, dtype=float)

        # save per-keypoint mse as npy for convenience

        kp_file = os.path.splitext(args.output)[0] + f'_{args.target}_per_kp_mse.npy'

        np.save(kp_file, res['per_kp_mse'])

        print(f"已保存每关键点 MSE 到: {kp_file}")def compute_ssim(img1, img2, data_range=255):                    break                if lm_list:



    # 计算图像质量指标（如果提供原始和变形视频）    """简单 SSIM 计算"""

    if args.original_video and args.warped_video:

        print("\n计算图像质量指标...")    C1 = (0.01 * data_range) ** 2        finally:                    for i, lm in enumerate(lm_list):

        img_quality = compute_image_quality(args.original_video, args.warped_video, max_frames=args.max_frames)

        if img_quality:    C2 = (0.03 * data_range) ** 2

            print("图像质量评估结果:")

            print(f"平均 SSIM: {img_quality['avg_ssim']:.4f}")                pbar.close()                        if i >= count:

            print(f"平均 PSNR: {img_quality['avg_psnr']:.4f}")

            print(f"平均 MSE: {img_quality['avg_mse']:.4f}")    img1 = img1.astype(float)

            print("\n每帧图像质量（前10）:")

            print(img_quality['per_frame'].head(10).to_string(index=False))    img2 = img2.astype(float)            cap.release()                            break

            

            if args.output:    

                img_file = os.path.splitext(args.output)[0] + '_image_quality.csv'

                img_quality['per_frame'].to_csv(img_file, index=False)    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)    kps = np.stack(kps_list, axis=0) if kps_list else np.zeros((0, count, 2))                        # MediaPipe face/pose uses normalized coords

                print(f"已保存图像质量统计到: {img_file}")

        else:    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

            print("无法计算图像质量指标")

        return frame_indices, kps                        pts[i,0] = lm.x * w

if __name__ == '__main__':

    main()    mu1_sq = mu1 ** 2

    mu2_sq = mu2 ** 2                        pts[i,1] = lm.y * h

    mu1_mu2 = mu1 * mu2

    def align_and_compute(gt_df, pred_frames, pred_kps, target, normalize_by=None):                kps_list.append(pts)

    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq

    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq    """                frame_indices.append(idx)

    sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2

        gt_df: ground-truth dataframe (must contain 'frame' column)                idx += 1

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)

    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)    pred_frames: list of frame indices (from video) or [] if pred_kps aligns by row index                pbar.update(1)

    

    ssim_map = numerator / denominator    pred_kps: (n_pred, count,2)                if max_frames and idx >= max_frames:

    return np.mean(ssim_map)

    target: 'pose' or 'face'                    break

def compute_image_quality(original_video, warped_video, max_frames=None):

    """    normalize_by: None or scalar to divide errors (e.g., image diagonal)        finally:

    计算两个视频之间的图像质量指标：SSIM, PSNR, MSE

    返回每帧的指标和平均值    """            pbar.close()

    """

    cap_orig = cv2.VideoCapture(original_video)    prefix = 'pose_' if target=='pose' else 'face_'            cap.release()

    cap_warp = cv2.VideoCapture(warped_video)

        count = infer_count_from_csv(gt_df, prefix)    kps = np.stack(kps_list, axis=0) if kps_list else np.zeros((0, count, 2))

    if not cap_orig.isOpened() or not cap_warp.isOpened():

        raise RuntimeError('无法打开视频文件')    gt_kps = load_kps_xy_from_csv(gt_df, prefix, count)    return frame_indices, kps

    

    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))    # build mapping: if pred_frames provided align by frame number, else assume same ordering and same length

    total_warp = int(cap_warp.get(cv2.CAP_PROP_FRAME_COUNT))

    total = min(total_orig, total_warp)    if pred_frames:def align_and_compute(gt_df, pred_frames, pred_kps, target, normalize_by=None):

    if max_frames:

        total = min(total, max_frames)        # create dict frame->index    """

    

    ssim_scores = []        frame_to_pred = {f:i for i,f in enumerate(pred_frames)}    gt_df: ground-truth dataframe (must contain 'frame' column)

    psnr_scores = []

    mse_scores = []        paired = []    pred_frames: list of frame indices (from video) or [] if pred_kps aligns by row index

    frame_indices = []

            gt_frames = gt_df['frame'].astype(int).values if 'frame' in gt_df.columns else np.arange(len(gt_df))    pred_kps: (n_pred, count,2)

    for idx in tqdm(range(total), desc='计算图像质量', unit='frame', ncols=80):

        ret_orig, frame_orig = cap_orig.read()        for gi, gframe in enumerate(gt_frames):    target: 'pose' or 'face'

        ret_warp, frame_warp = cap_warp.read()

                    if gframe in frame_to_pred:    normalize_by: None or scalar to divide errors (e.g., image diagonal)

        if not ret_orig or not ret_warp:

            break                pi = frame_to_pred[gframe]    """

        

        # 确保帧尺寸相同                paired.append((gi, pi))    prefix = 'pose_' if target=='pose' else 'face_'

        if frame_orig.shape != frame_warp.shape:

            h, w = min(frame_orig.shape[0], frame_warp.shape[0]), min(frame_orig.shape[1], frame_warp.shape[1])    else:    count = infer_count_from_csv(gt_df, prefix)

            frame_orig = cv2.resize(frame_orig, (w, h))

            frame_warp = cv2.resize(frame_warp, (w, h))        # align by index range    gt_kps = load_kps_xy_from_csv(gt_df, prefix, count)

        

        # 计算指标        n = min(gt_kps.shape[0], pred_kps.shape[0])    # build mapping: if pred_frames provided align by frame number, else assume same ordering and same length

        ssim_val = compute_ssim(frame_orig, frame_warp)

        psnr_val = cv2.PSNR(frame_orig, frame_warp)        paired = [(i,i) for i in range(n)]    if pred_frames:

        mse_val = np.mean((frame_orig.astype(float) - frame_warp.astype(float)) ** 2)

            if not paired:        # create dict frame->index

        ssim_scores.append(ssim_val)

        psnr_scores.append(psnr_val)        raise RuntimeError('没有重叠的帧用于比较')        frame_to_pred = {f:i for i,f in enumerate(pred_frames)}

        mse_scores.append(mse_val)

        frame_indices.append(idx)    # accumulate errors        paired = []

    

    cap_orig.release()    per_keypoint_errors = []  # list of (count, ) arrays of squared errors mean per kp per frame        gt_frames = gt_df['frame'].astype(int).values if 'frame' in gt_df.columns else np.arange(len(gt_df))

    cap_warp.release()

        per_frame_stats = []        for gi, gframe in enumerate(gt_frames):

    if not ssim_scores:

        return None    for gi, pi in paired:            if gframe in frame_to_pred:

    

    return {        gt_pts = gt_kps[gi]                pi = frame_to_pred[gframe]

        'per_frame': pd.DataFrame({

            'frame': frame_indices,        pr_pts = pred_kps[pi]                paired.append((gi, pi))

            'ssim': ssim_scores,

            'psnr': psnr_scores,        # valid mask where both present    else:

            'mse': mse_scores

        }),        valid = ~np.isnan(gt_pts[:,0]) & ~np.isnan(gt_pts[:,1]) & ~np.isnan(pr_pts[:,0]) & ~np.isnan(pr_pts[:,1])        # align by index range

        'avg_ssim': np.mean(ssim_scores),

        'avg_psnr': np.mean(psnr_scores),        if not np.any(valid):        n = min(gt_kps.shape[0], pred_kps.shape[0])

        'avg_mse': np.mean(mse_scores)

    }            per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,        paired = [(i,i) for i in range(n)]



def main():                                    'n_valid':0, 'mse':np.nan, 'mae':np.nan, 'rmse':np.nan})    if not paired:

    parser = argparse.ArgumentParser(description='评估关键点误差：支持从视频实时提取或用预测 CSV 对比 ground-truth CSV')

    parser.add_argument('--gt-csv', required=True, help='ground-truth CSV（extract_keypoints.py 输出）')            per_keypoint_errors.append(np.full((count,), np.nan))        raise RuntimeError('没有重叠的帧用于比较')

    parser.add_argument('--pred-csv', help='预测 CSV（同样格式）；若未提供则使用 --video 提取关键点')

    parser.add_argument('--video', help='视频文件路径（当未提供 pred-csv 时使用）')            continue    # accumulate errors

    parser.add_argument('--target', choices=['pose','face'], default='pose', help='评估目标关键点类型')

    parser.add_argument('--normalize', action='store_true', help='按图像对角线进行归一化（便于不同分辨率比较）')        dif = pr_pts[valid] - gt_pts[valid]    per_keypoint_errors = []  # list of (count, ) arrays of squared errors mean per kp per frame

    parser.add_argument('--max-frames', type=int, help='限制从视频提取的帧数（可选）')

    parser.add_argument('--output', help='保存评估结果 CSV 的路径（可选）')        se = np.sum(dif**2, axis=1)  # per valid kp squared error (x^2+y^2)    per_frame_stats = []

    parser.add_argument('--original-video', help='原始视频路径，用于计算图像质量指标（SSIM, PSNR, MSE）与变形视频比较')

    parser.add_argument('--warped-video', help='变形视频路径，用于计算图像质量指标（与 --original-video 一起使用）')        ae = np.sqrt(se)           # per-kp euclidean error    for gi, pi in paired:

    args = parser.parse_args()

        if normalize_by:        gt_pts = gt_kps[gi]

    gt_df = pd.read_csv(args.gt_csv)

    # load predictions            se = se / (normalize_by**2)        pr_pts = pred_kps[pi]

    if args.pred_csv:

        pred_df = pd.read_csv(args.pred_csv)            ae = ae / normalize_by        # valid mask where both present

        prefix = 'pose_' if args.target=='pose' else 'face_'

        count = infer_count_from_csv(gt_df, prefix)        mse = float(np.mean(se))        valid = ~np.isnan(gt_pts[:,0]) & ~np.isnan(gt_pts[:,1]) & ~np.isnan(pr_pts[:,0]) & ~np.isnan(pr_pts[:,1])

        pred_kps = load_kps_xy_from_csv(pred_df, prefix, count)

        pred_frames = None        mae = float(np.mean(ae))        if not np.any(valid):

    elif args.video:

        pred_frames, pred_kps = extract_from_video(args.video, target=args.target, max_frames=args.max_frames)        rmse = float(np.sqrt(np.mean(se)))            per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,

        # pred_kps shape (n, count,2)

    else:        # build per-keypoint: set NaN where invalid                                    'n_valid':0, 'mse':np.nan, 'mae':np.nan, 'rmse':np.nan})

        raise RuntimeError('需要提供 --pred-csv 或 --video 作为预测来源')

        pk = np.full((count,), np.nan)            per_keypoint_errors.append(np.full((count,), np.nan))

    # Determine normalization scalar if requested

    normalize_by = None        pk[valid] = se  # store squared error per kp            continue

    if args.normalize:

        # try get image size from video or gt info: prefer video        per_keypoint_errors.append(pk)        dif = pr_pts[valid] - gt_pts[valid]

        if args.video:

            cap = cv2.VideoCapture(args.video)        per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,        se = np.sum(dif**2, axis=1)  # per valid kp squared error (x^2+y^2)

            if cap.isOpened():

                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                                'n_valid': int(np.sum(valid)), 'mse':mse, 'mae':mae, 'rmse':rmse})        ae = np.sqrt(se)           # per-kp euclidean error

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                cap.release()    # aggregate across frames        if normalize_by:

                normalize_by = np.hypot(w, h)

        if normalize_by is None:    per_keypoint_errors = np.stack(per_keypoint_errors, axis=0)  # (n_pairs, count)            se = se / (normalize_by**2)

            # fallback: try to infer from first frame values in gt_df

            if 'frame' in gt_df.columns:    mean_se_per_kp = np.nanmean(per_keypoint_errors, axis=0)  # per-kp mean squared error            ae = ae / normalize_by

                # attempt to parse any pose_x values to find max width/height (not reliable)

                normalize_by = 1.0    overall_mse = np.nanmean(per_keypoint_errors)        mse = float(np.mean(se))



    res = align_and_compute(gt_df, pred_frames if not args.pred_csv else None, pred_kps, args.target, normalize_by=normalize_by)    overall_rmse = sqrt(float(overall_mse)) if not np.isnan(overall_mse) else np.nan        mae = float(np.mean(ae))

    print("评估结果摘要:")

    print(f"配对帧数: {res['n_pairs']}")    return {        rmse = float(np.sqrt(np.mean(se)))

    print(f"Overall MSE: {res['overall_mse']:.4f}")

    print(f"Overall RMSE: {res['overall_rmse']:.4f}")        'per_frame': pd.DataFrame(per_frame_stats),        # build per-keypoint: set NaN where invalid

    # per-frame stats head

    print("\n每帧统计（前10）:")        'per_kp_mse': mean_se_per_kp,        pk = np.full((count,), np.nan)

    print(res['per_frame'].head(10).to_string(index=False))

        'overall_mse': float(overall_mse),        pk[valid] = se  # store squared error per kp

    if args.output:

        out_dir = os.path.dirname(args.output) or '.'        'overall_rmse': float(overall_rmse),        per_keypoint_errors.append(pk)

        os.makedirs(out_dir, exist_ok=True)

        # save per-frame and per-keypoint summary        'n_pairs': len(paired)        per_frame_stats.append({'frame': int(gt_df.iloc[gi]['frame']) if 'frame' in gt_df.columns else gi,

        res['per_frame'].to_csv(args.output, index=False)

        print(f"已保存每帧统计到: {args.output}")    }                                'n_valid': int(np.sum(valid)), 'mse':mse, 'mae':mae, 'rmse':rmse})

        # save per-keypoint mse as npy for convenience

        kp_file = os.path.splitext(args.output)[0] + f'_{args.target}_per_kp_mse.npy'    # aggregate across frames

        np.save(kp_file, res['per_kp_mse'])

        print(f"已保存每关键点 MSE 到: {kp_file}")def compute_ssim(img1, img2, data_range=255):    per_keypoint_errors = np.stack(per_keypoint_errors, axis=0)  # (n_pairs, count)



    # 计算图像质量指标（如果提供原始和变形视频）    """简单 SSIM 计算"""    mean_se_per_kp = np.nanmean(per_keypoint_errors, axis=0)  # per-kp mean squared error

    if args.original_video and args.warped_video:

        print("\n计算图像质量指标...")    C1 = (0.01 * data_range) ** 2    overall_mse = np.nanmean(per_keypoint_errors)

        img_quality = compute_image_quality(args.original_video, args.warped_video, max_frames=args.max_frames)

        if img_quality:    C2 = (0.03 * data_range) ** 2    overall_rmse = sqrt(float(overall_mse)) if not np.isnan(overall_mse) else np.nan

            print("图像质量评估结果:")

            print(f"平均 SSIM: {img_quality['avg_ssim']:.4f}")        return {

            print(f"平均 PSNR: {img_quality['avg_psnr']:.4f}")

            print(f"平均 MSE: {img_quality['avg_mse']:.4f}")    img1 = img1.astype(float)        'per_frame': pd.DataFrame(per_frame_stats),

            print("\n每帧图像质量（前10）:")

            print(img_quality['per_frame'].head(10).to_string(index=False))    img2 = img2.astype(float)        'per_kp_mse': mean_se_per_kp,

            

            if args.output:            'overall_mse': float(overall_mse),

                img_file = os.path.splitext(args.output)[0] + '_image_quality.csv'

                img_quality['per_frame'].to_csv(img_file, index=False)    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)        'overall_rmse': float(overall_rmse),

                print(f"已保存图像质量统计到: {img_file}")

        else:    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)        'n_pairs': len(paired)

            print("无法计算图像质量指标")

        }

if __name__ == '__main__':

    main()    mu1_sq = mu1 ** 2

    mu2_sq = mu2 ** 2def compute_ssim(img1, img2, data_range=255):

    mu1_mu2 = mu1 * mu2    """简单 SSIM 计算"""

        C1 = (0.01 * data_range) ** 2

    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq    C2 = (0.03 * data_range) ** 2

    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq    

    sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2    img1 = img1.astype(float)

        img2 = img2.astype(float)

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)    

    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)

        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    ssim_map = numerator / denominator    

    return np.mean(ssim_map)    mu1_sq = mu1 ** 2

    mu2_sq = mu2 ** 2

def compute_image_quality(original_video, warped_video, max_frames=None):    mu1_mu2 = mu1 * mu2

    """    

    计算两个视频之间的图像质量指标：SSIM, PSNR, MSE    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq

    返回每帧的指标和平均值    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq

    """    sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2

    cap_orig = cv2.VideoCapture(original_video)    

    cap_warp = cv2.VideoCapture(warped_video)    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)

        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    if not cap_orig.isOpened() or not cap_warp.isOpened():    

        raise RuntimeError('无法打开视频文件')    ssim_map = numerator / denominator

        return np.mean(ssim_map)

    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    total_warp = int(cap_warp.get(cv2.CAP_PROP_FRAME_COUNT))def compute_image_quality(original_video, warped_video, max_frames=None):

    total = min(total_orig, total_warp)    """

    if max_frames:    计算两个视频之间的图像质量指标：SSIM, PSNR, MSE

        total = min(total, max_frames)    返回每帧的指标和平均值

        """

    ssim_scores = []    cap_orig = cv2.VideoCapture(original_video)

    psnr_scores = []    cap_warp = cv2.VideoCapture(warped_video)

    mse_scores = []    

    frame_indices = []    if not cap_orig.isOpened() or not cap_warp.isOpened():

            raise RuntimeError('无法打开视频文件')

    for idx in tqdm(range(total), desc='计算图像质量', unit='frame', ncols=80):    

        ret_orig, frame_orig = cap_orig.read()    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

        ret_warp, frame_warp = cap_warp.read()    total_warp = int(cap_warp.get(cv2.CAP_PROP_FRAME_COUNT))

            total = min(total_orig, total_warp)

        if not ret_orig or not ret_warp:    if max_frames:

            break        total = min(total, max_frames)

            

        # 确保帧尺寸相同    ssim_scores = []

        if frame_orig.shape != frame_warp.shape:    psnr_scores = []

            h, w = min(frame_orig.shape[0], frame_warp.shape[0]), min(frame_orig.shape[1], frame_warp.shape[1])    mse_scores = []

            frame_orig = cv2.resize(frame_orig, (w, h))    frame_indices = []

            frame_warp = cv2.resize(frame_warp, (w, h))    

            for idx in tqdm(range(total), desc='计算图像质量', unit='frame', ncols=80):

        # 计算指标        ret_orig, frame_orig = cap_orig.read()

        ssim_val = compute_ssim(frame_orig, frame_warp)        ret_warp, frame_warp = cap_warp.read()

        psnr_val = cv2.PSNR(frame_orig, frame_warp)        

        mse_val = np.mean((frame_orig.astype(float) - frame_warp.astype(float)) ** 2)        if not ret_orig or not ret_warp:

                    break

        ssim_scores.append(ssim_val)        

        psnr_scores.append(psnr_val)        # 确保帧尺寸相同

        mse_scores.append(mse_val)        if frame_orig.shape != frame_warp.shape:

        frame_indices.append(idx)            h, w = min(frame_orig.shape[0], frame_warp.shape[0]), min(frame_orig.shape[1], frame_warp.shape[1])

                frame_orig = cv2.resize(frame_orig, (w, h))

    cap_orig.release()            frame_warp = cv2.resize(frame_warp, (w, h))

    cap_warp.release()        

            # 计算指标

    if not ssim_scores:        ssim_val = compute_ssim(frame_orig, frame_warp)

        return None        psnr_val = cv2.PSNR(frame_orig, frame_warp)

            mse_val = np.mean((frame_orig.astype(float) - frame_warp.astype(float)) ** 2)

    return {        

        'per_frame': pd.DataFrame({        ssim_scores.append(ssim_val)

            'frame': frame_indices,        psnr_scores.append(psnr_val)

            'ssim': ssim_scores,        mse_scores.append(mse_val)

            'psnr': psnr_scores,        frame_indices.append(idx)

            'mse': mse_scores    

        }),    cap_orig.release()

        'avg_ssim': np.mean(ssim_scores),    cap_warp.release()

        'avg_psnr': np.mean(psnr_scores),    

        'avg_mse': np.mean(mse_scores)    if not ssim_scores:

    }        return None

    

def main():    return {

    parser = argparse.ArgumentParser(description='评估关键点误差：支持从视频实时提取或用预测 CSV 对比 ground-truth CSV')        'per_frame': pd.DataFrame({

    parser.add_argument('--gt-csv', required=True, help='ground-truth CSV（extract_keypoints.py 输出）')            'frame': frame_indices,

    parser.add_argument('--pred-csv', help='预测 CSV（同样格式）；若未提供则使用 --video 提取关键点')            'ssim': ssim_scores,

    parser.add_argument('--video', help='视频文件路径（当未提供 pred-csv 时使用）')            'psnr': psnr_scores,

    parser.add_argument('--target', choices=['pose','face'], default='pose', help='评估目标关键点类型')            'mse': mse_scores

    parser.add_argument('--normalize', action='store_true', help='按图像对角线进行归一化（便于不同分辨率比较）')        }),

    parser.add_argument('--max-frames', type=int, help='限制从视频提取的帧数（可选）')        'avg_ssim': np.mean(ssim_scores),

    parser.add_argument('--output', help='保存评估结果 CSV 的路径（可选）')        'avg_psnr': np.mean(psnr_scores),

    parser.add_argument('--original-video', help='原始视频路径，用于计算图像质量指标（SSIM, PSNR, MSE）与变形视频比较')        'avg_mse': np.mean(mse_scores)

    parser.add_argument('--warped-video', help='变形视频路径，用于计算图像质量指标（与 --original-video 一起使用）')    }

    args = parser.parse_args()

def main():

    gt_df = pd.read_csv(args.gt_csv)    parser = argparse.ArgumentParser(description='评估关键点误差：支持从视频实时提取或用预测 CSV 对比 ground-truth CSV')

    # load predictions    parser.add_argument('--gt-csv', required=True, help='ground-truth CSV（extract_keypoints.py 输出）')

    if args.pred_csv:    parser.add_argument('--pred-csv', help='预测 CSV（同样格式）；若未提供则使用 --video 提取关键点')

        pred_df = pd.read_csv(args.pred_csv)    parser.add_argument('--video', help='视频文件路径（当未提供 pred-csv 时使用）')

        prefix = 'pose_' if args.target=='pose' else 'face_'    parser.add_argument('--target', choices=['pose','face'], default='pose', help='评估目标关键点类型')

        count = infer_count_from_csv(gt_df, prefix)    parser.add_argument('--normalize', action='store_true', help='按图像对角线进行归一化（便于不同分辨率比较）')

        pred_kps = load_kps_xy_from_csv(pred_df, prefix, count)    parser.add_argument('--max-frames', type=int, help='限制从视频提取的帧数（可选）')

        pred_frames = None    parser.add_argument('--output', help='保存评估结果 CSV 的路径（可选）')

    elif args.video:    parser.add_argument('--original-video', help='原始视频路径，用于计算图像质量指标（SSIM, PSNR, MSE）与变形视频比较')

        pred_frames, pred_kps = extract_from_video(args.video, target=args.target, max_frames=args.max_frames)    parser.add_argument('--warped-video', help='变形视频路径，用于计算图像质量指标（与 --original-video 一起使用）')

        # pred_kps shape (n, count,2)    args = parser.parse_args()

    else:

        raise RuntimeError('需要提供 --pred-csv 或 --video 作为预测来源')    gt_df = pd.read_csv(args.gt_csv)

    # load predictions

    # Determine normalization scalar if requested    if args.pred_csv:

    normalize_by = None        pred_df = pd.read_csv(args.pred_csv)

    if args.normalize:        prefix = 'pose_' if args.target=='pose' else 'face_'

        # try get image size from video or gt info: prefer video        count = infer_count_from_csv(gt_df, prefix)

        if args.video:        pred_kps = load_kps_xy_from_csv(pred_df, prefix, count)

            cap = cv2.VideoCapture(args.video)        pred_frames = None

            if cap.isOpened():    elif args.video:

                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))        pred_frames, pred_kps = extract_from_video(args.video, target=args.target, max_frames=args.max_frames)

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))        # pred_kps shape (n, count,2)

                cap.release()    else:

                normalize_by = np.hypot(w, h)        raise RuntimeError('需要提供 --pred-csv 或 --video 作为预测来源')

        if normalize_by is None:

            # fallback: try to infer from first frame values in gt_df    # Determine normalization scalar if requested

            if 'frame' in gt_df.columns:    normalize_by = None

                # attempt to parse any pose_x values to find max width/height (not reliable)    if args.normalize:

                normalize_by = 1.0        # try get image size from video or gt info: prefer video

        if args.video:

    res = align_and_compute(gt_df, pred_frames if not args.pred_csv else None, pred_kps, args.target, normalize_by=normalize_by)            cap = cv2.VideoCapture(args.video)

    print("评估结果摘要:")            if cap.isOpened():

    print(f"配对帧数: {res['n_pairs']}")                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Overall MSE: {res['overall_mse']:.4f}")                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Overall RMSE: {res['overall_rmse']:.4f}")                cap.release()

    # per-frame stats head                normalize_by = np.hypot(w, h)

    print("\n每帧统计（前10）:")        if normalize_by is None:

    print(res['per_frame'].head(10).to_string(index=False))            # fallback: try to infer from first frame values in gt_df

            if 'frame' in gt_df.columns:

    if args.output:                # attempt to parse any pose_x values to find max width/height (not reliable)

        out_dir = os.path.dirname(args.output) or '.'                normalize_by = 1.0

        os.makedirs(out_dir, exist_ok=True)

        # save per-frame and per-keypoint summary    res = align_and_compute(gt_df, pred_frames if not args.pred_csv else None, pred_kps, args.target, normalize_by=normalize_by)

        res['per_frame'].to_csv(args.output, index=False)    print("评估结果摘要:")

        print(f"已保存每帧统计到: {args.output}")    print(f"配对帧数: {res['n_pairs']}")

        # save per-keypoint mse as npy for convenience    print(f"Overall MSE: {res['overall_mse']:.4f}")

        kp_file = os.path.splitext(args.output)[0] + f'_{args.target}_per_kp_mse.npy'    print(f"Overall RMSE: {res['overall_rmse']:.4f}")

        np.save(kp_file, res['per_kp_mse'])    # per-frame stats head

        print(f"已保存每关键点 MSE 到: {kp_file}")    print("\n每帧统计（前10）:")

    print(res['per_frame'].head(10).to_string(index=False))

    # 计算图像质量指标（如果提供原始和变形视频）

    if args.original_video and args.warped_video:    if args.output:

        print("\n计算图像质量指标...")        out_dir = os.path.dirname(args.output) or '.'

        img_quality = compute_image_quality(args.original_video, args.warped_video, max_frames=args.max_frames)        os.makedirs(out_dir, exist_ok=True)

        if img_quality:        # save per-frame and per-keypoint summary

            print("图像质量评估结果:")        res['per_frame'].to_csv(args.output, index=False)

            print(f"平均 SSIM: {img_quality['avg_ssim']:.4f}")        print(f"已保存每帧统计到: {args.output}")

            print(f"平均 PSNR: {img_quality['avg_psnr']:.4f}")        # save per-keypoint mse as npy for convenience

            print(f"平均 MSE: {img_quality['avg_mse']:.4f}")        kp_file = os.path.splitext(args.output)[0] + f'_{args.target}_per_kp_mse.npy'

            print("\n每帧图像质量（前10）:")        np.save(kp_file, res['per_kp_mse'])

            print(img_quality['per_frame'].head(10).to_string(index=False))        print(f"已保存每关键点 MSE 到: {kp_file}")

            

            if args.output:    # 计算图像质量指标（如果提供原始和变形视频）

                img_file = os.path.splitext(args.output)[0] + '_image_quality.csv'    if args.original_video and args.warped_video:

                img_quality['per_frame'].to_csv(img_file, index=False)        print("\n计算图像质量指标...")

                print(f"已保存图像质量统计到: {img_file}")        img_quality = compute_image_quality(args.original_video, args.warped_video, max_frames=args.max_frames)

        else:        if img_quality:

            print("无法计算图像质量指标")            print("图像质量评估结果:")

            print(f"平均 SSIM: {img_quality['avg_ssim']:.4f}")

if __name__ == '__main__':            print(f"平均 PSNR: {img_quality['avg_psnr']:.4f}")

    main()            print(f"平均 MSE: {img_quality['avg_mse']:.4f}")
            print("\n每帧图像质量（前10）:")
            print(img_quality['per_frame'].head(10).to_string(index=False))
            
            if args.output:
                img_file = os.path.splitext(args.output)[0] + '_image_quality.csv'
                img_quality['per_frame'].to_csv(img_file, index=False)
                print(f"已保存图像质量统计到: {img_file}")
        else:
            print("无法计算图像质量指标")

if __name__ == '__main__':
    main()
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

    # 计算图像质量指标（如果提供原始和变形视频）
    if args.original_video and args.warped_video:
        print("\n计算图像质量指标...")
        img_quality = compute_image_quality(args.original_video, args.warped_video, max_frames=args.max_frames)
        if img_quality:
            print("图像质量评估结果:")
            print(f"平均 SSIM: {img_quality['avg_ssim']:.4f}")
            print(f"平均 PSNR: {img_quality['avg_psnr']:.4f}")
            print(f"平均 MSE: {img_quality['avg_mse']:.4f}")
            print("\n每帧图像质量（前10）:")
            print(img_quality['per_frame'].head(10).to_string(index=False))
            
            if args.output:
                img_file = os.path.splitext(args.output)[0] + '_image_quality.csv'
                img_quality['per_frame'].to_csv(img_file, index=False)
                print(f"已保存图像质量统计到: {img_file}")
        else:
            print("无法计算图像质量指标")      try:
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

def compute_ssim(img1, img2, data_range=255):
    """简单 SSIM 计算"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1_mu2
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return np.mean(ssim_map)

def compute_image_quality(original_video, warped_video, max_frames=None):
    """
    计算两个视频之间的图像质量指标：SSIM, PSNR, MSE
    返回每帧的指标和平均值
    """
    cap_orig = cv2.VideoCapture(original_video)
    cap_warp = cv2.VideoCapture(warped_video)
    
    if not cap_orig.isOpened() or not cap_warp.isOpened():
        raise RuntimeError('无法打开视频文件')
    
    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    total_warp = int(cap_warp.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total_orig, total_warp)
    if max_frames:
        total = min(total, max_frames)
    
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    frame_indices = []
    
    for idx in tqdm(range(total), desc='计算图像质量', unit='frame', ncols=80):
        ret_orig, frame_orig = cap_orig.read()
        ret_warp, frame_warp = cap_warp.read()
        
        if not ret_orig or not ret_warp:
            break
        
        # 确保帧尺寸相同
        if frame_orig.shape != frame_warp.shape:
            h, w = min(frame_orig.shape[0], frame_warp.shape[0]), min(frame_orig.shape[1], frame_warp.shape[1])
            frame_orig = cv2.resize(frame_orig, (w, h))
            frame_warp = cv2.resize(frame_warp, (w, h))
        
        # 计算指标
        ssim_val = compute_ssim(frame_orig, frame_warp)
        psnr_val = cv2.PSNR(frame_orig, frame_warp)
        mse_val = np.mean((frame_orig.astype(float) - frame_warp.astype(float)) ** 2)
        
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
        mse_scores.append(mse_val)
        frame_indices.append(idx)
    
    cap_orig.release()
    cap_warp.release()
    
    if not ssim_scores:
        return None
    
    return {
        'per_frame': pd.DataFrame({
            'frame': frame_indices,
            'ssim': ssim_scores,
            'psnr': psnr_scores,
            'mse': mse_scores
        }),
        'avg_ssim': np.mean(ssim_scores),
        'avg_psnr': np.mean(psnr_scores),
        'avg_mse': np.mean(mse_scores)
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
    parser.add_argument('--original-video', help='原始视频路径，用于计算图像质量指标（SSIM, PSNR, MSE）与变形视频比较')
    parser.add_argument('--warped-video', help='变形视频路径，用于计算图像质量指标（与 --original-video 一起使用）')
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
    print("\n每帧统计（前10）：")
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

    # 计算图像质量指标（如果提供原始和变形视频）
    if args.original_video and args.warped_video:
        print("\n计算图像质量指标...")
        img_quality = compute_image_quality(args.original_video, args.warped_video, max_frames=args.max_frames)
        if img_quality:
            print("图像质量评估结果:")
            print(f"平均 SSIM: {img_quality['avg_ssim']:.4f}")
            print(f"平均 PSNR: {img_quality['avg_psnr']:.4f}")
            print(f"平均 MSE: {img_quality['avg_mse']:.4f}")
            print("\n每帧图像质量（前10）:")
            print(img_quality['per_frame'].head(10).to_string(index=False))
            
            if args.output:
                img_file = os.path.splitext(args.output)[0] + '_image_quality.csv'
                img_quality['per_frame'].to_csv(img_file, index=False)
                print(f"已保存图像质量统计到: {img_file}")
        else:
            print("无法计算图像质量指标")

if __name__ == '__main__':
    main()
