import os
import cv2
import numpy as np
import json

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

def ensure_debug_dir(args):
    debug_dir = os.path.join(os.path.dirname(args.output), "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir

def save_debug_log(log_data, args, filename="debug_log.json"):
    """保存调试日志到 JSON 文件"""
    debug_dir = ensure_debug_dir(args)
    log_path = os.path.join(debug_dir, filename)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ 保存调试日志失败: {e}")
    else:
        print(f"已保存调试日志: {log_path}")
    return log_path

def save_debug_frame(frame_with_points, args, idx):
    """保存带关键点的调试帧（第 idx 帧）。"""
    debug_dir = ensure_debug_dir(args)
    debug_path = os.path.join(debug_dir, f"debug_frame_{idx:06d}.png")
    try:
        cv2.imwrite(debug_path, frame_with_points)
    except Exception as e:
        print(f"⚠️ 保存调试帧失败: {e}")
    else:
        print(f"已保存带关键点的调试帧: {debug_path}")
    return debug_path

def save_first_last_comparison(src_img, src_kps, last_kps, args):
    """保存首帧与末帧关键点对比图（左右并排）。"""
    debug_dir = ensure_debug_dir(args)
    h, w = src_img.shape[:2]
    compare_img = np.hstack([src_img.copy(), src_img.copy()])
    # 左边标记首帧
    for i, (x, y) in enumerate(src_kps):
        if not (np.isnan(x) or np.isnan(y)):
            ix, iy = int(round(x)), int(round(y))
            cv2.circle(compare_img[0:h, 0:w], (ix, iy), 3, (0, 0, 255), -1)
            if i % 10 == 0:
                cv2.putText(compare_img[0:h, 0:w], f"{i}", (ix+3, iy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    # 右边标记末帧
    for i, (x, y) in enumerate(last_kps):
        if not (np.isnan(x) or np.isnan(y)):
            ix, iy = int(round(x)), int(round(y))
            cv2.circle(compare_img[0:h, w:2*w], (ix, iy), 3, (0, 255, 0), -1)
            if i % 10 == 0:
                cv2.putText(compare_img[0:h, w:2*w], f"{i}", (ix+3, iy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    path = os.path.join(debug_dir, "first_last_frame_comparison.png")
    try:
        cv2.imwrite(path, compare_img)
    except Exception as e:
        print(f"⚠️ 保存首末对比图失败: {e}")
    else:
        print(f"已保存首帧与末帧关键点对比图: {path}")
    return path

def save_first_frame_keypoints(src_img, src_kps, args):
    """保存首帧关键点标注图（便于调试）。"""
    debug_dir = ensure_debug_dir(args)
    img = src_img.copy()
    for i, (x, y) in enumerate(src_kps):
        if not (np.isnan(x) or np.isnan(y)):
            ix, iy = int(round(x)), int(round(y))
            cv2.circle(img, (ix, iy), 3, (0, 255, 0), -1)
            if i % 10 == 0:
                cv2.putText(img, f"{i}", (ix+3, iy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    path = os.path.join(debug_dir, "first_frame_keypoints.png")
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"⚠️ 保存首帧关键点图失败: {e}")
    else:
        print(f"已保存首帧关键点调试图: {path}")
    return path

def print_mediapipe_keypoint_diff(frame_img, csv_kps, warp_target, log_data, frame_idx):
    """
    使用 MediaPipe 检测当前帧关键点并与 CSV 关键点逐点比较并保存差异到日志。
    支持 'pose' 和 'face' 两种类型。
    """
    if not MEDIAPIPE_AVAILABLE:
        if str(frame_idx) not in log_data:
            log_data[str(frame_idx)] = {}
        log_data[str(frame_idx)]["error"] = "MediaPipe未安装"
        return
    
    if warp_target == 'pose':
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                if str(frame_idx) not in log_data:
                    log_data[str(frame_idx)] = {}
                log_data[str(frame_idx)]["error"] = "MediaPipe未检测到pose关键点"
                return
            mp_kps = np.array([[l.x * frame_img.shape[1], l.y * frame_img.shape[0]] for l in results.pose_landmarks.landmark])
    elif warp_target == 'face':
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                if str(frame_idx) not in log_data:
                    log_data[str(frame_idx)] = {}
                log_data[str(frame_idx)]["error"] = "MediaPipe未检测到face关键点"
                return
            mp_kps = np.array([[l.x * frame_img.shape[1], l.y * frame_img.shape[0]] for l in results.multi_face_landmarks[0].landmark])
    else:
        if str(frame_idx) not in log_data:
            log_data[str(frame_idx)] = {}
        log_data[str(frame_idx)]["error"] = "warp_target类型不支持"
        return

    n_csv = len(csv_kps)
    mp_kps = mp_kps[:n_csv]
    valid_mask = ~np.isnan(csv_kps).any(axis=1) & (np.arange(n_csv) < len(mp_kps))
    
    # 获取有效点的索引
    valid_indices = np.where(valid_mask)[0]
    diffs = np.linalg.norm(mp_kps[valid_mask] - csv_kps[valid_mask], axis=1)
    
    if str(frame_idx) not in log_data:
        log_data[str(frame_idx)] = {}
    
    # 保存所有点的详细信息到 JSON
    points_data = []
    for idx, (point_id, diff_val) in enumerate(zip(valid_indices, diffs)):
        point_info = {
            "point_id": int(point_id),
            "MediaPipe": {
                "x": float(mp_kps[point_id][0]),
                "y": float(mp_kps[point_id][1])
            },
            "CSV": {
                "x": float(csv_kps[point_id][0]),
                "y": float(csv_kps[point_id][1])
            },
            "差异_px": float(diff_val)
        }
        points_data.append(point_info)
    
    log_data[str(frame_idx)]["keypoint_points"] = points_data
    log_data[str(frame_idx)]["平均差异_px"] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    log_data[str(frame_idx)]["最大差异_px"] = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    log_data[str(frame_idx)]["最小差异_px"] = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    log_data[str(frame_idx)]["最大差异_px"] = float(np.max(diffs))
    log_data[str(frame_idx)]["最小差异_px"] = float(np.min(diffs))
    
    print(f"MediaPipe检测关键点与CSV差异统计（像素）：")
    print(f"  平均差异: {np.mean(diffs):.2f}px")
    print(f"  最大差异: {np.max(diffs):.2f}px")
    print(f"  最小差异: {np.min(diffs):.2f}px")
