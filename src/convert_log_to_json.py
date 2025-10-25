import json
import os
import numpy as np

def convert_debug_log_to_stats(debug_log_path, output_path=None):
    """
    将 debug_log.json 转换为统计格式，统计值放到开头。
    """
    if not os.path.exists(debug_log_path):
        print(f"文件不存在: {debug_log_path}")
        return

    with open(debug_log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    # 收集所有统计数据
    all_image_diffs = []
    all_keypoint_diffs = []
    all_frames_stats = []

    for frame_idx, frame_data in log_data.items():
        if isinstance(frame_data, dict):
            if 'diff' in frame_data:
                all_image_diffs.append(frame_data['diff'].get('mean_diff', 0))
            if 'keypoint_points' in frame_data:
                for point in frame_data['keypoint_points']:
                    all_keypoint_diffs.append(point.get('差异_px', 0))
            all_frames_stats.append(frame_data)

    # 计算全局统计
    stats = {
        "总帧数": len(log_data),
        "图像变形统计": {
            "平均差异_px": float(np.mean(all_image_diffs)) if all_image_diffs else 0.0,
            "最大差异_px": float(np.max(all_image_diffs)) if all_image_diffs else 0.0,
            "最小差异_px": float(np.min(all_image_diffs)) if all_image_diffs else 0.0,
            "标准差_px": float(np.std(all_image_diffs)) if all_image_diffs else 0.0
        },
        "关键点差异统计": {
            "平均差异_px": float(np.mean(all_keypoint_diffs)) if all_keypoint_diffs else 0.0,
            "最大差异_px": float(np.max(all_keypoint_diffs)) if all_keypoint_diffs else 0.0,
            "最小差异_px": float(np.min(all_keypoint_diffs)) if all_keypoint_diffs else 0.0,
            "标准差_px": float(np.std(all_keypoint_diffs)) if all_keypoint_diffs else 0.0,
            "总关键点数": len(all_keypoint_diffs)
        },
        "帧详情": {}
    }

    # 添加每帧详情
    for frame_idx in sorted(log_data.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        frame_data = log_data[frame_idx]
        if isinstance(frame_data, dict):
            stats["帧详情"][frame_idx] = {
                "图像差异": frame_data.get('diff', {}),
                "关键点平均差异_px": frame_data.get('平均差异_px', 0),
                "关键点最大差异_px": frame_data.get('最大差异_px', 0),
                "关键点最小差异_px": frame_data.get('最小差异_px', 0),
                "有效关键点数": len(frame_data.get('keypoint_points', []))
            }

    # 保存到输出文件
    if output_path is None:
        output_path = debug_log_path.replace('debug_log.json', 'debug_log_stats.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print(f"已保存统计 JSON: {output_path}")
    print(f"\n统计概览:")
    print(f"  总帧数: {stats['总帧数']}")
    print(f"  图像变形 - 平均差异: {stats['图像变形统计']['平均差异_px']:.2f}px")
    print(f"  关键点差异 - 平均: {stats['关键点差异统计']['平均差异_px']:.2f}px, 最大: {stats['关键点差异统计']['最大差异_px']:.2f}px")

    return stats

if __name__ == "__main__":
    # 使用示例
    debug_log_path = "debug_frames/debug_log.json"
    convert_debug_log_to_stats(debug_log_path)
