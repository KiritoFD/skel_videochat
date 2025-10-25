import json
import os
import numpy as np

def analyze_debug_log(log_path):
    """分析调试日志，统计关键点差异的平均值、最大值和最小值"""
    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    all_diffs = []
    for frame_idx, data in log_data.items():
        if "differences" in data:
            all_diffs.extend(data["differences"])

    if not all_diffs:
        print("日志中没有有效的差异数据")
        return

    all_diffs = np.array(all_diffs)
    print(f"关键点差异统计：")
    print(f"  平均差异: {np.mean(all_diffs):.2f}px")
    print(f"  最大差异: {np.max(all_diffs):.2f}px")
    print(f"  最小差异: {np.min(all_diffs):.2f}px")
    print(f"  差异标准差: {np.std(all_diffs):.2f}px")

if __name__ == "__main__":
    log_path = "debug_frames/debug_log.json"  # 默认日志路径
    analyze_debug_log(log_path)
