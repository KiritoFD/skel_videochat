import cv2
import os
import glob

def create_video_from_frames(output_path, frames_dir, frame_format='png', fps=30, width=None, height=None):
    # 获取所有帧文件
    frame_files = sorted(glob.glob(os.path.join(frames_dir, f'frame_*.{frame_format}')))
    if not frame_files:
        print(f"未找到帧文件在 {frames_dir}")
        return False

    # 读取第一帧获取尺寸
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"无法读取第一帧: {frame_files[0]}")
        return False
    h, w = first_frame.shape[:2]
    if width is None:
        width = w
    if height is None:
        height = h

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"跳过无法读取的帧: {frame_file}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()
    print(f"视频已保存到: {output_path}")
    return True

if __name__ == '__main__':
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else 'warped.mp4'
    frames_dir = sys.argv[2] if len(sys.argv) > 2 else 'warped_frames'
    create_video_from_frames(output, frames_dir, fps=30)