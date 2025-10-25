import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='基于关键点变化的图像变形视频生成')

    parser.add_argument('-s', '--source', help='源图像路径（图片或视频）',default='head.jpg')
    parser.add_argument('-c', '--csv',  help='输入 CSV 文件路径（由 extract_keypoints.py 生成）',default='head.csv')
    parser.add_argument('-o', '--output', default='warped.mp4', help='输出视频文件路径，默认 warped.mp4')

    parser.add_argument('--width', type=int, help='输出视频宽度（默认使用源图像宽度）')
    parser.add_argument('--height', type=int, help='输出视频高度（默认使用源图像高度）')
    parser.add_argument('--fps', type=float, default=0, help='覆盖 fps（若 <=0 则从 CSV time 推断或使用 30）')

    parser.add_argument('-a', type=float, default=0.05, help='Beier-Neely 权重参数 a，默认 0.05')
    parser.add_argument('-b', type=float, default=0.5, help='Beier-Neely 权重参数 b，默认 0.5')
    parser.add_argument('-p', type=float, default=1.0, help='Beier-Neely 长度权重指数 p，默认 1.0')

    parser.add_argument('--influence-radius', type=int, default=50, help='关键点线段影响半径（像素），超出该半径不受该线段影响，默认 50')
    parser.add_argument('--max-resolution', type=int, default=0, help='最大处理分辨率（>0 则缩放，0 表示不缩放），默认 0（保持原始分辨率）')

    parser.add_argument('--save-frames', action='store_true', help='保存为图像序列而不是视频')
    parser.add_argument('--frame-format', default='png', choices=['png', 'jpg'], help='图像格式，默认 png')

    parser.add_argument('--warp-target', default='face', choices=['pose', 'face'], help='选择用于变形的关键点类型')
    parser.add_argument('--roi-only', action='store_true', help='仅在关键点联合 ROI 内执行变形，加速面部处理')
    parser.add_argument('--roi-margin', type=int, default=80, help='ROI 边距像素，默认 80')

    parser.add_argument('--coarse-scale', type=float, default=1.0, help='下采样比例 (0,1]，减小变形分辨率提升速度')
    parser.add_argument('--threads', type=int, default=1, help='CPU 并行进程数')

    parser.add_argument('--mark-points', action='store_true', help='在输出图像上标出 CSV 中的像素点（保存到 marked 子目录）', default=True)
    parser.add_argument('--mark-radius', type=int, default=3, help='点标注半径（像素），默认 3')
    parser.add_argument('--mark-color', default='0,255,0', help='点标注颜色 B,G,R，例如 "0,255,0" 代表绿色')
    parser.add_argument('--marked-dir', default='marked', help='保存标注图像的子目录，默认 "marked"')

    # 自动评估相关
    parser.add_argument('--auto-eval', action='store_true', help='生成后自动运行评估（提取关键点并计算误差）')
    parser.add_argument('--eval-output', help='评估结果 CSV 保存路径（默认放在输出目录 evaluation.csv）')

    # 添加关键点增强选项
    parser.add_argument('--enhance-movement', action='store_true', default=False,
                        help='增强关键点的移动差异，放大变形效果')

    args = parser.parse_args()
    return args
