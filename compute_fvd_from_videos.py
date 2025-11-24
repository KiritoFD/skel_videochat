import os
import argparse
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torchvision.io import read_video
from torchvision import transforms
from PIL import Image

# Load the model (kept as before but wrapped)
def load_i3d_model(device='cpu'):
	model_path = hf_hub_download(
		repo_id="flateon/FVD-I3D-torchscript",
		filename="i3d_torchscript.pt"
	)
	i3d_model = torch.jit.load(model_path, map_location=device)
	i3d_model.eval()
	return i3d_model

# 新增：将视频文件读为帧（Tensor [T, H, W, C]，uint8）
def read_video_frames(video_path):
	# read_video 返回 (video_frames, audio, info)
	video, _, _ = read_video(video_path, pts_unit='sec')
	# video: Tensor [T, H, W, C], dtype uint8 or uint8-like
	return video

# 新增：将一段帧列表/张量处理为模型输入格式 [3, frames, H, W]
def preprocess_clip(frames_tensor, resize=256, crop=224):
	# frames_tensor: Tensor [T, H, W, C] 或 numpy array
	# 输出: Tensor [3, T, H, W], dtype float32, 已归一化
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	transform = transforms.Compose([
		transforms.Resize(resize),
		transforms.CenterCrop(crop),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])
	processed = []
	for i in range(frames_tensor.shape[0]):
		frame = frames_tensor[i].numpy() if isinstance(frames_tensor, torch.Tensor) else frames_tensor[i]
		# frame: H,W,C (uint8)
		img = Image.fromarray(frame)
		t = transform(img)  # [C, H, W]
		processed.append(t)
	# stack -> [T, C, H, W] then permute to [C, T, H, W]
	stack = torch.stack(processed, dim=0)  # [T, C, H, W]
	stack = stack.permute(1, 0, 2, 3)      # [C, T, H, W]
	return stack

# 新增：将视频分成若干长度为 clip_len 的片段（不足时用最后一帧补齐）
def make_clips(frames_tensor, clip_len=16):
	T = frames_tensor.shape[0]
	clips = []
	for start in range(0, T, clip_len):
		end = start + clip_len
		if end <= T:
			clip = frames_tensor[start:end]
		else:
			clip = frames_tensor[start:T]
			# pad by repeating last frame
			last = frames_tensor[T-1].unsqueeze(0).repeat(end - T, 1, 1, 1)
			clip = torch.cat([clip, last], dim=0)
		clips.append(clip)
	return clips

# 新增：对单个视频提取特征（按片段、批量计算并聚合）
def extract_features_from_video(video_path, model, device='cpu', clip_len=16, batch_size=8, agg='mean'):
	frames = read_video_frames(video_path)  # [T, H, W, C]
	if frames.shape[0] == 0:
		raise ValueError(f"No frames found in {video_path}")
	clips = make_clips(frames, clip_len=clip_len)
	# preprocess and batch
	all_feats = []
	batch_clips = []
	for clip in clips:
		inp = preprocess_clip(clip)  # [3, clip_len, H, W]
		batch_clips.append(inp)
		if len(batch_clips) == batch_size:
			b = torch.stack(batch_clips, dim=0)  # [B, 3, T, H, W]
			b = b.to(device)
			with torch.no_grad():
				feats = model(b, rescale=True, resize=True, return_features=True)
			all_feats.append(feats.cpu())
			batch_clips = []
	# leftover
	if batch_clips:
		b = torch.stack(batch_clips, dim=0).to(device)
		with torch.no_grad():
			feats = model(b, rescale=True, resize=True, return_features=True)
		all_feats.append(feats.cpu())
	if len(all_feats) == 0:
		return None
	all_feats = torch.cat(all_feats, dim=0)  # [num_clips, feat_dim]
	if agg == 'mean':
		video_feat = all_feats.mean(dim=0).numpy()
	elif agg == 'max':
		video_feat = all_feats.max(dim=0).values.numpy()
	else:
		video_feat = all_feats.numpy()  # 返回所有片段特征
	return video_feat

# 新增：安全的矩阵平方根（优先使用 scipy.linalg.sqrtm，若不可用则回退到特征分解）
def _sqrtm_safe(mat):
	try:
		from scipy.linalg import sqrtm
		m = sqrtm(mat)
		# 将数值误差产生的很小虚部去掉
		if np.iscomplexobj(m):
			m = m.real
		return m
	except Exception:
		# 回退实现：对称矩阵特征分解
		w, v = np.linalg.eigh(mat)
		# 防止负特征值的数值问题
		w[w < 0] = 0.0
		return (v * np.sqrt(w)) @ v.T

# 新增：计算 Frechet 距离（FVD）
def frechet_distance(feats1, feats2, eps=1e-6):
	"""
	feats1, feats2: np.ndarray shape [N, D]
	return: float
	"""
	feats1 = np.asarray(feats1, dtype=np.float64)
	feats2 = np.asarray(feats2, dtype=np.float64)
	if feats1.ndim != 2 or feats2.ndim != 2:
		raise ValueError("feats must be 2D arrays [N, D]")
	mu1 = np.mean(feats1, axis=0)
	mu2 = np.mean(feats2, axis=0)
	sigma1 = np.cov(feats1, rowvar=False)
	sigma2 = np.cov(feats2, rowvar=False)

	# numerical stability
	d = mu1.shape[0]
	sigma1 += np.eye(d) * eps
	sigma2 += np.eye(d) * eps

	# sqrt of product
	covmean = _sqrtm_safe(sigma1.dot(sigma2))
	# 有可能回退实现产生轻微虚部，取实部
	if np.iscomplexobj(covmean):
		covmean = covmean.real

	mean_diff = mu1 - mu2
	fd = mean_diff.dot(mean_diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return float(np.real(fd))

# 新增：命令行接口
def main():
	parser = argparse.ArgumentParser(description="Extract I3D features from videos (sharded by clips).")
	# 固定两个视频路径为参数，默认写死为示例路径
	parser.add_argument('--video-a', default='head.mp4', help='path to video A (default: head.mp4)')
	parser.add_argument('--video-b', default='generated_frames/output.mp4', help='path to video B (default: generated_frames/output.mp4)')
	parser.add_argument('--device', default='cpu', help='cpu or cuda')
	parser.add_argument('--clip_len', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--agg', choices=['mean','max','none'], default='mean', help='how to aggregate clip features per video')
	args = parser.parse_args()

	model = load_i3d_model(device=args.device)
	videos = [args.video_a, args.video_b]

	# 收集两侧特征（每个 clip 的特征），以便计算 FVD
	collected = {}
	for vp in videos:
		if not vp:
			continue
		try:
			# 传 agg=None（或 'none' 在 CLI 时）以返回所有片段特征 [num_clips, feat_dim]
			feats = extract_features_from_video(vp, model, device=args.device, clip_len=args.clip_len, batch_size=args.batch_size, agg=None)
			if feats is None:
				print(f"Skipping {vp}: no features")
				continue
			out_path = vp + '.feat.npy'
			np.save(out_path, feats)
			collected[vp] = feats
			print(f"Saved features for {vp} -> {out_path}, shape: {feats.shape}")
		except Exception as e:
			print(f"Error processing {vp}: {e}")

	# 若两个视频都有特征，则计算并打印 FVD
	if args.video_a in collected and args.video_b in collected:
		try:
			feats_a = collected[args.video_a]
			feats_b = collected[args.video_b]
			# 如果 extract 返回的是单个向量而不是多片段特征，强制变成 [1, D]
			if feats_a.ndim == 1:
				feats_a = feats_a[np.newaxis, :]
			if feats_b.ndim == 1:
				feats_b = feats_b[np.newaxis, :]
			fvd_value = frechet_distance(feats_a, feats_b)
			print(f"FVD between {args.video_a} and {args.video_b}: {fvd_value:.6f}")
		except Exception as e:
			print(f"Error computing FVD: {e}")

if __name__ == '__main__':
	main()
