import numpy as np
import cv2

# 尽量轻量实现 CPU 版本的“受限影响” Beier-Neely
def beier_neely_warp_cpu_limited(src_img, src_kps, tgt_kps, edges, a=0.1, b=1.0, max_dist=80, p=1.0):
	"""
	src_img: HxW[x3] uint8
	src_kps / tgt_kps: (N,2) 像素坐标（float），可能包含 NaN
	edges: [(i,j), ...] 关键点索引对
	max_dist: 影响半径（像素），超过该距离对该线段权重设为0
	p: 长度权重指数，默认 1.0（原始 Beier-Neely）
	"""
	h, w = src_img.shape[:2]
	# 预计算线段对
	line_data = []
	for i, j in edges:
		if i < len(src_kps) and j < len(src_kps) and i < len(tgt_kps) and j < len(tgt_kps):
			Pp = src_kps[i].astype(np.float32)
			Qp = src_kps[j].astype(np.float32)
			P = tgt_kps[i].astype(np.float32)
			Q = tgt_kps[j].astype(np.float32)
			if np.isnan(Pp).any() or np.isnan(Qp).any() or np.isnan(P).any() or np.isnan(Q).any():
				continue
			vec = Q - P
			len_sq = vec[0]*vec[0] + vec[1]*vec[1]
			if len_sq < 1e-6:
				continue
			vec_p = Qp - Pp
			len_p = np.linalg.norm(vec_p)
			if len_p < 1e-6:
				continue
			sqrt_len_sq = np.sqrt(len_sq)
			perp_p = np.array([-vec_p[1], vec_p[0]], dtype=np.float32) / len_p
			line_data.append((P, Q, Pp, vec, vec_p, len_sq, sqrt_len_sq, perp_p))
	if not line_data:
		return src_img.copy()
	# 网格
	y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
	DSUM_x = np.zeros((h, w), dtype=np.float32)
	DSUM_y = np.zeros((h, w), dtype=np.float32)
	weight_sum = np.zeros((h, w), dtype=np.float32)
	for (P, Q, Pp, vec, vec_p, len_sq, sqrt_len_sq, perp_p) in line_data:
		X_P_x = x_coords - P[0]
		X_P_y = y_coords - P[1]
		u = (X_P_x * vec[0] + X_P_y * vec[1]) / len_sq
		v = (X_P_x * vec[1] - X_P_y * vec[0]) / sqrt_len_sq
		dist_to_P = np.sqrt(X_P_x**2 + X_P_y**2)
		dist_to_Q = np.sqrt((x_coords - Q[0])**2 + (y_coords - Q[1])**2)
		dist = np.where(u < 0, dist_to_P, np.where(u > 1, dist_to_Q, np.abs(v)))
		# cutoff 超过 max_dist 的位置权重设为0（限制影响半径）
		mask = dist <= float(max_dist)
		# 计算 X'（源坐标）
		Xp_x = Pp[0] + u * vec_p[0] + v * perp_p[0]
		Xp_y = Pp[1] + u * vec_p[1] + v * perp_p[1]
		length = sqrt_len_sq
		# 避免除零
		with np.errstate(divide='ignore', invalid='ignore'):
			weight = ((length ** p) / (a + dist)) ** b
			weight[~mask] = 0.0
			weight = np.nan_to_num(weight, 0.0)
		DSUM_x += weight * (Xp_x - x_coords)
		DSUM_y += weight * (Xp_y - y_coords)
		weight_sum += weight
	# 构建最终映射，仅在 weight_sum>0 的位置应用偏移
	valid_mask = weight_sum > 0
	map_x = x_coords.copy()
	map_y = y_coords.copy()
	map_x[valid_mask] += DSUM_x[valid_mask] / weight_sum[valid_mask]
	map_y[valid_mask] += DSUM_y[valid_mask] / weight_sum[valid_mask]
	# clip
	map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
	map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
	# remap
	warped = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	# 合成：只在 valid_mask 区域使用 warped，否则使用原图
	out = src_img.copy()
	if src_img.ndim == 3:
		out[valid_mask] = warped[valid_mask]
	else:
		out[valid_mask] = warped[valid_mask]
	return out

def apply_warp_face_region(src_img, src_kps, tgt_kps, edges, a=0.1, b=1.0, use_cuda=False, max_dist=80, feather=12, p=1.0):
	"""
	仅对面部凸包内部区域应用变形并在边界处羽化混合。
	src_kps/tgt_kps: (N,2) 面部关键点（像素），可能包含 NaN
	edges: 面部网格边
	max_dist: 每条线段影响半径（像素）
	feather: 羽化半径（像素），为 0 则硬切换
	p: 长度权重指数
	"""
	h, w = src_img.shape[:2]
	# 收集有效点（优先使用 src_kps 来确定面部区域）
	valid_mask = ~np.isnan(src_kps).any(axis=1)
	if not np.any(valid_mask):
		return src_img.copy()
	points = src_kps[valid_mask].astype(np.int32)
	if len(points) < 3:
		return src_img.copy()
	# 计算凸包（面部边界）
	try:
		hull = cv2.convexHull(points)
	except Exception:
		return src_img.copy()
	# 生成二值掩码（uint8:0/255）
	mask = np.zeros((h, w), dtype=np.uint8)
	cv2.fillConvexPoly(mask, hull, 255)

	# 生成 warped 图（受限 Beier-Neely，实现会在 weight_sum>0 区域返回 remapped 像素）
	warped = beier_neely_warp_cpu_limited(src_img, src_kps, tgt_kps, edges, a=a, b=b, max_dist=max_dist, p=p)

	# 若 feather <= 0: 直接在掩码内替换
	if not feather or feather <= 0:
		out = src_img.copy()
		out[mask == 255] = warped[mask == 255]
		return out

	# 计算内部距离，用于羽化（距离越大 alpha 越接近 1）
	# distanceTransform 在非零区域计算到最近零像素的距离
	inside_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
	# 归一化到 [0,1]，在边界处 0，越远越接近 1
	alpha = np.clip(inside_dist.astype(np.float32) / float(max(1, feather)), 0.0, 1.0)
	# 扩展为三通道
	if warped.ndim == 3 and src_img.ndim == 3:
		alpha_3 = np.stack([alpha]*3, axis=-1)
		out = (warped.astype(np.float32) * alpha_3 + src_img.astype(np.float32) * (1.0 - alpha_3)).astype(np.uint8)
	else:
		out = (warped.astype(np.float32) * alpha + src_img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

	# 在掩码外使用原图（理论上 alpha==0 已处理）
	out[mask == 0] = src_img[mask == 0]
	return out

def apply_warp_limited(src_img, src_kps, tgt_kps, edges, a=0.1, b=1.0, use_cuda=False, max_dist=80, region_mode=None, feather=12, p=1.0):
	"""
	兼容入口：保留旧接口，同时支持 region_mode='face' 将调用 face 专用路径。
	添加 p 参数控制长度权重。
	"""
	if region_mode == 'face':
		return apply_warp_face_region(src_img, src_kps, tgt_kps, edges, a=a, b=b, use_cuda=use_cuda, max_dist=max_dist, feather=feather, p=p)
	# 默认行为：原先的受限实现
	return beier_neely_warp_cpu_limited(src_img, src_kps, tgt_kps, edges, a=a, b=b, max_dist=max_dist, p=p)
