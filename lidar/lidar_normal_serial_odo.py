# -*- coding: utf-8 -*-
"""
RPLIDAR A1 (A1M8-R6, fw=1.29) —— 纯 pyserial NORMAL@115200 + 自动端口 + 实时可视化 + 轻量 2D 里程计 + 录像
- 不控制电机（由机身开关控制），只控制扫描系统（START/STOP + 清空）
- 轻量 2D 里程计：角度分箱 + 小角度互相关粗配准 + SVD（Procrustes）精配准
- 按键：S(保存CSV/可选PLY), G(切换颜色: range/quality), R(轻量重同步), Q(退出)
"""

import os, sys, time, math, threading, argparse, datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
from serial.tools import list_ports

# ---------------------- 常量/默认参数 ----------------------
SYNC = b'\xA5'; STOP = b'\x25'; SCAN = b'\x20'   # NORMAL
DESC_LEN = 7

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 0.2
MIN_R_DEFAULT, MAX_R_DEFAULT = 0.15, 20.0
DOWN_DEG_DEFAULT = 1.0

# ODOM 默认参数
ANGLE_BIN_DEG_DEFAULT = 1.0   # 角度分箱步长
ODOM_MAX_ROT_DEG_DEFAULT = 25 # 粗配准最大旋转搜索范围
ODOM_MAX_RANGE_DIFF = 0.8     # 配对点允许的最大量程差（米）
ODOM_MIN_OVERLAP_BINS = 50    # 至少要有多少个有效角度格才能做配准
TUM_LOG_DEFAULT = None        # 输出 TUM 轨迹文件路径（可选）

'''经过实际测试这个参数设置大概率不能画出轨迹，可以把限制放宽后尝试'''

# 常见 USB-UART VID:PID
KNOWN_VIDPID = {('10C4','EA60'), ('1A86','7523'), ('067B','2303')}

# ---------------------- 串口工具 ----------------------
def read_exact(ser, n, t=1.0):
    buf = bytearray(); end = time.time() + t
    while len(buf) < n and time.time() < end:
        chunk = ser.read(n-len(buf))
        if chunk: buf += chunk
    return bytes(buf) if len(buf)==n else None

def stop_and_clear(ser):
    try: ser.write(SYNC+STOP); ser.flush()
    except: pass
    time.sleep(0.05)
    try: ser.reset_input_buffer(); ser.reset_output_buffer()
    except: pass

def start_normal_and_get_descriptor(ser):
    ser.write(SYNC+SCAN); ser.flush()
    desc = read_exact(ser, DESC_LEN, t=2.0)
    if not desc or len(desc)!=7: return (False, None, None)
    if not (desc[0]==0xA5 and desc[1]==0x5A): return (False, None, None)
    raw_len = desc[2] | (desc[3]<<8) | (desc[4]<<16) | (desc[5]<<24)
    payload_len = raw_len & 0x3FFFFFFF  # 低30位为真实长度
    dtype = desc[6]
    return (True, payload_len, dtype)

def parse_node(raw5):
    b0,b1,b2,b3,b4 = raw5
    new_flag = bool(b0 & 0x01)
    inv_new  = bool((b0>>1) & 0x01)
    if new_flag == inv_new: return (False, None, None, None, None)
    if (b1 & 0x01) != 1:    return (False, None, None, None, None)
    angle_q6 = ((b1>>1) | (b2<<7)); dist_q2 = (b3 | (b4<<8))
    ang = angle_q6/64.0;     dist_mm = dist_q2/4.0
    quality = (b0>>2)
    return (True, new_flag, ang, dist_mm, quality)

def parse_vidpid(hwid):
    import re
    m = re.search(r'VID:PID=([0-9A-Fa-f]{4}):([0-9A-Fa-f]{4})', hwid or '')
    return (m.group(1).upper(), m.group(2).upper()) if m else (None,None)

def candidate_ports():
    ports = list_ports.comports()
    good, maybe, others = [], [], []
    for p in ports:
        vidpid = parse_vidpid(p.hwid); desc=(p.description or '').upper()
        if vidpid in KNOWN_VIDPID: good.append(p.device)
        elif any(k in desc for k in ['CP210','CH340','SLAB_USBTOUART','USB-SERIAL']): maybe.append(p.device)
        else: others.append(p.device)
    return good + maybe + others

def handshake_port(port, baud, timeout):
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout,
                            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                            rtscts=False, dsrdtr=False)
        try:
            try: ser.setDTR(False)
            except: pass
            stop_and_clear(ser)
            ok, plen, dtype = start_normal_and_get_descriptor(ser)
            if ok and plen==5 and dtype==0x81: return True
        finally:
            try: ser.close()
            except: pass
    except Exception:
        pass
    return False

def auto_find_port(prefer, baud, timeout):
    tried=[]
    if prefer:
        if handshake_port(prefer, baud, timeout): return prefer
        tried.append(prefer)
    for p in candidate_ports():
        if p in tried: continue
        if handshake_port(p, baud, timeout): return p
        tried.append(p)
    raise RuntimeError(f'未找到可用端口；已尝试: {tried}')

# ---------------------- Lidar 读取线程 ----------------------
class LidarReader:
    """把每帧的原始极坐标(angles°, ranges m, quality)和 XY 点集传给可视化与里程计"""
    def __init__(self, port, baud, timeout, down_deg, rmin, rmax):
        self.port=port; self.baud=baud; self.timeout=timeout
        self.down_deg=down_deg; self.rmin=rmin; self.rmax=rmax
        self.stop_evt=threading.Event()
        self.lock = threading.Lock()
        self.xy  = np.zeros((0,2), np.float32)
        self.qual= np.zeros((0,),  np.float32)
        self.angles = np.zeros((0,), np.float32)  # deg
        self.ranges = np.zeros((0,), np.float32)  # m
        self.frame_id = 0
        self.save_dir = None; self.save_ply=False
        self.color_mode='range'

    def set_save_dir(self, d): self.save_dir=d; os.makedirs(d, exist_ok=True)
    def set_save_ply(self, flag): self.save_ply=bool(flag)
    def toggle_color(self):
        self.color_mode = 'quality' if self.color_mode=='range' else 'range'
        print(f'[INFO] 颜色映射 => {self.color_mode}')

    @staticmethod
    def _write_ascii_ply(path, xy, q):
        n = xy.shape[0]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {n}\n')
            f.write('property float x\nproperty float y\nproperty float z\nproperty float quality\n')
            f.write('end_header\n')
            for i in range(n):
                f.write(f'{xy[i,0]:.6f} {xy[i,1]:.6f} 0.0 {float(q[i]):.3f}\n')

    def _save_frame(self, xy, q, ang_deg, dist_m):
        if not self.save_dir: return
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # CSV：x,y,quality,angle_deg,range_m
        arr = np.column_stack((xy, q, ang_deg, dist_m))
        csv = os.path.join(self.save_dir, f'frame_{ts}.csv')
        np.savetxt(csv, arr, delimiter=',', fmt='%.6f', header='x,y,quality,angle_deg,range_m', comments='')
        if self.save_ply:
            ply = os.path.join(self.save_dir, f'frame_{ts}.ply')
            self._write_ascii_ply(ply, xy, q)

    def _light_resync(self, ser):
        stop_and_clear(ser); start_normal_and_get_descriptor(ser)

    def start(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
    def join(self, timeout=None): self.t.join(timeout=timeout)
    def stop(self): self.stop_evt.set()

    def _run(self):
        ser = serial.Serial(self.port, baudrate=self.baud, timeout=self.timeout,
                            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                            rtscts=False, dsrdtr=False)
        try:
            try: ser.setDTR(False)
            except: pass
            stop_and_clear(ser)
            ok, plen, dtype = start_normal_and_get_descriptor(ser)
            if not ok or plen!=5 or dtype!=0x81:
                print('[FATAL] NORMAL 握手失败'); return
            print(f'[INFO] 端口 {self.port} @ {self.baud} 握手成功 (NORMAL)')

            last_new=False; frame_pts=[]; frame_q=[]; frame_ang=[]; frame_dist=[]
            bads=0; last_frame_t=time.time()

            while not self.stop_evt.is_set():
                raw = read_exact(ser,5,t=1.0)
                if not raw:
                    if time.time()-last_frame_t>3.0:
                        print('[WDOG] 3s 无节点 -> 轻量重同步')
                        self._light_resync(ser); last_frame_t=time.time()
                    continue

                ok,newf,ang,dist_mm,q = parse_node(raw)
                if not ok:
                    bads+=1
                    if bads>=100:
                        print('[WARN] 坏节点过多 -> 轻量重同步')
                        self._light_resync(ser); bads=0
                    continue
                bads=0

                # 采样/量程过滤（保留原始角距用于里程计）
                if (ang % self.down_deg) <= 0.5 and (self.rmin*1000.0)<=dist_mm<=(self.rmax*1000.0):
                    r = dist_mm/1000.0; a = math.radians(ang)
                    frame_pts.append((r*math.cos(a), r*math.sin(a)))
                    frame_q.append(float(q))
                    frame_ang.append(float(ang))
                    frame_dist.append(r)

                # 帧边界：False->True
                if newf and not last_new and frame_pts:
                    xy = np.asarray(frame_pts, np.float32)
                    qq = np.asarray(frame_q,   np.float32)
                    ang_deg = np.asarray(frame_ang, np.float32)
                    dist_m  = np.asarray(frame_dist, np.float32)
                    rr = np.linalg.norm(xy, axis=1)
                    print(f'[SCAN] pts={xy.shape[0]:4d}  r=[{rr.min():.2f},{rr.max():.2f}] m  '
                          f'quality=[{qq.min():.0f},{qq.max():.0f}]')
                    with self.lock:
                        self.xy=xy; self.qual=qq; self.angles=ang_deg; self.ranges=dist_m; self.frame_id+=1
                    self._save_frame(xy, qq, ang_deg, dist_m)
                    frame_pts.clear(); frame_q.clear(); frame_ang.clear(); frame_dist.clear()
                    last_frame_t=time.time()
                last_new=newf
        finally:
            try: stop_and_clear(ser)
            except: pass
            ser.close()

# ---------------------- 2D 轻量里程计 ----------------------
class Odom2D:
    """
    轻量 2D 里程计（scan-to-scan）
    - 多尺度粗配准（4°→2°→1°）
    - 极坐标同角格配对 + 加权 SVD（Huber/截断权重）
    - 1~2 次细化（小范围“极域ICP”）
    - 跳变保护 + 指数平滑
    """
    def __init__(self, angle_bin_deg=1.0, max_rot_deg=25,
                 max_range_diff=0.8, min_overlap_bins=50,
                 tum_log_path=None, refine_iters=1,
                 huber_delta=0.15, smooth_alpha=0.25,
                 max_step_trans=0.8, max_step_rot_deg=12.0):
        self.bin_deg = float(angle_bin_deg)
        self.N = int(round(360.0 / self.bin_deg))
        self.K = int(round(max_rot_deg / self.bin_deg))
        self.max_range_diff = float(max_range_diff)
        self.min_overlap = int(min_overlap_bins)
        self.prev_bins = None           # 上一帧的 r(θ) 向量（长度 N；NaN 表示无效）
        self.pose = np.array([0.0, 0.0, 0.0], np.float64)  # x, y, θ
        self.traj = [self.pose.copy()]
        self.tum = open(tum_log_path, 'w', encoding='utf-8') if tum_log_path else None
        self.refine_iters = int(refine_iters)
        self.huber_delta = float(huber_delta)
        self.smooth_alpha = float(smooth_alpha)
        self.max_step_trans = float(max_step_trans)
        self.max_step_rot = math.radians(max_step_rot_deg)

        # 上一帧的“平滑增量”用于指数平滑
        self.prev_dxdyth = np.zeros(3, dtype=np.float64)

    # ---------- 工具 ----------
    def close(self):
        if self.tum: self.tum.close()

    def _bin_scan(self, angles_deg, ranges_m):
        bins = np.full((self.N,), np.nan, np.float32)
        idx = np.floor((angles_deg % 360.0) / self.bin_deg).astype(int) % self.N
        # 取同一角格的“最小距离”（靠近的更稳定）
        for i, r in zip(idx, ranges_m):
            if np.isnan(bins[i]) or r < bins[i]: bins[i] = r
        return bins

    def _corr_shift(self, prev, curr, K):
        """在 ±K 格内找最大互相关偏移"""
        best_k, best_score, best_overlap = 0, -1e18, 0
        for k in range(-K, K + 1):
            cur_shift = np.roll(curr, k)
            valid = (~np.isnan(prev)) & (~np.isnan(cur_shift))
            ov = int(valid.sum())
            if ov < max(self.min_overlap, 1):  # 保底
                continue
            score = np.nanmean(prev[valid] * cur_shift[valid])
            if score > best_score:
                best_score, best_k, best_overlap = score, k, ov
        return best_k, best_overlap

    def _multi_scale_shift(self, prev, curr):
        """
        4°→2°→1° 多尺度：把两向量按更粗的分箱聚合后相关搜索，再回到细粒度。
        """
        def aggregate(vec, m):
            if m == 1: return vec
            L = len(vec) // m
            if L == 0: return vec
            v = vec[:L*m].reshape(L, m)
            # 采用“最小值聚合”，与 _bin_scan 的策略一致
            out = np.nanmin(v, axis=1)
            return out

        # 4°
        m4 = int(round(4.0 / self.bin_deg))
        p4 = aggregate(prev, max(1, m4)); c4 = aggregate(curr, max(1, m4))
        K4 = max(1, self.K // max(1, m4))
        k4, _ = self._corr_shift(p4, c4, K4)

        # 2°
        m2 = int(round(2.0 / self.bin_deg))
        p2 = aggregate(np.roll(prev, k4*max(1,m4)), max(1, m2))
        c2 = aggregate(curr, max(1, m2))
        K2 = max(1, self.K // max(1, m2))
        k2, _ = self._corr_shift(p2, c2, K2)

        # 1°（或当前 bin_deg）
        p1 = np.roll(prev, k4*max(1,m4) + k2*max(1,m2))
        c1 = curr
        k1, ov = self._corr_shift(p1, c1, self.K)

        total_k = k4*max(1,m4) + k2*max(1,m2) + k1
        return total_k, ov

    @staticmethod
    def _weighted_svd(cur_xy, ref_xy, w):
        """加权 SVD（Procrustes）：w ≥ 0，形状 [M]"""
        w = w.reshape(-1, 1)  # (M,1)
        W = w / max(1e-9, w.sum())
        mu_c = (W * cur_xy).sum(axis=0)
        mu_r = (W * ref_xy).sum(axis=0)
        Cc = cur_xy - mu_c
        Cr = ref_xy - mu_r
        H = (W * Cc).T @ Cr
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_r - (R @ mu_c)
        return R, t

    def _build_pairs_and_svd(self, prev_bins, curr_bins, yaw_k):
        """按角格配对 + 距离差阈值 + Huber 权重，然后加权 SVD"""
        cur_shift = np.roll(curr_bins, yaw_k)
        valid = (~np.isnan(prev_bins)) & (~np.isnan(cur_shift))
        if valid.sum() < self.min_overlap:
            return None, None, None, None

        idx = np.where(valid)[0]
        th = (idx * self.bin_deg) * np.pi/180.0
        prev_r = prev_bins[idx]
        curr_r = cur_shift[idx]

        # 距离差筛选
        keep = np.abs(prev_r - curr_r) <= self.max_range_diff
        if keep.sum() < self.min_overlap:
            return None, None, None, None

        prev_r = prev_r[keep]; curr_r = curr_r[keep]; th = th[keep]
        prev_xy = np.stack([prev_r*np.cos(th), prev_r*np.sin(th)], axis=1)
        cur_xy0 = np.stack([curr_r*np.cos(th), curr_r*np.sin(th)], axis=1)

        # 初始 SVD
        R, t = self._weighted_svd(cur_xy0, prev_xy, np.ones((prev_r.size,), np.float64))

        # 细化：Huber 权重 + 再 SVD（1~2 次）
        for _ in range(self.refine_iters):
            cur_xy = (R @ cur_xy0.T).T + t  # 带入
            res = np.linalg.norm(prev_xy - cur_xy, axis=1)
            # Huber 权重
            d = self.huber_delta
            w = np.where(res <= d, 1.0, d / (res + 1e-9))
            # 再 SVD
            R, t = self._weighted_svd(cur_xy0, prev_xy, w)
        return R, t, prev_xy, cur_xy0

    def _log_tum(self, timestamp):
        if not self.tum: return
        t = timestamp if timestamp is not None else time.time()
        x, y, th = self.pose
        cz = math.cos(th/2.0); sz = math.sin(th/2.0)
        qx, qy, qz, qw = 0.0, 0.0, sz, cz
        self.tum.write(f'{t:.6f} {x:.6f} {y:.6f} 0.000000 0.000000 0.000000 {qz:.6f} {qw:.6f}\n')
        self.tum.flush()

    # ---------- 主接口 ----------
    def update(self, angles_deg, ranges_m, timestamp=None):
        bins = self._bin_scan(angles_deg, ranges_m)

        if self.prev_bins is None:
            self.prev_bins = bins.copy()
            self._log_tum(timestamp)
            return self.pose.copy(), np.eye(3)

        # 多尺度粗配准
        yaw_k, overlap = self._multi_scale_shift(self.prev_bins, bins)
        if overlap < self.min_overlap:
            return self.pose.copy(), np.eye(3)

        # 同角格配对 + 加权SVD（含细化）
        R, t, ref_xy, cur_xy0 = self._build_pairs_and_svd(self.prev_bins, bins, yaw_k)
        if R is None:
            return self.pose.copy(), np.eye(3)

        dtheta = math.atan2(R[1,0], R[0,0])
        dx, dy = t

        # 跳变保护 + 指数平滑在“相对增量”层面
        step = np.array([dx, dy, dtheta], np.float64)
        # 限幅
        step[0] = np.clip(step[0], -self.max_step_trans, self.max_step_trans)
        step[1] = np.clip(step[1], -self.max_step_trans, self.max_step_trans)
        step[2] = np.clip(step[2], -self.max_step_rot,   self.max_step_rot)
        # 平滑
        step = (1.0 - self.smooth_alpha) * self.prev_dxdyth + self.smooth_alpha * step
        self.prev_dxdyth = step.copy()

        # 累加到世界坐标
        x, y, theta = self.pose
        c, s = math.cos(theta), math.sin(theta)
        Rx = c*step[0] - s*step[1]
        Ry = s*step[0] + c*step[1]
        x += Rx; y += Ry; theta = (theta + step[2] + np.pi)%(2*np.pi)-np.pi
        self.pose[:] = (x, y, theta)
        self.traj.append(self.pose.copy())

        # 设置“上一帧”
        self.prev_bins = bins.copy()
        self._log_tum(timestamp)

        T = np.array([[math.cos(step[2]), -math.sin(step[2]), step[0]],
                      [math.sin(step[2]),  math.cos(step[2]), step[1]],
                      [0,0,1]], dtype=np.float64)
        return self.pose.copy(), T

# ---------------------- 主程序：可视化 + 里程计 ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=str, default=None, help='串口（不指定则自动识别）')
    ap.add_argument('--baud', type=int, default=DEFAULT_BAUD, help='波特率（默认 115200）')
    ap.add_argument('--rmin', type=float, default=MIN_R_DEFAULT)
    ap.add_argument('--rmax', type=float, default=MAX_R_DEFAULT)
    ap.add_argument('--down', type=float, default=DOWN_DEG_DEFAULT)
    ap.add_argument('--save-dir', type=str, default=None, help='保存帧到目录（CSV；--save-ply 可另存 PLY）')
    ap.add_argument('--save-ply', action='store_true')
    ap.add_argument('--color', type=str, default='range', choices=['range','quality'])
    # ODOM 参数
    ap.add_argument('--angle-bin', type=float, default=ANGLE_BIN_DEG_DEFAULT)
    ap.add_argument('--odom-max-rot', type=float, default=ODOM_MAX_ROT_DEG_DEFAULT)
    ap.add_argument('--odom-max-diff', type=float, default=ODOM_MAX_RANGE_DIFF)
    ap.add_argument('--odom-min-overlap', type=int, default=ODOM_MIN_OVERLAP_BINS)
    ap.add_argument('--tum', type=str, default=TUM_LOG_DEFAULT, help='TUM 轨迹输出路径（可选）')
    args = ap.parse_args()

    # 自动识别端口
    if args.port:
        port = args.port
    else:
        print('[INFO] 正在自动识别端口...')
        port = auto_find_port(None, args.baud, DEFAULT_TIMEOUT)
    print(f'[INFO] 使用端口 {port} @ {args.baud}')

    # 读取线程
    reader = LidarReader(port, args.baud, DEFAULT_TIMEOUT, args.down, args.rmin, args.rmax)
    if args.save_dir: reader.set_save_dir(args.save_dir)
    if args.save_ply: reader.set_save_ply(True)
    reader.color_mode = args.color
    reader.start()

    # 里程计
    odom = Odom2D(angle_bin_deg=args.angle_bin,
                  max_rot_deg=args.odom_max_rot,
                  max_range_diff=args.odom_max_diff,
                  min_overlap_bins=args.odom_min_overlap,
                  tum_log_path=args.tum)

    # 画布：左=扫描点云，右=轨迹
    plt.style.use('dark_background')
    fig, (ax_scan, ax_traj) = plt.subplots(1, 2, figsize=(11,6))
    # 左：点云
    sc = ax_scan.scatter([], [], s=2, c=[], cmap='viridis', vmin=0, vmax=1, alpha=0.95)
    lim = args.rmax*1.05
    ax_scan.set_xlim(-lim, lim); ax_scan.set_ylim(-lim, lim); ax_scan.set_aspect('equal')
    ax_scan.set_title('RPLIDAR A1 (NORMAL@115200)')
    for r in np.linspace(max(0.5, args.rmin), args.rmax, num=6):
        ax_scan.add_patch(plt.Circle((0,0), r, color='gray', alpha=0.2, fill=False, lw=0.6))
    ax_scan.axhline(0.0, color='gray', lw=0.4, alpha=0.4)
    ax_scan.axvline(0.0, color='gray', lw=0.4, alpha=0.4)
    textL = ax_scan.text(0.02, 0.98, '', transform=ax_scan.transAxes, va='top', ha='left', fontsize=9)

    # 右：轨迹
    traj_ln, = ax_traj.plot([], [], 'o-', ms=2, lw=1.0, color='#6cf')
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.set_title('2D Odometry (scan-to-scan)')
    ax_traj.grid(True, alpha=0.2)
    textR = ax_traj.text(0.02, 0.98, '', transform=ax_traj.transAxes, va='top', ha='left', fontsize=9)

    last_frame_done = -1
    last_t = time.time(); frames = 0

    def frame_to_color(xy, q, mode, rmax):
        if xy.size == 0: return np.array([])
        if mode == 'quality':
            if q.size == 0: return np.zeros((xy.shape[0],))
            qn = (q - q.min()) / max(1e-6, (q.max() - q.min()))
            return qn
        else:
            r = np.linalg.norm(xy, axis=1)
            rn = np.clip(r / max(1e-6, rmax), 0.0, 1.0)
            return rn

    def on_key(evt):
        if evt.key in ('q','Q'):
            plt.close(fig)
        elif evt.key in ('s','S'):
            if reader.save_dir is None:
                d = os.path.join(os.getcwd(), 'lidar_frames')
                reader.set_save_dir(d)
                print(f'[INFO] 开始保存帧到 {d} （CSV{"+PLY" if reader.save_ply else ""}）')
            else:
                print('[INFO] 停止保存帧'); reader.set_save_dir(None)
        elif evt.key in ('g','G'):
            reader.toggle_color()
        elif evt.key in ('r','R'):
            print('[INFO] 轻量重同步将在后台自动执行（无须关闭窗口）')

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_):
        nonlocal last_frame_done, last_t, frames
        # 1) 读取最新一帧
        with reader.lock:
            fid = reader.frame_id
            xy  = reader.xy.copy()
            q   = reader.qual.copy()
            ang = reader.angles.copy()
            rng = reader.ranges.copy()

        # 2) 左图：点云
        if xy.size:
            c = frame_to_color(xy, q, reader.color_mode, args.rmax)
            sc.set_offsets(xy)
            if c.size:
                sc.set_array(c)
                sc.set_cmap('viridis' if reader.color_mode=='range' else 'plasma')

        # 3) 右图：里程计（仅在“新帧到达”时更新一次）
        if fid != last_frame_done and ang.size and rng.size:
            pose, Trel = odom.update(ang, rng, timestamp=time.time())
            last_frame_done = fid

        # 4) 轨迹绘制 —— 即使只有 1 个点也要画！
        traj = np.array(odom.traj, np.float64)
        if traj.shape[0] >= 1:
            traj_ln.set_data(traj[:,0], traj[:,1])
            if traj.shape[0] >= 2:
                xmin, xmax = traj[:,0].min(), traj[:,0].max()
                ymin, ymax = traj[:,1].min(), traj[:,1].max()
                dx = max(1.0, (xmax - xmin) * 0.2)
                dy = max(1.0, (ymax - ymin) * 0.2)
                ax_traj.set_xlim(xmin - dx, xmax + dx)
                ax_traj.set_ylim(ymin - dy, ymax + dy)
            else:
                # 只有 1 个姿态点时，给一个 2m 的视窗，保证“看得见”
                x, y = traj[0,0], traj[0,1]
                ax_traj.set_xlim(x - 1.0, x + 1.0)
                ax_traj.set_ylim(y - 1.0, y + 1.0)

        # 5) HUD
        frames += 1
        now = time.time()
        if now - last_t >= 1.0:
            textL.set_text(f'points: {xy.shape[0]:d}  FPS: {frames/(now-last_t):.1f}  color={reader.color_mode}')
            x, y, th = odom.pose
            textR.set_text(f'pose: x={x:.2f} y={y:.2f} th={math.degrees(th):.1f}°  trajN={len(odom.traj)}')
            frames = 0
            last_t = now
        return (sc, traj_ln, textL, textR)

    ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop(); reader.join(timeout=2); odom.close()

if __name__ == '__main__':
    main()