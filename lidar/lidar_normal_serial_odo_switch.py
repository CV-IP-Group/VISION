# -*- coding: utf-8 -*-
"""
RPLIDAR A1 (A1M8-R6, fw=1.29) —— 纯 pyserial NORMAL@115200
+ 自动端口 + 实时点云 + 2D 里程计（--odom-method {p2p,p2l} 可切换）
+ 朝向箭头 + 速度估计 + 录像（CSV/可选 PLY）

使用示例：
  python lidar_normal_serial_odo_switch.py                          # 自动找口，默认 p2l
  python lidar_normal_serial_odo_switch.py --odom-method p2p        # 点-点 SVD
  python lidar_normal_serial_odo_switch.py --save-dir ./logs --save-ply
  python lidar_normal_serial_odo_switch.py --angle-bin 1.0 --odom-max-rot 25 --odom-max-diff 0.8 --odom-min-overlap 50
"""

import json
import os, sys, time, math, threading, argparse, datetime
import requests
import pyttsx3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
from serial.tools import list_ports
try:
    import winsound
except ImportError:
    winsound = None

# ---------------------- 常量/默认参数 ----------------------
SYNC = b'\xA5'; STOP = b'\x25'; SCAN = b'\x20'   # NORMAL
DESC_LEN = 7

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 0.2
MIN_R_DEFAULT, MAX_R_DEFAULT = 0.15, 20.0
DOWN_DEG_DEFAULT = 1.0

# ODOM 默认
ANGLE_BIN_DEG_DEFAULT = 1.0
ODOM_MAX_ROT_DEG_DEFAULT = 25
ODOM_MAX_RANGE_DIFF_DEFAULT = 0.8
ODOM_MIN_OVERLAP_BINS_DEFAULT = 50
TUM_LOG_DEFAULT = None

# 常见 USB-UART VID:PID
KNOWN_VIDPID = {('10C4','EA60'), ('1A86','7523'), ('067B','2303')}
HAZARD_ANGLE_RANGE = 30.0
HAZARD_DIST_THRESHOLD = 0.8
SNAPSHOT_FILE = "runs/lidar_snapshot.json"
MCP_API_URL = "http://127.0.0.1:9000/api/v1/mcp"
MCP_COOLDOWN_SECONDS = 5.0

_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", 150)
_tts_engine.setProperty("volume", 0.9)

# ------------------------------------------------------------------
def _forward_hazard_to_mcp_and_speak() -> None:
    """Send a hazard check to MCP and speak the returned message. """
    payload = {
        "api_version": "v1",
        "text": "check hazards",
        "camera_index": None,
        "tts": False,
    }
    try:
        response = requests.post(MCP_API_URL, json=payload, timeout=0.5)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or data.get("intent", {}).get("msg")
        if message:
            print("[LIDAR MCP]", message)
            _tts_engine.say(message)
            _tts_engine.runAndWait()
    except Exception as exc:
        print(f"[LIDAR MCP ERROR] {exc}")

# ---------------------- 通用小工具 ----------------------
def huber_weights(res, delta=0.15):
    res = np.asarray(res, float)
    w = np.ones_like(res)
    mask = np.abs(res) > delta
    w[mask] = delta / (np.abs(res[mask]) + 1e-12)
    return w

def read_exact(ser, n, t=1.0):
    buf = bytearray(); end = time.time() + t
    while len(buf) < n and time.time() < end:
        chunk = ser.read(n - len(buf))
        if chunk: buf += chunk
    return bytes(buf) if len(buf) == n else None

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
    payload_len = raw_len & 0x3FFFFFFF  # 低30位为真实长度；高2位是应答标志
    dtype = desc[6]
    return (True, payload_len, dtype)

def parse_node(raw5):
    b0,b1,b2,b3,b4 = raw5
    newf = bool(b0 & 0x01); inv_new = bool((b0>>1) & 0x01)
    if newf == inv_new: return (False, None, None, None, None)
    if (b1 & 0x01) != 1:   return (False, None, None, None, None)
    angle_q6 = ((b1>>1) | (b2<<7)); dist_q2 = (b3 | (b4<<8))
    ang = angle_q6/64.0;   dist_mm = dist_q2/4.0
    quality = (b0>>2)
    return (True, newf, ang, dist_mm, quality)

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

# ---------------------- Reader：串口读取 → 帧数据 ----------------------
class LidarReader:
    """
    线程读取 NORMAL@115200 的 5 字节节点，按“新帧”边界切帧。
    输出：xy（N,2）、qual（N,）、angles_deg（N,）、ranges_m（N,）
    """
    def __init__(self, port, baud, timeout, down_deg, rmin, rmax):
        self.port=port; self.baud=baud; self.timeout=timeout
        self.down_deg=down_deg; self.rmin=rmin; self.rmax=rmax
        self.stop_evt=threading.Event()
        self.lock = threading.Lock()
        self.xy  = np.zeros((0,2), np.float32)
        self.qual= np.zeros((0,),  np.float32)
        self.angles = np.zeros((0,), np.float32)
        self.ranges = np.zeros((0,), np.float32)
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

            in_hazard_alert = False
            last_mcp_ts = 0.0

            def _emit_hazard_beep() -> None:
                if winsound:
                    try:
                        winsound.Beep(1000, 200)
                    except RuntimeError:
                        print('\a', end='', flush=True)
                else:
                    print('\a', end='', flush=True)

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

                if (ang % self.down_deg) <= 0.5 and (self.rmin*1000.0)<=dist_mm<=(self.rmax*1000.0):
                    r = dist_mm/1000.0; a = math.radians(ang)
                    frame_pts.append((r*math.cos(a), r*math.sin(a)))
                    frame_q.append(float(q))
                    frame_ang.append(float(ang))
                    frame_dist.append(r)

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
                    # Sentry logic + Provider snapshot
                    forward_mask = (ang_deg < HAZARD_ANGLE_RANGE) | (ang_deg > (360.0 - HAZARD_ANGLE_RANGE))
                    forward_min_dist = None
                    if forward_mask.any():
                        forward_min_dist = float(np.min(dist_m[forward_mask]))
                    now = time.time()
                    is_hazard = (
                        forward_min_dist is not None
                        and forward_min_dist < HAZARD_DIST_THRESHOLD
                    )
                    if is_hazard:
                        if not in_hazard_alert:
                            _emit_hazard_beep()
                            if now - last_mcp_ts >= MCP_COOLDOWN_SECONDS:
                                _forward_hazard_to_mcp_and_speak()
                                last_mcp_ts = now
                            in_hazard_alert = True
                        else:
                            _emit_hazard_beep()
                    else:
                        in_hazard_alert = False
                    snapshot = {
                        'forward_min_distance': forward_min_dist,
                        'source': 'hardware',
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    tmp_snapshot = SNAPSHOT_FILE + '.tmp'
                    try:
                        snapshot_dir = os.path.dirname(SNAPSHOT_FILE)
                        if snapshot_dir:
                            os.makedirs(snapshot_dir, exist_ok=True)
                        with open(tmp_snapshot, 'w', encoding='utf-8') as f:
                            json.dump(snapshot, f, ensure_ascii=False)
                            f.flush(); os.fsync(f.fileno())
                        os.replace(tmp_snapshot, SNAPSHOT_FILE)
                    except Exception as exc:
                        print(f'[WARN] 快照写入失败: {exc}')
                    self._save_frame(xy, qq, ang_deg, dist_m)
                    frame_pts.clear(); frame_q.clear(); frame_ang.clear(); frame_dist.clear()
                    last_frame_t=time.time()
                last_new=newf
        finally:
            try: stop_and_clear(ser)
            except: pass
            ser.close()

# ---------------------- 里程计工具：分箱/法线/XY ----------------------
def bins_to_xy(bins, bin_deg):
    N = len(bins)
    th = (np.arange(N)*bin_deg)*np.pi/180.0
    x = bins*np.cos(th); y = bins*np.sin(th)
    return np.stack([x,y], axis=1).astype(np.float32)

def estimate_normals_from_bins(prev_bins, bin_deg, win=2):
    N=len(prev_bins)
    normals = np.full((N,2), np.nan, np.float32)
    th = (np.arange(N)*bin_deg)*np.pi/180.0
    xy = np.stack([prev_bins*np.cos(th), prev_bins*np.sin(th)], axis=1)
    for k in range(N):
        a=(k-win)%N; b=(k+win)%N
        if not (np.isfinite(xy[a]).all() and np.isfinite(xy[b]).all() and np.isfinite(xy[k]).all()):
            continue
        t = xy[b]-xy[a]; nt=np.linalg.norm(t)
        if nt<1e-6: continue
        t=t/nt
        normals[k]=np.array([-t[1], t[0]], np.float32)
    return normals

# ---------------------- Odom：点-点（p2p） ----------------------
class Odom2D_P2P:
    """角度分箱 + 多尺度相关粗配准 + 点-点 SVD 精配 + 限幅 + 平滑"""
    def __init__(self, angle_bin_deg=1.0, max_rot_deg=25,
                 max_range_diff=0.8, min_overlap_bins=50,
                 tum_log_path=None, smooth_alpha=0.25,
                 max_step_trans=0.8, max_step_rot_deg=12.0):
        self.bin_deg=float(angle_bin_deg); self.N=int(round(360.0/self.bin_deg))
        self.K=int(round(max_rot_deg/self.bin_deg))
        self.max_range_diff=float(max_range_diff); self.min_overlap=int(min_overlap_bins)
        self.pose=np.array([0.0,0.0,0.0],np.float64); self.traj=[self.pose.copy()]
        self.prev_bins=None; self.prev_step=np.zeros(3); self.last_dt=0.0
        self.smooth_alpha=float(smooth_alpha)
        self.max_step_trans=float(max_step_trans); self.max_step_rot=math.radians(max_step_rot_deg)
        self.tum=open(tum_log_path,'w',encoding='utf-8') if tum_log_path else None
        self._last_ts=None

    def close(self): 
        if self.tum: self.tum.close()

    def _log_tum(self, ts):
        if not self.tum: return
        t = ts if ts is not None else time.time()
        x,y,th=self.pose; cz,sz=math.cos(th/2), math.sin(th/2)
        self.tum.write(f'{t:.6f} {x:.6f} {y:.6f} 0 0 0 {sz:.6f} {cz:.6f}\n'); self.tum.flush()

    def _bin_scan(self, ang, rng):
        bins = np.full((self.N,), np.nan, np.float32)
        idx = np.floor((ang%360.0)/self.bin_deg).astype(int)%self.N
        for i,r in zip(idx,rng):
            if np.isnan(bins[i]) or r<bins[i]: bins[i]=r
        return bins

    def _aggregate(self, vec, m):
        if m<=1: return vec
        L=len(vec)//m
        if L==0: return vec
        v=vec[:L*m].reshape(L,m)
        return np.nanmin(v,axis=1)

    def _corr_shift(self, prev,curr,K,min_overlap):
        best_k=0; best_s=-1e18; best_ov=0
        for k in range(-K,K+1):
            cs=np.roll(curr,k)
            valid=(~np.isnan(prev)) & (~np.isnan(cs))
            ov=int(valid.sum())
            if ov<min_overlap: continue
            s=float(np.nanmean(prev[valid]*cs[valid]))
            if s>best_s: best_s, best_k, best_ov = s, k, ov
        return best_k, best_ov

    def _multi_scale_shift(self, prev, curr):
        m4=max(1,int(round(4.0/self.bin_deg))); p4=self._aggregate(prev,m4); c4=self._aggregate(curr,m4)
        K4=max(1,self.K//m4); k4,_=self._corr_shift(p4,c4,K4,max(1,self.min_overlap//m4))
        m2=max(1,int(round(2.0/self.bin_deg))); p2=self._aggregate(np.roll(prev,k4*m4),m2); c2=self._aggregate(curr,m2)
        K2=max(1,self.K//m2); k2,_=self._corr_shift(p2,c2,K2,max(1,self.min_overlap//m2))
        p1=np.roll(prev,k4*m4+k2*m2); c1=curr
        k1,ov=self._corr_shift(p1,c1,self.K,self.min_overlap)
        return k4*m4+k2*m2+k1, ov

    def _svd(self, cur, ref):
        mu_c=cur.mean(axis=0); mu_r=ref.mean(axis=0)
        Cc=cur-mu_c; Cr=ref-mu_r
        H=Cc.T@Cr
        U,S,Vt=np.linalg.svd(H)
        R=Vt.T@U.T
        if np.linalg.det(R)<0:
            Vt[-1,:]*=-1; R=Vt.T@U.T
        t = mu_r - R@mu_c
        dth=math.atan2(R[1,0],R[0,0])
        return t[0],t[1],dth

    def update(self, angles_deg, ranges_m, timestamp=None):
        bins = self._bin_scan(angles_deg, ranges_m)
        if self.prev_bins is None:
            self.prev_bins=bins.copy(); self._last_ts=timestamp; self._log_tum(timestamp)
            return self.pose.copy(), np.eye(3)

        k,ov = self._multi_scale_shift(self.prev_bins, bins)
        if ov < self.min_overlap:
            return self.pose.copy(), np.eye(3)

        cur_rot=np.roll(bins,k)
        valid=(~np.isnan(self.prev_bins)) & (~np.isnan(cur_rot))
        valid &= (np.abs(self.prev_bins-cur_rot) <= self.max_range_diff)
        idx=np.where(valid)[0]
        if idx.size < self.min_overlap:
            return self.pose.copy(), np.eye(3)

        th=(idx*self.bin_deg)*np.pi/180.0
        ref_r=self.prev_bins[idx]; cur_r=cur_rot[idx]
        ref=np.stack([ref_r*np.cos(th), ref_r*np.sin(th)], axis=1)
        cur=np.stack([cur_r*np.cos(th),  cur_r*np.sin(th)], axis=1)

        dx,dy,dth = self._svd(cur, ref)

        # 限幅 + 平滑
        step=np.array([dx,dy,dth],np.float64)
        step[0]=np.clip(step[0],-self.max_step_trans,self.max_step_trans)
        step[1]=np.clip(step[1],-self.max_step_trans,self.max_step_trans)
        step[2]=np.clip(step[2],-self.max_step_rot,  self.max_step_rot)
        step = (1.0-self.smooth_alpha)*self.prev_step + self.smooth_alpha*step
        self.prev_step=step.copy()

        # 时间差
        if self._last_ts is not None and timestamp is not None:
            self.last_dt = max(1e-3, float(timestamp - self._last_ts))
        else:
            self.last_dt = 0.0
        self._last_ts = timestamp

        # 累加世界坐标
        x,y,th_pose=self.pose
        c,s=math.cos(th_pose), math.sin(th_pose)
        Rx=c*step[0]-s*step[1]; Ry=s*step[0]+c*step[1]
        x+=Rx; y+=Ry; th_pose=(th_pose+step[2]+np.pi)%(2*np.pi)-np.pi
        self.pose[:]=[x,y,th_pose]; self.traj.append(self.pose.copy())

        self.prev_bins=bins.copy(); self._log_tum(timestamp)
        T=np.array([[math.cos(step[2]),-math.sin(step[2]), step[0]],
                    [math.sin(step[2]), math.cos(step[2]), step[1]],
                    [0,0,1]], dtype=np.float64)
        return self.pose.copy(), T

# ---------------------- Odom：点-线（p2l） ----------------------
class Odom2D_P2L:
    """角度分箱 + 多尺度粗配准 + point-to-line 最小二乘 + Huber + 小步细化 + 限幅 + 平滑"""
    def __init__(self, angle_bin_deg=1.0, max_rot_deg=25,
                 max_range_diff=0.8, min_overlap_bins=50,
                 tum_log_path=None, refine_iters=2,
                 huber_delta=0.15, smooth_alpha=0.25,
                 max_step_trans=0.8, max_step_rot_deg=12.0):
        self.bin_deg=float(angle_bin_deg); self.N=int(round(360.0/self.bin_deg))
        self.K=int(round(max_rot_deg/self.bin_deg))
        self.max_range_diff=float(max_range_diff); self.min_overlap=int(min_overlap_bins)
        self.refine_iters=int(refine_iters); self.huber_delta=float(huber_delta)
        self.pose=np.array([0.0,0.0,0.0],np.float64); self.traj=[self.pose.copy()]
        self.prev_bins=None; self.prev_step=np.zeros(3); self.last_dt=0.0
        self.smooth_alpha=float(smooth_alpha)
        self.max_step_trans=float(max_step_trans); self.max_step_rot=math.radians(max_step_rot_deg)
        self.tum=open(tum_log_path,'w',encoding='utf-8') if tum_log_path else None
        self._last_ts=None

    def close(self):
        if self.tum: self.tum.close()

    def _log_tum(self, ts):
        if not self.tum: return
        t = ts if ts is not None else time.time()
        x,y,th=self.pose; cz,sz=math.cos(th/2), math.sin(th/2)
        self.tum.write(f'{t:.6f} {x:.6f} {y:.6f} 0 0 0 {sz:.6f} {cz:.6f}\n'); self.tum.flush()

    def _bin_scan(self, ang, rng):
        bins = np.full((self.N,), np.nan, np.float32)
        idx = np.floor((ang%360.0)/self.bin_deg).astype(int)%self.N
        for i,r in zip(idx,rng):
            if np.isnan(bins[i]) or r<bins[i]: bins[i]=r
        return bins

    def _aggregate(self, vec, m):
        if m<=1: return vec
        L=len(vec)//m
        if L==0: return vec
        v=vec[:L*m].reshape(L,m)
        return np.nanmin(v,axis=1)

    def _corr_shift(self, prev,curr,K,min_overlap):
        best_k=0; best_s=-1e18; best_ov=0
        for k in range(-K,K+1):
            cs=np.roll(curr,k)
            valid=(~np.isnan(prev)) & (~np.isnan(cs))
            ov=int(valid.sum())
            if ov<min_overlap: continue
            s=float(np.nanmean(prev[valid]*cs[valid]))
            if s>best_s: best_s, best_k, best_ov = s, k, ov
        return best_k, best_ov

    def _multi_scale_shift(self, prev, curr):
        m4=max(1,int(round(4.0/self.bin_deg))); p4=self._aggregate(prev,m4); c4=self._aggregate(curr,m4)
        K4=max(1,self.K//m4); k4,_=self._corr_shift(p4,c4,K4,max(1,self.min_overlap//m4))
        m2=max(1,int(round(2.0/self.bin_deg))); p2=self._aggregate(np.roll(prev,k4*m4),m2); c2=self._aggregate(curr,m2)
        K2=max(1,self.K//m2); k2,_=self._corr_shift(p2,c2,K2,max(1,self.min_overlap//m2))
        p1=np.roll(prev,k4*m4+k2*m2); c1=curr
        k1,ov=self._corr_shift(p1,c1,self.K,self.min_overlap)
        return k4*m4+k2*m2+k1, ov

    def _pt2line_once(self, prev_bins, curr_bins, yaw_k):
        dx=dy=dth=0.0
        cur_rot=np.roll(curr_bins, yaw_k)
        normals = estimate_normals_from_bins(prev_bins, self.bin_deg, win=2)
        prev_xy = bins_to_xy(prev_bins, self.bin_deg)
        curr_xy0= bins_to_xy(cur_rot,    self.bin_deg)

        valid=(np.isfinite(prev_bins) & np.isfinite(cur_rot) &
               np.isfinite(normals).all(axis=1) & np.isfinite(prev_xy).all(axis=1) &
               np.isfinite(curr_xy0).all(axis=1))
        valid &= (np.abs(prev_bins-cur_rot) <= self.max_range_diff)
        idx=np.where(valid)[0]
        if idx.size < self.min_overlap:
            return 0.0,0.0,0.0,0

        n=normals[idx]; q=prev_xy[idx]; p0=curr_xy0[idx]

        for _ in range(max(1,self.refine_iters)):
            c,s=math.cos(dth), math.sin(dth)
            R=np.array([[c,-s],[s,c]])
            p=(R @ p0.T).T + np.array([dx,dy])
            r=np.sum(n*(p-q), axis=1)                 # 法向残差
            w=huber_weights(r, self.huber_delta)      # Huber 权重
            Jp=np.stack([-p[:,1], p[:,0]], axis=1)    # Jp
            A =np.stack([n[:,0], n[:,1], np.sum(n*Jp, axis=1)], axis=1)
            b =-r
            W=np.sqrt(w)[:,None]; Aw=A*W; bw=b*W[:,0]
            try:
                dxi,*_ = np.linalg.lstsq(Aw, bw, rcond=None)
            except np.linalg.LinAlgError:
                break
            dx+=float(dxi[0]); dy+=float(dxi[1]); dth+=float(dxi[2])
            if np.linalg.norm(dxi) < 1e-4: break

        return dx,dy,dth, idx.size

    def update(self, angles_deg, ranges_m, timestamp=None):
        bins = self._bin_scan(angles_deg, ranges_m)
        if self.prev_bins is None:
            self.prev_bins=bins.copy(); self._last_ts=timestamp; self._log_tum(timestamp)
            return self.pose.copy(), np.eye(3)

        k,ov=self._multi_scale_shift(self.prev_bins, bins)
        if ov < self.min_overlap:
            return self.pose.copy(), np.eye(3)

        dx,dy,dth, used = self._pt2line_once(self.prev_bins, bins, k)

        # 限幅 + 平滑
        step=np.array([dx,dy,dth],np.float64)
        step[0]=np.clip(step[0],-self.max_step_trans,self.max_step_trans)
        step[1]=np.clip(step[1],-self.max_step_trans,self.max_step_trans)
        step[2]=np.clip(step[2],-self.max_step_rot,  self.max_step_rot)
        step = (1.0-self.smooth_alpha)*self.prev_step + self.smooth_alpha*step
        self.prev_step=step.copy()

        # 时间差
        if hasattr(self, '_last_ts') and (self._last_ts is not None) and (timestamp is not None):
            self.last_dt = max(1e-3, float(timestamp - self._last_ts))
        else:
            self.last_dt = 0.0
        self._last_ts = timestamp

        # 累加世界坐标
        x,y,th_pose=self.pose
        c,s=math.cos(th_pose), math.sin(th_pose)
        Rx=c*step[0]-s*step[1]; Ry=s*step[0]+c*step[1]
        x+=Rx; y+=Ry; th_pose=(th_pose+step[2]+np.pi)%(2*np.pi)-np.pi
        self.pose[:]=[x,y,th_pose]; self.traj.append(self.pose.copy())

        self.prev_bins=bins.copy(); self._log_tum(timestamp)
        T=np.array([[math.cos(step[2]),-math.sin(step[2]), step[0]],
                    [math.sin(step[2]), math.cos(step[2]), step[1]],
                    [0,0,1]], dtype=np.float64)
        return self.pose.copy(), T

# ---------------------- 主程序：开关/可视化/速度/朝向箭头 ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=str, default=None, help='串口（不指定则自动识别）')
    ap.add_argument('--baud', type=int, default=DEFAULT_BAUD)
    ap.add_argument('--rmin', type=float, default=MIN_R_DEFAULT)
    ap.add_argument('--rmax', type=float, default=MAX_R_DEFAULT)
    ap.add_argument('--down', type=float, default=DOWN_DEG_DEFAULT)
    ap.add_argument('--save-dir', type=str, default=None)
    ap.add_argument('--save-ply', action='store_true')
    ap.add_argument('--color', type=str, default='range', choices=['range','quality'])
    # ODOM 参数
    ap.add_argument('--odom-method', type=str, default='p2l', choices=['p2p','p2l'], help='里程计方法：点-点/点-线')
    ap.add_argument('--angle-bin', type=float, default=ANGLE_BIN_DEG_DEFAULT)
    ap.add_argument('--odom-max-rot', type=float, default=ODOM_MAX_ROT_DEG_DEFAULT)
    ap.add_argument('--odom-max-diff', type=float, default=ODOM_MAX_RANGE_DIFF_DEFAULT)
    ap.add_argument('--odom-min-overlap', type=int, default=ODOM_MIN_OVERLAP_BINS_DEFAULT)
    ap.add_argument('--smooth-alpha', type=float, default=0.25, help='增量指数平滑系数 (0..1)')
    ap.add_argument('--max-step-trans', type=float, default=0.8, help='单步平移限幅 (m)')
    ap.add_argument('--max-step-rot', type=float, default=12.0, help='单步转角限幅 (deg)')
    ap.add_argument('--refine-iters', type=int, default=2, help='p2l 细化迭代次数')
    ap.add_argument('--huber-delta', type=float, default=0.15, help='p2l Huber 阈值 (m)')
    ap.add_argument('--tum', type=str, default=TUM_LOG_DEFAULT)
    args = ap.parse_args()

    # 自动识别端口
    if args.port:
        port = args.port
    else:
        print('[INFO] 正在自动识别端口...')
        port = auto_find_port(None, args.baud, DEFAULT_TIMEOUT)
    print(f'[INFO] 使用端口 {port} @ {args.baud}  |  Odom={args.odom_method}')

    # 读取线程
    reader = LidarReader(port, args.baud, DEFAULT_TIMEOUT, args.down, args.rmin, args.rmax)
    if args.save_dir: reader.set_save_dir(args.save_dir)
    if args.save_ply: reader.set_save_ply(True)
    reader.color_mode = args.color
    reader.start()

    # 里程计实例
    if args.odom_method == 'p2p':
        odom = Odom2D_P2P(angle_bin_deg=args.angle_bin,
                          max_rot_deg=args.odom_max_rot,
                          max_range_diff=args.odom_max_diff,
                          min_overlap_bins=args.odom_min_overlap,
                          tum_log_path=args.tum,
                          smooth_alpha=args.smooth_alpha,
                          max_step_trans=args.max_step_trans,
                          max_step_rot_deg=args.max_step_rot)
    else:
        odom = Odom2D_P2L(angle_bin_deg=args.angle_bin,
                          max_rot_deg=args.odom_max_rot,
                          max_range_diff=args.odom_max_diff,
                          min_overlap_bins=args.odom_min_overlap,
                          tum_log_path=args.tum,
                          refine_iters=args.refine_iters,
                          huber_delta=args.huber_delta,
                          smooth_alpha=args.smooth_alpha,
                          max_step_trans=args.max_step_trans,
                          max_step_rot_deg=args.max_step_rot)

    # 画布：左=点云，右=轨迹（含朝向箭头与速度）
    plt.style.use('dark_background')
    fig, (ax_scan, ax_traj) = plt.subplots(1, 2, figsize=(11,6))
    # 左：点云
    sc = ax_scan.scatter([], [], s=2, c=[], cmap='viridis', vmin=0, vmax=1, alpha=0.95)
    lim = args.rmax*1.05
    ax_scan.set_xlim(-lim, lim); ax_scan.set_ylim(-lim, lim); ax_scan.set_aspect('equal')
    ax_scan.set_title(f'RPLIDAR A1 (NORMAL@{args.baud})')
    for r in np.linspace(max(0.5, args.rmin), args.rmax, num=6):
        ax_scan.add_patch(plt.Circle((0,0), r, color='gray', alpha=0.2, fill=False, lw=0.6))
    ax_scan.axhline(0.0, color='gray', lw=0.4, alpha=0.4)
    ax_scan.axvline(0.0, color='gray', lw=0.4, alpha=0.4)
    textL = ax_scan.text(0.02, 0.98, '', transform=ax_scan.transAxes, va='top', ha='left', fontsize=9)

    # 右：轨迹 + 朝向箭头
    traj_ln, = ax_traj.plot([], [], 'o-', ms=2, lw=1.0, color='#6cf')
    # 初始朝向箭头（长度 0.5m）
    heading_len = 0.5
    arrow = ax_traj.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1.0, color='#ffcc66', width=0.008)
    ax_traj.set_aspect('equal', adjustable='box'); ax_traj.grid(True, alpha=0.2)
    ax_traj.set_title(f'2D Odom ({args.odom_method})')
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
        nonlocal last_frame_done, last_t, frames, arrow
        with reader.lock:
            fid = reader.frame_id
            xy  = reader.xy.copy(); q = reader.qual.copy()
            ang = reader.angles.copy(); rng = reader.ranges.copy()

        # 左图：点云
        if xy.size:
            c = frame_to_color(xy, q, reader.color_mode, args.rmax)
            sc.set_offsets(xy)
            if c.size: sc.set_array(c); sc.set_cmap('viridis' if reader.color_mode=='range' else 'plasma')

        # 右图：里程计（新帧时更新）
        if fid != last_frame_done and ang.size and rng.size:
            ts = time.time()
            pose, Trel = odom.update(ang, rng, timestamp=ts)
            last_frame_done = fid

        # 轨迹 + 朝向箭头
        traj = np.asarray(odom.traj, np.float64)
        if traj.shape[0] >= 1:
            traj_ln.set_data(traj[:,0], traj[:,1])
            # 箭头：以当前姿态 (x,y,θ) 朝向为单位向量
            x, y, th = odom.pose
            hx = math.cos(th) * heading_len
            hy = math.sin(th) * heading_len
            if arrow: arrow.remove()
            arrow = ax_traj.quiver([x], [y], [hx], [hy], angles='xy',
                                   scale_units='xy', scale=1.0, color='#ffcc66', width=0.008)
            # 视野
            if traj.shape[0] >= 2:
                xmin,xmax = traj[:,0].min(), traj[:,0].max()
                ymin,ymax = traj[:,1].min(), traj[:,1].max()
                dx = max(1.0, (xmax-xmin)*0.2); dy = max(1.0, (ymax-ymin)*0.2)
                ax_traj.set_xlim(xmin-dx, xmax+dx); ax_traj.set_ylim(ymin-dy, ymax+dy)
            else:
                ax_traj.set_xlim(x-1.0, x+1.0); ax_traj.set_ylim(y-1.0, y+1.0)

        # HUD：左侧 FPS；右侧 pose + 速度
        frames += 1; now=time.time()
        if now-last_t >= 1.0:
            textL.set_text(f'points: {xy.shape[0]:d}  FPS: {frames/(now-last_t):.1f}  color={reader.color_mode}')
            # 速度估计
            dt = getattr(odom, 'last_dt', 0.0)
            if dt and dt>0:
                # 注意：odom.prev_step 是里程计坐标系下的增量，近似线速度用欧氏长度/时间
                dx,dy,dth = getattr(odom, 'prev_step', np.zeros(3))
                v  = math.hypot(dx,dy) / dt
                w  = math.degrees(dth) / dt
                textR.set_text(f'pose: x={odom.pose[0]:.2f} y={odom.pose[1]:.2f} th={math.degrees(odom.pose[2]):.1f}° '
                               f'| trajN={len(odom.traj)}\n'
                               f'v={v:.2f} m/s,  w={w:.1f} °/s')
            else:
                textR.set_text(f'pose: x={odom.pose[0]:.2f} y={odom.pose[1]:.2f} th={math.degrees(odom.pose[2]):.1f}° '
                               f'| trajN={len(odom.traj)}\n'
                               f'v=-- m/s, w=-- °/s')
            frames=0; last_t=now
        return (sc, traj_ln, arrow, textL, textR)

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop(); reader.join(timeout=2); odom.close()

if __name__ == '__main__':
    main()
