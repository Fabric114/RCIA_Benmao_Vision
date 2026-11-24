
import sys, os
import math
import shutil
from pathlib import Path

import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FOUNDATIONPOSE_DIR = PROJECT_ROOT / 'foundation_pose'
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'runtime/matplotlib'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / 'runtime/ultralytics'))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

from estimater import *
from datareader import *
import argparse
from scripts.yaw_canonicalization import YawCanonicalizer, canonicalize_yaw, mesh_center_from_vertices
from scripts.pose_end_disambiguation import disambiguate_zone_pose, mesh_endpoints_from_vertices
from scripts.bbox_depth_mask import bbox_depth_mask
try:
    from auto_register import AutoRegisterGate
except ImportError:
    from scripts.auto_register import AutoRegisterGate

# D435 摄像头内参
K_D435 = np.array([
    [606.1544849121768, 0, 320.71911360697],
    [0, 606.2304160969798, 239.7231533262084],
    [0, 0, 1]
], dtype=np.float64)


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _bbox_iou(a, b):
    inter_w = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    inter_h = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    inter = max(0.0, inter_w) * max(0.0, inter_h)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(bbox):
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def select_detection_for_request(detections, request_bbox=None, min_iou=0.2):
    """从 (x1, y1, x2, y2, conf, cls) 检测列表中选出一个框，无合适框时返回 None。

    request_bbox 为 None 时选面积最大的框。否则选 IoU 最大且 >= min_iou 的框；
    没有这样的框时，选中心距离最近且距离不超过 request_bbox 对角线的框。
    """
    if not detections:
        return None
    if request_bbox is None:
        best_det, best_area = None, 0.0
        for det in detections:
            area = _bbox_area(det)
            if area > best_area:
                best_det, best_area = det, area
        return best_det

    best_iou, best_det = max(((_bbox_iou(det, request_bbox), det) for det in detections),
                             key=lambda item: item[0])
    if best_iou >= min_iou:
        return best_det

    rx, ry = _bbox_center(request_bbox)
    diag = math.hypot(float(request_bbox[2]) - float(request_bbox[0]),
                      float(request_bbox[3]) - float(request_bbox[1]))
    nearest_det, nearest_dist = None, None
    for det in detections:
        cx, cy = _bbox_center(det)
        dist = math.hypot(cx - rx, cy - ry)
        if nearest_dist is None or dist < nearest_dist:
            nearest_det, nearest_dist = det, dist
    if nearest_dist is not None and nearest_dist <= diag:
        return nearest_det
    return None


class MaskGenerator:
    """YOLO 检测 + 框内深度窗口，生成目标物体遮罩"""

    def __init__(self, yolo_path, device='cuda', center_frac=0.35, half_window_m=0.11):
        from ultralytics import YOLO

        self.device = device
        self.yolo = YOLO(yolo_path)
        self.center_frac = center_frac
        self.half_window_m = half_window_m

    def detect_only(self, color_bgr, target_class=None, conf=0.3):
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        results = self.yolo.predict(color_rgb, conf=conf, verbose=False)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        dets = []
        for box in results[0].boxes:
            cls = int(box.cls[0]) if box.cls is not None else None
            if target_class is not None and cls != target_class:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0]) if box.conf is not None else 1.0
            dets.append((int(x1), int(y1), int(x2), int(y2), conf_val, cls))
        return dets

    def generate_from_detections(self, depth_m, detections, request_bbox=None):
        """从已有检测结果中选框，并在框内用深度窗口生成遮罩；无可用框时返回 (空遮罩, None)"""
        H, W = depth_m.shape[:2]
        det = select_detection_for_request(detections, request_bbox)
        if det is None:
            return np.zeros((H, W), dtype=bool), None

        bbox_int = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
        mask = bbox_depth_mask(depth_m, bbox_int, self.center_frac, self.half_window_m)
        return mask, bbox_int

    def generate(self, color_bgr, depth_m, target_class=None, conf=0.7,
                 request_bbox=None, detections=None):
        """对 BGR 图像运行 YOLO，选框后在框内用深度窗口生成遮罩

        detections 不为 None 时直接复用（按 target_class 和 conf 过滤），不再运行 YOLO。
        request_bbox 不为 None 时选与其匹配的框，否则选面积最大的框。
        """
        if detections is None:
            detections = self.detect_only(color_bgr, target_class=target_class, conf=conf)
        else:
            detections = [
                det for det in detections
                if (target_class is None or det[5] == target_class) and det[4] >= conf
            ]
        return self.generate_from_detections(depth_m, detections, request_bbox=request_bbox)


def read_frames_from_dir(data_dir):
    """从数据目录读取彩色和深度帧（rgb/ 和 depth/ 子目录）"""
    rgb_files = sorted(glob.glob(f'{data_dir}/rgb/*.png')) or \
                sorted(glob.glob(f'{data_dir}/color/*.png'))
    depth_files = sorted(glob.glob(f'{data_dir}/depth/*.png'))

    if len(rgb_files) == 0:
        raise RuntimeError(f'在 {data_dir}/rgb/ 或 {data_dir}/color/ 中未找到 RGB 帧')
    if len(depth_files) == 0:
        raise RuntimeError(f'在 {data_dir}/depth/ 中未找到深度帧')

    logging.info(f'找到 {len(rgb_files)} 个 RGB 帧, {len(depth_files)} 个深度帧')
    return rgb_files, depth_files


def read_frame_pair(rgb_file, depth_file):
    """读取一对 RGB 和深度图像"""
    color = imageio.imread(rgb_file)
    if color.shape[-1] == 4:
        color = color[..., :3]

    depth_raw = cv2.imread(depth_file, -1)
    if depth_raw.dtype == np.uint16:
        depth = depth_raw.astype(np.float32) / 1e3  # 毫米 -> 米
    else:
        depth = depth_raw.astype(np.float32)
        if depth.max() > 100:
            depth /= 1e3
    depth[(depth < 0.001) | (~np.isfinite(depth))] = 0
    return color, depth


def zone_disambiguate(est, pose, depth, K, endpoints, radius_m):
    """按端点区域点数消歧注册位姿；翻转时同步 est.pose_last，供后续 track_one 使用"""
    model_center, large_tip, small_tip = endpoints
    res = disambiguate_zone_pose(pose=pose, depth_m=depth, K=K, model_center=model_center,
                                 large_tip=large_tip, small_tip=small_tip, radius_m=radius_m)
    if res.flipped:
        logging.info(f'端点区域消歧: 位姿已翻转 (n_large={res.n_large}, n_small={res.n_small}, radius_m={radius_m})')
        last = getattr(est, 'pose_last', None)
        if last is not None:
            est.pose_last = (_torch.as_tensor(res.pose, dtype=last.dtype, device=last.device)
                             @ _torch.linalg.inv(est.get_tf_to_centered_mesh()))
    return res.pose


def run_offline(args, est, mesh, to_origin, bbox_vis, K):
    """离线模式：处理磁盘上保存的帧"""
    mask_gen = MaskGenerator(
        yolo_path=args.yolo_model,
        device=args.device,
        center_frac=args.mask_center_frac,
        half_window_m=args.depth_window_half_m,
    )
    rgb_files, depth_files = read_frames_from_dir(args.data_dir)
    yaw = None if args.no_canonicalize_yaw else YawCanonicalizer(mesh_center_from_vertices(mesh.vertices))
    endpoints = mesh_endpoints_from_vertices(mesh.vertices) if args.zone_disambiguation_enabled else None

    for i in range(len(rgb_files)):
        logging.info(f'帧 {i}/{len(rgb_files)}: {rgb_files[i]}')
        color, depth = read_frame_pair(rgb_files[i], depth_files[i])

        if i == 0:
            mask, bbox = mask_gen.generate(color, depth, target_class=args.target_class)
            if mask.sum() == 0:
                logging.warning('YOLO 在第一帧未检测到有效深度的目标！使用中心裁剪。')
                H, W = color.shape[:2]
                mask[H//4:3*H//4, W//4:3*W//4] = 1
            else:
                logging.info(f'检测框: {bbox}, 遮罩面积: {mask.sum()}')
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                              iteration=args.est_refine_iter, bbox_xyxy=bbox)
            if endpoints is not None:
                pose = zone_disambiguate(est, pose, depth, K, endpoints, args.zone_disambiguation_radius_m)
            if yaw is not None:
                pose = canonicalize_yaw(pose, yaw.model_center, yaw.fallback_deg)
            pose_output = Path(args.pose_output)
            pose_output.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(pose_output, pose.reshape(4, 4))
            logging.info(f'初始位姿已保存到 {pose_output}')
        else:
            pose = est.track_one(rgb=color, depth=depth, K=K,
                               iteration=args.track_refine_iter)
            if yaw is not None:
                pose = yaw.apply(pose)

        # 保存位姿
        os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{args.debug_dir}/ob_in_cam/{i:06d}.txt', pose.reshape(4, 4))
        logging.info(f'位姿[{i}]:\n{pose}')

        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox_vis)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K,
                              thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('Pose', vis[..., ::-1])
            if args.save_vis:
                os.makedirs(f'{args.debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{args.debug_dir}/track_vis/{i:06d}.png', vis)
            cv2.waitKey(1)


def run_live(args, est, mesh, to_origin, bbox_vis, K):
    """实时模式：先检测目标 -> 确认后开始位姿跟踪"""
    import pyrealsense2 as rs

    yaw = None if args.no_canonicalize_yaw else YawCanonicalizer(mesh_center_from_vertices(mesh.vertices))
    endpoints = mesh_endpoints_from_vertices(mesh.vertices) if args.zone_disambiguation_enabled else None

    mask_gen = MaskGenerator(
        yolo_path=args.yolo_model,
        device=args.device,
        center_frac=args.mask_center_frac,
        half_window_m=args.depth_window_half_m,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # 获取深度比例
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)
    logging.info(f'D435 已启动, 深度比例: {depth_scale}')

    # 创建可缩放窗口
    cv2.namedWindow('FoundationPose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FoundationPose', 960, 720)

    pose = None
    phase = 'detect'  # detect -> register -> track
    mask = None
    bbox = None
    frame_idx = 0
    auto_gate = AutoRegisterGate(
        min_conf=args.detect_conf,
        stable_frames=args.auto_register_stable_frames
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            depth[(depth < 0.001) | (~np.isfinite(depth))] = 0

            # ---- 阶段1：YOLO 检测，显示画面和检测框 ----
            if phase == 'detect':
                dets = mask_gen.detect_only(color, target_class=args.target_class, conf=args.detect_conf)
                vis = color.copy()

                if dets:
                    for (x1, y1, x2, y2, conf_val, cls_id) in dets:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'cls={cls_id} {conf_val:.2f}'
                        cv2.putText(vis, label, (x1, max(y1 - 8, 0)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                prompt = 'AUTO DETECT: registering after stable high-conf target' \
                    if args.auto_register else 'DETECT: Press SPACE to register, ESC to quit'
                cv2.putText(vis, prompt,
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if dets:
                    cv2.putText(vis, f'Found {len(dets)} target(s)',
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, 'Searching...', (10, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if args.auto_register:
                    selected = auto_gate.update(dets)
                    if selected is not None:
                        phase = 'register'
                        logging.info(
                            '自动确认检测，开始注册位姿: '
                            f'bbox={selected[:4]}, conf={selected[4]:.3f}, cls={selected[5]}'
                        )
                    else:
                        cv2.imshow('FoundationPose', vis[..., ::-1])
                        if (cv2.waitKey(1) & 0xFF) == 27:
                            break
                        continue
                else:
                    cv2.imshow('FoundationPose', vis[..., ::-1])
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == 32 and dets:  # SPACE 确认检测
                        phase = 'register'
                        logging.info(f'用户确认检测，开始注册位姿...')
                    continue

            # ---- 阶段2：注册初始位姿（遮罩由 YOLO 框内深度窗口生成） ----
            if phase == 'register':
                mask, bbox = mask_gen.generate(
                    color,
                    depth,
                    target_class=args.target_class,
                    conf=args.register_conf
                )
                if mask.sum() == 0:
                    logging.warning('YOLO 未检测到有效深度的目标！使用中心裁剪。')
                    H, W = color.shape[:2]
                    mask[H//4:3*H//4, W//4:3*W//4] = 1
                else:
                    logging.info(f'检测框: {bbox}, 遮罩面积: {mask.sum()}')
                pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                                  iteration=args.est_refine_iter, bbox_xyxy=bbox)
                if endpoints is not None:
                    pose = zone_disambiguate(est, pose, depth, K, endpoints, args.zone_disambiguation_radius_m)
                if yaw is not None:
                    yaw.reset()
                    pose = yaw.apply(pose)
                logging.info(f'初始位姿:\n{pose}')
                # 保存位姿供 M3T 跟踪器读取
                pose_output = Path(args.pose_output)
                pose_output.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(pose_output, pose.reshape(4, 4))
                logging.info(f'初始位姿已保存到 {pose_output}')

                # 可视化初始化结果（只显示这一帧）
                if args.debug >= 1:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox_vis)
                    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K,
                                      thickness=3, transparency=0, is_input_rgb=True)
                    # 绘制遮罩轮廓
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    vis_bgr = vis[..., ::-1].copy()
                    cv2.drawContours(vis_bgr, contours, -1, (255, 255, 0), 2)
                    cv2.putText(vis_bgr, 'INITIALIZATION COMPLETE | Press any key to continue',
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('FoundationPose', vis_bgr)
                    logging.info('初始化完成，按任意键继续...')
                    cv2.waitKey(0)  # 等待用户确认

                if args.init_only:
                    logging.info('--init_only 模式，退出')
                    break
                phase = 'track'
                continue

            # ---- 阶段3：实时跟踪 ----
            if phase == 'track':
                pose = est.track_one(rgb=color, depth=depth, K=K,
                                   iteration=args.track_refine_iter)
                if yaw is not None:
                    pose = yaw.apply(pose)
                print(f'帧 {frame_idx:04d} 位姿:\n{pose}\n')
                frame_idx += 1

            # 可视化：位姿叠加
            if pose is not None and args.debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox_vis)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K,
                                  thickness=3, transparency=0, is_input_rgb=True)
                cv2.putText(vis, 'TRACKING | ESC to quit',
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('FoundationPose', vis[..., ::-1])
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str,
                        default=str(PROJECT_ROOT / 'models/object/Str3D.obj'))
    parser.add_argument('--yolo_model', type=str,
                        default=str(PROJECT_ROOT / 'models/yolo/1117_openvino_model'))
    parser.add_argument('--mask_center_frac', type=float, default=0.35,
                        help='框中心区域比例，用于取参考深度')
    parser.add_argument('--depth_window_half_m', type=float, default=0.08,
                        help='遮罩深度窗口半宽（米）')
    parser.add_argument('--target_class', type=int, default=None,
                        help='要跟踪的 YOLO 类别 ID（None=选择最大检测框）')
    parser.add_argument('--detect_conf', type=float, default=0.7,
                        help='检测阶段 YOLO 最低置信度')
    parser.add_argument('--register_conf', type=float, default=None,
                        help='送入 FoundationPose 注册前的 YOLO 最低置信度（默认同 detect_conf）')
    parser.add_argument('--auto_register', action='store_true',
                        help='检测置信度稳定达标后自动注册，不等待 SPACE')
    parser.add_argument('--auto_register_stable_frames', type=int, default=3,
                        help='自动注册前需要连续检测达标的帧数')

    # 输入模式
    parser.add_argument('--live', action='store_true', default=True,
                        help='通过 D435 摄像头实时采集（默认：开启）')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='包含 rgb/ 和 depth/ 子目录的数据路径（离线模式）')

    # 优化迭代次数
    parser.add_argument('--est_refine_iter', type=int, default=2)
    parser.add_argument('--symmetry_mode', type=str, default='z12', choices=['none', 'z12'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--render_backend', type=str, default='moderngl',
                        choices=['nvdiffrast', 'moderngl'])
    parser.add_argument('--net_backend', type=str, default='openvino', choices=['torch', 'openvino'])
    parser.add_argument('--ov_device', type=str, default='CPU', choices=['CPU', 'GPU'])
    parser.add_argument('--ov_precision', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--ov_num_threads', type=int, default=0,
                        help='OpenVINO 推理线程数，0 表示由 OpenVINO 自动决定（仅 ov_device=CPU）')
    parser.add_argument('--ov_cache_dir', type=str, default='',
                        help='OpenVINO 编译缓存目录，空字符串表示不启用')
    parser.add_argument('--no_canonicalize_yaw', action='store_true')
    parser.add_argument('--zone_disambiguation_enabled', action='store_true',
                        help='注册后用端点区域点数消歧，修正位姿翻转')
    parser.add_argument('--zone_disambiguation_radius_m', type=float, default=0.14,
                        help='端点区域半径（米）')
    parser.add_argument('--track_refine_iter', type=int, default=2)

    # 调试
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str,
                        default=str(PROJECT_ROOT / 'runtime/foundationpose_debug'))
    parser.add_argument('--pose_output', type=str,
                        default=str(PROJECT_ROOT / 'runtime/foundation_pose.txt'),
                        help='FoundationPose 初始位姿输出文件，供 M3T 读取')
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--init_only', action='store_true',
                        help='仅获取初始位姿后退出，供 M3T 启动脚本使用')
    args = parser.parse_args()
    if args.register_conf is None:
        args.register_conf = args.detect_conf

    # --- 环境检查 ---
    import torch
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用。--device cuda 需要 GPU。')
        # 清理GPU缓存
        torch.cuda.empty_cache()

    # 设置内存分配策略
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # 检查 mycpp 是否已编译
    try:
        import mycpp.build.mycpp as mycpp
    except ImportError:
        raise RuntimeError(
            'mycpp 未编译。请运行: bash build_all_conda.sh'
        )

    # 检查权重文件
    weight_dir = str(FOUNDATIONPOSE_DIR / 'weights')
    if not os.path.exists(f'{weight_dir}/2024-01-11-20-02-45/model_best.pth') or \
       not os.path.exists(f'{weight_dir}/2023-10-28-18-33-37/model_best.pth'):
        logging.warning('未找到预训练权重！请从 FoundationPose 仓库下载。')
        logging.warning('期望路径:')
        logging.warning(f'  {weight_dir}/2024-01-11-20-02-45/model_best.pth  (评分器)')
        logging.warning(f'  {weight_dir}/2023-10-28-18-33-37/model_best.pth  (优化器)')

    # --- 初始化 ---
    set_logging_format()
    set_seed(0)
    K = K_D435.copy()

    debug_dir = args.debug_dir
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(debug_path / 'track_vis', ignore_errors=True)
    shutil.rmtree(debug_path / 'ob_in_cam', ignore_errors=True)
    (debug_path / 'track_vis').mkdir(parents=True, exist_ok=True)
    (debug_path / 'ob_in_cam').mkdir(parents=True, exist_ok=True)

    # 加载 3D 模型并初始化 FoundationPose
    from scripts.foundationpose_factory import EstimatorConfig, build_estimator
    bundle = build_estimator(EstimatorConfig(mesh_file=args.mesh_file, debug=args.debug, debug_dir=debug_dir,
                                             symmetry_mode=args.symmetry_mode, device=args.device,
                                             render_backend=args.render_backend,
                                             net_backend=args.net_backend, ov_device=args.ov_device,
                                             ov_precision=args.ov_precision,
                                             ov_num_threads=args.ov_num_threads if args.ov_num_threads > 0 else None,
                                             ov_cache_dir=(str(PROJECT_ROOT / args.ov_cache_dir)
                                                           if args.ov_cache_dir else None)))
    est, mesh, to_origin, bbox_vis = bundle.estimator, bundle.mesh, bundle.to_origin, bundle.bbox_vis
    logging.info(f'Mesh: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面')
    logging.info('FoundationPose 初始化完成')

    # --- 运行 ---
    if args.data_dir:
        run_offline(args, est, mesh, to_origin, bbox_vis, K)
    else:
        run_live(args, est, mesh, to_origin, bbox_vis, K)
