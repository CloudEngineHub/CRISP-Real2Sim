#!/usr/bin/env python3
import argparse
import time
import os
from pathlib import Path
import numpy as np
import yaml
import cv2
import imageio.v3 as iio
from contextlib import suppress

import h5py
from utils.viser_visualizer_z import ViserHelper
from utils.robot_viser_z import RobotMjcfViser


class ExternalCameraHandler:
    """Handles external camera data from NPZ files."""
    
    def __init__(self, npz_data: dict, npz_cam_data: dict, 
                     T_align: np.array, 
                 conf_threshold: float = 1.0, 
                 foreground_conf_threshold: float = 0.1, 
                 no_mask: bool = False, 
                 xyzw=True, init_conf=False, extra_obj=False):
        
        # Camera intrinsics
        self.K = np.expand_dims(npz_cam_data['intrinsic'], 0)
        ax, ay = 1, 1  # Aspect ratio adjustments if needed
        
        # Adjust intrinsics
        self.K[0][0][0] = ax * self.K[0][0][0]
        self.K[0][0][-1] = ax * self.K[0][0][-1]
        self.K[0][1][1] = ay * self.K[0][1][1]
        self.K[0][1][-1] = ay * self.K[0][1][-1]
        self.K[0][1][1] = self.K[0][0][0]  # Make fy = fx
        
        # Scale factor
        self.S = npz_cam_data.get('scale', 1.0)
        
        # Repeat K for all frames
        num_frames = npz_data['images'].shape[0]
        self.K = np.repeat(self.K, num_frames, axis=0)  # (1,3,3) -> (N,3,3)
        
        # Camera poses (c2w)
        T_world_cameras = npz_cam_data['cam_c2w'].copy()
        T_world_cameras[..., :3, 3] *= self.S  # Scale translation

        if T_align is not None:
          self.T_world_cameras = np.matmul(T_align, T_world_cameras)


      
        
        # Frame data
        self.images = npz_data['images']  # (N,H,W,3)
        try:
            self.depths = npz_data['depth'] * self.S
        except:
            self.depths = npz_data.get('depths', np.zeros_like(self.images[..., 0])) * self.S
        
        self.fps = npz_cam_data.get('fps', 30)
        
        # Optional data
        self.masks = npz_data.get('enlarged_dynamic_mask', np.zeros_like(self.images[..., 0]))
        self.confidences = np.array(npz_data.get('uncertainty', []))
        
        # Resize masks if needed
        if self.masks.shape != self.images.shape[:3]:
            import skimage.transform
            self.masks = skimage.transform.resize(
                self.masks, self.images.shape[:3], order=0
            ).astype(np.uint8)
        
        # Set confidence threshold
        if len(self.confidences):
            self.conf_threshold = np.quantile(self.confidences, 0.0)
        else:
            self.conf_threshold = conf_threshold
    
    def get_camera_at_frame(self, frame_idx: int):
        """Get camera parameters for a specific frame."""
        if frame_idx >= len(self.T_world_cameras):
            frame_idx = len(self.T_world_cameras) - 1
        
        return {
            'K': self.K[frame_idx],
            'T_c2w': self.T_world_cameras[frame_idx],
            'image': self.images[frame_idx],
            'depth': self.depths[frame_idx] if frame_idx < len(self.depths) else None,
            'H': self.images[frame_idx].shape[0],
            'W': self.images[frame_idx].shape[1]
        }


def _k_to_fov_y(K: np.ndarray, H: int) -> float:
    """Convert intrinsics to vertical FOV in radians."""
    fy = float(K[1, 1])
    return float(2.0 * np.arctan2(H, 2.0 * fy))


def _twc_to_cam_pose(T_wc: np.ndarray):
    """Convert cam->world (4x4) to (position, quaternion_wxyz) for viser."""
    import viser.transforms as vtf
    R_wc = np.asarray(T_wc[:3, :3], dtype=float)
    t_wc = np.asarray(T_wc[:3, 3], dtype=float)
    q_wxyz = vtf.SO3.from_matrix(R_wc).wxyz
    return t_wc.astype(float), np.asarray(q_wxyz, dtype=float).reshape(4,)


def _get_client_or_launch(server, timeout_sec: float = 20.0):
    """Get or launch a client for the Viser server."""
    host, port = server.get_host(), server.get_port()
    url = f"http://{host}:{port}"
    
    # Check for existing clients
    clients = server.get_clients()
    if clients:
        return list(clients.values())[0], None
    
    # Try to launch headless browser
    launched = None
    with suppress(Exception):
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
        page = browser.new_page(
            viewport={"width": 1280, "height": 800, "deviceScaleFactor": 1.0}
        )
        page.goto(url, wait_until="load")
        launched = (pw, browser, page)
    
    # Wait for client connection
    t0 = time.time()
    while True:
        clients = server.get_clients()
        if clients:
            return list(clients.values())[0], launched
        if time.time() - t0 > timeout_sec:
            if launched is not None:
                pw, browser, page = launched
                with suppress(Exception):
                    page.close()
                    browser.close()
                    pw.stop()
            raise TimeoutError(f"No client connected. Open {url} in browser or install playwright.")
        time.sleep(0.05)


def _close_launched(launched):
    """Close launched browser if any."""
    if launched is None:
        return
    pw, browser, page = launched
    with suppress(Exception):
        page.close()
        browser.close()
        pw.stop()


def render_robot_with_external_camera(
    viser: ViserHelper,
    robot: RobotMjcfViser,
    camera_handler: ExternalCameraHandler,
    robot_pos: np.ndarray,  # (T, B, 3)
    robot_rot: np.ndarray,  # (T, B, 4) xyzw
    output_dir: str,
    render_frames: list = None,
    save_video: bool = True,
    save_images: bool = True,
    dt: float = 1.0/30.0,  # 添加 dt 参数
):
    """Render robot using external camera data and save outputs."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get viser client
    client, launched = _get_client_or_launch(viser.server)
    
    T = len(robot_pos)
    if render_frames is None:
        render_frames = list(range(T))
    
    rendered_images = []
    
    print(f"[Renderer] Starting to render {len(render_frames)} frames...")
    print(f"[Render] Total robot frames: {len(robot_pos)}")
    print(f"[Render] Total camera frames: {len(camera_handler.T_world_cameras)}")
    print(f"[Render] Frames to render: {len(render_frames)}")
    print(f"[Renderer] Playback FPS: {1.0/dt:.1f} Hz (dt={dt:.4f}s)")

    for idx, t in enumerate(render_frames):
        # Update robot pose
        robot.update(robot_pos[t], robot_rot[t])
        
        # Get camera parameters for this frame
        cam_data = camera_handler.get_camera_at_frame(t)
        
        # Set camera pose and FOV
        pos, quat = _twc_to_cam_pose(cam_data['T_c2w'])
        fov = _k_to_fov_y(cam_data['K'], cam_data['H'])
        # print(cam_data)
        
        # Update camera in atomic operation
        with client.atomic():
            client.camera.position = pos
            client.camera.wxyz = quat
            client.camera.fov = float(fov)
        time.sleep(dt)  # 按照指定的 dt 等待
        rgb = client.get_render(height=int(cam_data['H']), width=int(cam_data['W']))
        time.sleep(0.2)
        
        # Save individual image if requested
        if save_images:
            img_path = os.path.join(output_dir, f"frame_{t:05d}.png")
            cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if idx % 10 == 0:
                print(f"  Saved frame {t} -> {img_path}")
        
        rendered_images.append(rgb)
    

    # ---------- 写视频（OpenCV） ----------
    if save_video and rendered_images:
        video_path = os.path.join(output_dir, "robot_render.mp4")
        H, W = rendered_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # 可换 'avc1' 看环境
        writer = cv2.VideoWriter(video_path, fourcc, float(camera_handler.fps), (W, H))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter. Try fourcc 'avc1' or install codecs.")

        for rgb in rendered_images:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if bgr.shape[0] != H or bgr.shape[1] != W:
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
            writer.write(bgr)
        writer.release()
        print(f"[Renderer] Saved video: {video_path}")

    _close_launched(launched)
    print(f"[Renderer] Rendering complete. Output saved to: {output_dir}")
    return rendered_images
    
    # Close launched browser if any
    _close_launched(launched)
    
    print(f"[Renderer] Rendering complete. Output saved to: {output_dir}")
    
    return rendered_images
def _load_npz_to_dict(path: Path, *, allow_pickle: bool = False) -> dict:
    """Utility: load .npz into a writable dict."""
    with np.load(path, allow_pickle=allow_pickle) as f:
        return {k: f[k] for k in f}
def load_scene_mesh(scene_path):
    """
    Load scene mesh, handling both single file and multi-part formats.
    
    Args:
        scene_path: Path to scene file or directory containing parts
    
    Returns:
        tuple: (vertices, faces) as numpy arrays, or (None, None) if failed
    """
    scene_path = Path(scene_path)
    
    # Check if it's a directory with parts
    if scene_path.is_dir():
        parts_dir = scene_path / "parts" if (scene_path / "parts").exists() else scene_path
        part_files = sorted(parts_dir.glob("part_*.obj"))
        
        if not part_files:
            # Try other patterns
            part_files = sorted(parts_dir.glob("part_*.obj"))
            if not part_files:
                part_files = sorted(parts_dir.glob("*.obj"))
        
        if part_files:
            print(f"[Scene] Found {len(part_files)} parts to concatenate")
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            
            for part_file in part_files:
                with open(part_file, 'r') as f:
                    vertices = []
                    for line in f:
                        if line.startswith('v '):
                            parts = line.strip().split()
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif line.startswith('f '):
                            face_parts = line.strip().split()[1:]
                            face_indices = []
                            for p in face_parts:
                                vidx = int(p.split('/')[0]) - 1 + vertex_offset
                                face_indices.append(vidx)
                            # Triangulate if needed
                            for i in range(1, len(face_indices) - 1):
                                all_faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
                    
                    all_vertices.extend(vertices)
                    vertex_offset += len(vertices)
            
            return np.array(all_vertices, dtype=np.float32), np.array(all_faces, dtype=np.int32)
    
    # Single file format
    elif scene_path.exists() and scene_path.suffix == '.obj':
        print(f"[Scene] Loading single OBJ file: {scene_path}")
        vertices = []
        faces = []
        with open(scene_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    face_parts = line.strip().split()[1:]
                    face_indices = []
                    for p in face_parts:
                        vidx = int(p.split('/')[0]) - 1
                        face_indices.append(vidx)
                    # Triangulate
                    for i in range(1, len(face_indices) - 1):
                        faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)
    
    print(f"[Scene] Could not load scene from: {scene_path}")
    return None, None

    
def main():
    ap = argparse.ArgumentParser(description="Visualize robot with external camera data.")
    ap.add_argument(
        "record_dir",
        type=str,
        nargs="?",
        help="Directory containing robot recordings",
    )
    ap.add_argument(
        "--data",
        type=str,
        help="Path to NPZ file containing camera data (intrinsic, cam_c2w, scale)",
    )
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--env_idx", type=int, default=0, help="Environment index")
    ap.add_argument("--port", type=int, default=8080, help="Viser server port")
    ap.add_argument("--output_dir", type=str, default="./rendered_output", help="Output directory")
    ap.add_argument("--save_images", action="store_true", help="Save individual frame images")
    ap.add_argument("--save_video", action="store_true", default=True, help="Save video")
    ap.add_argument("--max_frames", type=int, default=None, help="Maximum frames to render")
    ap.add_argument("--frame_stride", type=int, default=1, help="Frame stride for rendering")
    ap.add_argument("--scene_obj",type=str, default="/data3/zihanwa3/_Robotics/_vision/mega-sam/post_results/pkr_c/gv/scene_mesh_sqs/scene_mesh_sqs.obj", help="Optional path to a scene OBJ file to visualize")
    ap.add_argument('--scene', required=True, help='Sequence name (e.g., 000)')
    ap.add_argument('--types', required=True, help='Sequence name (e.g., 000)')
    ap.add_argument('--parent_name', default='emdb_new', help='Parent folder name')
    ap.add_argument('--method', default='emdb_new', help='Parent folder name')
    ap.add_argument('--data_path', default='/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/motion_data/', help='Root path to motion data')

    args = ap.parse_args()

    args.output_dir = os.path.join(args.output_dir, f'{args.parent_name}_{args.scene}_{args.method}')
    os.makedirs(args.output_dir, exist_ok=True)

    types = args.types
    args.data = f'/data3/zihanwa3/_Robotics/_vision/mega-sam/postprocess/{args.scene}_gv_sgd_cvd_hr.npz'
    args.record_dir = os.path.join(f'/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/post_asset_{args.method}_{args.parent_name}', args.scene)
    parent_name = args.parent_name
    data_path = args.data_path
    seq_name = args.scene
    data = args.data

    # /data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/motion_data
    args.scene_obj = f'/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/parkour_anim/data/assets/urdf/{args.parent_name}/{seq_name}/{args.method}'


    if args.method == 'videomimic':
      calibrated_path = f"/data3/zihanwa3/_Robotics/_baselines/VideoMimic/real2sim/_videomimic_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{args.scene}_cam01_frame_0_-1_subsample_1/gravity_calibrated_keypoints.h5" 
      with h5py.File(calibrated_path, 'r') as f:
          world_rotation = f['world_rotation'][:]  # (3, 3)

      world_rot_tensor = (world_rotation)
      T_align = np.eye(4, dtype=np.float32)
      T_align[:3, :3] = world_rotation
      T_align[2, 3] = 1.7017016  # Add the z-translation from your data

    

    else:
      raw_data_path = f"{data_path}/{parent_name}/{seq_name}_{types}.npz"
      raw_data = np.load(raw_data_path)
      T_align = raw_data['T_align'].astype(np.float32)

    
    moge_base_path = '/data3/zihanwa3/_Robotics/_vision/TAPIP3D/_raw_mega_priors'
    tgt_name = str(data).split('_sgd')[0].split('/')[-1]   
    tgt_name = "_".join(tgt_name.split("_")[:-1]) 
    moge_data = os.path.join(moge_base_path, f'{tgt_name}.npz')
    tgt_folder = os.path.join('/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/post_results', tgt_name)

    # npz_cam_data = np.load(moge_data)
    # npz_data = np.load(moge_data)
    data = _load_npz_to_dict(data)
    moge_data = np.load(moge_data)
    data["depths"] = moge_data["depths"]
    data["images"] = moge_data["images"]
    data['cam_c2w'] = moge_data['cam_c2w']
    data['intrinsic'] = moge_data['intrinsic']
    npz_cam_data = data
    npz_data = data


    
    # Initialize camera handler
    camera_handler = ExternalCameraHandler(
        npz_data=dict(npz_data),
        npz_cam_data=dict(npz_cam_data), 
        T_align=T_align
    )
    
    # Load robot data
    rec = Path(args.record_dir)
    if not rec.exists():
        raise FileNotFoundError(f"Record dir not found: {rec}")
    
    # Load robot info
    info_path = rec / "robot_vis_info.yaml"
    if info_path.exists():
        info = yaml.safe_load(info_path.read_text())
        body_names = info.get("body_names", [])
        asset_xml_rel = info.get("asset_xml", None)
    else:
        print("[Main] robot_vis_info.yaml not found; using defaults.")
        body_names = []
        asset_xml_rel = None

    # Find MJCF path
    if asset_xml_rel and (rec / asset_xml_rel).exists():
        mjcf_path = str(rec / asset_xml_rel)
    else:
        cand = list(rec.glob("*.xml"))
        mjcf_path = str(cand[0]) if cand else None
        if mjcf_path is None:
            raise FileNotFoundError("No MJCF xml found in record dir.")
    
    # Load robot trajectory
    npz_path = rec / f"rigid_bodies_{args.env_idx}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Rigid body file not found: {npz_path}")
    
    robot_data = np.load(npz_path)
    robot_pos = robot_data["pos"]  # (T, B, 3)
    robot_rot = robot_data["rot"]  # (T, B, 4) xyzw
    
    # Initialize Viser

    import random

    def generate_four_digit():
        return random.randint(1000, 9999)
    port=generate_four_digit()
    viser = ViserHelper(port=port)
    if not viser.ok():
        print("[Main] Viser not available; exiting.")
        return
    
    # Initialize robot
    robot = RobotMjcfViser(viser, mjcf_path, body_names if body_names else None)
    
    # Load optional scene
    if args.scene_obj:
      verts, faces = load_scene_mesh(args.scene_obj)
      if verts is not None and faces is not None:
          viser.add_mesh_simple("/scene", verts, faces, color=(0.6, 0.7, 0.9))
          print(f"[Main] Loaded scene with {len(verts)} vertices, {len(faces)} faces")
    
    # Determine frames to render
    T = min(len(robot_pos), len(camera_handler.images))
    if args.max_frames:
        T = min(T, args.max_frames)
    
    render_frames = list(range(0, T, args.frame_stride))
    
    print(f"[Main] Ready to render {len(render_frames)} frames")
    print(f"[Main] Output will be saved to: {args.output_dir}")
    
    # Render and save
    rendered = render_robot_with_external_camera(
        viser=viser,
        robot=robot,
        camera_handler=camera_handler,
        robot_pos=robot_pos,
        robot_rot=robot_rot,
        output_dir=args.output_dir,
        render_frames=render_frames,
        save_video=args.save_video,
        save_images=True
    )
    
    print(f"[Main] Complete! Rendered {len(rendered)} frames.")


if __name__ == "__main__":
    main()