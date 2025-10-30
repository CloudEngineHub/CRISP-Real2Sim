import numpy as np
import torch
import trimesh
import cv2
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, TexturesVertex, PointLights
)

def combine_all_geometry(
    hmr_results: Dict,
    frame_data: Dict,
    sqs_mesh: trimesh.Trimesh,
    frame_idx: int,
    downsample_factor: int = 1,
    bg_downsample_factor: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine all geometry (HMR meshes, point clouds, scene mesh) into single arrays.
    
    Returns:
        combined_points: (N, 3) array of all 3D points
        combined_colors: (N, 3) array of RGB colors (0-255)
    """
    all_points = []
    all_colors = []
    
    # Add HMR mesh vertices
    for hmr_type, hmr_data in hmr_results.items():
        if frame_idx < len(hmr_data['pred_vert']):
            verts = hmr_data['pred_vert'][frame_idx]
            color = np.array(hmr_data['color']).reshape(1, 3)
            colors = np.repeat(color, len(verts), axis=0)
            
            all_points.append(verts)
            all_colors.append(colors)
    
    # Add scene mesh vertices if available
    if sqs_mesh is not None:
        scene_verts = np.asarray(sqs_mesh.vertices)
        scene_colors = np.full((len(scene_verts), 3), [180, 180, 180])  # Gray
        all_points.append(scene_verts)
        all_colors.append(scene_colors)
    
    # Add point cloud from frame
    if frame_data is not None:
        # Extract point cloud from frame
        output_pts, _, _ = frame_data.get_point_cloud(downsample_factor, bg_downsample_factor)
        position, color, bg_position, bg_color, _, _, _, _ = output_pts
        
        # Foreground points
        if position is not None and len(position) > 0:
            all_points.append(position)
            all_colors.append(color * 255)  # Convert to 0-255 range
        
        # Background points
        if bg_position is not None and len(bg_position) > 0:
            all_points.append(bg_position)
            all_colors.append(bg_color * 255)
    
    # Concatenate all
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors).astype(np.uint8)
    else:
        combined_points = np.empty((0, 3))
        combined_colors = np.empty((0, 3), dtype=np.uint8)
    
    return combined_points, combined_colors


def project_points_to_image(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    img_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image plane.
    
    Args:
        points_3d: (N, 3) 3D points in world coordinates
        R: (3, 3) rotation matrix (world to camera)
        t: (3,) translation vector (world to camera)
        K: (3, 3) intrinsic matrix
        img_shape: (H, W) image dimensions
        
    Returns:
        pixels: (M, 2) valid pixel coordinates
        valid_mask: (N,) boolean mask indicating which points are visible
    """
    # Transform to camera coordinates
    points_cam = (R @ points_3d.T).T + t
    
    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0.1  # Small threshold to avoid numerical issues
    points_cam = points_cam[valid_mask]
    
    if len(points_cam) == 0:
        return np.empty((0, 2)), np.zeros(len(points_3d), dtype=bool)
    
    # Project to image plane
    points_2d_homo = (K @ points_cam.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    # Filter points outside image bounds
    H, W = img_shape
    in_bounds = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
    )
    
    # Update valid mask
    full_valid = np.zeros(len(points_3d), dtype=bool)
    valid_indices = np.where(valid_mask)[0]
    full_valid[valid_indices[in_bounds]] = True
    
    return points_2d[in_bounds], full_valid


def render_to_camera_view(
    points_3d: np.ndarray,
    colors: np.ndarray,
    camera_params: Dict,
    img_shape: Tuple[int, int] = (1080, 1920),
    point_size: int = 2
) -> np.ndarray:
    """
    Render 3D points to a camera view.
    
    Args:
        points_3d: (N, 3) 3D points in world coordinates
        colors: (N, 3) RGB colors (0-255)
        camera_params: Dict containing R, T, K matrices
        img_shape: Output image shape (H, W)
        point_size: Size of rendered points
        
    Returns:
        rendered_img: (H, W, 3) rendered image
    """
    # Initialize blank image
    img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255  # White background
    
    # Project points
    pixels, valid_mask = project_points_to_image(
        points_3d,
        camera_params['R'],
        camera_params['T'],
        camera_params['K'],
        img_shape
    )
    
    if len(pixels) == 0:
        return img
    
    # Get valid colors
    valid_colors = colors[valid_mask]
    
    # Sort by depth for proper occlusion
    cam_points = (camera_params['R'] @ points_3d[valid_mask].T).T + camera_params['T']
    depth_order = np.argsort(-cam_points[:, 2])  # Sort back to front
    
    # Draw points
    for idx in depth_order:
        px, py = pixels[idx].astype(int)
        color = valid_colors[idx].tolist()
        
        # Draw as a filled circle
        cv2.circle(img, (px, py), point_size, color, -1)
    
    return img


def render_all_training_views(
    hmr_results: Dict,
    loader,  # Your data loader
    sqs_mesh: trimesh.Trimesh,
    world_cam_R: torch.Tensor,
    world_cam_T: torch.Tensor,
    intrinsics: np.ndarray,
    num_frames: int,
    output_dir: str,
    interval: int = 10
):
    """
    Render combined geometry to all training view cameras and save images.
    
    Args:
        hmr_results: Dictionary of HMR results
        loader: Data loader with frame information
        sqs_mesh: Scene mesh (SQS or NKSR)
        world_cam_R: Camera rotations (N, 3, 3)
        world_cam_T: Camera translations (N, 3)
        intrinsics: Camera intrinsic matrix (3, 3)
        num_frames: Number of frames to render
        output_dir: Directory to save rendered images
        interval: Frame interval for rendering
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert torch tensors to numpy if needed
    if isinstance(world_cam_R, torch.Tensor):
        world_cam_R = world_cam_R.cpu().numpy()
    if isinstance(world_cam_T, torch.Tensor):
        world_cam_T = world_cam_T.cpu().numpy()
    
    rendered_images = []
    
    for frame_idx in range(0, num_frames, interval):
        print(f"Rendering frame {frame_idx}/{num_frames}")
        
        # Get frame data
        frame = loader.get_frame(frame_idx) if loader else None
        
        # Combine all geometry for this frame
        combined_points, combined_colors = combine_all_geometry(
            hmr_results=hmr_results,
            frame_data=frame,
            sqs_mesh=sqs_mesh,
            frame_idx=frame_idx,
            downsample_factor=1,
            bg_downsample_factor=1
        )
        
        # Camera parameters for this frame
        camera_params = {
            'R': world_cam_R[frame_idx] if frame_idx < len(world_cam_R) else world_cam_R[-1],
            'T': world_cam_T[frame_idx] if frame_idx < len(world_cam_T) else world_cam_T[-1],
            'K': intrinsics
        }
        
        # Render to camera view
        rendered_img = render_to_camera_view(
            points_3d=combined_points,
            colors=combined_colors,
            camera_params=camera_params,
            img_shape=(1080, 1920),  # Adjust as needed
            point_size=2
        )
        
        # Save rendered image
        output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
        rendered_images.append(rendered_img)
        
        # Optional: Also render with meshes using trimesh
        if frame_idx % (interval * 5) == 0:  # Every 5th interval, create mesh rendering
            scene = trimesh.Scene()
            
            # Add HMR meshes
            for hmr_type, hmr_data in hmr_results.items():
                if frame_idx < len(hmr_data['pred_vert']):
                    mesh = trimesh.Trimesh(
                        vertices=hmr_data['pred_vert'][frame_idx],
                        faces=faces,  # You need to pass faces from your main code
                        vertex_colors=hmr_data['color']
                    )
                    scene.add_geometry(mesh)
            
            # Add scene mesh
            if sqs_mesh is not None:
                scene.add_geometry(sqs_mesh)
            
            # Set camera
            scene.camera.K = intrinsics
            scene.camera.transform = np.linalg.inv(
                np.vstack([
                    np.hstack([camera_params['R'], camera_params['T'].reshape(-1, 1)]),
                    [0, 0, 0, 1]
                ])
            )
            
            # Render mesh view
            mesh_img = scene.save_image(resolution=(1920, 1080))
            mesh_output_path = os.path.join(output_dir, f"mesh_frame_{frame_idx:05d}.png")
            with open(mesh_output_path, 'wb') as f:
                f.write(mesh_img)
    
    print(f"Saved {len(rendered_images)} rendered images to {output_dir}")
    return rendered_images


# Integration into your main code:
# Add this after loading all geometry (around line 800 in your code)

def integrate_rendering(
    hmr_results,
    loader,
    sqs_mesh_original,
    world_cam_R,
    world_cam_T,
    camera,
    num_frames,
    tgt_name,
    faces,
    s_first_=None,
    R_first_=None,
    t_first_=None
):
    """
    Integrate rendering into your main pipeline.
    Call this after all geometry is loaded.
    """
    # Build intrinsic matrix
    K = np.array([
        [camera['img_focal'], 0, camera['img_center'][0]],
        [0, camera['img_focal'], camera['img_center'][1]],
        [0, 0, 1]
    ])
    
    # Apply transformation to scene mesh if alignment was performed
    if sqs_mesh_original is not None and s_first_ is not None:
        from scipy.spatial.transform import Rotation
        transform = np.eye(4)
        if isinstance(R_first_, torch.Tensor):
            R_first_ = R_first_.cpu().numpy()
        if isinstance(t_first_, torch.Tensor):
            t_first_ = t_first_.cpu().numpy()
        
        transform[:3, :3] = R_first_ * s_first_
        transform[:3, 3] = t_first_
        
        sqs_mesh_transformed = sqs_mesh_original.copy()
        sqs_mesh_transformed.apply_transform(transform)
    else:
        sqs_mesh_transformed = sqs_mesh_original
    
    # Render to all training views
    output_dir = f"/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/rendered_views/{tgt_name}"
    
    render_all_training_views(
        hmr_results=hmr_results,
        loader=loader,
        sqs_mesh=sqs_mesh_transformed,
        world_cam_R=world_cam_R,
        world_cam_T=world_cam_T,
        intrinsics=K,
        num_frames=num_frames,
        output_dir=output_dir,
        interval=10  # Adjust based on your needs
    )
    
    print(f"Rendering complete! Check {output_dir} for results.")
    
    # Optional: Create a video from rendered frames
    create_video_from_frames(output_dir, f"{output_dir}/{tgt_name}_rendered.mp4")


def create_video_from_frames(frames_dir: str, output_path: str, fps: int = 30):
    """Create a video from rendered frames."""
    import cv2
    import glob
    
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_paths:
        print("No frames found for video creation")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")


# Add these imports at the top of your file (around line 20-30)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
)
from pytorch3d.renderer.cameras import look_at_view_transform
import torch

def render_mesh_with_pytorch3d(
    vertices_list,  # List of vertex arrays
    faces_list,     # List of face arrays
    colors_list,    # List of colors for each mesh
    K,              # Intrinsic matrix (3, 3)
    R_cam,          # Camera rotation (3, 3)
    T_cam,          # Camera translation (3,)
    img_height=1080,
    img_width=1920,
    device='cuda'
):
    """
    Render meshes using PyTorch3D instead of trimesh.
    
    Args:
        vertices_list: List of numpy arrays, each (V_i, 3)
        faces_list: List of numpy arrays, each (F_i, 3)
        colors_list: List of RGB colors, each (V_i, 3) or single (3,)
        K: Camera intrinsic matrix (3, 3)
        R_cam: Camera rotation matrix (3, 3)
        T_cam: Camera translation vector (3,)
        img_height, img_width: Output image dimensions
        
    Returns:
        rendered_img: (H, W, 3) numpy array (uint8)
    """
    # Convert to torch tensors
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Combine all meshes into one
    all_verts = []
    all_faces = []
    all_colors = []
    vert_offset = 0
    
    for verts, faces, color in zip(vertices_list, faces_list, colors_list):
        verts_torch = torch.tensor(verts, dtype=torch.float32)
        faces_torch = torch.tensor(faces, dtype=torch.long) + vert_offset
        
        # Handle colors
        if len(color.shape) == 1:  # Single color for whole mesh
            colors_torch = torch.tensor(color, dtype=torch.float32).unsqueeze(0).repeat(len(verts), 1) / 255.0
        else:  # Per-vertex colors
            colors_torch = torch.tensor(color, dtype=torch.float32) / 255.0
            
        all_verts.append(verts_torch)
        all_faces.append(faces_torch)
        all_colors.append(colors_torch)
        vert_offset += len(verts)
    
    # Concatenate all
    verts_combined = torch.cat(all_verts, dim=0).unsqueeze(0).to(device)  # (1, V_total, 3)
    faces_combined = torch.cat(all_faces, dim=0).unsqueeze(0).to(device)  # (1, F_total, 3)
    colors_combined = torch.cat(all_colors, dim=0).unsqueeze(0).to(device)  # (1, V_total, 3)
    
    # Create texture
    textures = TexturesVertex(verts_features=colors_combined)
    
    # Create mesh
    mesh = Meshes(verts=verts_combined, faces=faces_combined, textures=textures)
    
    # Set up PyTorch3D camera
    # Convert from OpenCV convention to PyTorch3D
    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]
    
    # PyTorch3D uses NDC coordinates, need to convert
    focal_length = torch.tensor([[fx, fy]], dtype=torch.float32).to(device)
    principal_point = torch.tensor([[px, py]], dtype=torch.float32).to(device)
    image_size = torch.tensor([[img_height, img_width]], dtype=torch.float32).to(device)
    
    # Camera extrinsics
    # Convert from world-to-camera to camera-to-world for PyTorch3D
    R_torch = torch.tensor(R_cam.T, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, 3)
    T_torch = -torch.tensor(R_cam.T @ T_cam, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3)
    
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R_torch,
        T=T_torch,
        image_size=image_size,
        device=device,
        in_ndc=False
    )
    
    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(img_height, img_width),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
    )
    
    # Create renderer
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, -3.0]],
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.5, 0.5, 0.5),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    
    # Render
    images = renderer(mesh)  # (1, H, W, 4) RGBA
    
    # Convert to numpy uint8
    rendered = images[0, ..., :3].cpu().numpy()  # Remove alpha, get RGB
    rendered = (rendered * 255).clip(0, 255).astype(np.uint8)
    
    return rendered