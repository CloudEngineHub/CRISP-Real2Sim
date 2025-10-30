import cv2
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt

from PIL import Image


def detect_edges_and_segment_planes(depth_map):
    # 1. Load Depth Map
    # depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    #if depth_map is None:
    #    raise FileNotFoundError('Depth map image not found!')
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_min == depth_max:
        # Handle case where entire image is a single value
        depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min) * 255
        depth_map_normalized = depth_map_normalized.astype(np.uint8)
    depth_map = depth_map_normalized
    edges = cv2.Canny(depth_map, 50, 150)

    # 3. Morphological Operations to enhance edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)

    # 4. Find all contours
    contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Identify the contour with the highest position (topmost point)
    highest_contour = None
    highest_y = float('inf')
    
    for contour in contours:
        contour_topmost = contour[contour[:, :, 1].argmin()][0]
        if contour_topmost[1] < highest_y:
            highest_y = contour_topmost[1]
            highest_contour = contour

    # 6. Create a mask with the region **above** the highest contour
    height, width = edges_eroded.shape
    mask_above_line = np.zeros_like(edges_eroded)

    if highest_contour is not None:
        # Draw the contour on the mask
        cv2.drawContours(mask_above_line, [highest_contour], -1, 255, thickness=1)
        
        # Fill above the contour by flood-filling from the top
        for x in range(0, width, 10):
            if mask_above_line[0, x] == 0:  # Ensure starting from the very top
                cv2.floodFill(mask_above_line, None, (x, 0), 255)
        
        # Invert the mask to keep only the region **above** the contour
        mask_above_line = cv2.bitwise_not(mask_above_line)

    # 7. Apply the mask to the depth map
    masked_depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask_above_line)

    return masked_depth_map