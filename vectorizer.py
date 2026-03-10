"""
vectorizer.py — Image-to-VKB Vectorizer
========================================
Converts a raster photograph or graphic into a compact VKB binary scene
using an adaptive quadtree subdivision algorithm.

Algorithm overview:
  1. Compute a structural gradient map (np.gradient) for edge orientation.
  2. Detect the dominant background colour from image border pixels.
  3. Recursively subdivide the image into quadtree cells:
       - Cells with low edge magnitude and colour variance stay large.
       - Cells near edges subdivide down to --block minimum size.
  4. Skip cells whose average colour matches the background (subject isolation).
  5. Pack each surviving cell as a 20-byte kernel record into a VKB file.

Usage:
    python vectorizer.py <input> <output.vkb> [--block N] [--max_block N]
"""
import argparse
import json
import struct
import numpy as np
from PIL import Image
import os
import gzip

def vectorize_image(input_path, output_path, block_size=10, max_block=32):
    """
    Vectorize *input_path* into a VKB scene saved at *output_path*.

    Parameters
    ----------
    input_path  : str   Path to a PNG or JPG source image.
    output_path : str   Destination path. ``.vkb`` produces a gzip-compressed
                        binary scene; ``.json`` produces a plain JSON scene.
    block_size  : int   Minimum quadtree cell size in pixels (default 10).
    max_block   : int   Maximum quadtree cell size in pixels (default 32).
                        Cells with low detail stop subdividing once they reach
                        this size, keeping flat areas compressed.
    """
    print(f"Loading '{input_path}'...")
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    print(f"Image size: {w}x{h}. Min block size: {block_size}")
    
    # Compute Structural Image Gradients for edge-alignment orientation
    print("Pre-computing structural gradient vectors...")
    gray_img = np.array(img.convert('L'), dtype=float)
    gy, gx = np.gradient(gray_img)
    mag = np.sqrt(gx**2 + gy**2) # Recalculate magnitude after new gx, gy

    # Detect Background Color (Median of border pixels to avoid edge noise)
    border_pixels = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    detected_bg = np.median(border_pixels, axis=0)
    
    gray = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Calculate gradients for edge detection
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)
    
    nodes = []
    
    def subdivide(x_start, y_start, w_box, h_box, depth):
        if w_box <= 0 or h_box <= 0:
            return
            
        region = arr[y_start:y_start+h_box, x_start:x_start+w_box]
        region_mag = mag[y_start:y_start+h_box, x_start:x_start+w_box]
        
        # Adaptive detail detection:
        # Instead of strict max-min (which breaks on noise and gradients),
        # we check the Structural edge magnitude (m_mean, m_max) and standard color deviation.
        color_std = np.mean(np.std(region, axis=(0,1)))
        m_mean = np.mean(region_mag)
        m_max = np.max(region_mag)
        
        # Stop condition: If lacks sharp physical edges and heavy texture noise, keep block large!
        is_detail_low = (m_mean < 2.0) and (m_max < 8.0) and (color_std < 8.0)
        
        if (is_detail_low and w_box <= max_block and h_box <= max_block) or w_box <= block_size or h_box <= block_size or depth >= 12:
            avg_color = np.mean(region, axis=(0,1))
            
            # Subject Isolation: Cull flat geometry matching the global background!
            if is_detail_low:
                color_dist = np.linalg.norm(avg_color - detected_bg)
                if color_dist < 8.0: # Tolerance for noise
                    return
                    
            avg_gx = np.mean(gx[y_start:y_start+h_box, x_start:x_start+w_box])
            avg_gy = np.mean(gy[y_start:y_start+h_box, x_start:x_start+w_box])
            nodes.append((x_start, y_start, w_box, h_box, avg_color, avg_gx, avg_gy, m_mean))
        else:
            w1 = w_box // 2
            w2 = w_box - w1
            h1 = h_box // 2
            h2 = h_box - h1
            if w1 > 0 and h1 > 0: subdivide(x_start, y_start, w1, h1, depth+1)
            if w2 > 0 and h1 > 0: subdivide(x_start+w1, y_start, w2, h1, depth+1)
            if w1 > 0 and h2 > 0: subdivide(x_start, y_start+h1, w1, h2, depth+1)
            if w2 > 0 and h2 > 0: subdivide(x_start+w1, y_start+h1, w2, h2, depth+1)

    print("Subdividing image via quadtree...")
    subdivide(0, 0, w, h, 0)
    print(f"Quadtree generated {len(nodes)} regions.")
        
    kernels = []
    
    print("Generating geometry from quadtree regions...")
    for (nx, ny, nw, nh, color, agx, agy, amag) in nodes:
        cx = nx + nw / 2.0
        cy = ny + nh / 2.0
        
        # Calculate local gradient for angle alignment (brush stroke direction)
        mean_gx = np.mean(gx[ny:ny+nh, nx:nx+nw])
        mean_gy = np.mean(gy[ny:ny+nh, nx:nx+nw])
        # We want the brush stroke to be parallel to the edge, so we rotate 90 degrees from the gradient
        angle_rad = np.arctan2(mean_gy, mean_gx)
        angle_deg = (np.degrees(angle_rad) + 90.0) % 360.0
        
        # For exact raster reconstruction, we use 'bounds' to limit the stretch
        bound_x = nw / 2.0
        bound_y = nh / 2.0
        
        kernel = {
            'x': float(np.round(cx, 2)),
            'y': float(np.round(cy, 2)),
            'color': color.astype(int).tolist(),
            'angle': float(np.round(angle_deg, 2)),
            'bounds': [float(np.round(-bound_x, 2)), float(np.round(-bound_y, 2)), float(np.round(bound_x, 2)), float(np.round(bound_y, 2))]
        }
        kernels.append(kernel)

    print(f"Generated {len(kernels)} kernels.")
    
    # Use the isolated border median as the engine background exactly like traditional rasters!
    bg_color = detected_bg.astype(int).tolist()
    
    scene = {
        'width': w,
        'height': h,
        'background_color': [bg_color[0], bg_color[1], bg_color[2], 255],
        'layers': [{'blend_mode': 'normal', 'kernels': kernels}]
    }
    
    if output_path.endswith('.vkb'):
        print(f"Compressing to VKB6 binary: {output_path}...")
        with gzip.open(output_path, "wb") as f:
            # VKB6 Header: 4s (VKB6), H (w), H (h), B B B B (bg_rgba), I (total_count) = 16 bytes
            f.write(struct.pack('<4sHHBBBBI', b'VKB6', w, h, bg_color[0], bg_color[1], bg_color[2], 255, len(kernels)))
            
            # Group kernels by 8-bit prefix of 16-bit quantized XY
            groups = {}
            for k in kernels:
                # 16-bit quantization (0-65535)
                xq = int(np.clip(np.round(k['x'] / w * 65535), 0, 65535))
                yq = int(np.clip(np.round(k['y'] / h * 65535), 0, 65535))
                
                px, sx = xq >> 8, xq & 0xFF
                py, sy = yq >> 8, yq & 0xFF
                
                prefix = (px, py)
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append((sx, sy, k))
            
            for (px, py), k_list in groups.items():
                f.write(struct.pack('<BBH', px, py, len(k_list)))
                for sx, sy, k in k_list:
                    # Determine flags
                    flags = 0
                    
                    # Use ceil to ensure quantized bounding box always covers the source pixels
                    hwq = int(np.clip(np.ceil(k['bounds'][2] / w * 65535), 0, 65535))
                    hhq = int(np.clip(np.ceil(k['bounds'][3] / h * 65535), 0, 65535))
                    # Default is now something unlikely to collide
                    if hwq != 1 or hhq != 1: flags |= 1
                    
                    rgb = k['color']
                    if rgb[0] != 255 or rgb[1] != 255 or rgb[2] != 255: flags |= 2
                    
                    alpha = int(k.get('alpha', 1.0) * 255)
                    if alpha != 255: flags |= 4
                    
                    decay = int(k.get('clamp_decay', 0.0) * 255)
                    if decay != 0: flags |= 8
                    
                    theta = int((k['angle'] / 360.0) * 255.0)
                    if theta != 0: flags |= 16
                    
                    # Pack record
                    f.write(struct.pack('<B', flags))
                    f.write(struct.pack('<BB', sx, sy))
                    
                    if flags & 1: f.write(struct.pack('<HH', hwq, hhq))
                    if flags & 2: f.write(struct.pack('<BBB', rgb[0], rgb[1], rgb[2]))
                    if flags & 4: f.write(struct.pack('<B', alpha))
                    if flags & 8: f.write(struct.pack('<B', decay))
                    if flags & 16: f.write(struct.pack('<B', theta))
                    
        print(f"Saved VKB6 binary scene to {output_path}")
    else:
        with open(output_path, "w") as f:
            json.dump(scene, f, indent=2)
        print(f"Saved JSON VKE scene to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VKE Image Vectorizer")
    parser.add_argument("input_image", help="Path to input image (PNG/JPG)")
    parser.add_argument("output_json", help="Path to output VKE JSON file")
    parser.add_argument("--block", type=int, default=7, help="Minimum pixel size of each sampled block (default: 7)")
    parser.add_argument("--max_block", type=int, default=30, help="Maximum pixel size of a low-detail block (default: 30)")
    
    args = parser.parse_args()
    vectorize_image(args.input_image, args.output_json, args.block, args.max_block)
