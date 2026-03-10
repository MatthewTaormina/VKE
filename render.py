"""
render.py — VKB/JSON Scene Renderer
=====================================
Loads a VKB binary scene (or plain JSON scene) and renders it to a PNG
through the VKE numpy rendering engine (main.py).

Supported input formats
-----------------------
* **Struct VKB** (``.vkb``, magic ``VKB1``) — compact 20-byte-per-kernel
  binary produced by ``vectorizer.py``.  Each kernel is axis-aligned with
  zero decay (solid rectangular fill).
* **JSON VKB** (``.vkb``, other magic) — gzip-compressed VKE JSON scene.
  Supports the full kernel schema: modulation, decay, logic, bounds, etc.
* **Plain JSON** (``.json``) — uncompressed VKE JSON scene.

Usage:
    python render.py <input.vkb|input.json> <output.png>
"""
import sys
import json
import time
import struct
from main import process_vke_scene
import numpy as np
from PIL import Image
import gzip

def render(json_path, output_png):
    """
    Render *json_path* (VKB or JSON) to *output_png*.

    Parameters
    ----------
    json_path  : str  Path to a ``.vkb`` or ``.json`` scene file.
    output_png : str  Destination PNG path.
    """
    print(f"Rendering {json_path}...")
    if json_path.endswith('.vkb'):
        with gzip.open(json_path, "rb") as f:
            magic = f.read(4)
            if magic == b'VKB6' or magic == b'VKB5' or magic == b'VKB3':
                # Struct VKB6/VKB5/VKB3 (Grouped 16-bit Quantized)
                w, h, br, bg, bb, ba, total_k = struct.unpack('<HHBBBBI', f.read(12))
                kernels = []
                k_read = 0
                while k_read < total_k:
                    # Read group header
                    px, py, g_count = struct.unpack('<BBH', f.read(4))
                    for _ in range(g_count):
                        # Defaults for VKB6
                        hwq, hhq = 8, 8 # Default from vectorizer logic if not provided
                        r, g, b = 255, 255, 255
                        a, d, t = 255, 0, 0
                        
                        if magic == b'VKB6':
                            flags = struct.unpack('<B', f.read(1))[0]
                            sx, sy = struct.unpack('<BB', f.read(2))
                            if flags & 1:
                                hwq, hhq = struct.unpack('<HH', f.read(4))
                            if flags & 2:
                                r, g, b = struct.unpack('<BBB', f.read(3))
                            if flags & 4:
                                a = struct.unpack('<B', f.read(1))[0]
                            if flags & 8:
                                d = struct.unpack('<B', f.read(1))[0]
                            if flags & 16:
                                t = struct.unpack('<B', f.read(1))[0]
                        else:
                            # VKB5/VKB3 is fixed 12 bytes
                            sx, sy, hwq, hhq, r, g, b, a, d, t = struct.unpack('<BBHHBBBBBB', f.read(12))
                        
                        xq = (px << 8) | sx
                        yq = (py << 8) | sy
                        
                        kx = (xq / 65535.0) * w
                        ky = (yq / 65535.0) * h
                        # Use a more generous epsilon (0.5px) to bridge quantization gaps
                        hw = (hwq / 65535.0) * w + 0.5
                        hh = (hhq / 65535.0) * h + 0.5
                        angle = (t / 255.0) * 360.0
                        
                        kernels.append({
                            'x': kx, 'y': ky, 'color': [r, g, b], 'alpha': a/255.0,
                            'clamp_decay': d/255.0,
                            'angle': angle if d > 0 else 0,
                            'bounds': [-hw, -hh, hw, hh],
                            'decay': {k: 0.0 for k in ['east_inward', 'east_outward', 'west_inward', 'west_outward',
                                                      'north_inward', 'north_outward', 'south_inward', 'south_outward']}
                        })
                        k_read += 1
                scene = {
                    'width': w, 'height': h, 'background_color': [br, bg, bb, ba],
                    'layers': [{'kernels': kernels}]
                }
            elif magic == b'VKB4':
                # Struct VKB4 (Grouped 12-bit Quantized)
                w, h, br, bg, bb, ba, total_k = struct.unpack('<HHBBBBI', f.read(12))
                kernels = []
                k_read = 0
                while k_read < total_k:
                    px, py, g_count = struct.unpack('<BBH', f.read(4))
                    for _ in range(g_count):
                        s_xy, hwq, hhq, r, g, b, a, d, t = struct.unpack('<BHHBBBBBB', f.read(11))
                        
                        # Unpack 12-bit positions (8-bit prefix + 4-bit suffix)
                        xq = (px << 4) | (s_xy >> 4)
                        yq = (py << 4) | (s_xy & 0x0F)
                        
                        kx = (xq / 4095.0) * w
                        ky = (yq / 4095.0) * h
                        hw = (hwq / 65535.0) * w + 0.01
                        hh = (hhq / 65535.0) * h + 0.01
                        angle = (t / 255.0) * 360.0
                        
                        kernels.append({
                            'x': kx, 'y': ky, 'color': [r, g, b], 'alpha': a/255.0,
                            'clamp_decay': d/255.0,
                            'angle': angle if d > 0 else 0,
                            'bounds': [-hw, -hh, hw, hh],
                            'decay': {k: 0.0 for k in ['east_inward', 'east_outward', 'west_inward', 'west_outward',
                                                      'north_inward', 'north_outward', 'south_inward', 'south_outward']}
                        })
                        k_read += 1
                scene = {
                    'width': w, 'height': h, 'background_color': [br, bg, bb, ba],
                    'layers': [{'kernels': kernels}]
                }
            elif magic == b'VKB3':
                # Struct VKB3 (Grouped Quantized)
                w, h, br, bg, bb, ba, total_k = struct.unpack('<HHBBBBI', f.read(12))
                kernels = []
                k_read = 0
                while k_read < total_k:
                    px, py, g_count = struct.unpack('<BBH', f.read(4))
                    for _ in range(g_count):
                        sx, sy, hwq, hhq, r, g, b, a, d, t = struct.unpack('<BBHHBBBBBB', f.read(12))
                        
                        xq = (px << 8) | sx
                        yq = (py << 8) | sy
                        
                        kx = (xq / 65535.0) * w
                        ky = (yq / 65535.0) * h
                        hw = (hwq / 65535.0) * w + 0.01
                        hh = (hhq / 65535.0) * h + 0.01
                        angle = (t / 255.0) * 360.0
                        
                        kernels.append({
                            'x': kx, 'y': ky, 'color': [r, g, b], 'alpha': a/255.0,
                            'clamp_decay': d/255.0,
                            'angle': angle if d > 0 else 0,
                            'bounds': [-hw, -hh, hw, hh],
                            'decay': {k: 0.0 for k in ['east_inward', 'east_outward', 'west_inward', 'west_outward',
                                                      'north_inward', 'north_outward', 'south_inward', 'south_outward']}
                        })
                        k_read += 1
                scene = {
                    'width': w, 'height': h, 'background_color': [br, bg, bb, ba],
                    'layers': [{'kernels': kernels}]
                }
            elif magic == b'VKB2':
                # Struct VKB2 (Modern Quantized)
                w, h, br, bg, bb, ba, k_count = struct.unpack('<HHBBBBI', f.read(12))
                kernels = []
                for _ in range(k_count):
                    xq, yq, hwq, hhq, r, g, b, a, d, t = struct.unpack('<HHHHBBBBBB', f.read(14))
                    # De-quantize with tiny epsilon for seamless tiling
                    kx = (xq / 65535.0) * w
                    ky = (yq / 65535.0) * h
                    hw = (hwq / 65535.0) * w + 0.01
                    hh = (hhq / 65535.0) * h + 0.01
                    angle = (t / 255.0) * 360.0
                    
                    kernels.append({
                        'x': kx, 'y': ky, 'color': [r, g, b], 'alpha': a/255.0,
                        'clamp_decay': d/255.0,
                        'angle': angle if d > 0 else 0,
                        'bounds': [-hw, -hh, hw, hh],
                        'decay': {k: 0.0 for k in ['east_inward', 'east_outward', 'west_inward', 'west_outward',
                                                  'north_inward', 'north_outward', 'south_inward', 'south_outward']}
                    })
                scene = {
                    'width': w, 'height': h, 'background_color': [br, bg, bb, ba],
                    'layers': [{'kernels': kernels}]
                }
            elif magic == b'VKB1':
                header_data = f.read(11)
                w, h, bg_r, bg_g, bg_b, k_count = struct.unpack('<HHBBBI', header_data)
                
                kernels = []
                for _ in range(k_count):
                    k_data = f.read(20)
                    kx, ky, hw, hh, kr, kg, kb, theta = struct.unpack('<ffffBBBB', k_data)
                    angle = (theta / 255.0) * 360.0
                    
                    kernels.append({
                        'x': float(kx), 'y': float(ky),
                        'color': [int(kr), int(kg), int(kb)],
                        'bounds': [float(-hw), float(-hh), float(hw), float(hh)],
                        'angle': angle
                    })
                
                scene = {
                    'width': int(w),
                    'height': int(h),
                    'background_color': [int(bg_r), int(bg_g), int(bg_b), 255],
                    'layers': [{'blend_mode': 'normal', 'kernels': kernels}]
                }
            else:
                # Try fallback just in case it's an old JSON .vkb
                f.seek(0)
                json_str = f.read().decode('utf-8')
                scene = json.loads(json_str)
    else:
        with open(json_path, 'r') as f:
            scene = json.load(f)

    t0 = time.time()
    w = scene.get('width', 200)
    h = scene.get('height', 200)
    bg = scene.get('background_color', [0, 0, 0, 255])
    img_data = process_vke_scene(scene, w, h, bg)
    img = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
    img.save(output_png)
    print(f"Rendered {output_png} in {time.time()-t0:.2f}s")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python render.py <input.json> <output.png>")
        sys.exit(1)
    render(sys.argv[1], sys.argv[2])
