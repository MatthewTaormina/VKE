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
        import struct
        with gzip.open(json_path, "rb") as f:
            magic = f.read(4)
            if magic != b'VKB1':
                # Try fallback just in case it's an old JSON .vkb
                f.seek(0)
                json_str = f.read().decode('utf-8')
                scene = json.loads(json_str)
            else:
                header_data = f.read(11)
                w, h, bg_r, bg_g, bg_b, k_count = struct.unpack('<HHBBBI', header_data)
                
                kernels = []
                for _ in range(k_count):
                    k_data = f.read(20)
                    kx, ky, hw, hh, kr, kg, kb, theta = struct.unpack('<ffffBBBB', k_data)
                    
                    # Axis-aligned bounds + zero decay = perfect seamless solid tiling.
                    # Rotation with bounds leaves uncovered corner gaps, so no angle here.
                    kernels.append({
                        'x': float(kx), 'y': float(ky),
                        'color': [int(kr), int(kg), int(kb)],
                        'bounds': [float(-hw), float(-hh), float(hw), float(hh)],
                        'decay': {
                            'east_inward':   0.0, 'east_outward':  0.0,
                            'west_inward':   0.0, 'west_outward':  0.0,
                            'north_inward':  0.0, 'north_outward': 0.0,
                            'south_inward':  0.0, 'south_outward': 0.0,
                        }
                    })
                
                scene = {
                    'width': int(w),
                    'height': int(h),
                    'background_color': [int(bg_r), int(bg_g), int(bg_b), 255],
                    'layers': [{'blend_mode': 'normal', 'kernels': kernels}]
                }
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
