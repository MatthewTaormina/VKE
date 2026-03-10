"""
main.py — VKE Core Rendering Engine
=====================================
Numpy-vectorised renderer for Vector Kernel Engine (VKE) scenes.  Each
"Influence Kernel" contributes a per-pixel weight field that blends its
colour onto the image buffer through a pipeline of:

  rotation → modulation → shift → 8-way decay → logic metric → alpha blend

The engine also functions as the sample-scene generator: ``python main.py``
saves all 15 built-in scenes to ``samples/`` as gzip-VKB files and renders
them to ``output/`` as PNG files.
"""
import json
import gzip
import numpy as np
from PIL import Image
import math
import os
import time

def process_vke_scene(scene_data, width, height, bg_color):
    """
    Render a VKE scene dictionary to a numpy RGBA image array.

    Parameters
    ----------
    scene_data : dict   Parsed VKE scene (layers, templates, groups, kernels).
    width      : int    Canvas width in pixels.
    height     : int    Canvas height in pixels.
    bg_color   : list   Background colour as ``[R, G, B]`` or ``[R, G, B, A]``.

    Returns
    -------
    numpy.ndarray  Shape ``(height, width, 4)`` uint8 RGBA image.
    """
    # Initialize the main image buffer (RGBA float for accumulation)
    image_buffer = np.zeros((height, width, 4), dtype=np.float32)
    image_buffer[:, :, 0] = bg_color[0]
    image_buffer[:, :, 1] = bg_color[1]
    image_buffer[:, :, 2] = bg_color[2]
    image_buffer[:, :, 3] = bg_color[3] if len(bg_color) > 3 else 255.0

    # Precompute coordinate grids
    # Y is rows (0 to height-1), X is cols (0 to width-1)
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    layers = scene_data.get('layers', [])
    templates = scene_data.get('templates', {})

    for layer in layers:
        layer_buffer = np.zeros((height, width, 4), dtype=np.float32)
        kernels = layer.get('kernels', [])
        groups = layer.get('groups', [])

        all_kernels = list(kernels)
        for g in groups:
            group_x = g.get('x', 0)
            group_y = g.get('y', 0)
            group_angle = g.get('angle', 0)
            for k in g.get('kernels', []):
                k_copy = dict(k)
                k_copy['x'] = k_copy.get('x', 0) + group_x
                k_copy['y'] = k_copy.get('y', 0) + group_y
                k_copy['angle'] = k_copy.get('angle', 0) + group_angle
                all_kernels.append(k_copy)

        for kernel_def in all_kernels:
            # Resolve template
            template_name = kernel_def.get('template')
            kernel = {}
            if template_name and template_name in templates:
                kernel.update(templates[template_name])
            kernel.update(kernel_def)

            # Properties
            kx = kernel.get('x', width / 2)
            ky = kernel.get('y', height / 2)
            angle_deg = kernel.get('angle', 0)
            angle_rad = np.radians(angle_deg)

            min_clamp = kernel.get('min_clamp', 0)
            max_clamp = kernel.get('max_clamp', max(width, height) * 2)

            mod_type = kernel.get('modulation', 'CONTINUOUS').upper()
            mod_period = kernel.get('period', 50)

            shift = kernel.get('shift', 0)

            decays = kernel.get('decay', {})
            d_e_in = decays.get('east_inward', 0.05)
            d_e_out = decays.get('east_outward', 0.05)
            d_w_in = decays.get('west_inward', 0.05)
            d_w_out = decays.get('west_outward', 0.05)
            d_n_in = decays.get('north_inward', 0.05)
            d_n_out = decays.get('north_outward', 0.05)
            d_s_in = decays.get('south_inward', 0.05)
            d_s_out = decays.get('south_outward', 0.05)

            logic = kernel.get('logic', 'MULTIPLY').upper()
            color = np.array(kernel.get('color', [255, 255, 255]), dtype=np.float32)
            alpha = kernel.get('alpha', 1.0)

            # Calculate fast bounding box to avoid full-screen numpy maths
            min_decay = min([d for d in [d_e_in, d_e_out, d_w_in, d_w_out, d_n_in, d_n_out, d_s_in, d_s_out] if d > 0] or [0.0001])
            r_decay = (1.0 / min_decay) + shift
            
            R = min(max_clamp, r_decay)
            if 'bounds' in kernel:
                bx1, by1, bx2, by2 = kernel['bounds']
                rb = np.sqrt(max(bx1**2, bx2**2) + max(by1**2, by2**2))
                R = min(R, rb)
                
            y1 = max(0, int(ky - R - 2))
            y2 = min(height, int(ky + R + 3))
            x1 = max(0, int(kx - R - 2))
            x2 = min(width, int(kx + R + 3))
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            y_slice = y_coords[y1:y2, x1:x2]
            x_slice = x_coords[y1:y2, x1:x2]

            # Step 1: Coordinates
            dx_global = x_slice - kx
            dy_global = y_slice - ky

            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            local_x = dx_global * cos_a - dy_global * sin_a
            local_y = dx_global * sin_a + dy_global * cos_a

            abs_x = np.abs(local_x)
            abs_y = np.abs(local_y)
            d_euc = np.sqrt(local_x**2 + local_y**2)

            # Step 2: Visibility Check
            valid_mask = (d_euc >= min_clamp) & (d_euc <= max_clamp)

            if 'bounds' in kernel:
                bx1, by1, bx2, by2 = kernel['bounds']
                valid_mask &= (local_x >= bx1) & (local_x <= bx2) & (local_y >= by1) & (local_y <= by2)

            if not np.any(valid_mask):
                continue

            # Compute Modulators before shift
            if mod_type == 'RAMP':
                d_mod_x = abs_x % mod_period
                d_mod_y = abs_y % mod_period
            elif mod_type == 'TRIANGLE':
                half_p = mod_period / 2.0
                d_mod_x = half_p - np.abs((abs_x % mod_period) - half_p)
                d_mod_y = half_p - np.abs((abs_y % mod_period) - half_p)
            elif mod_type == 'PARABOLIC':
                half_p = mod_period / 2.0
                # Scale so it has a reasonable size. Spec says x**2 / 100. Let's do scale factor.
                scale = kernel.get('parabola_scale', mod_period)
                v_x = (abs_x % mod_period) - half_p
                d_mod_x = (v_x ** 2) / scale
                v_y = (abs_y % mod_period) - half_p
                d_mod_y = (v_y ** 2) / scale
            else:  # CONTINUOUS
                d_mod_x = abs_x
                d_mod_y = abs_y

            # Step 3: Shifted Distance
            d_shifted_x = d_mod_x - shift
            d_shifted_y = d_mod_y - shift

            d_final_x = np.abs(d_shifted_x)
            d_final_y = np.abs(d_shifted_y)

            # Step 5: Coefficients
            is_east = local_x >= 0
            is_x_outward = d_shifted_x >= 0

            coeff_x = np.zeros_like(local_x)
            coeff_x[is_east & is_x_outward] = d_e_out
            coeff_x[is_east & ~is_x_outward] = d_e_in
            coeff_x[~is_east & is_x_outward] = d_w_out
            coeff_x[~is_east & ~is_x_outward] = d_w_in

            is_south = local_y >= 0
            is_y_outward = d_shifted_y >= 0

            coeff_y = np.zeros_like(local_y)
            coeff_y[is_south & is_y_outward] = d_s_out
            coeff_y[is_south & ~is_y_outward] = d_s_in
            coeff_y[~is_south & is_y_outward] = d_n_out
            coeff_y[~is_south & ~is_y_outward] = d_n_in

            # Compute Weights
            weight_x = np.maximum(0.0, 1.0 - (d_final_x * coeff_x))
            weight_y = np.maximum(0.0, 1.0 - (d_final_y * coeff_y))

            # Logic Metric
            if logic == 'MULTIPLY':
                weight_comb = weight_x * weight_y
            elif logic == 'MAX':
                weight_comb = np.maximum(weight_x, weight_y)
            elif logic == 'MIN':
                weight_comb = np.minimum(weight_x, weight_y)
            elif logic == 'SQUARESUM':
                weight_comb = np.clip(np.sqrt(weight_x**2 + weight_y**2), 0.0, 1.0)
            else:
                weight_comb = weight_x * weight_y

            final_weight = weight_comb * alpha
            final_weight[~valid_mask] = 0.0

            # Accumulate on layer view
            active = final_weight > 0
            if np.any(active):
                fw = final_weight[active, np.newaxis]
                lb_view = layer_buffer[y1:y2, x1:x2]
                lb_view[active, 0:3] = color * fw + lb_view[active, 0:3] * (1.0 - fw)
                lb_view[active, 3] = np.maximum(lb_view[active, 3], fw.squeeze() * 255.0)

        # Merge layer
        layer_alpha = layer_buffer[:, :, 3:4] / 255.0
        image_buffer[:, :, 0:3] = layer_buffer[:, :, 0:3] * layer_alpha + image_buffer[:, :, 0:3] * (1.0 - layer_alpha)
        image_buffer[:, :, 3:4] = np.maximum(image_buffer[:, :, 3:4], layer_buffer[:, :, 3:4])

    return np.clip(image_buffer, 0, 255).astype(np.uint8)


def generate_samples():
    """Generate 15 showcase scenes demonstrating VKE's range."""
    os.makedirs('samples', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    samples = []
    W, H = 400, 400
    CX, CY = W // 2, H // 2

    def solid(v): return {k: v for k in ['east_inward','east_outward','west_inward','west_outward',
                                          'north_inward','north_outward','south_inward','south_outward']}

    # ----------------------------------------------------------
    # 1. SOLAR ECLIPSE — dramatic ring + corona glow
    # ----------------------------------------------------------
    s1_kernels = [
        # Wide orange corona glow
        {'x': CX, 'y': CY, 'logic': 'SQUARESUM',
         'decay': solid(0.005), 'color': [255, 140, 20], 'alpha': 0.9},
        # Inner bright ring
        {'x': CX, 'y': CY, 'logic': 'MAX', 'min_clamp': 70, 'max_clamp': 90,
         'decay': solid(0.0), 'color': [255, 220, 80]},
        # Outer faint halo ring
        {'x': CX, 'y': CY, 'logic': 'MAX', 'min_clamp': 100, 'max_clamp': 115,
         'decay': solid(0.0), 'color': [255, 180, 60], 'alpha': 0.6},
        # Black moon disc
        {'x': CX, 'y': CY, 'logic': 'MAX', 'max_clamp': 72,
         'decay': solid(0.0), 'color': [5, 5, 10]},
    ]
    samples.append({'width': W, 'height': H, 'background_color': [5, 5, 15, 255],
                    'layers': [{'kernels': s1_kernels}]})

    # ----------------------------------------------------------
    # 2. NEON LOGO RING — multi-ring glowing emblem
    # ----------------------------------------------------------
    s2_kernels = []
    ring_colors = [[255, 0, 120], [0, 220, 255], [255, 200, 0], [120, 0, 255]]
    for i, col in enumerate(ring_colors):
        inner = 30 + i * 28
        s2_kernels.append({'x': CX, 'y': CY, 'logic': 'MAX',
                           'min_clamp': inner, 'max_clamp': inner + 16,
                           'decay': solid(0.0), 'color': col, 'alpha': 0.95})
        # Soft glow around each ring
        s2_kernels.append({'x': CX, 'y': CY, 'logic': 'SQUARESUM',
                           'min_clamp': inner - 4, 'max_clamp': inner + 20,
                           'decay': solid(0.04), 'color': col, 'alpha': 0.25})
    # Central star
    for angle_offset in range(0, 360, 45):
        s2_kernels.append({'x': CX, 'y': CY, 'angle': angle_offset,
                           'logic': 'MAX',
                           'decay': {'east_inward': 0.018, 'east_outward': 0.018,
                                     'west_inward': 0.018, 'west_outward': 0.018,
                                     'north_inward': 0.3, 'north_outward': 0.3,
                                     'south_inward': 0.3, 'south_outward': 0.3},
                           'color': [255, 255, 255], 'max_clamp': 25, 'alpha': 0.9})
    samples.append({'width': W, 'height': H, 'background_color': [8, 8, 20, 255],
                    'layers': [{'kernels': s2_kernels}]})

    # ----------------------------------------------------------
    # 3. PLASMA WEB — organic parabolic cell field
    # ----------------------------------------------------------
    import random
    random.seed(42)
    s3_kernels = []
    plasma_colors = [[80, 0, 200], [0, 180, 255], [200, 0, 180], [0, 255, 150], [255, 80, 0]]
    for _ in range(25):
        col = plasma_colors[random.randint(0, len(plasma_colors)-1)]
        s3_kernels.append({
            'x': random.randint(20, W-20), 'y': random.randint(20, H-20),
            'modulation': 'PARABOLIC', 'period': random.randint(80, 160),
            'parabola_scale': random.randint(30, 60),
            'decay': solid(0.008 + random.random() * 0.006),
            'color': col, 'logic': 'MULTIPLY', 'alpha': 0.75
        })
    samples.append({'width': W, 'height': H, 'background_color': [5, 0, 20, 255],
                    'layers': [{'kernels': s3_kernels}]})

    # ----------------------------------------------------------
    # 4. SUNSET GRADIENT — warm layered horizontal sweeps
    # ----------------------------------------------------------
    s4_kernels = [
        # Sky — top violet bloom
        {'x': CX, 'y': -20, 'logic': 'MAX', 'decay': {'east_inward': 0.0, 'east_outward': 0.0,
         'west_inward': 0.0, 'west_outward': 0.0,
         'north_inward': 0.006, 'north_outward': 0.006,
         'south_inward': 0.003, 'south_outward': 0.003},
         'color': [120, 60, 200], 'alpha': 1.0},
        # Horizon orange
        {'x': CX, 'y': 200, 'logic': 'MAX', 'decay': solid(0.003),
         'color': [255, 100, 20], 'alpha': 0.9},
        # Sun disc
        {'x': CX, 'y': 260, 'logic': 'MAX', 'max_clamp': 55, 'decay': solid(0.0),
         'color': [255, 220, 80]},
        # Sun glow
        {'x': CX, 'y': 260, 'logic': 'SQUARESUM', 'decay': solid(0.007),
         'color': [255, 160, 20], 'alpha': 0.5},
        # Ground — deep dark teal
        {'x': CX, 'y': H + 50, 'logic': 'MAX', 'decay': solid(0.004),
         'color': [10, 40, 60], 'alpha': 0.95},
    ]
    samples.append({'width': W, 'height': H, 'background_color': [15, 10, 40, 255],
                    'layers': [{'kernels': s4_kernels}]})

    # ----------------------------------------------------------
    # 5. 8-POINT STAR LOGO — classic sharp vector star
    # ----------------------------------------------------------
    s5_kernels = []
    star_col = [255, 215, 0]
    for a in range(0, 180, 20):
        s5_kernels.append({'x': CX, 'y': CY, 'angle': a, 'logic': 'MAX',
                           'decay': {'east_inward': 0.009, 'east_outward': 0.009,
                                     'west_inward': 0.009, 'west_outward': 0.009,
                                     'north_inward': 0.18, 'north_outward': 0.18,
                                     'south_inward': 0.18, 'south_outward': 0.18},
                           'color': star_col, 'alpha': 0.92})
    # Solid center circle
    s5_kernels.append({'x': CX, 'y': CY, 'logic': 'MAX', 'max_clamp': 22,
                       'decay': solid(0.0), 'color': star_col})
    samples.append({'width': W, 'height': H, 'background_color': [15, 15, 40, 255],
                    'layers': [{'kernels': s5_kernels}]})

    # ----------------------------------------------------------
    # 6. CRYSTAL LATTICE — triangle-modulated grid
    # ----------------------------------------------------------
    s6_kernels = []
    for gx in range(40, W, 60):
        for gy in range(40, H, 60):
            s6_kernels.append({
                'x': gx, 'y': gy, 'logic': 'SQUARESUM',
                'modulation': 'TRIANGLE', 'period': 50, 'shift': 15,
                'decay': solid(0.08), 'color': [100, 220, 255], 'alpha': 0.85
            })
    samples.append({'width': W, 'height': H, 'background_color': [5, 10, 30, 255],
                    'layers': [{'kernels': s6_kernels}]})

    # ----------------------------------------------------------
    # 7. OCEAN RIPPLES — RAMP concentric rings across canvas
    # ----------------------------------------------------------
    s7_kernels = [
        {'x': CX, 'y': CY + 60, 'logic': 'SQUARESUM',
         'modulation': 'RAMP', 'period': 28, 'shift': 12,
         'decay': {'east_inward': 0.004, 'east_outward': 0.004,
                   'west_inward': 0.004, 'west_outward': 0.004,
                   'north_inward': 0.01, 'north_outward': 0.01,
                   'south_inward': 0.002, 'south_outward': 0.002},
         'color': [30, 150, 255], 'alpha': 0.9},
        # Water surface tint
        {'x': CX, 'y': H + 100, 'logic': 'MAX', 'decay': solid(0.004),
         'color': [0, 60, 120], 'alpha': 0.6},
        # Sky
        {'x': CX, 'y': -100, 'logic': 'MAX', 'decay': solid(0.004),
         'color': [80, 160, 255], 'alpha': 0.6},
    ]
    samples.append({'width': W, 'height': H, 'background_color': [10, 30, 80, 255],
                    'layers': [{'kernels': s7_kernels}]})

    # ----------------------------------------------------------
    # 8. ABSTRACT FACE — larger, smoother geometric portrait
    # ----------------------------------------------------------
    s8 = {
        'width': W, 'height': H, 'background_color': [18, 18, 28, 255],
        'layers': []
    }
    # Skin base — large warm ellipse
    s8['layers'].append({'kernels': [
        {'x': CX, 'y': CY + 20, 'logic': 'SQUARESUM',
         'decay': {'east_inward': 0.012, 'east_outward': 0.012,
                   'west_inward': 0.012, 'west_outward': 0.012,
                   'north_inward': 0.014, 'north_outward': 0.014,
                   'south_inward': 0.01, 'south_outward': 0.01},
         'color': [220, 170, 120]},
        # Shadow bottom of face
        {'x': CX, 'y': CY + 120, 'logic': 'SQUARESUM', 'decay': solid(0.008),
         'color': [150, 100, 70], 'alpha': 0.6},
        # Highlight forehead
        {'x': CX, 'y': CY - 80, 'logic': 'SQUARESUM', 'decay': solid(0.025),
         'color': [255, 220, 180], 'alpha': 0.4},
    ]})
    # Hair — dark curved cap
    hair_kernels = []
    for dx in range(-120, 130, 18):
        hair_kernels.append({'x': CX + dx, 'y': CY - 80 + int(dx**2 / 400),
                             'logic': 'SQUARESUM', 'decay': solid(0.04),
                             'color': [40, 20, 10]})
    s8['layers'].append({'kernels': hair_kernels})
    # Eyes
    eye_kernels = []
    for ex in [CX - 55, CX + 55]:
        # Eye white
        eye_kernels.append({'x': ex, 'y': CY - 20, 'angle': 10,
                            'logic': 'SQUARESUM', 'max_clamp': 22,
                            'decay': {'east_inward': 0.06, 'east_outward': 0.06,
                                      'west_inward': 0.06, 'west_outward': 0.06,
                                      'north_inward': 0.12, 'north_outward': 0.12,
                                      'south_inward': 0.12, 'south_outward': 0.12},
                            'color': [240, 240, 240]})
        # Iris
        eye_kernels.append({'x': ex, 'y': CY - 18, 'logic': 'MAX', 'max_clamp': 12,
                            'decay': solid(0.0), 'color': [50, 100, 200]})
        # Pupil
        eye_kernels.append({'x': ex, 'y': CY - 18, 'logic': 'MAX', 'max_clamp': 6,
                            'decay': solid(0.0), 'color': [10, 10, 10]})
        # Brow
        for bdx in range(-20, 25, 8):
            eye_kernels.append({'x': ex + bdx, 'y': CY - 50 + int(abs(bdx) * 0.15),
                                'logic': 'MAX', 'max_clamp': 5,
                                'decay': solid(0.0), 'color': [50, 25, 10]})
    s8['layers'].append({'kernels': eye_kernels})
    # Nose + mouth
    s8['layers'].append({'kernels': [
        # Nose shadow
        {'x': CX, 'y': CY + 30, 'logic': 'SQUARESUM',
         'decay': {'east_inward': 0.15, 'east_outward': 0.15,
                   'west_inward': 0.15, 'west_outward': 0.15,
                   'north_inward': 0.04, 'north_outward': 0.04,
                   'south_inward': 0.04, 'south_outward': 0.04},
         'color': [170, 110, 80], 'alpha': 0.7},
        # Top lip
        {'x': CX, 'y': CY + 80, 'angle': 0, 'logic': 'SQUARESUM',
         'decay': {'east_inward': 0.05, 'east_outward': 0.05,
                   'west_inward': 0.05, 'west_outward': 0.05,
                   'north_inward': 0.2, 'north_outward': 0.2,
                   'south_inward': 0.2, 'south_outward': 0.2},
         'color': [180, 70, 70]},
        # Bottom lip
        {'x': CX, 'y': CY + 96, 'logic': 'SQUARESUM',
         'decay': {'east_inward': 0.04, 'east_outward': 0.04,
                   'west_inward': 0.04, 'west_outward': 0.04,
                   'north_inward': 0.18, 'north_outward': 0.18,
                   'south_inward': 0.22, 'south_outward': 0.22},
         'color': [200, 90, 90]},
    ]})
    samples.append(s8)

    # ----------------------------------------------------------
    # 9. RADIAL BURST — 36 rotated rays from center
    # ----------------------------------------------------------
    s9_kernels = []
    burst_cols = [[255, 60, 0], [255, 140, 0], [255, 220, 0], [200, 255, 0], [0, 200, 255]]
    for i in range(36):
        col = burst_cols[i % len(burst_cols)]
        s9_kernels.append({'x': CX, 'y': CY, 'angle': i * 10, 'logic': 'MAX',
                           'decay': {'east_inward': 0.008, 'east_outward': 0.008,
                                     'west_inward': 0.008, 'west_outward': 0.008,
                                     'north_inward': 0.5, 'north_outward': 0.5,
                                     'south_inward': 0.5, 'south_outward': 0.5},
                           'color': col, 'max_clamp': 190, 'alpha': 0.8})
    # Center circle
    s9_kernels.append({'x': CX, 'y': CY, 'logic': 'MAX', 'max_clamp': 18,
                       'decay': solid(0.0), 'color': [255, 255, 255]})
    samples.append({'width': W, 'height': H, 'background_color': [10, 10, 10, 255],
                    'layers': [{'kernels': s9_kernels}]})

    # ----------------------------------------------------------
    # 10. NEON GRID — glowing cross-hair pattern
    # ----------------------------------------------------------
    s10_kernels = []
    for gx in range(50, W, 80):
        for gy in range(50, H, 80):
            hue = ((gx + gy) % 360)
            r_c = int(128 + 127 * math.sin(math.radians(hue)))
            g_c = int(128 + 127 * math.sin(math.radians(hue + 120)))
            b_c = int(128 + 127 * math.sin(math.radians(hue + 240)))
            # Horizontal bar
            s10_kernels.append({'x': gx, 'y': gy, 'angle': 0, 'logic': 'MAX',
                                'decay': {'east_inward': 0.005, 'east_outward': 0.005,
                                          'west_inward': 0.005, 'west_outward': 0.005,
                                          'north_inward': 0.25, 'north_outward': 0.25,
                                          'south_inward': 0.25, 'south_outward': 0.25},
                                'color': [r_c, g_c, b_c], 'alpha': 0.9})
            # Vertical bar
            s10_kernels.append({'x': gx, 'y': gy, 'angle': 90, 'logic': 'MAX',
                                'decay': {'east_inward': 0.005, 'east_outward': 0.005,
                                          'west_inward': 0.005, 'west_outward': 0.005,
                                          'north_inward': 0.25, 'north_outward': 0.25,
                                          'south_inward': 0.25, 'south_outward': 0.25},
                                'color': [r_c, g_c, b_c], 'alpha': 0.9})
            # Glow dot at intersection
            s10_kernels.append({'x': gx, 'y': gy, 'logic': 'SQUARESUM',
                                'decay': solid(0.05), 'color': [255, 255, 255], 'alpha': 0.4})
    samples.append({'width': W, 'height': H, 'background_color': [5, 5, 15, 255],
                    'layers': [{'kernels': s10_kernels}]})

    # ----------------------------------------------------------
    # 11. VORTEX — stacked multi-ring spiral
    # ----------------------------------------------------------
    s11_kernels = []
    vortex_cols = [[200, 0, 255], [0, 200, 255], [255, 0, 120], [255, 200, 0], [0, 255, 150]]
    for i in range(18):
        radius = 18 + i * 10
        angle = i * 20
        col = vortex_cols[i % len(vortex_cols)]
        # Full ring using MAX + min/max clamp + soft glow layer
        s11_kernels.append({'x': CX, 'y': CY, 'angle': angle, 'logic': 'MAX',
                            'min_clamp': radius, 'max_clamp': radius + 12,
                            'decay': solid(0.0),
                            'color': col, 'alpha': 1.0})
    samples.append({'width': W, 'height': H, 'background_color': [5, 0, 15, 255],
                    'layers': [{'kernels': s11_kernels}]})

    # ----------------------------------------------------------
    # 12. ATOM DIAGRAM — nucleus + 3 orbital ellipses
    # ----------------------------------------------------------
    s12_kernels = [
        # Nucleus glow
        {'x': CX, 'y': CY, 'logic': 'SQUARESUM', 'decay': solid(0.04),
         'color': [255, 160, 0], 'alpha': 0.9},
        {'x': CX, 'y': CY, 'logic': 'MAX', 'max_clamp': 14,
         'decay': solid(0.0), 'color': [255, 220, 80]},
    ]
    # 3 orbital rings at different angles
    for orb_angle, orb_col in [(0, [100, 220, 255]), (60, [200, 100, 255]), (120, [100, 255, 180])]:
        s12_kernels.append({'x': CX, 'y': CY, 'angle': orb_angle, 'logic': 'SQUARESUM',
                            'min_clamp': 95, 'max_clamp': 110,
                            'decay': {'east_inward': 0.0, 'east_outward': 0.0,
                                      'west_inward': 0.0, 'west_outward': 0.0,
                                      'north_inward': 0.5, 'north_outward': 0.5,
                                      'south_inward': 0.0, 'south_outward': 0.0},
                            'color': orb_col, 'alpha': 0.9})
        s12_kernels.append({'x': CX, 'y': CY, 'angle': orb_angle, 'logic': 'SQUARESUM',
                            'min_clamp': 95, 'max_clamp': 110,
                            'decay': {'east_inward': 0.0, 'east_outward': 0.0,
                                      'west_inward': 0.0, 'west_outward': 0.0,
                                      'north_inward': 0.0, 'north_outward': 0.0,
                                      'south_inward': 0.5, 'south_outward': 0.5},
                            'color': orb_col, 'alpha': 0.9})
    samples.append({'width': W, 'height': H, 'background_color': [5, 5, 20, 255],
                    'layers': [{'kernels': s12_kernels}]})

    # ----------------------------------------------------------
    # 13. FIRE BLOOM — deep red-orange layered flame
    # ----------------------------------------------------------
    s13_kernels = [
        {'x': CX, 'y': H - 20, 'logic': 'MAX',
         'decay': {'east_inward': 0.006, 'east_outward': 0.006,
                   'west_inward': 0.006, 'west_outward': 0.006,
                   'north_inward': 0.004, 'north_outward': 0.006,
                   'south_inward': 0.01, 'south_outward': 0.01},
         'color': [255, 80, 0], 'alpha': 1.0},
        {'x': CX, 'y': H - 40, 'logic': 'SQUARESUM', 'decay': solid(0.006),
         'color': [255, 160, 0], 'alpha': 0.75},
        {'x': CX, 'y': H - 80, 'logic': 'SQUARESUM', 'decay': solid(0.01),
         'color': [255, 220, 30], 'alpha': 0.5},
        {'x': CX, 'y': H - 120, 'logic': 'SQUARESUM', 'decay': solid(0.018),
         'color': [255, 255, 200], 'alpha': 0.3},
        # Embers left/right
        {'x': CX - 60, 'y': H - 60, 'logic': 'SQUARESUM', 'decay': solid(0.025),
         'color': [255, 120, 0], 'alpha': 0.5},
        {'x': CX + 60, 'y': H - 60, 'logic': 'SQUARESUM', 'decay': solid(0.025),
         'color': [255, 120, 0], 'alpha': 0.5},
    ]
    samples.append({'width': W, 'height': H, 'background_color': [10, 5, 5, 255],
                    'layers': [{'kernels': s13_kernels}]})

    # ----------------------------------------------------------
    # 14. MOUNTAIN LANDSCAPE — layered horizon scene
    # ----------------------------------------------------------
    s14_kernels = [
        # Sky gradient
        {'x': CX, 'y': -50, 'logic': 'MAX', 'decay': solid(0.003),
         'color': [30, 80, 180], 'alpha': 1.0},
        # Moon
        {'x': 80, 'y': 80, 'logic': 'MAX', 'max_clamp': 30,
         'decay': solid(0.0), 'color': [240, 240, 200]},
        # Moon glow
        {'x': 80, 'y': 80, 'logic': 'SQUARESUM', 'decay': solid(0.025),
         'color': [200, 200, 160], 'alpha': 0.35},
        # Stars (tiny bright dots)
        *[{'x': random.randint(0, W), 'y': random.randint(0, int(H * 0.55)),
           'logic': 'MAX', 'max_clamp': random.randint(1, 3),
           'decay': solid(0.0), 'color': [255, 255, 255], 'alpha': random.uniform(0.4, 1.0)}
          for _ in range(40)],
        # Far mountain (blue-grey)
        {'x': CX, 'y': int(H * 0.58), 'logic': 'SQUARESUM',
         'decay': {'east_inward': 0.007, 'east_outward': 0.007,
                   'west_inward': 0.007, 'west_outward': 0.007,
                   'north_inward': 0.012, 'north_outward': 0.012,
                   'south_inward': 0.03, 'south_outward': 0.03},
         'color': [70, 90, 130], 'alpha': 0.95},
        # Near mountain (dark)
        {'x': CX + 60, 'y': int(H * 0.68), 'logic': 'SQUARESUM',
         'decay': {'east_inward': 0.008, 'east_outward': 0.008,
                   'west_inward': 0.008, 'west_outward': 0.008,
                   'north_inward': 0.014, 'north_outward': 0.014,
                   'south_inward': 0.025, 'south_outward': 0.025},
         'color': [30, 40, 60], 'alpha': 1.0},
        # Foreground ground
        {'x': CX, 'y': H + 80, 'logic': 'MAX', 'decay': solid(0.005),
         'color': [15, 25, 40], 'alpha': 1.0},
        # Water reflection
        {'x': CX, 'y': int(H * 0.78), 'logic': 'SQUARESUM',
         'modulation': 'RAMP', 'period': 12, 'shift': 4,
         'decay': {'east_inward': 0.005, 'east_outward': 0.005,
                   'west_inward': 0.005, 'west_outward': 0.005,
                   'north_inward': 0.05, 'north_outward': 0.05,
                   'south_inward': 0.02, 'south_outward': 0.02},
         'color': [60, 100, 180], 'alpha': 0.55},
    ]
    random.seed(99)
    samples.append({'width': W, 'height': H, 'background_color': [10, 15, 35, 255],
                    'layers': [{'kernels': s14_kernels}]})

    # ----------------------------------------------------------
    # 15. PSYCHEDELIC EYE — deep nested alternating rings
    # ----------------------------------------------------------
    s15_kernels = [{'x': CX, 'y': CY, 'logic': 'SQUARESUM',
                    'decay': solid(0.003), 'color': [20, 0, 40], 'alpha': 1.0}]
    ring_palette = [
        [255, 0, 120], [255, 100, 0], [255, 220, 0], [0, 220, 100],
        [0, 180, 255], [120, 0, 255], [255, 255, 255]
    ]
    for i in range(18):
        inner = 8 + i * 10
        outer = inner + 8
        col = ring_palette[i % len(ring_palette)]
        s15_kernels.append({
            'x': CX, 'y': CY, 'logic': 'MAX',
            'min_clamp': inner, 'max_clamp': outer,
            'decay': solid(0.0), 'color': col
        })
    # Pupil
    s15_kernels.append({'x': CX, 'y': CY, 'logic': 'MAX', 'max_clamp': 7,
                        'decay': solid(0.0), 'color': [5, 5, 5]})
    # Radial iris lines
    for i in range(24):
        s15_kernels.append({'x': CX, 'y': CY, 'angle': i * 15, 'logic': 'MAX',
                            'min_clamp': 8, 'max_clamp': 185,
                            'decay': {'east_inward': 0.25, 'east_outward': 0.25,
                                      'west_inward': 0.25, 'west_outward': 0.25,
                                      'north_inward': 0.0, 'north_outward': 0.0,
                                      'south_inward': 0.0, 'south_outward': 0.0},
                            'color': [0, 0, 0], 'alpha': 0.35})
    samples.append({'width': W, 'height': H, 'background_color': [5, 0, 10, 255],
                    'layers': [{'kernels': s15_kernels}]})

    # ----------------------------------------------------------
    # Render all
    # ----------------------------------------------------------
    for idx, scn in enumerate(samples):
        name = f"sample{idx+1:02d}"
        with gzip.open(f"samples/{name}.vkb", "wb") as f:
            f.write(json.dumps(scn).encode('utf-8'))
        t0 = time.time()
        print(f"Rendering {name}...", end=' ', flush=True)
        img_arr = process_vke_scene(scn, scn['width'], scn['height'], scn['background_color'])
        img = Image.fromarray(img_arr, 'RGBA')
        img.save(f"output/{name}.png")
        print(f"done in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    generate_samples()

    os.makedirs('samples', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    tight_decay = {k: 0.1 for k in ['east_inward', 'east_outward', 'west_inward', 'west_outward', 
                                     'north_inward', 'north_outward', 'south_inward', 'south_outward']}
    
    samples = []

    # 1. Circle (solid, clipped to circle shape)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 30, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': {'east_inward': 0, 'east_outward': 0, 'west_inward': 0, 'west_outward': 0,
                      'north_inward': 0, 'north_outward': 0, 'south_inward': 0, 'south_outward': 0},
            'color': [255, 100, 100], 'logic': 'MAX', 'max_clamp': 60
        }]}]
    })

    # 2. Donut (using min_clamp to create hole)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 30, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': {'east_inward': 0, 'east_outward': 0, 'west_inward': 0, 'west_outward': 0,
                      'north_inward': 0, 'north_outward': 0, 'south_inward': 0, 'south_outward': 0},
            'color': [100, 255, 100], 'logic': 'MAX', 'min_clamp': 40, 'max_clamp': 80
        }]}]
    })

    # 3. Half Circle (uses max clamp for circle, bounds for cut)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 30, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': {'east_inward': 0, 'east_outward': 0, 'west_inward': 0, 'west_outward': 0,
                      'north_inward': 0, 'north_outward': 0, 'south_inward': 0, 'south_outward': 0},
            'color': [100, 100, 255], 'logic': 'MAX', 'max_clamp': 80,
            'bounds': [0, -100, 100, 100] # Right half only (local_x >= 0)
        }]}]
    })

    # 4. Moon (Crescent - donut clipped)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 30, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': {'east_inward': 0, 'east_outward': 0, 'west_inward': 0, 'west_outward': 0,
                      'north_inward': 0, 'north_outward': 0, 'south_inward': 0, 'south_outward': 0},
            'color': [255, 255, 100], 'logic': 'MAX', 'min_clamp': 40, 'max_clamp': 80,
            'bounds': [-100, 0, 100, 100] # Bottom half of donut
        }]}]
    })

    # 5. Rectangle
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 30, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'angle': 45, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': tight_decay, 'color': [255, 100, 255], 'logic': 'MIN'
        }]}]
    })

    # 6. Ellipse (Stretched by different decays)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 30, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': {'east_inward': 0.02, 'east_outward': 0.02, 'west_inward': 0.02, 'west_outward': 0.02,
                      'north_inward': 0.1, 'north_outward': 0.1, 'south_inward': 0.1, 'south_outward': 0.1},
            'color': [100, 255, 255], 'logic': 'SQUARESUM'
        }]}]
    })

    # 7. Line
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 30, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'angle': 30, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': {'east_inward': 0.01, 'east_outward': 0.01, 'west_inward': 0.01, 'west_outward': 0.01,
                      'north_inward': 0.5, 'north_outward': 0.5, 'south_inward': 0.5, 'south_outward': 0.5},
            'color': [255, 255, 255], 'logic': 'MIN'
        }]}]
    })

    # 8. Gradient Over Image
    samples.append({
        'width': 200, 'height': 200, 'background_color': [0, 0, 0, 255],
        'layers': [{'kernels': [
            {
                'x': 0, 'y': 0, 'modulation': 'CONTINUOUS', 'shift': 0,
                'decay': {k: 0.005 for k in tight_decay.keys()},
                'color': [255, 50, 50], 'logic': 'MAX', 'alpha': 0.8
            },
            {
                'x': 200, 'y': 200, 'modulation': 'CONTINUOUS', 'shift': 0,
                'decay': {k: 0.005 for k in tight_decay.keys()},
                'color': [50, 50, 255], 'logic': 'MAX', 'alpha': 0.8
            }
        ]}]
    })

    # 9. Organic Cells (Parabolic)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [10, 40, 20, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'PARABOLIC', 'period': 40, 'shift': 0, 'parabola_scale': 15,
            'decay': {k: 0.05 for k in tight_decay.keys()}, 'color': [100, 200, 255], 'logic': 'MULTIPLY'
        }]}]
    })

    # 10. Ripples (RAMP)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [255, 255, 255, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'RAMP', 'period': 20, 'shift': 10,
            'decay': {k: 0.05 for k in tight_decay.keys()}, 'color': [50, 50, 50], 'logic': 'MIN'
        }]}]
    })

    # 11. Waves (TRIANGLE)
    samples.append({
        'width': 200, 'height': 200, 'background_color': [10, 10, 10, 255],
        'layers': [{'kernels': [{
            'x': 100, 'y': 100, 'modulation': 'TRIANGLE', 'period': 30, 'shift': 0,
            'decay': {k: 0.02 for k in tight_decay.keys()}, 'color': [255, 100, 100], 'logic': 'SQUARESUM'
        }]}]
    })

    # 12. Star
    star_layer = {'kernels': []}
    for i in range(4):
        star_layer['kernels'].append({
            'x': 100, 'y': 100, 'angle': i * 45, 'modulation': 'CONTINUOUS', 'shift': 0,
            'decay': {'east_inward': 0.02, 'east_outward': 0.02, 'west_inward': 0.02, 'west_outward': 0.02,
                      'north_inward': 0.2, 'north_outward': 0.2, 'south_inward': 0.2, 'south_outward': 0.2},
            'color': [255, 255, 0], 'logic': 'MAX', 'alpha': 0.8
        })
    samples.append({
        'width': 200, 'height': 200, 'background_color': [20, 20, 50, 255],
        'layers': [star_layer]
    })

    # 13. Grid / Templates
    samples.append({
        'width': 200, 'height': 200, 'background_color': [0, 0, 0, 255],
        'templates': {
            'dot': {
                'modulation': 'CONTINUOUS', 'shift': 5,
                'decay': {k: 0.2 for k in tight_decay.keys()}, 'logic': 'SQUARESUM', 'color': [200, 200, 200]
            }
        },
        'layers': [{'groups': [
            {'x': cx, 'y': cy, 'kernels': [{'template': 'dot'}]}
            for cx in range(20, 200, 40) for cy in range(20, 200, 40)
        ]}]
    })

    # 14. Fractal (Sample 5 original)
    complex_layer = {'kernels': []}
    for i in range(5):
        complex_layer['kernels'].append({
            'x': 100, 'y': 100, 'angle': i * 15,
            'modulation': 'PARABOLIC', 'period': 60, 'shift': i*5, 'parabola_scale': 10,
            'color': [50 * i, 255 - 40 * i, 150 + 20 * i], 
            'decay': {k: 0.05 for k in tight_decay.keys()},
            'logic': 'SQUARESUM', 'alpha': 0.5
        })
    samples.append({
        'width': 200, 'height': 200, 'background_color': [10, 10, 10, 255],
        'layers': [complex_layer]
    })

    # 15. Face / Object Art Piece
    face_layers = []
    
    # 15. Face / Object Art Piece (Dotted Art Style)
    face_layers = []
    
    # We will use templates for the "dots"
    # A dot is just a small circle kernel
    # By using templates, we save space.
    templates = {
        'dot_base': {
            'modulation': 'CONTINUOUS', 'shift': 0, 'logic': 'MAX', 
            'decay': {'east_inward': 0, 'east_outward': 0, 'west_inward': 0, 'west_outward': 0,
                      'north_inward': 0, 'north_outward': 0, 'south_inward': 0, 'south_outward': 0},
            'max_clamp': 4
        },
        'dot_skin': { 'template': 'dot_base', 'color': [240, 200, 160] },
        'dot_shadow': { 'template': 'dot_base', 'color': [180, 130, 100] },
        'dot_eye_white': { 'template': 'dot_base', 'color': [255, 255, 255] },
        'dot_pupil': { 'template': 'dot_base', 'color': [20, 20, 20] },
        'dot_lip': { 'template': 'dot_base', 'color': [200, 80, 80] },
        'dot_hair': { 'template': 'dot_base', 'color': [50, 20, 10] }
    }

    # Generate a dotted circle for the face base
    face_dots = []
    
    # helper to add a dot
    def add_dot(x, y, temp):
        face_dots.append({'x': x, 'y': y, 'template': temp})

    # Hair Outline
    for r in range(120, 160, 8):
        for angle in range(-40, 220, 5):
            rad = math.radians(angle)
            add_dot(200 + r * math.cos(rad), 200 - r * math.sin(rad), 'dot_hair')

    # Face Base (filled circle with dots)
    for r in range(0, 100, 8):
        steps = int(r * 2) if r > 0 else 1
        for i in range(steps):
            angle = i * (2 * math.pi / max(1, steps))
            x, y = 200 + r * math.cos(angle), 200 + r * math.sin(angle)
            
            # Create a shadow gradient effect towards the edges and bottom
            if r > 80 or (y > 240 and r > 60):
                add_dot(x, y, 'dot_shadow')
            else:
                add_dot(x, y, 'dot_skin')

    # Eyes
    for ex in [160, 240]:
        # Eye white
        for dx, dy in [(-8, 0), (0, -4), (0, 4), (8, 0), (-4, -2), (4, -2), (-4, 2), (4, 2)]:
            add_dot(ex + dx, 170 + dy, 'dot_eye_white')
        
        # Pupil
        add_dot(ex, 170, 'dot_pupil')
        
        # Eyebrow
        for dx in range(-15, 20, 6):
            add_dot(ex + dx, 150 - (5 if abs(dx) < 10 else 0), 'dot_hair')

    # Nose
    for dy in range(180, 220, 8):
        add_dot(200, dy, 'dot_shadow')
    add_dot(192, 215, 'dot_shadow')
    add_dot(208, 215, 'dot_shadow')

    # Mouth
    for dx in range(-20, 25, 6):
        # curve the smile
        y = 250 + (abs(dx) * 0.2)
        add_dot(200 + dx, y, 'dot_lip')
        if abs(dx) < 15:
            add_dot(200 + dx, y + 6, 'dot_lip')

    samples.append({
        'width': 400, 'height': 400, 'background_color': [20, 20, 30, 255],
        'templates': templates,
        'layers': [{'groups': [{'kernels': face_dots}]}]
    })

    for idx, scn in enumerate(samples):
        name = f"sample{idx+1:02d}"
        # Write as gzip-compressed .vkb (JSON payload, fallback-compatible with render.py)
        with gzip.open(f"samples/{name}.vkb", "wb") as f:
            f.write(json.dumps(scn).encode('utf-8'))
        
        t0 = time.time()
        print(f"Rendering {name}...")
        img_arr = process_vke_scene(scn, scn['width'], scn['height'], scn['background_color'])
        img = Image.fromarray(img_arr, 'RGBA')
        img.save(f"output/{name}.png")
        print(f"Done {name} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    generate_samples()
