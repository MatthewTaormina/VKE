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
