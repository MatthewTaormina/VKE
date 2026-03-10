# VKB Binary Format Specification

**Version:** `VKB1`  
**Extension:** `.vkb`  
**Compression:** gzip (RFC 1952)

---

## Overview

The **Vector Kernel Binary (VKB)** format is a compact binary encoding for Vector Kernel Engine (VKE) scenes. It stores the geometric and colour data for every Influence Kernel in a fixed-size record, compressed with gzip, achieving extreme compression ratios compared to equivalent JSON.

The format has **two sub-types** distinguished by a magic header:

| Sub-type | Magic | Description |
|----------|-------|-------------|
| **Struct VKB** | `VKB1` | Pure binary struct records (vectorizer output) |
| **JSON VKB**   | _(any other)_ | gzip-compressed JSON scene (hand-crafted scenes) |

Both are stored with a `.vkb` extension and loaded identically by `render.py`.

---

## Sub-type 1: Struct VKB (`VKB1`)

Used by `vectorizer.py` when converting a photo or image to a compact binary scene.

### File Layout

```
[FILE HEADER — 15 bytes]
[KERNEL RECORDS — 20 bytes × kernel_count]
```

All multi-byte integers are **little-endian**.

---

### File Header (15 bytes)

| Offset | Size | Type     | Field           | Description |
|--------|------|----------|-----------------|-------------|
| 0      | 4    | `char[4]`| `magic`         | Always `VKB1` (ASCII). Identifies Struct VKB format. |
| 4      | 2    | `uint16` | `width`         | Canvas width in pixels. |
| 6      | 2    | `uint16` | `height`        | Canvas height in pixels. |
| 8      | 1    | `uint8`  | `bg_r`          | Background colour — red channel (0–255). |
| 9      | 1    | `uint8`  | `bg_g`          | Background colour — green channel (0–255). |
| 10     | 1    | `uint8`  | `bg_b`          | Background colour — blue channel (0–255). |
| 11     | 4    | `uint32` | `kernel_count`  | Total number of kernel records that follow. |

**Python struct format:** `'<4sHHBBBI'` (15 bytes total)

---

### Kernel Record (20 bytes)

One record per kernel, packed sequentially immediately after the header.

| Offset | Size | Type      | Field     | Description |
|--------|------|-----------|-----------|-------------|
| 0      | 4    | `float32` | `x`       | Kernel centre X position in canvas pixels. |
| 4      | 4    | `float32` | `y`       | Kernel centre Y position in canvas pixels. |
| 8      | 4    | `float32` | `half_w`  | Half-width of the kernel's quadtree cell in pixels. |
| 12     | 4    | `float32` | `half_h`  | Half-height of the kernel's quadtree cell in pixels. |
| 16     | 1    | `uint8`   | `r`       | Kernel colour — red channel (0–255). |
| 17     | 1    | `uint8`   | `g`       | Kernel colour — green channel (0–255). |
| 18     | 1    | `uint8`   | `b`       | Kernel colour — blue channel (0–255). |
| 19     | 1    | `uint8`   | `theta`   | Kernel orientation angle, mapped: `angle_deg = (theta / 255.0) × 360.0`. |

**Python struct format:** `'<ffffBBBB'` (20 bytes total)

> **Note:** `float32` is used for geometry to guarantee sub-pixel accuracy across canvas sizes up to 65 535 × 65 535 pixels (the `uint16` canvas dimension limit).

---

### Rendering Contract

When `render.py` reads a Struct VKB file it reconstructs each kernel as:

```python
{
    'x':      float(x),
    'y':      float(y),
    'color':  [int(r), int(g), int(b)],
    'bounds': [-half_w, -half_h, half_w, half_h],  # axis-aligned clip
    'decay':  { all 8 directions: 0.0 }             # solid fill, no gradient
}
```

- **`bounds`** clips the kernel exactly to its quadtree cell — ensures zero-gap coverage.
- **Zero decay** keeps `weight = 1.0` everywhere inside the cell — no dark centre dot.
- The `theta` field is stored but currently unused in the axis-aligned rendering path (reserved for future brush-stroke rendering modes).

---

### Annotated Hex Example

A 3-kernel file for a 10×10 canvas with white background:

```
Offset  Hex Bytes                                       Description
------  -----------------------------------------------  -------------------
00      56 4B 42 31                                      magic = "VKB1"
04      0A 00                                            width = 10
06      0A 00                                            height = 10
08      FF FF FF                                         bg = [255, 255, 255]
0B      03 00 00 00                                      kernel_count = 3

-- Kernel 0 --
0F      00 00 20 41                                      x = 10.0
13      00 00 20 41                                      y = 10.0
17      00 00 80 40                                      half_w = 4.0
1B      00 00 80 40                                      half_h = 4.0
1F      FF 00 00                                         rgb = [255, 0, 0]
22      5A                                               theta = 90° → 90/360*255 ≈ 64
... (kernels 1 and 2 follow identically)
```

---

## Sub-type 2: JSON VKB (gzip-compressed JSON)

Used for hand-crafted `.vkb` scenes (sample scenes, artworks). The file is simply the JSON text of a full VKE scene, gzip-compressed, with no `VKB1` magic header.

### Detection

`render.py` reads the first 4 bytes. If they are **not** `VKB1`, it seeks back to position 0 and decodes the entire gzip stream as UTF-8 JSON.

### JSON Schema

The JSON payload follows the full VKE scene schema:

```jsonc
{
  "width": 400,
  "height": 400,
  "background_color": [20, 20, 30, 255],

  // Optional: reusable kernel parameter sets
  "templates": {
    "ring": {
      "modulation": "RAMP",
      "period": 40,
      "shift": 20,
      "logic": "SQUARESUM"
    }
  },

  // Sequential rendering layers
  "layers": [
    {
      "kernels": [
        {
          "template": "ring",     // Optional: inherit from a template
          "x": 200, "y": 200,
          "color": [80, 160, 255],
          "angle": 0,             // Rotation in degrees
          "alpha": 1.0,           // Opacity (0.0–1.0)
          "modulation": "RAMP",   // CONTINUOUS | RAMP | TRIANGLE | PARABOLIC
          "period": 40,
          "shift": 20,
          "min_clamp": 0,         // Minimum visible distance
          "max_clamp": 200,       // Maximum visible distance
          "clamp_decay": 0.4,     // Soft-edge slope for min/max clamp (0 = hard mask)
          "logic": "SQUARESUM",   // MULTIPLY | MAX | MIN | SQUARESUM
          "decay": {
            "east_inward":  0.05, "east_outward":  0.05,
            "west_inward":  0.05, "west_outward":  0.05,
            "north_inward": 0.05, "north_outward": 0.05,
            "south_inward": 0.05, "south_outward": 0.05
          },
          "bounds": [-50, -50, 50, 50]  // Local [x1, y1, x2, y2] clip rect
        }
      ],

      // Optional: groups of kernels with a shared offset
      "groups": [
        {
          "x": 0, "y": 0, "angle": 0,
          "kernels": [ /* ... */ ]
        }
      ]
    }
  ]
}
```

#### Kernel Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `x`, `y` | number | canvas centre | Centre position in pixels |
| `color` | `[R, G, B]` | `[255,255,255]` | RGB fill colour |
| `angle` | number | `0` | Rotation in degrees (counter-clockwise) |
| `alpha` | number | `1.0` | Blending opacity |
| `modulation` | string | `"CONTINUOUS"` | Distance modulation function |
| `period` | number | `50` | Modulation period in pixels |
| `shift` | number | `0` | Distance zero-point offset |
| `min_clamp` | number | `0` | Minimum visible euclidean distance |
| `max_clamp` | number | `∞` | Maximum visible euclidean distance |
| `clamp_decay` | number | `0` | Soft-edge slope for the clamp boundaries. When `> 0`, replaces the hard binary mask with a smooth linear fade: the inner edge fades in over `1/clamp_decay` pixels from `min_clamp`, and the outer edge fades out over `1/clamp_decay` pixels from `max_clamp`. The two fades are multiplied together to produce a smooth "bubble" cross-section. When `0` (default), hard mask behaviour is preserved. |
| `logic` | string | `"MULTIPLY"` | X/Y weight combination metric |
| `decay` | object | `0.05` all | 8 directional decay coefficients |
| `bounds` | `[x1,y1,x2,y2]` | none | Local bounding-box clip |
| `template` | string | none | Name of template to inherit from |

---

## Compression

Both sub-types are wrapped in **gzip** (Python `gzip.open(..., 'wb')`). The gzip header itself is not part of the VKB specification — it is transparently handled by the Python `gzip` module.

**Typical compression ratios:**

| Content | Raw Size | `.vkb` Size | Ratio |
|---------|----------|-------------|-------|
| 1024×1024 photo (24 523 kernels, struct) | ~14 MB JSON | ~120 KB | 99.1% |
| Hand-crafted face scene (JSON) | 220 KB | ~42 KB | 81% |
| Simple 3-kernel scene (JSON) | 800 B | ~280 B | 65% |

---

## Versioning

The magic string `VKB1` encodes the version number as the last character. Future breaking format changes should use `VKB2`, `VKB3`, etc. The JSON VKB sub-type has no version field — the JSON schema itself is the versioned contract.

---

## Reading a VKB File (Python Pseudocode)

```python
import gzip, struct, json

with gzip.open('scene.vkb', 'rb') as f:
    magic = f.read(4)
    if magic == b'VKB1':
        # Struct VKB
        w, h, bg_r, bg_g, bg_b, k_count = struct.unpack('<HHBBBI', f.read(11))
        kernels = []
        for _ in range(k_count):
            x, y, hw, hh, r, g, b, theta = struct.unpack('<ffffBBBB', f.read(20))
            kernels.append({'x': x, 'y': y, 'color': [r, g, b],
                            'half_w': hw, 'half_h': hh, 'theta': theta})
    else:
        # JSON VKB — seek back to start of gzip payload
        f.seek(0)
        scene = json.loads(f.read().decode('utf-8'))
```
