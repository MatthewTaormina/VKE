# VKE — Vector Kernel Engine

A high-performance Python renderer that constructs images entirely out of mathematical **Influence Kernels** defined in a compact binary scene format (`.vkb`). No raster grids, no SVG paths — pure kernel mathematics.

---

## What Is a Kernel?

Each **Influence Kernel** defines a region of influence centred on a point `(x, y)`. The distance from that centre is:
1. **Modulated** into repeating waves (`CONTINUOUS`, `RAMP`, `TRIANGLE`, `PARABOLIC`)
2. **Shifted** to offset the peak
3. **Decayed** independently in each cardinal direction (N / S / E / W, inward / outward)
4. **Combined** across X and Y axes using a logic metric (`MULTIPLY`, `MAX`, `MIN`, `SQUARESUM`)
5. **Clipped** against an optional local bounding box

The resulting per-pixel weight blends the kernel's colour onto the image buffer.

---

## Repository Structure

```
VKE/
├── main.py            # Core rendering engine (numpy-vectorized)
├── render.py          # CLI: load .vkb or .json → PNG
├── vectorizer.py      # CLI: photo/image → .vkb binary scene
├── samples/           # 15 hand-crafted .json scene files
├── output/            # 15 rendered sample PNGs
├── test_input.png     # Small synthetic test image (200×200)
├── test_out_1.vkb     # Vectorized binary of test_input.png
├── test_out_1.png     # Rendered output of test_out_1.vkb
├── 6.png              # Portrait photograph source
├── test_out_3.vkb     # Vectorized binary of 6.png (~116 KB for 1024×1024)
├── test_out_3.png     # Rendered output of test_out_3.vkb
├── VKB_FORMAT_SPEC.md # Formal specification of the .vkb binary format
└── Vector Kernel Specification.md  # Original VKE engine specification
```

---

## Quick Start

**Install dependencies:**
```bash
pip install numpy pillow
```

**Render all 15 sample scenes:**
```bash
python main.py
```
Outputs render to `output/`.

**Render a specific scene:**
```bash
python render.py samples/sample12.json output/sample12.png
python render.py test_out_3.vkb test_out_3.png
```

**Vectorize a photo into a compact `.vkb` binary:**
```bash
python vectorizer.py 6.png output.vkb --block 7 --max_block 14
python render.py output.vkb output.png
```

---

## CLI Reference

### `vectorizer.py` — Image → VKB

```
python vectorizer.py <input_image> <output.vkb> [options]

Arguments:
  input_image       Path to source PNG or JPG
  output.vkb        Output binary scene file (.vkb) or JSON (.json)

Options:
  --block N         Minimum quadtree cell size in pixels (default: 8)
  --max_block N     Maximum quadtree cell size in pixels (default: 64)
```

The vectorizer:
1. Detects the background colour from border pixels and sets it as the scene background
2. Subdivides the image via an adaptive quadtree — large flat areas use big cells, detailed edges use small cells
3. Skips cells whose colour matches the background (subject isolation)
4. Packs every kernel into the compact 20-byte `.vkb` binary format (gzip-compressed)

**Typical results:** A 1024×1024 photo with ~24 000 kernels compresses to ~120 KB.

### `render.py` — VKB/JSON → PNG

```
python render.py <input.vkb|input.json> <output.png>
```

Handles both `.vkb` binary files and `.json` scene files transparently.

---

## Scene Format: `.vke.json`

Human-readable JSON for hand-crafted scenes and debugging:

```json
{
  "width": 400, "height": 400,
  "background_color": [20, 20, 30, 255],
  "templates": {
    "ring": { "modulation": "RAMP", "period": 40, "shift": 20 }
  },
  "layers": [
    {
      "kernels": [
        {
          "template": "ring",
          "x": 200, "y": 200,
          "color": [80, 160, 255],
          "decay": { "east_inward": 0.02, "west_inward": 0.02,
                     "north_inward": 0.02, "south_inward": 0.02,
                     "east_outward": 0.05, "west_outward": 0.05,
                     "north_outward": 0.05, "south_outward": 0.05 },
          "logic": "SQUARESUM"
        }
      ]
    }
  ]
}
```

## Binary Format: `.vkb`

A compact gzip-compressed binary format. A 1024×1024 image with 24 000 kernels fits in **~120 KB** (vs. ~14 MB for equivalent JSON).

See **[VKB_FORMAT_SPEC.md](VKB_FORMAT_SPEC.md)** for the complete field-by-field specification.

---

## Sample Scenes

| # | Scene | Key Features |
|---|-------|-------------|
| 01 | Eclipse | `SQUARESUM` logic, corona + moon disc |
| 02 | Plasma Web | `clamp_decay` bubble merging, weighted-average blending |
| 03 | Organic Cells | `PARABOLIC` modulation tessellation |
| 04 | Gradient Ring | `RAMP` modulation, `SQUARESUM` circular logic |
| 05 | Layered Rings | `min_clamp`/`max_clamp` annulus shaping |
| 06 | Sunset | Horizontal gradient layers, asymmetric decay |
| 07 | Wave Interference | 3D bevelled ripples, `clamp_decay` soft circles, multi-source interference |
| 08 | Portrait | Hand-drawn face from clipped kernels |
| 09 | Fractal | Deep abstract layered geometry |
| 10 | Grid | `templates` + `groups` scatter |
| 11 | Star | Four rotated ellipses at offsets |
| 12 | Waves | `TRIANGLE` frequency modulation |
| 13 | Ellipse | Independent N/S vs E/W decay |
| 14 | Moon | Clipped shifted ring |
| 15 | Portrait (Hi-Res) | Complex face from many layered kernels |

---

## Key Features

- **Adaptive Quadtree Vectorization** — large flat areas → few big cells; fine edges → many small cells
- **Zero-gap Binary Tiling** — solid-fill rectangular kernels perfectly tile with axis-aligned bounds
- **Subject Isolation** — background-matching cells are automatically culled from the binary output
- **Extreme Compression** — 20 bytes per kernel + gzip; typical photos at < 0.1% of raw raster size
- **Full JSON Scene Language** — modulators, shifts, 8-way decay, logic metrics, layers, groups, templates
- **`clamp_decay`** — replaces hard `min_clamp`/`max_clamp` binary masks with a smooth euclidean fade, enabling circles and rings to merge organically like bubbles where their edges meet
- **Weighted-Average Compositor** — kernels within a layer blend via `Σ(color × weight) / Σ(weight)`, producing order-independent, physically correct overlaps and constructive wave interference
