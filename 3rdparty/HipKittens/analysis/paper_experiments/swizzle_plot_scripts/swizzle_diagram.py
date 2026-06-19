#!/usr/bin/env python3
"""
swizzle_diagram.py
Create colored table diagrams for memory swizzles.

- Default: 16 columns × 16 rows. Each cell has a distinct color based on its
  row-major index modulo 64, so colors repeat starting from the 64th cell.
  Labels show the same 0..63 index and repeat.
- You can pass a column permutation or a callable mapping to customize labels.

Usage examples:
    python swizzle_diagram.py                            # saves swizzle_16x16.png
    python swizzle_diagram.py --rows 16 --cols 16 --out my.png
    # With a column permutation (CSV string):
    python swizzle_diagram.py --perm "0,1,2,3,4,5,6,7,8,9,10,11,...,15"
    # With a simple XOR swizzle demo:
    python swizzle_diagram.py --demo-xor

Requires: numpy, matplotlib
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

def save_swizzle_diagram(filename, rows=16, cols=16, mapping=None, dpi=300,
                         darken=None, darken_scale=0.6, palette="turbo",
                         label_fontsize=16, secondary_darken=None,
                         side_by_side=False, cell_aspect=1.0):
    """Save a swizzle diagram image.

    Args:
        filename (str): Output file path (PNG recommended).
        rows (int): Number of rows (default 16).
        cols (int): Number of columns (default 16).
        mapping: None (labels = 0..cols-1), a sequence of length `cols` giving
                 per-column labels, or a callable f(r, c) -> int returning the
                 label to draw in each cell.
        dpi (int): Figure DPI.
        darken: Optional specification of cells to darken. One of:
                - 2D boolean array of shape (rows, cols)
                - Iterable of (row, col) integer pairs
                - Callable f(r, c) -> bool
        darken_scale (float): Multiplier in [0, 1] applied to RGB of darkened
                              cells (e.g., 0.6 makes them 40% darker).
    """

    # Per-cell coloring: 64 distinct colors repeating by row-major index
    repeat_period = 64
    num_colors = repeat_period

    cmap = cm.get_cmap(palette)
    colors = cmap(np.linspace(0.02, 0.98, num_colors))[:, :3]
    colors = 0.10 + 0.90 * colors

    linear = np.arange(rows * cols).reshape(rows, cols)
    color_idx = (linear % repeat_period).astype(int)
    rgb_base = colors[color_idx]

    # Labels
    if mapping is None:
        labels_base = (linear % 64).astype(int)
    elif callable(mapping):
        labels_base = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                labels_base[r, c] = int(mapping(r, c))
    else:
        mp = np.array(mapping, dtype=int)
        if mp.shape[0] != cols:
            raise ValueError("mapping length must equal number of columns")
        labels_base = np.tile(mp, (rows, 1))

    # Darken selected cells if requested
    img_left = rgb_base.copy()
    labels_left = labels_base
    if darken is not None:
        if callable(darken):
            mask = np.zeros((rows, cols), dtype=bool)
            for r in range(rows):
                for c in range(cols):
                    mask[r, c] = bool(darken(r, c))
        else:
            arr = np.array(darken, dtype=object)
            if arr.dtype == bool and arr.shape == (rows, cols):
                mask = arr
            else:
                mask = np.zeros((rows, cols), dtype=bool)
                for pair in darken:
                    rr, cc = int(pair[0]), int(pair[1])
                    if 0 <= rr < rows and 0 <= cc < cols:
                        mask[rr, cc] = True
        img_left[mask] = np.clip(img_left[mask] * float(darken_scale), 0.0, 1.0)

    if side_by_side and secondary_darken is not None:
        img_right = rgb_base.copy()
        labels_right = labels_base
        if callable(secondary_darken):
            mask2 = np.zeros((rows, cols), dtype=bool)
            for r in range(rows):
                for c in range(cols):
                    mask2[r, c] = bool(secondary_darken(r, c))
        else:
            arr2 = np.array(secondary_darken, dtype=object)
            if arr2.dtype == bool and arr2.shape == (rows, cols):
                mask2 = arr2
            else:
                mask2 = np.zeros((rows, cols), dtype=bool)
                for pair in secondary_darken:
                    rr, cc = int(pair[0]), int(pair[1])
                    if 0 <= rr < rows and 0 <= cc < cols:
                        mask2[rr, cc] = True
        img_right[mask2] = np.clip(img_right[mask2] * float(darken_scale), 0.0, 1.0)
        # Prepare for separate subplot rendering
        render_cols = cols
    else:
        rgb_img = img_left
        labels = labels_left
        render_cols = cols

    # (flip logic removed per user request)

    # Draw
    fig_w, fig_h = 20, 6
    if side_by_side and secondary_darken is not None:
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)
        panels = [
            (axes[0], img_left, labels_left),
            (axes[1], img_right, labels_right),
        ]
        for ax, panel_img, panel_labels in panels:
            ax.imshow(panel_img, interpolation='nearest')
            ax.set_aspect(cell_aspect)
            for r in range(rows):
                for c in range(cols):
                    ax.text(c, r, f"{panel_labels[r, c]:02d}", ha='center', va='center',
                            fontsize=label_fontsize, color='black')
            ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
            ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
        fig.tight_layout(pad=0.5)
        fig.subplots_adjust(wspace=0.1)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(rgb_img, interpolation='nearest')
        ax.set_aspect(cell_aspect)
        for r in range(rows):
            for c in range(render_cols):
                ax.text(c, r, f"{labels[r, c]:02d}", ha='center', va='center',
                        fontsize=label_fontsize, color='black')
        ax.set_xticks(np.arange(-.5, render_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        fig.tight_layout(pad=0.5)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser(description="Generate memory swizzle diagrams.")
    p.add_argument("--rows", type=int, default=16, help="Number of rows (default: 16)")
    p.add_argument("--cols", type=int, default=16, help="Number of columns (default: 16)")
    p.add_argument("--out", type=str, default="swizzle_16x16.png", help="Output filename (PNG)")
    p.add_argument("--perm", type=str, default=None,
                   help="Comma-separated list of per-column labels (length == cols)")
    p.add_argument("--demo-xor", action="store_true",
                   help="Use a demo XOR swizzle (labels = c ^ 0x10)")
    p.add_argument("--dark", type=str, default=None,
                   help="Cells to darken; formats: 'r:c;r:c' or 'all:even' etc.")
    p.add_argument("--dark-scale", type=float, default=0.6,
                   help="RGB multiplier for darkened cells (0..1, default 0.6)")
    p.add_argument("--palette", type=str, default="turbo",
                   help="Matplotlib palette name (e.g., turbo, viridis, plasma)")
    p.add_argument("--font-size", type=int, default=16,
                   help="Font size for cell labels (default: 16)")
    p.add_argument("--compare", action="store_true",
                   help="Render side-by-side: default mask (left) vs swizzled (right)")
    p.add_argument("--cell-aspect", type=float, default=1.0,
                   help="Height/width ratio per cell; >1 makes cells taller")
    return p.parse_args()

def main():
    args = parse_args()

    mapping = None
    if args.demo_xor:
        # Demonstration: labels are column XOR 0x10 (adjust as desired)
        def f(r, c): return c ^ 0x10
        mapping = f
    elif args.perm:
        # Parse a CSV list into int labels
        mapping = [int(x.strip()) for x in args.perm.split(",")]
        if len(mapping) != args.cols:
            raise SystemExit(f"--perm must have exactly {args.cols} entries")

    default = lambda r, c: (
        (r < 4 and c < 4) or               # first 4 rows, first 4 cols
        (r >= 12 and c < 4) or             # last 4 rows, first 4 cols (rows=16 → 12..15)
        (4 <= r <= 11 and 4 <= c <= 7)    # rows 5..12, cols 5..7
    )
    swizzled = lambda r, c: (
        (r < 4 and c < 4) or
        (4 <= r < 8 and 4 <= c < 8) or
        (8 <= r < 12 and 12 <= c) or
        (r >= 12 and 8 <= c < 12)
    )

    darken = None
    secondary = None
    side_by_side = False

    if args.compare:
        darken = default
        secondary = swizzled
        side_by_side = True
    elif args.dark:
        spec = args.dark.strip()
        if spec.lower() in ("all", "*", "true"):
            darken = lambda r, c: True
        elif spec.lower() in ("even-cols", "all:even-cols"):
            darken = lambda r, c: (c % 2) == 0
        elif spec.lower() in ("odd-cols", "all:odd-cols"):
            darken = lambda r, c: (c % 2) == 1
        elif spec.lower() in ("even-rows", "all:even-rows"):
            darken = lambda r, c: (r % 2) == 0
        elif spec.lower() in ("odd-rows", "all:odd-rows"):
            darken = lambda r, c: (r % 2) == 1
        else:
            pairs = []
            for token in spec.split(";"):
                token = token.strip()
                if not token:
                    continue
                if ":" not in token:
                    raise SystemExit("--dark entries must be 'row:col' pairs or a known pattern")
                r_str, c_str = token.split(":", 1)
                pairs.append((int(r_str), int(c_str)))
            darken = pairs

    save_swizzle_diagram(
        args.out,
        rows=args.rows,
        cols=args.cols,
        mapping=mapping,
        dpi=300,
        darken=darken,
        darken_scale=args.dark_scale,
        palette=args.palette,
        label_fontsize=args.font_size,
        cell_aspect=args.cell_aspect,
        secondary_darken=secondary,
        side_by_side=side_by_side,
    )
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
