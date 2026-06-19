#!/usr/bin/env python3
"""
table_values_64x16.py
Render a 64×16 table where each cell displays a provided value.

Features:
- Robust CSV/TSV parsing (ignores blank lines, trims trailing delimiters)
- Optional blank token/number to leave cells empty
- Optional row-block shading using custom colors
- Optional targeted cell darkening
- Optional swizzle-like color backgrounds (pair-by-2) with multiple modes

Usage:
    python3 table_values_64x16.py --values values.csv              # saves table_64x16.png
    python3 table_values_64x16.py --values vals.txt --delimiter ' ' --out my.png
    python3 table_values_64x16.py --values values.csv --swizzle-colors --swizzle-mode by-col
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from typing import Optional


def save_value_table(
    filename: str,
    values_path: str,
    rows: int = 16,
    cols: int = 64,
    delimiter: str = ",",
    font_size: int = 16,
    dpi: int = 300,
    cell_aspect: float = 1.0,
    blank_token: Optional[str] = None,
    blank_number: Optional[float] = None,
    shade: bool = False,
    shade_colors: Optional[list] = None,
    shade_size: int = 4,
    darken: Optional[list] = None,
    darken_scale: float = 0.6,
    swizzle_colors: bool = False,
    swizzle_palette: str = "turbo",
    swizzle_mode: str = "by-col",
) -> None:
    # Load values from file (robust to trailing delimiters and blank lines)
    try:
        with open(values_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        raise SystemExit(f"Failed to read values file '{values_path}': {e}")

    data = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue  # skip completely blank lines
        parts = [p.strip() for p in line.split(delimiter)]
        # Trim trailing empty fields (e.g., lines ending with a delimiter)
        while parts and parts[-1] == "":
            parts.pop()
        if not parts:
            continue
        row_vals = []
        for j, token in enumerate(parts):
            if token == "":
                raise SystemExit(f"Empty value at row {len(data)}, column {j}")
            # Check for blank placeholders
            is_blank = False
            if blank_token is not None and token == str(blank_token):
                is_blank = True
            if not is_blank and blank_number is not None and token == str(blank_number):
                is_blank = True
            if not is_blank and token.lower() == "nan":
                is_blank = True
            if is_blank:
                row_vals.append(float("nan"))
                continue
            try:
                v = float(token)
            except ValueError as ve:
                raise SystemExit(
                    f"Invalid number '{token}' at row {len(data)}, column {j}"
                ) from ve
            row_vals.append(v)
        data.append(row_vals)

    if len(data) != rows:
        raise SystemExit(f"Values row count {len(data)} does not match expected {rows}")
    for i, row_vals in enumerate(data):
        if len(row_vals) != cols:
            raise SystemExit(
                f"Row {i} has {len(row_vals)} columns; expected {cols}."
            )
    vals = np.array(data, dtype=float)

    # Base image colors
    if swizzle_colors:
        # Swizzle-like palette: every 2 cells share a color
        cmap = plt.cm.get_cmap(swizzle_palette)
        if swizzle_mode == "by-col":
            # Pair by columns within each row (c//2)
            num_colors = max(1, cols // 2)
            colors = cmap(np.linspace(0.02, 0.98, num_colors))[:, :3]
            colors = 0.10 + 0.90 * colors
            idx_cols = (np.arange(cols) // 2) % num_colors
            row_pattern = colors[idx_cols]  # (cols, 3)
            img = np.tile(row_pattern[None, :, :], (rows, 1, 1))
        elif swizzle_mode == "linear":
            # linear mode: pair by row-major linear index ((r*cols+c)//2)
            repeat_period = 64
            num_colors = max(1, repeat_period // 2)
            colors = cmap(np.linspace(0.02, 0.98, num_colors))[:, :3]
            colors = 0.10 + 0.90 * colors
            linear = np.arange(rows * cols).reshape(rows, cols)
            idx = ((linear % repeat_period) // 2).astype(int)
            img = colors[idx]
        else:  # row-linear: pairs within each row, colors advance across rows
            repeat_period = 64
            num_colors = max(1, repeat_period // 2)
            colors = cmap(np.linspace(0.02, 0.98, num_colors))[:, :3]
            colors = 0.10 + 0.90 * colors
            r_idx = np.arange(rows)[:, None]
            c_idx = np.arange(cols)[None, :]
            pairs_per_row = max(1, cols // 2)
            pair_idx = (r_idx * pairs_per_row + (c_idx // 2)) % num_colors
            img = colors[pair_idx]
    else:
        # White background with optional row-block shading
        img = np.ones((rows, cols, 3), dtype=float)
        if shade:
            default_hex = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]
            palette = shade_colors if shade_colors else default_hex
            rgb_list = [to_rgb(h) for h in palette]
            for r in range(rows):
                block = (r // max(1, shade_size)) % len(rgb_list)
                img[r, :, :] = rgb_list[block]

    # Darken specific cells if requested
    if darken:
        mask = np.zeros((rows, cols), dtype=bool)
        try:
            for rr, cc in darken:
                r_i, c_i = int(rr), int(cc)
                if 0 <= r_i < rows and 0 <= c_i < cols:
                    mask[r_i, c_i] = True
        except Exception:
            raise SystemExit("--dark must be a list of (row,col) pairs")
        img[mask] = np.clip(img[mask] * float(darken_scale), 0.0, 1.0)

    # Draw
    fig_w, fig_h = 20, 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img, interpolation='nearest')
    ax.set_aspect(cell_aspect)

    # Draw values as text
    for r in range(rows):
        for c in range(cols):
            v = vals[r, c]
            if np.isnan(v):
                continue  # leave cell blank
            # Render integers without .0, zero-pad single digits to two chars
            if float(v).is_integer():
                vi = int(v)
                label = f"{vi:02d}" if 0 <= vi < 10 else f"{vi}"
            else:
                label = f"{v:g}"
            ax.text(c, r, label, ha='center', va='center', fontsize=font_size, color='black')

    # Grid & axes aesthetics
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
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
    p = argparse.ArgumentParser(description="Render a 64x16 value-labeled grid image.")
    p.add_argument("--values", type=str, required=True, help="Path to values file (text/CSV)")
    p.add_argument("--rows", type=int, default=16, help="Number of rows (default: 16)")
    p.add_argument("--cols", type=int, default=64, help="Number of columns (default: 64)")
    p.add_argument("--delimiter", type=str, default=",", help="Column delimiter in file (default ',')")
    p.add_argument("--font-size", type=int, default=16, help="Font size for labels (default: 16)")
    p.add_argument("--cell-aspect", type=float, default=1.0, help="Height/width ratio per cell (default: 1.0)")
    p.add_argument("--out", type=str, default="table_64x16.png", help="Output filename (PNG)")
    p.add_argument("--blank-token", type=str, default="-", help="Token that renders a cell blank (default '-')")
    p.add_argument("--blank-number", type=float, default=None, help="Numeric value that renders a cell blank (optional)")
    p.add_argument("--shade", action="store_true", help="Shade rows in blocks using specified colors")
    p.add_argument("--shade-colors", type=str, default="#8E69B8,#E59952,#68AC5A,#7CB9BC",
                   help="Comma-separated hex colors for shading blocks (default preset)")
    p.add_argument("--shade-size", type=int, default=4, help="Number of rows per shaded block (default: 4)")
    p.add_argument("--dark", type=str, default=None,
                   help="Semicolon-separated list of row:col pairs to darken, e.g., '0:0;1:5' ")
    p.add_argument("--dark-scale", "--darken-scale", type=float, default=0.6,
                   help="RGB multiplier for darkened cells (0..1)")
    p.add_argument("--swizzle-colors", action="store_true",
                   help="Use swizzle-like palette background; every two cells share a color")
    p.add_argument("--swizzle-palette", type=str, default="turbo",
                   help="Colormap for swizzle colors (e.g., turbo, viridis)")
    p.add_argument("--swizzle-mode", type=str, default="by-col", choices=["by-col", "linear", "row-linear"],
                   help="Pair colors by columns (by-col), row-major (linear), or row-linear")
    return p.parse_args()


def main():
    args = parse_args()
    dark_pairs = [tuple(map(int, p.split(":"))) for p in args.dark.split(";") if p.strip()] if args.dark else None
    shade_palette = [s.strip() for s in args.shade_colors.split(",")] if args.shade_colors else None
    save_value_table(
        filename=args.out,
        values_path=args.values,
        rows=args.rows,
        cols=args.cols,
        delimiter=args.delimiter,
        font_size=args.font_size,
        cell_aspect=args.cell_aspect,
        blank_token=args.blank_token,
        blank_number=args.blank_number,
        shade=args.shade,
        shade_colors=shade_palette,
        shade_size=args.shade_size,
        darken=dark_pairs,
        darken_scale=args.dark_scale,
        swizzle_colors=args.swizzle_colors,
        swizzle_palette=args.swizzle_palette,
        swizzle_mode=args.swizzle_mode,
        dpi=300,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()


