import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B", "#55555A"]

# Color mapping to match BF16 plot
color_map = {
    "cutlass": "#E59952",      # Orange (same as BF16)
    "hipkittens": "#7CB9BC",     # Light Blue/Teal (same as BF16) 
    "composable_kernel": "#DE836B"  # Coral/Light Red (same as BF16)
}


baselines = {
    "ck": {
        "1024": 584,
        "2048": 1227,
        "4096": 1181,
        "8192": 1312,
        "16384": 1355,
    },
    "cutlass": {
        "1024": 261.42,
        "2048": 1336.32,
        "4096": 2757.18,
        "8192": 3038.77,
        "16384": 2524.63,
    }
}


def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi350x', 'mi355x']:

    # Read data
    try:
        with open(f'{device}_fp6_gemm.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_fp6_gemm.json: {e}")
        continue

    # Extract data for plotting - only 8192 and 16384
    matrix_sizes = [8192, 16384]
    tk_tflops = [data[str(size)] for size in matrix_sizes if str(size) in data]
    cutlass_tflops = [data[str(size)] for size in matrix_sizes if str(size) in data]

    ck_tflops = []
    if device == 'mi355x':
        ck_tflops = [baselines['ck'][str(size)] for size in matrix_sizes]
        cutlass_tflops = [baselines['cutlass'][str(size)] for size in matrix_sizes]

    # Process data to separate OOM values
    cutlass_vals, cutlass_oom = process_data(cutlass_tflops)
    tk_vals, tk_oom = process_data(tk_tflops)
    ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])

    max_tflops = max(max(cutlass_vals), max(tk_vals), max(ck_vals))

    # Create bar chart
    x = np.arange(len(matrix_sizes)) # Reduce spacing between clusters
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 6))
    first_bar = x - width
    second_bar = x
    third_bar = x + width
    bars2 = ax.bar(second_bar, cutlass_vals, width, label='CUTLASS (B200)', color=color_map["cutlass"])
    bars3 = ax.bar(third_bar, tk_vals, width, label='HipKittens', color=color_map["hipkittens"])
    bars5 = ax.bar(first_bar, ck_vals, width, label='Composable Kernel', color=color_map["composable_kernel"])

    # Plot X markers for OOM
    oom_height = 120
    markersize = 18
    markeredgewidth = 3
    fontsize = 13

    for idx in cutlass_oom:
        ax.plot(x[idx] - width, oom_height, 'x', color=color_map["cutlass"], markersize=markersize, markeredgewidth=markeredgewidth)
        ax.text(x[idx] - width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=color_map["cutlass"])

    for idx in tk_oom:
        ax.plot(x[idx], oom_height, 'x', color=color_map["hipkittens"], markersize=markersize, markeredgewidth=markeredgewidth)
        ax.text(x[idx], oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=color_map["hipkittens"])

    for idx in ck_oom:
        ax.plot(x[idx] - width, oom_height, 'x', color=color_map["composable_kernel"], markersize=markersize, markeredgewidth=markeredgewidth)
        ax.text(x[idx] - width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=color_map["composable_kernel"])

    # Add value labels on bars
    for bar, value in zip(bars2, cutlass_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars3, tk_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars5, ck_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    # Set y-axis with more spaced out ticks
    ax.set_ylim(0, max_tflops * 1.15)
    # Create more spaced out y-axis ticks (every 200 TFLOPS instead of default)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'FP6 GEMM Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # Order legend to match bar order (left to right): Composable Kernel, CUTLASS, HipKittens
    ax.legend([bars5, bars2, bars3], 
              ['Composable Kernel', 'CUTLASS (B200)', 'HipKittens'],
              fontsize=13, loc='upper center', bbox_to_anchor=(0.6, 1))

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_fp6_gemm_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"CUTLASS TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in cutlass_vals]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in tk_vals]}")
    print(f"Composable Kernel TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in ck_vals]}")
