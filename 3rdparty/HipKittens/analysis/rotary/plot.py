import json
import matplotlib.pyplot as plt
import numpy as np


colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B"]

for device in ['mi300x', 'mi325x', 'mi350x', 'mi355x']:

    # Read data
    try:
        with open(f'mi350x/{device}_rotary.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_rotary.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    pytorch_tflops = [data[str(size)]['tflops_pytorch'] for size in matrix_sizes]
    compiled_tflops = [data[str(size)]['tflops_pytorch_compiled'] for size in matrix_sizes]
    aiter_tflops = [data[str(size)]['tflops_aiter'] for size in matrix_sizes]
    tk_tflops = [data[str(size)]['tflops_tk'] for size in matrix_sizes]

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars0 = ax.bar(x - width, pytorch_tflops, width, label='PyTorch', color=colors[4])
    bars1 = ax.bar(x, compiled_tflops, width, label='Compiled PyTorch', color=colors[2])
    bars2 = ax.bar(x + width, aiter_tflops, width, label='AITER (ASM)', color=colors[0])
    bars3 = ax.bar(x + 2*width, tk_tflops, width, label='HipKittens', color=colors[3])

    max_tflops = max(max(pytorch_tflops), max(compiled_tflops), max(aiter_tflops), max(tk_tflops))

    # Add value labels on bars
    for bar, value in zip(bars0, pytorch_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars1, compiled_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars2, aiter_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars3, tk_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Sequence Length', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'Rotary Embedding Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16, loc='upper left')
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_rotary_plot.png'
    #save high quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"PyTorch TFLOPS: {[f'{t:.2f}' for t in pytorch_tflops]}")
    print(f"Compiled PyTorch TFLOPS: {[f'{t:.2f}' for t in compiled_tflops]}")
    print(f"AITER TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
    print(f"TK TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

