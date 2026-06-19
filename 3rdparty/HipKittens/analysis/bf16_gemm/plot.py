import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B", "#55555A"]


mi355x_baselines = {
    "triton": {
        "1024": 59.378,
        "2048": 270.523,
        "4096": 934.393,
        "8192": 1090.526,
        "16384": 1086.317,
    },
    "hipblaslt": {
        "1024": 167.903,
        "2048": 639.608,
        "4096": 1432.70,
        "8192": 1574.69,
        "16384": 1579.20,
    },
    "ck": {
        "1024": 188.799,
        "2048": 262.674,
        "4096": 930.504,
        "8192": 504.206,
        "16384": 477.777,
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


for device in ['mi355x']:

    # Read data
    try:
        with open(f'mi350x/{device}_bf16_gemm.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading mi350x/{device}_bf16_gemm.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    pytorch_tflops = [data[str(size)]['tflops_pytorch'] for size in matrix_sizes]
    aiter_tflops = [data[str(size)]['tflops_aiter'] for size in matrix_sizes]
    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

    triton_tflops = []
    # rocblas_tflops = []
    ck_tflops = []
    hipblaslt_tflops = []
    if device == 'mi355x':
        triton_tflops = [mi355x_baselines['triton'][str(size)] for size in matrix_sizes]
        # rocblas_tflops = [mi355x_baselines['rocblas'][str(size)] for size in matrix_sizes]
        ck_tflops = [mi355x_baselines['ck'][str(size)] for size in matrix_sizes]
        hipblaslt_tflops = [mi355x_baselines['hipblaslt'][str(size)] for size in matrix_sizes]
    elif device == 'mi350x':
        triton_tflops = [mi350x_baselines['triton'][str(size)] for size in matrix_sizes]
        ck_tflops = [mi350x_baselines['ck'][str(size)] for size in matrix_sizes]
        hipblaslt_tflops = [mi350x_baselines['hipblaslt'][str(size)] for size in matrix_sizes]
        # rocblas_tflops = [mi350x_baselines['rocblas'][str(size)] for size in matrix_sizes]

    # Process data to separate OOM values
    # pytorch_vals, pytorch_oom = process_data(pytorch_tflops)
    aiter_vals, aiter_oom = process_data(aiter_tflops)
    hipblaslt_vals, hipblaslt_oom = process_data(hipblaslt_tflops)
    tk_vals, tk_oom = process_data(tk_tflops)
    triton_vals, triton_oom = process_data(triton_tflops) if triton_tflops else ([], [])
    # rocblas_vals, rocblas_oom = process_data(rocblas_tflops) if rocblas_tflops else ([], [])
    ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])

    max_tflops = max(max(aiter_vals), max(hipblaslt_vals), max(tk_vals), max(triton_vals), max(ck_vals))

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 6))
    first_bar = x - 3*width
    second_bar = x - 2*width
    third_bar = x - width
    fourth_bar = x
    fifth_bar = x + width
    sixth_bar = x + 2*width
    # bars0 = ax.bar(first_bar, pytorch_vals, width, label='PyTorch', color=colors[4])
    bars1 = ax.bar(fourth_bar, aiter_vals, width, label='AITER (ASM)', color=colors[0])
    bars2 = ax.bar(third_bar, hipblaslt_vals, width, label='HipblasLT', color=colors[1])
    bars3 = ax.bar(fifth_bar, tk_vals, width, label='HipKittens', color=colors[3])
    bars4 = ax.bar(first_bar, triton_vals, width, label='Triton', color=colors[2])
    bars5 = ax.bar(second_bar, ck_vals, width, label='Composable Kernel', color=colors[4])


    fontsize = 10
    
    # Plot X markers for OOM
    oom_height = max_tflops * 0.95

    # for idx in pytorch_oom:
    #     ax.plot(x[idx] - 3*width, oom_height, 'x', color=colors[0], markersize=15, markeredgewidth=3)
    #     ax.text(x[idx] - 3*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[0])

    for idx in aiter_oom:
        ax.plot(x[idx] - 2*width, oom_height, 'x', color=colors[1], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - 2*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=colors[1])

    for idx in hipblaslt_oom:
        ax.plot(x[idx] - width, oom_height, 'x', color=colors[2], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=colors[2])

    for idx in tk_oom:
        ax.plot(x[idx], oom_height, 'x', color=colors[3], markersize=15, markeredgewidth=3)
        ax.text(x[idx], oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=colors[3])

    for idx in triton_oom:
        ax.plot(x[idx] + width, oom_height, 'x', color=colors[4], markersize=15, markeredgewidth=3)
        ax.text(x[idx] + width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=colors[4])

    for idx in ck_oom:
        ax.plot(x[idx] + 2*width, oom_height, 'x', color=colors[5], markersize=15, markeredgewidth=3)
        ax.text(x[idx] + 2*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=fontsize, color=colors[5])

    # Add value labels on bars
        # for bar, value in zip(bars0, pytorch_vals):
        #     if value > 0:
        #         height = bar.get_height()
        #         ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
        #                 f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars1, aiter_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars2, hipblaslt_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars3, tk_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars4, triton_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars5, ck_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'BF16 GEMM Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # Order legend to match bar order (left to right): Triton, Composable Kernel, HipblasLT, AITER (ASM), HipKittens
    ax.legend([bars4, bars5, bars2, bars1, bars3], 
              ['Triton', 'Composable Kernel', 'HipblasLT', 'AITER (ASM)', 'HipKittens'],
              fontsize=14)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_bf16_gemm_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    # print(f"PyTorch TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in pytorch_vals]}")
    print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in aiter_vals]}")
    print(f"HipblasLT TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in hipblaslt_vals]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in tk_vals]}")
    print(f"Triton TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in triton_vals]}")
    # print(f"rocBLAS TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in rocblas_vals]}")
    print(f"Composable Kernel TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in ck_vals]}")