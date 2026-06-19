import json
import matplotlib.pyplot as plt
import numpy as np


# B = 16, H = 64, HK8, D = 128.
mi355x_gqa_baselines_causal = {
    "triton": {
        "1024": 156.883795,
        "2048": 248.866051,
        "4096": 367.821190,
        "8192": 593.577963,
        "16384": 793.799216,
    },
    "ck": {
        "1024": 603.19,
        "2048": 701.25,
        "4096": 813.36,
        "8192": 865.59,
        "16384": 888.64,
    },
    "torch": {
        "1024": 158.805348,
        "2048": 353.431751,
        "4096": 499.219058,
        "8192": 705.247090,
        "16384": 789.283990,
    }
}

mi355x_gqa_baselines_non_causal = {
    "triton": {
        "1024": 827.544143,
        "2048": 905.804361,
        "4096": 967.666252,
        "8192": 993.507993,
        "16384":989.718846,
    },
    "ck": {
        "1024": 787.99,
        "2048": 826.05,
        "4096": 876.55,
        "8192": 895.94,
        "16384": 896.47,
    },
    "torch": {
        "1024": 468.169993,
        "2048": 625.294982,
        "4096": 610.161908,
        "8192": 841.134461,
        "16384": 878.907987,
    }
}

#**************************************#
#**************************************#
#**************************************#


# B = 16, H = 16, D = 128.
mi355x_mha_baselines_causal = {
    "triton": {
        "1024": 333.074017,
        "2048": 435.433397,
        "4096": 481.786402,
        "8192": 653.114342,
        "16384": 749.760119,
    },
    "ck": {
        "1024": 494.69,
        "2048": 610.53,
        "4096": 754.87,
        "8192": 843.54,
        "16384": 902.62,
    },
    "torch": {
        "1024": 54.832384,
        "2048": 246.376588,
        "4096": 421.775249,
        "8192": 630.386987,
        "16384": 761.564515,
    }
}

# B = 16, H = 16, D = 128.
mi355x_mha_baselines_non_causal = {
    "triton": {
        "1024": 686.733941,
        "2048": 839.167226,
        "4096": 923.082293,
        "8192": 973.608837,
        "16384": 987.687075,
    },
    "ck": {
        "1024": 761.47,
        "2048": 733.48,
        "4096": 810.28,
        "8192": 893.89,
        "16384": 908.30,
    },
    "torch": {
        "1024": 452.941180,
        "2048": 579.431411,
        "4096": 593.913331,
        "8192": 821.786381,
        "16384": 872.787694,
    }
}

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B"]


def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)  # Use 0 for bar height
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi355x']:

    for setting in ['mha_causal_fwd', 'mha_non_causal_fwd', 'gqa_causal_fwd', 'gqa_non_causal_fwd']:

        # Read data
        try:
            with open(f'benchmark/{device}_{setting}.json', 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {device}_{setting}.json: {e}")
            continue

        # Extract data for plotting
        matrix_sizes = sorted([int(size) for size in data.keys()])
        aiter_tflops = [data[str(size)]['tflops_ref'] for size in matrix_sizes]
        tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

        triton_tflops = []
        torch_tflops = []
        ck_tflops = []
        if setting == 'mha_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_gqa_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_gqa_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
            
        # Process data to separate OOM values
        triton_vals, triton_oom = process_data(triton_tflops) if triton_tflops else ([], [])
        torch_vals, torch_oom = process_data(torch_tflops) if torch_tflops else ([], [])
        ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])

        # Calculate max for numeric values only
        numeric_vals = aiter_tflops + tk_tflops
        if triton_vals:
            numeric_vals.extend([v for v in triton_vals if v != 0])
        if torch_vals:
            numeric_vals.extend([v for v in torch_vals if v != 0])
        if ck_vals:
            numeric_vals.extend([v for v in ck_vals if v != 0])
        max_tflops = max(numeric_vals) if numeric_vals else 100

        # Create bar chart
        x = np.arange(len(matrix_sizes))
        width = 0.19

        fig, ax = plt.subplots(figsize=(10, 6))
        first_bar_start = x - 2*width
        second_bar_start = x - width
        third_bar_start = x
        fourth_bar_start = x + width
        fifth_bar_start = x + 2*width
        bars0 = ax.bar(fourth_bar_start, aiter_tflops, width, label='AITER (ASM)', color=colors[0])
        bars1 = ax.bar(fifth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])
        bars2 = ax.bar(second_bar_start, triton_vals, width, label='Triton', color=colors[2])
        bars3 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[1])
        bars4 = ax.bar(third_bar_start, ck_vals, width, label='Composable Kernel', color=colors[4])


        fontsize = 11
        # Plot X markers for OOM
        oom_height = 35  # Position X near top of chart
        if torch_oom:
            for idx in torch_oom:
                ax.plot(x[idx] -  2*width, oom_height, 'x', color=colors[1], 
                       markersize=13, markeredgewidth=3)
                ax.text(x[idx] -  2*width, oom_height + max_tflops * 0.03,
                       'OOM', ha='center', va='bottom', fontsize=9, color=colors[1])

        # Add value labels on bars
        for bar, value in zip(bars0, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        for bar, value in zip(bars1, tk_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        if len(triton_vals) > 0:
            for bar, value in zip(bars2, triton_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)
        
        if len(torch_vals) > 0:
            for i, (bar, value) in enumerate(zip(bars3, torch_vals)):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)
                        
        if len(ck_vals) > 0:
            for bar, value in zip(bars4, ck_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        # Parse setting name for title
        setting_parts = setting.split('_')
        attn_type = setting_parts[0].upper()  # MHA or GQA
        causal_mode = 'Causal' if 'causal' in setting and 'non_causal' not in setting else 'Non-Causal'

        # add some padding to the top of the y-axis to prevent label overlap
        ax.set_ylim(0, max_tflops * 1.15)
        ax.set_xlabel('Sequence Length', fontsize=16)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
        ax.set_title(f'(D=128) {attn_type} {causal_mode} Forward Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_sizes, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        # Order legend to match bar order (left to right): PyTorch SDPA, Triton, Composable Kernel, AITER (ASM), HipKittens
        ax.legend([bars3, bars2, bars4, bars0, bars1], 
                  ['PyTorch SDPA', 'Triton', 'Composable Kernel', 'AITER (ASM)', 'HipKittens'],
                  fontsize=14)

        plt.tight_layout()
        plt.show()

        output_file = f'{device}_{setting}_attn_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        
        # Compute max and min win for HK over AITER
        tk_aiter_wins = [tk_tflops[i] / aiter_tflops[i] for i in range(len(matrix_sizes))]
        tk_torch_wins = [tk_tflops[i] / torch_vals[i] for i in range(len(matrix_sizes))]
        tk_ck_wins = [tk_tflops[i] / ck_vals[i] for i in range(len(matrix_sizes))]
        tk_triton_wins = [tk_tflops[i] / triton_vals[i] for i in range(len(matrix_sizes))]
        print(f"Max win for HK over AITER: {max(tk_aiter_wins):.2f}, Min: {min(tk_aiter_wins):.2f}, Avg: {np.mean(tk_aiter_wins):.2f}")
        print(f"Max win for HK over PyTorch: {max(tk_torch_wins):.2f}, Min: {min(tk_torch_wins):.2f}, Avg: {np.mean(tk_torch_wins):.2f}")
        print(f"Max win for HK over Composable Kernel: {max(tk_ck_wins):.2f}, Min: {min(tk_ck_wins):.2f}, Avg: {np.mean(tk_ck_wins):.2f}")
        print(f"Max win for HK over Triton: {max(tk_triton_wins):.2f}, Min: {min(tk_triton_wins):.2f}, Avg: {np.mean(tk_triton_wins):.2f}\n")



