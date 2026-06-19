import json
import matplotlib.pyplot as plt
import numpy as np



mi355x_gqa_baselines_causal = {
    "ck": {
        "1024": 542.83,
        "2048": 669.05,
        "4096": 735.92,
        "8192": 783.53,
        "16384": 804.24,
    },
    "triton": {
        "1024": 355.077782,
        "2048": 450.211863,
        "4096": 491.634949,
        "8192": 655.760443,
        "16384": 745.645985,
    },
    "torch": {
        "1024": 226.41,
        "2048": 383.09,
        "4096": 509.33,
        "8192": 609.27,
        "16384": 666.56,
    }
}

mi355x_gqa_baselines_non_causal = {
    "ck": {
        "1024": 726.69,
        "2048": 777.92,
        "4096": 795.13,
        "8192": 805.35,
        "16384": 806.48,
    },
    "triton": {
        "1024": 735.516728,
        "2048": 802.428351,
        "4096": 857.724312,
        "8192": 865.103837,
        "16384": 861.238370,
    },
    "torch": {
        "1024": 522.29,
        "2048": 668.73,
        "4096": 680.48,
        "8192": 686.72,
        "16384": 704.23,
    }
}

# B = 16, H = 16, D = 128.
mi355x_mha_baselines_causal = {
    "ck": {
        "1024": 418.61,
        "2048": 601.80,
        "4096": 675.84,
        "8192": 755.79,
        "16384": 799.67,
    },
    "triton": {
        "1024": 284.631187,
        "2048": 372.463081,
        "4096": 444.170321,
        "8192": 609.744767,
        "16384": 697.219085,
    },
    "torch": {
        "1024": 226.41,
        "2048": 383.09,
        "4096": 509.33,
        "8192": 609.27,
        "16384": 666.56,
    }
}

# B = 16, H = 16, D = 128.
mi355x_mha_baselines_non_causal = {
    "ck": {
        "1024": 670.53,
        "2048": 766.45,
        "4096": 771.84,
        "8192": 762.47,
        "16384": 821.81,
    },
     "triton": {
        "1024": 603.598816,
        "2048": 731.640900,
        "4096": 817.375312,
        "8192": 853.588397,
        "16384": 865.080935,
    },
    "torch": {
        "1024": 522.29,
        "2048": 668.73,
        "4096": 680.48,
        "8192": 686.72,
        "16384": 704.23,
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

    for setting in ['d64_mha_causal_fwd', 'd64_mha_non_causal_fwd', 'd64_gqa_causal_fwd', 'd64_gqa_non_causal_fwd']:

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

        torch_tflops = []
        ck_tflops = []
        triton_tflops = []
        if setting == 'd64_mha_causal_fwd' and device == 'mi355x':
            torch_tflops = [mi355x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
            triton_tflops = [mi355x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
        elif setting == 'd64_mha_non_causal_fwd' and device == 'mi355x':
            torch_tflops = [mi355x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
            triton_tflops = [mi355x_mha_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
        elif setting == 'd64_gqa_causal_fwd' and device == 'mi355x':
            torch_tflops = [mi355x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
            triton_tflops = [mi355x_gqa_baselines_causal['triton'][str(size)] for size in matrix_sizes]
        elif setting == 'd64_gqa_non_causal_fwd' and device == 'mi355x':
            torch_tflops = [mi355x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
            triton_tflops = [mi355x_gqa_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]

        # Process data to separate OOM values
        torch_vals, torch_oom = process_data(torch_tflops) if torch_tflops else ([], [])
        ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])
        triton_vals, triton_oom = process_data(triton_tflops) if triton_tflops else ([], [])

        # Calculate max for numeric values only
        numeric_vals = aiter_tflops + tk_tflops
        if torch_vals:
            numeric_vals.extend([v for v in torch_vals if v != 0])
        if ck_vals:
            numeric_vals.extend([v for v in ck_vals if v != 0])
        if triton_vals:
            numeric_vals.extend([v for v in triton_vals if v != 0])
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

        for bar, value in zip(bars2, triton_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        for bar, value in zip(bars3, torch_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)
                        
        for bar, value in zip(bars4, ck_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        # Parse setting name for title
        setting_parts = setting.split('_')
        attn_type = setting_parts[1].upper()  # MHA or GQA
        causal_mode = 'Causal' if 'causal' in setting and 'non_causal' not in setting else 'Non-Causal'

        # add some padding to the top of the y-axis to prevent label overlap
        ax.set_ylim(0, max_tflops * 1.15)
        ax.set_xlabel('Sequence Length', fontsize=16)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
        ax.set_title(f'(D = 64) {attn_type} {causal_mode} Forward Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_sizes, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        # Order legend to match bar order (left to right): PyTorch SDPA, Triton, Composable Kernel, AITER (ASM), HipKittens
        ax.legend([bars3, bars2, bars4, bars0, bars1], 
                  ['PyTorch SDPA', 'Triton', 'Composable Kernel', 'AITER (ASM)', 'HipKittens'],
                  fontsize=14)

        plt.tight_layout()
        plt.show()

        output_file = f'{device}_{setting}_attn_d64_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        # Compute max and min win for HK over AITER
        tk_aiter_wins = [tk_tflops[i] / aiter_tflops[i] for i in range(len(matrix_sizes))]
        tk_torch_wins = [tk_tflops[i] / torch_vals[i] for i in range(len(matrix_sizes))]
        tk_ck_wins = [tk_tflops[i] / ck_vals[i] for i in range(len(matrix_sizes))]
        print(f"Max win for HK over AITER: {max(tk_aiter_wins):.2f}, Min: {min(tk_aiter_wins):.2f}")
        print(f"Max win for HK over PyTorch: {max(tk_torch_wins):.2f}, Min: {min(tk_torch_wins):.2f}")
        print(f"Max win for HK over Composable Kernel: {max(tk_ck_wins):.2f}, Min: {min(tk_ck_wins):.2f}\n")