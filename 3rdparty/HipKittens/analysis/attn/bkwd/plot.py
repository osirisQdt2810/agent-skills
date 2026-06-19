import json
import matplotlib.pyplot as plt
import numpy as np


def flops(batch, seqlen, nheads, headdim, causal, mode="bwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(time, flop):
    """Calculate efficiency in TFLOPS given time in ms and flop count in FLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time

# Note that CK computes the flops differently than the other frameworks, so we instead use their reported wall clock time and adjust for the consistent flop count.
# See: https://github.com/ROCm/composable_kernel/blob/9f33b7cfd3df3fcfd540f7633b0abd7019935761/example/ck_tile/01_fmha/fmha_bwd_runner.hpp#L212

# MHA baselines (B=16, H=16, D=128)
mi355x_mha_baselines_causal = {
    "triton": {
        "1024": 147,
        "2048": 179,
        "4096": 215,
        "8192": 237,
        "16384": 251,
    },
    "ck": {
        "1024": efficiency(0.666,flops(batch=16, seqlen=1024, nheads=16, headdim=128, causal=True, mode="bwd")),
        "2048": efficiency(1.833,flops(batch=16, seqlen=2048, nheads=16, headdim=128, causal=True, mode="bwd")),
        "4096": efficiency(6.032,flops(batch=16, seqlen=4096, nheads=16, headdim=128, causal=True, mode="bwd")),
        "8192": efficiency(22.922,flops(batch=16, seqlen=8192, nheads=16, headdim=128, causal=True, mode="bwd")),
        "16384": efficiency(81.486,flops(batch=16, seqlen=16384, nheads=16, headdim=128, causal=True, mode="bwd")),
    },
    "torch": {
        "1024": 109.51,
        "2048": 156.71,
        "4096": 142.82,
        "8192": 224.01,
        "16384": 259.14,
    },
}

mi355x_mha_baselines_non_causal = { 
    # triton not available for non-causal bwd attn
    "ck": {
        "1024": efficiency(0.942,flops(batch=16, seqlen=1024, nheads=16, headdim=128, causal=False, mode="bwd")),
        "2048": efficiency(3.214,flops(batch=16, seqlen=2048, nheads=16, headdim=128, causal=False, mode="bwd")),
        "4096": efficiency(12.345,flops(batch=16, seqlen=4096, nheads=16, headdim=128, causal=False, mode="bwd")),
        "8192": efficiency(46.844,flops(batch=16, seqlen=8192, nheads=16, headdim=128, causal=False, mode="bwd")),
        "16384": efficiency(186.751,flops(batch=16, seqlen=16384, nheads=16, headdim=128, causal=False, mode="bwd")),
    },
    "torch": {
        "1024": 220.27,
        "2048": 273.06,
        "4096": 301.24,
        "8192": 309.30,
        "16384": 311.66,
    },
}

# GQA baselines (B=16, Q_HEADS=64, KV_HEADS=8, D=128)
mi355x_gqa_baselines_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": efficiency(2.524, flops(batch=16, seqlen=1024, nheads=64, headdim=128, causal=True, mode="bwd")),
        "2048": efficiency(6.807, flops(batch=16, seqlen=2048, nheads=64, headdim=128, causal=True, mode="bwd")),
        "4096": efficiency(23.290, flops(batch=16, seqlen=4096, nheads=64, headdim=128, causal=True, mode="bwd")),
        "8192": efficiency(89.926, flops(batch=16, seqlen=8192, nheads=64, headdim=128, causal=True, mode="bwd")),
        "16384": efficiency(325.060, flops(batch=16, seqlen=16384, nheads=64, headdim=128, causal=True, mode="bwd")),
    },
    "torch": {
        "1024": 109.51,
        "2048": 156.71,
        "4096": 142.82,
        "8192": 224.01,
        "16384": 259.14,
    },
}

mi355x_gqa_baselines_non_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": efficiency(3.737,flops(batch=16, seqlen=1024, nheads=64, headdim=128, causal=False, mode="bwd")),
        "2048": efficiency(12.845,flops(batch=16, seqlen=2048, nheads=64, headdim=128, causal=False, mode="bwd")),
        "4096": efficiency(49.213,flops(batch=16, seqlen=4096, nheads=64, headdim=128, causal=False, mode="bwd")),
        "8192": efficiency(190.167,flops(batch=16, seqlen=8192, nheads=64, headdim=128, causal=False, mode="bwd")),
        "16384": efficiency(757.375,flops(batch=16, seqlen=16384, nheads=64, headdim=128, causal=False, mode="bwd")),
    },
    "torch": {
        "1024": 220.27,
        "2048": 273.06,
        "4096": 301.24,
        "8192": 309.30,
        "16384": 311.66,
    },
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


for device in ['mi350x', 'mi355x']:
    for setting in ['mha_causal_bkwd', 'mha_non_causal_bkwd', 'gqa_causal_bkwd', 'gqa_non_causal_bkwd']:
        
        # Read data
        try:
            # Map setting to filename
            if setting == 'mha_causal_bkwd':
                filename = f'benchmark/{device}_mha_bkwd_causal.json'
            elif setting == 'mha_non_causal_bkwd':
                filename = f'benchmark/{device}_mha_bkwd_non_causal.json'
            elif setting == 'gqa_causal_bkwd':
                filename = f'benchmark/{device}_gqa_bkwd_causal.json'
            elif setting == 'gqa_non_causal_bkwd':
                filename = f'benchmark/{device}_gqa_bkwd_non_causal.json'
            
            with open(filename, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Extract data for plotting
        matrix_sizes = sorted([int(size) for size in data.keys()])
        aiter_tflops = [data[str(size)]['tflops_ref'] for size in matrix_sizes]
        tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

        # Get baseline data based on setting
        triton_tflops = []
        torch_tflops = []
        ck_tflops = []
        
        if setting == 'mha_causal_bkwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_bkwd' and device == 'mi355x':
            # No triton for non-causal backward
            torch_tflops = [mi355x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_bkwd' and device == 'mi355x':
            # No triton for GQA
            torch_tflops = [mi355x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_bkwd' and device == 'mi355x':
            torch_tflops = [mi355x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_causal_bkwd' and device == 'mi350x':
            triton_tflops = [mi350x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi350x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_bkwd' and device == 'mi350x':
            torch_tflops = [mi350x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_bkwd' and device == 'mi350x':
            torch_tflops = [mi350x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_bkwd' and device == 'mi350x':
            torch_tflops = [mi350x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]

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

        if triton_vals:
            # 5 bars: PyTorch, Triton, CK, AITER, HipKittens
            first_bar_start = x - 2*width
            second_bar_start = x - width
            third_bar_start = x
            fourth_bar_start = x + width
            fifth_bar_start = x + 2*width

            bars3 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[1])
            bars2 = ax.bar(second_bar_start, triton_vals, width, label='Triton', color=colors[2])
            bars4 = ax.bar(third_bar_start, ck_vals, width, label='Composable Kernel', color=colors[4])
            bars0 = ax.bar(fourth_bar_start, aiter_tflops, width, label='AITER (ASM)', color=colors[0])
            bars1 = ax.bar(fifth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])
        else:
            # 4 bars: PyTorch, CK, AITER, HipKittens
            first_bar_start = x - 1.5*width
            second_bar_start = x - 0.5*width
            third_bar_start = x + 0.5*width
            fourth_bar_start = x + 1.5*width

            bars3 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[1])
            bars4 = ax.bar(second_bar_start, ck_vals, width, label='Composable Kernel', color=colors[4])
            bars0 = ax.bar(third_bar_start, aiter_tflops, width, label='AITER', color=colors[0])
            bars1 = ax.bar(fourth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])

        fontsize = 11

        # Plot X markers for OOM
        oom_height = 50  # Position X near top of chart
        if torch_oom:
            for idx in torch_oom:
                offset = -2*width if triton_vals else -1.5*width
                ax.plot(x[idx] + offset, oom_height, 'x', color=colors[1],
                       markersize=15, markeredgewidth=3)
                ax.text(x[idx] + offset, oom_height + max_tflops * 0.03,
                       'OOM', ha='center', va='bottom', fontsize=fontsize, color=colors[1])

        # Add value labels on bars
        for bar, value in zip(bars0, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        for bar, value in zip(bars1, tk_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        if triton_vals:
            for bar, value in zip(bars2, triton_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        if torch_vals:
            for i, (bar, value) in enumerate(zip(bars3, torch_vals)):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=fontsize)

        if ck_vals:
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
        ax.set_title(f'{attn_type} {causal_mode} Backward Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_sizes, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

        output_file = f'{device}_{setting}_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        # Print summary
        print(f"Sequence lengths tested: {matrix_sizes}")
        print(f"AITER TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
        print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

