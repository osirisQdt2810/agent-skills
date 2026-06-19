
# the goal of this plot is to show how the producer-consumer baseline compares to 8-wave ping pong and pytorch. 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


all_data = {
    "3072x3072x3072": {
        "0.67": 453.65,  # (3072x3072 by 3072x3072)
        "0.75": 683.61,  # (3072x3072 by 3072x3072)
        "1": 808.16,     # (3072x3072 by 3072x3072)
    },
    "3072x4096x4096": {
        "0.67": 708.06,  # (3072x4096 by 4096x4096)
        "0.75": 1006.45,  # (3072x4096 by 4096x4096)
        "1": 1102.71,     # (3072x4096 by 4096x4096)
    },
    "7680x7680x7680": {
        "0.67": 872.48,  # (7680x7680 by 7680x7680)
        "0.75": 1128.48,  # (7680x7680 by 7680x7680)
        "1": 1443.73,    # (7680x7680 by 7680x7680)
    },
    "7680x8192x8192": {
        "0.67": 964,   # (7680x8192 by 8192x8192)
        "0.75": 1251.18,  # (7680x8192 by 8192x8192)
        "1": 1521.94,     # (7680x8192 by 8192x8192)
    },
    "9216x8192x8192": {
        "0.67": 1010.59,   # (9216x8192 by 8192x8192)
        "0.75": 1229.37,  # (9216x8192 by 8192x8192)
        "1": 1490.14,     # (9216x8192 by 8192x8192)
    },
    "9216x9216x9216": {
        "0.67": 900.17,   # (8192x8192 by 8192x8192)
        "0.75": 1129.15,   # (9216x9216 by 9216x9216)
        "1": 1418.23,     # (9216x9216 by 9216x9216)
    },
}

colors = ["#7CB9BC", "#8E69B8", "#E59952", "#68AC5A"]

# plot all three in one figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, key in enumerate(["3072x3072x3072", "7680x8192x8192", "9216x9216x9216"]):
    data = all_data[key]
    axes[idx].bar(data.keys(), data.values(), color=colors)
    for i, v in enumerate(data.values()):
        axes[idx].text(i, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=14)
    axes[idx].set_title(f"GEMM (M={key.split('x')[0]}, K={key.split('x')[1]}, N={key.split('x')[2]})", fontsize=16)
    axes[idx].set_xlabel("% consumer workers", fontsize=14)
    axes[idx].set_ylabel("TFLOPs", fontsize=14)
    axes[idx].tick_params(axis='both', which='major', labelsize=12)
    # Add some padding to the top of the y-axis to prevent label overlap
    axes[idx].set_ylim(0, max(data.values()) * 1.15)
plt.tight_layout()
plt.savefig("pc_micro.png", dpi=300, bbox_inches="tight")
plt.close()

