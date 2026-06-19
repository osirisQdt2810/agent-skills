
# the goal of this plot is to show how the producer-consumer baseline compares to 8-wave ping pong and pytorch. 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


all_data = {
    "BF16 GEMM": {
        "8-pp w/ m=32": 1291,
        "8-pp w/ m=16": 1580,  
        "4-wave": 1507,
    },
    "FP8 GEMM": {
        "8-pp w/ m=32": 2500,  
        "8-pp w/ m=16": 3066,  
        "4-wave": 3303,   
    },
}

colors = ["#7CB9BC", "#8E69B8", "#E59952", "#68AC5A"]

# plot all three in one figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, key in enumerate(["BF16 GEMM", "FP8 GEMM"]):
    data = all_data[key]
    axes[idx].bar(data.keys(), data.values(), color=colors)
    for i, v in enumerate(data.values()):
        axes[idx].text(i, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=16)
    axes[idx].set_title(key, fontsize=16)
    # axes[idx].set_xlabel("% consumer workers", fontsize=14)
    axes[idx].set_ylabel("TFLOPs", fontsize=14)
    axes[idx].tick_params(axis='both', which='major', labelsize=14)
    # Add some padding to the top of the y-axis to prevent label overlap
    axes[idx].set_ylim(0, max(data.values()) * 1.15)
plt.tight_layout()
plt.savefig("pp_micro.png", dpi=300, bbox_inches="tight")
plt.close()

