# %%
from lir_proteome_screen_pssm import environment as env
from lir_proteome_screen_pssm import pssms
import pandas as pd
import numpy as np
import logomaker as lm
import matplotlib.pyplot as plt
import seaborn as sns
import lir_proteome_screen_pssm.sequence_utils as seqtools
import copy
from pathlib import Path
import re
import lir_proteome_screen_pssm.data_loaders as dl
import lir_proteome_screen_pssm.stats as stats
plt.style.use("lir_proteome_screen_pssm.lir")
from adjustText import adjust_text
mm = 1 / 25.4

labels = {
    "pssm - all binders": "all binders\nPSSM",
    "pssm - low z-score": "low z-score\nPSSM",
    "pssm - high z-score": "high z-score\nPSSM",
    "pssm - ilir": "$\mathregular{iLIR_{27}}$\nPSSM",
    "rf - esm2_train_all": "all binders\nRF - ESM2",
    "rf - esm2_train_high_z": "high z-score\nRF - ESM2",
    "rf - esm2_train_low_z": "low z-score\nRF - ESM2",
    "rf - onehot_train_all": "all binders\nRF - OneHot",
    "rf - onehot_train_high_z": "high z-score\nRF - OneHot",
    "rf - onehot_train_low_z": "low z-score\nRF - OneHot",
    "rf - pssm_train_all": "all binders\nRF - PSSM",
    "rf - pssm_train_high_z": "high z-score\nRF - PSSM",
    "rf - pssm_train_low_z": "low z-score\nRF - PSSM",
    "brf - esm2_train_all": "all binders\nBRF - ESM2",
    "brf - esm2_train_high_z": "high z-score\nBRF - ESM2",
    "brf - esm2_train_low_z": "low z-score\nBRF - ESM2",
    "brf - onehot_train_all": "all binders\nBRF - OneHot",
    "brf - onehot_train_high_z": "high z-score\nBRF - OneHot",
    "brf - onehot_train_low_z": "low z-score\nBRF - OneHot",
    "brf - pssm_train_all": "all binders\nBRF - PSSM",
    "brf - pssm_train_high_z": "high z-score\nBRF - PSSM",
    "brf - pssm_train_low_z": "low z-score\nBRF - PSSM",
}

cong_filepaths = [
    env.DATA_DIR / "from_cong/BRF_train_test_on_screening_ready_for_plot",
    env.DATA_DIR / "from_cong/train_test_on_screening_ready_for_plot",
]

my_path = env.PROJ_ROOT / "notebooks" / "v2_data" / "05-output" / "screening_performance_summary.csv"

filename_key = {
    "brf_test_on_all_screen_all.csv": "all binders",
    "brf_test_on_high_z_screen_all.csv": "high z-score",
    "brf_test_on_low_z_screen_all.csv": "low z-score",
    "test_on_all_screen_all.csv": "all binders",
    "test_on_high_z_screen_all.csv": "high z-score",
    "test_on_low_z_screen_all.csv": "low z-score",
}

df = pd.read_csv(my_path)
df['foreground'] = df['foreground'].apply(lambda x: f"pssm - {x}")



rf = pd.read_csv(list(cong_filepaths[0].glob("*.csv"))[0])
rf['Unnamed: 0'] = rf['Unnamed: 0'].apply(lambda x: f"rf - {x}")
rf = rf.rename(columns={"Unnamed: 0": "foreground", "mean": "mean auROC", "std": "std auROC"})
rf["test set"] = filename_key[list(cong_filepaths[0].glob("*.csv"))[0].name]


def import_cong_table(filepath, model_prefix, filename_key):
    df = pd.read_csv(filepath)
    df['Unnamed: 0'] = df['Unnamed: 0'].apply(lambda x: f"{model_prefix} - {x}")
    df = df.rename(columns={"Unnamed: 0": "foreground", "mean": "mean auROC", "std": "std auROC"})
    df["test set"] = filename_key[filepath.name]
    return df



dfs = []
for d in cong_filepaths:
    for file in d.glob("*.csv"):
        if file.name not in filename_key:
            print(f"Skipping {file.name} as not in filename_key")
            continue
        print(file)
        temp = import_cong_table(file, "brf" if "brf" in file.name else "rf", filename_key)
        dfs.append(temp)
df = pd.concat([df] + dfs, axis=0, ignore_index=True)#.reset_index(drop=True)
# df2 = pd.concat([df] + dfs, axis=0)#.reset_index(drop=True)

# %%

order = [
    "pssm - all binders",
    "pssm - low z-score",
    "pssm - high z-score",
    "pssm - ilir",
    "rf - esm2_train_all",
    "rf - esm2_train_high_z",
    "rf - esm2_train_low_z",
    "rf - onehot_train_all",
    "rf - onehot_train_high_z",
    "rf - onehot_train_low_z",
    "rf - pssm_train_all",
    "rf - pssm_train_high_z",
    "rf - pssm_train_low_z",
    "brf - esm2_train_all",
    "brf - esm2_train_high_z",
    "brf - esm2_train_low_z",
    "brf - onehot_train_all",
    "brf - onehot_train_high_z",
    "brf - onehot_train_low_z",
    "brf - pssm_train_all",
    "brf - pssm_train_high_z",
    "brf - pssm_train_low_z",
]



fig, axes = plt.subplots(3, 1, figsize=(200 * mm, 200 * mm))
test_sets = df["test set"].unique()

for i, test_set in enumerate(test_sets):
    data = df[df["test set"] == test_set].copy()
    data = data.set_index("foreground").reindex(order).reset_index()

    axes[i].bar(
        data["foreground"], data["mean auROC"], yerr=data["std auROC"], capsize=5
    )
    axes[i].set_title(f"Test Set: {test_set}")
    # axes[i].set_xticklabels(axes[i].get_xticklabels(), ha="right", va="top")
    axes[i].set_xlabel("Foreground")
    axes[i].set_ylabel("Mean auROC")
    # axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim([0.4, 1.05])
    axes[i].set_xlabel("")
    for bar, value in zip(axes[i].patches, data["mean auROC"]):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.09,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    # Specify the order of x-axis values
    current_labels = [tick.get_text() for tick in axes[i].get_xticklabels()]
    new_labels = [labels.get(label, label) for label in current_labels]
    axes[i].tick_params(axis="x", rotation=90)
    axes[i].set_xticklabels(new_labels, ha="center", va="top")
    axes[i].set_xticklabels(
        axes[i].get_xticklabels(),
        ha="right",         # right-align the text
        va="center",        # vertically center the text
        rotation_mode="anchor"  # anchor the rotation to the alignment point
    )


plt.tight_layout()
plt.savefig("screening_test_sets.svg", dpi=300, bbox_inches="tight")






# %%






# %%
