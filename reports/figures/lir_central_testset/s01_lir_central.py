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

cong_filepaths = [
    env.DATA_DIR / "from_cong/LIRCentral_test_auc_ready_for_plot/RF_LIRCentral_test_auc.csv",
    env.DATA_DIR / "from_cong/LIRCentral_test_auc_ready_for_plot/BRF_LIRCentral_test_auc.csv",
]
my_path = env.PROJ_ROOT / "notebooks" / "v2_data" / "04-output" / "lir_central_performance.csv"

# %%
labels = {
    "screen all binders": "PSSM - all binders",
    "screen low z-score": "PSSM - low z-score",
    "screen high z-score": "PSSM - high z-score",
    "ilir": "$\mathregular{iLIR_{27}}$",
    "lir central augmented": "lir central augmented",
    "RF-all_binders-onehot": "RF - all binders - onehot",
    "RF-low_z_score-onehot": "RF - low z-score - onehot",
    "RF-high_z_score-onehot": "RF - high z-score - onehot",
    "BRF-all_binders-onehot": "BRF - all binders - onehot",
    "BRF-low_z_score-onehot": "BRF - low z-score - onehot",
    "BRF-high_z_score-onehot": "BRF - high z-score - onehot",
    "RF-all_binders-pssm": "RF - all binders\nPSSM encoding",
    "RF-low_z_score-pssm": "RF - low z-score\nPSSM encoding",
    "RF-high_z_score-pssm": "RF - high z-score\nPSSM encoding",
    "BRF-all_binders-pssm": "BRF - all binders\nPSSM encoding",
    "BRF-low_z_score-pssm": "BRF - low z-score\nPSSM encoding",
    "BRF-high_z_score-pssm": "BRF - high z-score\nPSSM encoding",
    "RF-all_binders-esm2": "RF - all binders - ESM2",
    "RF-low_z_score-esm2": "RF - low z-score - ESM2",
    "RF-high_z_score-esm2": "RF - high z-score - ESM2",
    "BRF-all_binders-esm2": "BRF - all binders - ESM2",
    "BRF-low_z_score-esm2": "BRF - low z-score - ESM2",
    "BRF-high_z_score-esm2": "BRF - high z-score - ESM2",
}
order = [
    "screen all binders",
    "screen high z-score",
    "screen low z-score",
    "ilir",
    # "RF-all_binders-onehot",
    "RF-all_binders-pssm",
    # "RF-all_binders-esm2",
    # "RF-high_z_score-onehot",
    "RF-high_z_score-pssm",
    # "RF-high_z_score-esm2",
    # "RF-low_z_score-onehot",
    "RF-low_z_score-pssm",
    # "RF-low_z_score-esm2",
    # "BRF-all_binders-onehot",
    "BRF-all_binders-pssm",
    # "BRF-all_binders-esm2",
    # "BRF-high_z_score-onehot",
    "BRF-high_z_score-pssm",
    # "BRF-high_z_score-esm2",
    # "BRF-low_z_score-onehot",
    "BRF-low_z_score-pssm",
    # "BRF-low_z_score-esm2",
]

pssm_perf = pd.read_csv(my_path)
pssm_perf["std"] = 0.0

# rf_perf = pd.read_csv(cong_filepaths[0])
# rf_perf[['encoding', 'foreground']] = rf_perf['Unnamed: 0'].str.extract(r'RF_train_w_(?P<encoding>[^_]+)_(?P<foreground>\w+_\w+)_auc')
# brf_perf = pd.read_csv(cong_filepaths[1])
# brf_perf[['encoding', 'foreground']] = brf_perf['Unnamed: 0'].str.extract(r'BRF_train_w_(?P<encoding>[^_]+)_(?P<foreground>\w+_\w+)_auc')


def import_cong_table(filepath, balanced=False):
    df = pd.read_csv(filepath)
    df[['encoding', 'foreground']] = df['Unnamed: 0'].str.extract(r'RF_train_w_(?P<encoding>[^_]+)_(?P<foreground>\w+_\w+)_auc')
    if balanced:
        df['foreground'] = df.apply(lambda x: f"BRF-{x['foreground']}-{x['encoding']}", axis=1)
    if not balanced:
        df['foreground'] = df.apply(lambda x: f"RF-{x['foreground']}-{x['encoding']}", axis=1)
    df = df.rename(columns={"mean": "auROC"})
    df = df.drop(columns=["Unnamed: 0", "encoding"])
    return df

rf_perf = import_cong_table(cong_filepaths[0], balanced=False)
brf_perf = import_cong_table(cong_filepaths[1], balanced=True)


df = pd.concat([pssm_perf, rf_perf, brf_perf], axis=0).reset_index(drop=True)


# %%
# h = 100
h = 75
# w = 200
w = 100
fig, ax = plt.subplots(figsize=(w*mm, h*mm))
data = df.copy()
data = data.set_index("foreground").reindex(order).reset_index()
ax.bar(
    data["foreground"], data["auROC"], yerr=data["std"], capsize=5
)
ax.set_xlabel("Foreground")
ax.set_ylabel("auROC")
# ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 0.85])
ax.set_xlabel("")
for bar, value in zip(ax.patches, data["auROC"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
# Specify the order of x-axis values
current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
new_labels = [labels.get(label, label) for label in current_labels]
ax.tick_params(axis="x", rotation=90)
ax.set_xticklabels(new_labels, ha="center", va="top")
ax.set_xticklabels(
    ax.get_xticklabels(),
    ha="right",         # right-align the text
    va="center",        # vertically center the text
    rotation_mode="anchor"  # anchor the rotation to the alignment point
)
plt.tight_layout()
fig.savefig("./lir_central_testset_results.svg")










# %%
