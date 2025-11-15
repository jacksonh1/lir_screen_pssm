# %%
from lir_proteome_screen_pssm import environment as env
from lir_proteome_screen_pssm import pssms
import pandas as pd
import numpy as np
import logomaker as lm
import matplotlib.pyplot as plt

plt.style.use("custom_standard")
plt.style.use("custom_small")
import seaborn as sns
import lir_proteome_screen_pssm.sequence_utils as seqtools
import copy
from pathlib import Path
import re
# import umap
from sklearn.preprocessing import OneHotEncoder
import lir_proteome_screen_pssm.data_loaders as dl
import lir_proteome_screen_pssm.stats as stats
plt.style.use("lir_proteome_screen_pssm.lir")
from adjustText import adjust_text
mm = 1 / 25.4

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # scoring ilir and screening PSSMs - test set 1
#
# Test sets: 7mers
# - lir central - (the augmented set)
#
#
# PSSM foregrounds:
# - ilir
# - screening data:
#     - all binders (z>=1.7)
#     - high z-score (z>=2.3)
#     - low z-score (1.7<=z<2.3)
#
# PSSM background:
# - proteome
#
# psuedocounts:
# - 0

# %%

def check_tables(processed_sequence_tables, test_sets):
    assert (processed_sequence_tables.screen_nonbinders["7mer"].str.len() == 7).all()
    assert (processed_sequence_tables.screen_binders["7mer"].str.len() == 7).all()
    assert (processed_sequence_tables.ilir_binders["7mer"].str.len() == 7).all()
    assert (test_sets.lir_central["7mer"].str.len() == 7).all()
    assert (test_sets.lir_central_augmented["7mer"].str.len() == 7).all()


def check_for_fg_test_overlap(
    foregrounds: dict[str, list[str]], test_set_dict: dict[str, pd.DataFrame]
):
    for fg_name, fg_seqs in foregrounds.items():
        fg_set = set(fg_seqs)
        for test_name, test_set in test_set_dict.items():
            test_set_seqs = set(test_set["7mer"].to_list())
            overlap = fg_set.intersection(test_set_seqs)
            if len(overlap) > 0:
                if 'cheating' in fg_name:
                    continue
                raise ValueError(
                    f"Overlap between foreground '{fg_name}' and test set '{test_name}'"
                )

def score_test_sets_with_pssms(
    test_set_dict: dict[str, pd.DataFrame], pssm_dict: dict[str, pd.DataFrame]
):
    auc_results = []
    for foreground, pssm in pssm_dict.items():
        for test_set_name, test_set in test_set_dict.items():
            temp_df = test_set.copy()
            temp_df["pssm_score"] = temp_df["7mer"].apply(
                pssms.PSSM_score_sequence, PSSM=pssm
            )
            auroc = stats.df_2_roc_auc(temp_df, "true label", "pssm_score")
            auc_results.append(
                {
                    "foreground": foreground,
                    "test set": test_set_name,
                    "auROC": auroc,
                }
            )
    return pd.DataFrame(auc_results)

# %%
version = "v2"
PROCESSED_SEQUENCE_TABLES = dl.get_processed_sequence_tables(version)
TEST_SETS = dl.get_test_sets(version)   
BGFREQS = dl.get_background_frequencies(version)
PSEUDOCOUNT = 0
output_dir = Path("./04-output")
output_dir.mkdir(exist_ok=True, parents=True)
plot_count=1

check_tables(PROCESSED_SEQUENCE_TABLES, TEST_SETS)

lca_testset = TEST_SETS.lir_central_augmented.copy()
binder_df = PROCESSED_SEQUENCE_TABLES.screen_binders.copy()

fgs = {
    "screen all binders": binder_df['7mer'].to_list(),
    "screen low z-score": binder_df[binder_df['avg_z_score'] < 2.3]['7mer'].to_list(),
    "screen high z-score": binder_df[binder_df['avg_z_score'] >= 2.3]['7mer'].to_list(),
    "ilir": PROCESSED_SEQUENCE_TABLES.ilir_binders["7mer"].to_list(),
}

testsets = {
    "lir central augmented": TEST_SETS.lir_central_augmented.copy(),
}

labels = {
    "screen all binders": "all binders",
    "screen low z-score": "low z-score",
    "screen high z-score": "high z-score",
    "ilir": "$\mathregular{iLIR_{27}}$",
    "lir central augmented": "lir central augmented",
}
# %%
check_for_fg_test_overlap(fgs, testsets)
pssm_dict = {}
fg_counts = {}
for k, v in fgs.items():
    counts = pssms.seqlist_2_counts_matrix(v, pseudocount=PSEUDOCOUNT)
    fg_counts[k] = counts
    pssm = pssms.make_pssm(
        df_counts=counts,
        bg=BGFREQS.proteome,
    )
    pssm_dict[k] = pssm
auc_results = score_test_sets_with_pssms(testsets, pssm_dict)



# %%
fig, ax = plt.subplots(constrained_layout=True, figsize=(60*mm, 60*mm))

sns.barplot(
    auc_results,
    x='foreground',
    y='auROC',
    ax = ax
)
ax.set_ylim([0.4, 0.85])
# rotate the xtick labels by 90 degrees
ax.tick_params(axis='x', rotation=90)
current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
new_labels = [labels.get(label, label) for label in current_labels]
ax.set_xticklabels(new_labels, rotation=90) 
ax.set_xlabel('')
# Add text labels above bars
for bar, value in zip(ax.patches, auc_results['auROC']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f"{value:.2f}", ha='center', va='bottom', fontsize=8)
fig.savefig(output_dir / "lir_central_performance.png", bbox_inches='tight', dpi=300, format="png")
auc_results.to_csv(output_dir / "lir_central_performance.csv", index=False)


# %%
fsize = (80*mm, 30*mm)
for k, m in fg_counts.items():
    m.to_csv(output_dir / f"counts_matrix-{k.replace(' ', '_')}.csv")
    fig, ax = plt.subplots(figsize=fsize)
    pssms.plot_logo(m, ax=ax)
    ax.set_title(f"{labels.get(k, k)}")
    fig.savefig(output_dir / f"logo-{k.replace(' ', '_')}.png", bbox_inches='tight', dpi=300, format="png")

for k, m in testsets.items():
    bs = m[m['true label'] == 1]
    nbs = m[m['true label'] == 0]
    bcounts = pssms.seqlist_2_counts_matrix(bs["7mer"].to_list(), pseudocount=0)
    bcounts.to_csv(output_dir / f"counts_matrix-{k.replace(' ', '_')}-binders.csv")
    fig, ax = plt.subplots(figsize=fsize)
    pssms.plot_logo(bcounts, ax=ax)
    ax.set_title(f"{labels.get(k, k)} - binders")
    fig.savefig(output_dir / f"logo-{k.replace(' ', '_')}-binders.png", bbox_inches='tight', dpi=300, format="png")

    nbcounts = pssms.seqlist_2_counts_matrix(nbs["7mer"].to_list(), pseudocount=0)
    nbcounts.to_csv(output_dir / f"counts_matrix-{k.replace(' ', '_')}-nonbinders.csv")
    fig, ax = plt.subplots(figsize=fsize)
    pssms.plot_logo(nbcounts, ax=ax)
    ax.set_title(f"{labels.get(k, k)} - nonbinders")
    fig.savefig(output_dir / f"logo-{k.replace(' ', '_')}-nonbinders.png", bbox_inches='tight', dpi=300, format="png")

# %%


nonbinder_df = PROCESSED_SEQUENCE_TABLES.screen_nonbinders.copy()
counts = pssms.seqlist_2_counts_matrix(nonbinder_df['7mer'].to_list(), pseudocount=PSEUDOCOUNT)
fig, ax = plt.subplots(figsize=fsize)
pssms.plot_logo(counts, ax=ax)
ax.set_title("screen non-binders")
fig.savefig(output_dir / "screen non-binders.png", bbox_inches='tight', dpi=300, format="png")










# %%