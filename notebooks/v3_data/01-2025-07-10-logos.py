# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: lir_proteome_screen_pssm
#     language: python
#     name: python3
# ---

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

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # scoring ilir and screening PSSMs
#
# Test sets: 7mers
# - screening - [FWY]xx[LVI]
# - screening - [FWY]xx[FWY]
# - lir central - (not the augmented set)
#
#
# PSSM foregrounds:
# - ilir
# - screening data:
#     - different z-score ranges
#
# PSSM background:
# - proteome
#
# psuedocounts:
# - 1.0

# %%
plt.rcParams.update({"font.size": 14})

# %%
version = "v3"
PROCESSED_SEQUENCE_TABLES = dl.get_processed_sequence_tables(version)
TEST_SETS = dl.get_test_sets(version)
BGFREQS = dl.get_background_frequencies(version)
output_dir = Path("./01-02-plots")
output_dir.mkdir(exist_ok=True, parents=True)
plot_count=1

# %%
ts = {
    "lir_central":TEST_SETS.lir_central,
    "lir_central_augmented": TEST_SETS.lir_central_augmented
}
n_plots = len(ts)
fig, axes = plt.subplots(
    nrows=n_plots,
    ncols=2,
    figsize=(13, 2.5 * n_plots),
    sharex=True,
)
for axs, (name, testset) in zip(axes.T, ts.items()):
    axs[0].set_title(f"{name} - binders")
    pssms.plot_logo(
        pssms.seqlist_2_counts_matrix(testset[testset['true label']==1]["7mer"].to_list()),
        ax=axs[0],
    )
    axs[0].set_ylabel("count")
    pssms.plot_logo(
        pssms.seqlist_2_counts_matrix(testset[testset['true label']==0]["7mer"].to_list()),
        ax=axs[1],
    )
    axs[1].set_title(f"{name} - nonbinders")
axs[0].set_xlabel("Position")
axs[1].set_xlabel("Position")
plt.tight_layout()
plt.savefig(output_dir / f"01-{plot_count}-lir_central-logos.png", dpi=300, bbox_inches="tight")
plot_count += 1


# %%
def plot_logo_at_z_score_range(binder_df, z_score_range, ax = None):
    binders = binder_df[(binder_df['avg_z_score']>=z_score_range[0]) & (binder_df['avg_z_score']<z_score_range[1])]['7mer'].to_list()
    c_binders = pssms.alignment_2_counts(binders)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    pssms.plot_logo(c_binders, ax = ax)
    ax.set_title(f'z-score range: {z_score_range[0]} - {z_score_range[1]}');
    # increase title font size
    ax.title.set_fontsize(16)
    return ax


# %%
# cutoffs = np.arange(1.7, 3.2, 0.15)
# define (start, stop) ranges spanning the z-score cutoffs
cutoffs = [(1.7, 2.5), (2.5, 4.2)]

# plot each cutoff on a subplot
n_plots = len(cutoffs)
fig, axs = plt.subplots(n_plots, 1, figsize=(7, 2.5*n_plots))
for axis, cutoff in zip(axs, cutoffs):
    plot_logo_at_z_score_range(PROCESSED_SEQUENCE_TABLES.screen_binders, z_score_range=cutoff, ax=axis)
plt.tight_layout()

# %%
df_zs = PROCESSED_SEQUENCE_TABLES.screen_binders[~PROCESSED_SEQUENCE_TABLES.screen_binders["avg_z_score"].isna()].copy()
df_zs = df_zs.sort_values(by = "avg_z_score", ascending = True)
df_zs = df_zs.reset_index(drop=True)
# create 5 ranges from df_zs.index
# df_zs["z_score_range"] = pd.cut(df_zs.index, bins=4, labels=False)
df_zs["z_score_range"] = pd.cut(df_zs.index, bins=3, labels=False)
n_plots = len(df_zs["z_score_range"].unique())
fig, axes = plt.subplots(nrows = n_plots, ncols = 1, figsize=(7, 2.5*n_plots))
for i, ax in zip(df_zs["z_score_range"].unique(), axes):
    df_filt = df_zs[df_zs["z_score_range"] == i]
    pssms.plot_logo(pssms.alignment_2_counts(df_filt["7mer"]), ax=ax)
    ax.set_title(f"Z-score range: {df_filt['avg_z_score'].min()} - {df_filt['avg_z_score'].max()}")
plt.tight_layout()
plt.savefig(output_dir / f"01-{plot_count}-screen_z-score_ranges.png", dpi=300, bbox_inches="tight")
plot_count += 1

# %%
screen_binders = PROCESSED_SEQUENCE_TABLES.screen_binders.copy()
screen_binders['lir_type'].value_counts()

# %%
fig, ax = plt.subplots()
sns.stripplot(
    data=screen_binders,
    x="lir_type",
    y="avg_z_score",
    ax=ax,
    jitter=True,
    alpha=0.5,
    color="black",
)
sns.boxplot(
    data=screen_binders,
    x="lir_type",
    y="avg_z_score",
    ax=ax,
)

# Add number of datapoints above each box
group_counts = screen_binders.groupby("lir_type")["avg_z_score"].count()
for i, (lir_type, count) in enumerate(group_counts.items()):
    y_max = screen_binders[screen_binders["lir_type"] == lir_type]["avg_z_score"].max()
    ax.text(i, y_max + 0.2, str(count), ha="center", va="bottom", fontsize=16, fontweight="bold")
fig.savefig(output_dir / f"01-{plot_count}-motif_counts.png", dpi=300, bbox_inches="tight")
plot_count += 1

# %%
fig, ax = plt.subplots(figsize=(7, 2.5))
pssms.plot_logo(pssms.seqlist_2_counts_matrix(screen_binders[screen_binders["lir_type"]=="...[FWY]..[WFY]"]["7mer"].to_list()), ax=ax)
ax.set_title("...[FWY]..[WFY] LIRs")
ax.set_ylabel("count")
ax.set_xlabel("Position")
fig.savefig(output_dir / f"01-{plot_count}-logo-aromatic.png", dpi=300, bbox_inches="tight")
plot_count += 1
fig, ax = plt.subplots(figsize=(7, 2.5))
pssms.plot_logo(pssms.seqlist_2_counts_matrix(screen_binders[screen_binders["lir_type"]=="...[FWY]..[LVI]"]["7mer"].to_list()), ax=ax)
ax.set_title("...[FWY]..[LVI] LIRs")
ax.set_ylabel("count")
ax.set_xlabel("Position")
fig.savefig(output_dir / f"01-{plot_count}-logo-classic.png", dpi=300, bbox_inches="tight")
plot_count += 1


# %%
fig, ax = plt.subplots(figsize=(7, 2.5))
ax = pssms.plot_logo(pssms.seqlist_2_counts_matrix(screen_binders["7mer"].to_list()), ax=ax)
ax.set_title("all LIRs; z-score >= 1.7")
ax.set_ylabel("count")
ax.set_xlabel("Position")
fig.savefig(output_dir / f"01-{plot_count}-logo-all-screen.png", dpi=300, bbox_inches="tight")
plot_count += 1

# %%
fig, ax = plt.subplots(figsize=(7, 2.5))
ax = pssms.plot_logo(pssms.seqlist_2_counts_matrix(PROCESSED_SEQUENCE_TABLES.screen_nonbinders["7mer"].to_list()), ax=ax)
ax.set_title("nonbinder LIRs\nnegative enrichment for round 1 and 3; dropped after 3; >= 50 input count")
ax.set_ylabel("count")
ax.set_xlabel("Position")
fig.savefig(output_dir / f"01-{plot_count}-logo-nonbinders.png", dpi=300, bbox_inches="tight")
plot_count += 1

# %%
fig, ax = plt.subplots(figsize=(7, 2.5))
ax = pssms.plot_logo(pssms.seqlist_2_counts_matrix(PROCESSED_SEQUENCE_TABLES.ilir_binders["7mer"].to_list()), ax=ax)
ax.set_title("ilir binders")
ax.set_ylabel("count")
ax.set_xlabel("Position")
fig.savefig(output_dir / f"01-{plot_count}-logo-ilir.png", dpi=300, bbox_inches="tight")
plot_count += 1
# %%
