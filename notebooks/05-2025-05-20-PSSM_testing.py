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
import umap
from sklearn.preprocessing import OneHotEncoder
import lir_proteome_screen_pssm.data_loaders as dl
import lir_proteome_screen_pssm.stats as stats

# %load_ext autoreload
# %autoreload 2


# %% [markdown]
# 1. load test set
# 2. make score functions - aucPRC, roc_auc
# 3. set up parameters to screen
# 4. build dataframe of results

# %% [markdown]
# # load test set

# %%
df = dl.TEST_SETS.lir_central_augmented.copy()
df = df.drop_duplicates(subset=["sequence"])

# %% [markdown]
# screening pssm parameters:
# - foreground set
#     - all binders
#     - all binders weighted
#     - binders with > 2.4 z-score
# - background set
#     - input library
#     - proteome
#     - nonbinders
# - low count cutoff
# - pseudocounts
#
# ilir parameters:
# - foreground set
#     - ilir
# - background set
#     - ilir
#     - proteome
# - low count cutoff
# - pseudocounts

# %% [markdown]
# # set up grid search

# %%
psi_blast_pssm_dict = {
    "screen_all_binders-psiblast": pd.read_csv(
        env.DATA_DIR / "processed" / "pssms" / "screen-all_binders_psiblast_pssm.csv"
    ),
    "screen_z_score_above_2_4-psiblast": pd.read_csv(
        env.DATA_DIR
        / "processed"
        / "pssms"
        / "screen-z_score_above_2_4_psiblast_pssm.csv"
    ),
    "ilir_binders-psiblast": pd.read_csv(
        env.DATA_DIR / "processed" / "pssms" / "fg-ilir_bg-proteome_psiblast.csv"
    ),
}

# %%
data_origin_map = {
    "screen_all_binders": "screen",
    "screen_all_binders_weighted": "screen",
    "screen_z_score_above_2_4": "screen",
    "ilir_binders": "ilir",
    "screen_all_binders-psiblast": "psiblast",
    "screen_z_score_above_2_4-psiblast": "psiblast",
    "ilir_binders-psiblast": "psiblast",
}

# %%
# fg_sets = [i for i in dl.COUNT_MATRICES.__dict__.keys() if 'file' not in i]
# bg_sets = list(dl.BGFREQS.__dict__.keys())

fg_sets = [
    "screen_all_binders",
    "screen_all_binders_weighted",
    "screen_z_score_above_2_4",
]
bg_sets = ["proteome", "input_library", "nonbinders_undef_pos"]
pseudo_counts = [0, 0.1, 1, 10]
low_count_cutoffs = [0, 2, 5, 10]
parameter_combinations = []
for fg_set in fg_sets:
    for bg_set in bg_sets:
        for pseudo_count in pseudo_counts:
            if pseudo_count != 0:
                parameter_combinations.append((fg_set, bg_set, pseudo_count, 0))
                continue
            for low_count_cutoff in low_count_cutoffs:
                parameter_combinations.append(
                    (fg_set, bg_set, pseudo_count, low_count_cutoff)
                )
fg_sets = ["ilir_binders"]
bg_sets = ["proteome", "ilir_bg"]
pseudo_counts = [0, 0.1, 1, 10]
low_count_cutoffs = [0, 2, 5]
for fg_set in fg_sets:
    for bg_set in bg_sets:
        for pseudo_count in pseudo_counts:
            if pseudo_count != 0:
                parameter_combinations.append((fg_set, bg_set, pseudo_count, 0))
                continue
            for low_count_cutoff in low_count_cutoffs:
                parameter_combinations.append(
                    (fg_set, bg_set, pseudo_count, low_count_cutoff)
                )
parameter_combinations.append(("screen_all_binders-psiblast", "proteome", 0, 0))
parameter_combinations.append(("screen_z_score_above_2_4-psiblast", "proteome", 0, 0))
parameter_combinations.append(("ilir_binders-psiblast", "proteome", 0, 0))
print(len(parameter_combinations))


# %% [markdown]
# # score test set


# %%
def fractional_score(seq, pssm):
    max_score = pssm.max(axis=1).sum()
    min_score = pssm.min(axis=1).sum()
    score_mag = max_score - min_score
    score = pssms.PSSM_score_sequence(seq, pssm)
    return (score - min_score) / score_mag


# %%
auc_results = []
scores = []
pssms_dict = {}

for p in parameter_combinations:
    fg_set, bg_set, pseudo_count, low_count_cutoff = p
    if fg_set in psi_blast_pssm_dict:
        pssm = psi_blast_pssm_dict[fg_set]
    else:
        fg = getattr(dl.COUNT_MATRICES, fg_set)
        bg = getattr(dl.BGFREQS, bg_set)
        pssm = pssms.make_pssm(
            fg, bg, min_count=low_count_cutoff, pseudocount=pseudo_count, plot=False
        )
    temp_df = df.copy()
    temp_df["pssm_score"] = temp_df["sequence"].apply(
        pssms.PSSM_score_sequence, PSSM=pssm
    )
    temp_df["pssm_score_fraction"] = temp_df["sequence"].apply(
        fractional_score, pssm=pssm
    )
    temp_df["pssm_score_normalized"] = (temp_df["pssm_score"] - temp_df["pssm_score"].min())/(temp_df["pssm_score"].max() - temp_df["pssm_score"].min())
    _, _, _, auprc = stats.df_2_precision_recall_curve(
        temp_df, "true label", "pssm_score"
    )
    auroc = stats.df_2_roc_auc(temp_df, "true label", "pssm_score")
    # Store results for summary
    auc_results.append(
        {
            "foreground": fg_set,
            "background": bg_set,
            "low_count_mask": low_count_cutoff,
            "pseudocount": pseudo_count,
            "auROC": auroc,
            "auPRC": auprc,
        }
    )

    all_scored_df = temp_df[["sequence", "pssm_score", "pssm_score_fraction", "pssm_score_normalized"]].copy()
    all_scored_df["foreground"] = fg_set
    all_scored_df["background"] = bg_set
    all_scored_df["low_count_mask"] = low_count_cutoff
    all_scored_df["pseudocount"] = pseudo_count
    scores.append(all_scored_df)
    pssms_dict['-'.join([fg_set, bg_set, str(low_count_cutoff), str(float(pseudo_count))])] = pssm

all_scored_df = scores[0]
for s in scores[1:]:
    all_scored_df = pd.concat([all_scored_df, s], ignore_index=True)

auc_results_df = pd.DataFrame(auc_results)

all_scored_df["data_source"] = all_scored_df["foreground"].map(data_origin_map)
all_scored_df.to_csv("./all_scores.csv", index=False)
auc_results_df["data_source"] = auc_results_df["foreground"].map(data_origin_map)
auc_results_df.to_csv("./results.csv", index=False)

# %%
print(
    "Best parameters (auROC):\n", auc_results_df.loc[auc_results_df["auROC"].idxmax()]
)
print("++++++++++++++++++++++++++++++++++++++")
print(
    "Best parameters (auPRC):\n", auc_results_df.loc[auc_results_df["auPRC"].idxmax()]
)

# %%
psiblast_scores = all_scored_df[all_scored_df["data_source"] == "psiblast"].copy()
other_scores = all_scored_df[all_scored_df["data_source"] != "psiblast"].copy()

psiblast_auc_results = auc_results_df[
    auc_results_df["data_source"] == "psiblast"
].copy()
other_auc_results_df = auc_results_df[
    auc_results_df["data_source"] != "psiblast"
].copy()

# %%
other_scores

# %% [markdown]
# # explore grid search performance results

# %%
for i in auc_results_df["data_source"].unique():
    print(f"        best results for {i}")
    print("$++++++++++++++++++++++++++++++++++++++$")
    temp_df = auc_results_df[auc_results_df["data_source"] == i].copy()
    print("Best parameters (auROC):\n", temp_df.loc[temp_df["auROC"].idxmax()])
    print("--------------------------------------------")
    print("Best parameters (auPRC):\n", temp_df.loc[temp_df["auPRC"].idxmax()])
    print("\n")


# %% [markdown]
# ## distributions of performance metrics - ilir vs screen


# %%
sns.displot(
    data=auc_results_df,
    x="auROC",
    hue="data_source",
    common_norm=False,
)

# %%
plt.style.use("custom_standard")
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["font.size"] = 16
fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
sns.stripplot(
    data=auc_results_df[auc_results_df["low_count_mask"] == 0],
    x="auROC",
    y="data_source",
    hue="pseudocount",
    alpha=0.5,
    ax=axes[0],
    palette="deep",
)
sns.stripplot(
    data=auc_results_df[auc_results_df["low_count_mask"] == 0],
    x="auPRC",
    y="data_source",
    hue="pseudocount",
    alpha=0.5,
    ax=axes[1],
    palette="deep",
)
axes[0].set_title("auROC")
axes[1].set_title("auPRC")
for ax in axes:
    ax.set_xlim(0.5, 1)
plt.tight_layout()

# %%
fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
sns.stripplot(
    data=other_auc_results_df[other_auc_results_df["pseudocount"] == 0],
    x="auROC",
    y="data_source",
    hue="low_count_mask",
    alpha=0.5,
    ax=axes[0],
    palette="deep",
)
sns.stripplot(
    data=other_auc_results_df[other_auc_results_df["pseudocount"] == 0],
    x="auPRC",
    y="data_source",
    hue="low_count_mask",
    alpha=0.5,
    ax=axes[1],
    palette="deep",
    jitter=True,
)
axes[0].set_title("auROC")
axes[1].set_title("auPRC")
for ax in axes:
    ax.set_xlim(0.5, 1)
    legend = ax.get_legend()
    # legend.set_bbox_to_anchor((1, 1))  # Move legend outside plot
    # legend.set_frame_on(False)         # Remove legend box
    plt.setp(legend.get_texts(), fontsize=12)
    plt.tight_layout()


# %% [markdown]
# ## compare ilir background to proteome background

# %%
auc_results_df_long = auc_results_df.melt(
    id_vars=[
        "foreground",
        "background",
        "low_count_mask",
        "pseudocount",
        "data_source",
    ],
    value_vars=["auROC", "auPRC"],
    var_name="metric",
    value_name="score",
)
temp = auc_results_df_long[auc_results_df_long["data_source"] == "ilir"].copy()
# turn background into separate columns
temp = temp.pivot_table(
    index=["foreground", "low_count_mask", "pseudocount", "metric"],
    columns="background",
    values="score",
).reset_index()
fig, ax = plt.subplots()
sns.scatterplot(
    data=temp,
    x="proteome",
    y="ilir_bg",
    hue="metric",
    ax=ax,
)
ax.plot(
    [0.6, 0.85],
    [0.6, 0.85],
    linestyle="--",
    color="black",
)


# %% [markdown]
# # explore scores of individual sequences

# %%
other_scores["pssm_id"] = (
    other_scores[["foreground", "background", "low_count_mask", "pseudocount"]]
    .astype(str)
    .agg("-".join, axis=1)
)


# %% [markdown]
# ## score distributions


# %%
sns.displot(
    data=other_scores,
    x="pssm_score_fraction",
    hue="foreground",
    kind="hist",
    stat="density",
    common_norm=False,
    palette="deep",
    element="step",  # step histogram (not filled)
    fill=False,  # do not fill the histogram
)
plt.xlim(0, 1)

# %%
plt.rcParams["legend.fontsize"] = 10
fig, ax = plt.subplots(figsize=(8, 4))
ax = sns.ecdfplot(
    data=other_scores,
    x="pssm_score_fraction",
    hue="foreground",
    stat="proportion",
    palette="deep",
    ax=ax,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("PSSM score (fractional)")
ax.set_ylabel("Cumulative proportion")
# put the legend outside the plot
legend = ax.get_legend()
legend.set_bbox_to_anchor((1, 1))  # Move legend outside plot
legend.set_frame_on(False)  # Remove legend box
plt.setp(legend.get_texts(), fontsize=12)
plt.tight_layout()

# %%
sns.displot(
    data=other_scores,
    x="pssm_score",
    hue="foreground",
    kind="hist",
    stat="density",
    common_norm=False,
    palette="deep",
    element="step",  # step histogram (not filled)
    fill=False,  # do not fill the histogram
    aspect=1.5,  # aspect ratio of the plot
)


# %%
sns.displot(
    data=other_scores,
    x="pssm_score_normalized",
    hue="foreground",
    kind="hist",
    stat="density",
    common_norm=False,
    palette="deep",
    element="step",  # step histogram (not filled)
    fill=False,  # do not fill the histogram
    aspect=1.5,  # aspect ratio of the plot
)

# %%
other_scores.groupby(["foreground", "background", "low_count_mask", "pseudocount"])[
    "sequence"
].count().value_counts()

# %% [markdown]
# ## individual sequences

# %%
label_map = df[["sequence", "true label"]].set_index("sequence").to_dict()["true label"]

# raw pssm scores
other_scores_pivot = other_scores.pivot(
    columns="pssm_id", index="sequence", values="pssm_score"
)
correlations = other_scores_pivot.corr(method="spearman")
other_scores_pivot["label"] = other_scores_pivot.index.map(label_map)

# fractional pssm scores
other_scores_pivot_f = other_scores.pivot(
    columns="pssm_id", index="sequence", values="pssm_score_fraction"
)
correlations_f = other_scores_pivot_f.corr(method="spearman")
other_scores_pivot_f["label"] = other_scores_pivot_f.index.map(label_map)

# normalized pssm scores
other_scores_pivot_n = other_scores.pivot(
    columns="pssm_id", index="sequence", values="pssm_score_normalized"
)
correlations_f = other_scores_pivot_n.corr(method="spearman")
other_scores_pivot_n["label"] = other_scores_pivot_n.index.map(label_map)

for i in other_scores_pivot_f.columns:
    print(i)


# %%
def match_regex(seq, re_pattern1, re_pattern2):
    if re.fullmatch(re_pattern1, seq):
        return re_pattern1
    elif re.fullmatch(re_pattern2, seq):
        return re_pattern2
    else:
        return np.nan

exp_f = other_scores_pivot_f.copy()
exp_f = exp_f.reset_index(names="sequence")
exp_n = other_scores_pivot_n.copy()
exp_n = exp_n.reset_index(names="sequence")

old_regex = "...[FWY]..[LVI]"
new_regex = "...[FWY]..[WFMY]"

exp_f["lir_type"] = exp_f["sequence"].apply(
    lambda x: match_regex(x, old_regex, new_regex)
)
exp_f.to_csv("./pssm_scores_pivot_fraction.csv", index=False)
exp_n["lir_type"] = exp_n["sequence"].apply(
    lambda x: match_regex(x, old_regex, new_regex)
)
exp_n.to_csv("./pssm_scores_pivot_normalized.csv", index=False)

# %%
exp_f

# %% [markdown]
# ### correlation of pssm scores


# %%
sns.heatmap(
    correlations_f,
    cmap='coolwarm',
    center=0,
    annot=False,
    fmt='.2f',
    linewidths=0.5,
    cbar_kws={"shrink": .8},
)


# %%
fig, ax = plt.subplots()
sns.scatterplot(
    data=other_scores_pivot_f,
    x="screen_all_binders-proteome-0-1.0",
    y="ilir_binders-proteome-0-1.0",
    alpha=0.5,
    hue="label",
    ax=ax,
)
ax.plot(
    np.linspace(0.3, 1, 100),
    np.linspace(0.3, 1, 100),
    linestyle="-",
    color="black",
)

# %%
fig, ax = plt.subplots()
sns.scatterplot(
    data=other_scores_pivot_n,
    x="screen_all_binders-proteome-0-1.0",
    y="ilir_binders-proteome-0-1.0",
    alpha=0.5,
    hue="label",
    ax=ax,
)
ax.plot(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100),
    linestyle="-",
    color="black",
)

# %%
columns2compare = ["screen_all_binders-proteome-0-1.0", "ilir_binders-proteome-0-1.0"]
pssms.plot_logo(pssms_dict[columns2compare[0]])
pssms.plot_logo(pssms_dict[columns2compare[1]])

# %%
pssms_dict.keys()

# %%

temp = other_scores_pivot_f.copy()
temp = temp[columns2compare + ["label"]]
for i in columns2compare:
    temp[f"{i}_rank"] = temp[i].rank(ascending=False, method="min")

fig, ax = plt.subplots()
sns.scatterplot(
    data=temp,
    x=columns2compare[0] + "_rank",
    y=columns2compare[1] + "_rank",
    alpha=0.5,
    hue="label",
    ax=ax,
)
temp = temp.reset_index(names="sequence")
temp2 = temp.copy().melt(
    id_vars=["sequence", "label"],
    value_vars=columns2compare,
    var_name="pssm_id",
    value_name="score",
)


# %%
sns.boxplot(
    data=temp2,
    x="pssm_id",
    y="score",
    hue="label",
)
# %%
sns.displot(
    data=temp2,
    x="score",
    hue="label",
    col="pssm_id",
    kind="hist",
    common_norm=False,
    stat="density",
    fill=True,
    palette="deep",
)

# %% [markdown]
# ## split sequences into "old" and "new" lirs


# %%
temp_n = other_scores_pivot_n.copy()
temp_n = temp_n[columns2compare + ["label"]]
temp_n = temp_n.reset_index(names="sequence")
temp_n_long = temp_n.copy().melt(
    id_vars=["sequence", "label"],
    value_vars=columns2compare,
    var_name="pssm_id",
    value_name="score",
)

# %%
old_regex = "...[FWY]..[LVI]"
new_regex = "...[FWY]..[WFMY]"


def match_regex(seq, re_pattern1, re_pattern2):
    if re.fullmatch(re_pattern1, seq):
        return re_pattern1
    elif re.fullmatch(re_pattern2, seq):
        return re_pattern2
    else:
        return np.nan


temp_n_long["lir_type"] = temp_n_long["sequence"].apply(
    lambda x: match_regex(x, old_regex, new_regex)
)
temp_n["lir_type"] = temp_n["sequence"].apply(
    lambda x: match_regex(x, old_regex, new_regex)
)

# %%
sns.stripplot(
    data=temp_n_long[temp_n_long["label"] == 1],
    x="lir_type",
    y="score",
    hue="pssm_id",
    alpha=0.5,
    jitter=True,
    palette="muted",
    dodge=True,
)


# %%
sns.stripplot(
    data=temp_n_long,
    x="pssm_id",
    y="score",
    hue="label",
    alpha=0.5,
    jitter=True,
    dodge=True,
)
# rotate x-axis labels
plt.xticks(rotation=90)

# %%
sns.boxplot(
    data=temp_n_long,
    x="pssm_id",
    y="score",
    hue="label",
)
sns.displot(
    data=temp_n_long,
    x="score",
    hue="label",
    col="pssm_id",
    kind="hist",
    common_norm=False,
    stat="density",
    fill=True,
    palette="deep",
)

# %%
temp

# %%
import altair as alt
from vega_datasets import data
import pandas as pd

temp_long = pd.melt(
    temp,
    id_vars=["sequence", "label"],
    value_vars=[i + "_rank" for i in columns2compare],
    var_name="PSSM",
    value_name="Rank",
)
chart = (
    alt.Chart(temp_long)
    .mark_line(point=True)
    .encode(
        x=alt.X("PSSM:N", title=None),
        y=alt.Y("Rank:O", sort="ascending", title="Rank"),
        color="sequence:N",
        detail="sequence:N",
    )
    .properties(
        title="Bump Chart: Sequence Rankings by Two Scores", width=800, height=900
    )
)
chart

# %%
temp3 = other_scores_pivot.copy()
temp3 = temp3[columns2compare + ["label"]].reset_index(names="sequence")


temp3[['sequence', 'screen_all_binders-proteome-0-1.0']].values[:,0]

def pssm_weighted_logo(
    df: pd.DataFrame,
    score_column: str,
):
    x = df[['sequence', score_column]].values
    c = pssms.seqlist_2_counts_matrix(x[:, 0], x[:, 1])
    pssms.plot_logo(c)



pssm_weighted_logo(temp3, "screen_all_binders-proteome-0-1.0")
pssm_weighted_logo(temp3, "ilir_binders-proteome-0-1.0")

pssm_weighted_logo(temp3[temp3['label']==1], "screen_all_binders-proteome-0-1.0")
pssm_weighted_logo(temp3[temp3['label']==1], "ilir_binders-proteome-0-1.0")


# %%
temp3

# %%
# Example: 2D embedding of sequences using UMAP on one-hot encoded sequences

# Prepare sequences for embedding
seqs = other_scores_pivot_f.index.tolist()
# One-hot encode sequences (assuming all are same length)
encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(np.array([list(s) for s in seqs]))

# Compute 2D UMAP embedding
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)

# Add embedding to DataFrame
embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=seqs)
embedding_df["label"] = other_scores_pivot_f["label"]

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=embedding_df, x="UMAP1", y="UMAP2", hue="label", alpha=0.7, palette="deep"
)
plt.title("2D UMAP embedding of sequences")
plt.tight_layout()
plt.show()


# %%
fractional_score("".join(pssm.idxmax(axis=1).values), pssm)
fractional_score("".join(pssm.idxmin(axis=1).values), pssm)


pssm.idxmin(axis=1)
# %%
embedding_df

# %%
