# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
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
plt.style.use('custom_standard')
plt.style.use('custom_small')
import seaborn as sns
import lir_proteome_screen_pssm.sequence_utils as seqtools

from pathlib import Path

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# Thoughts from first notebook
#
#
# - more sequence information from hits at z-scores greater than ~2.4. But sequences in this range don't have many "new-lir" hits, so they look like the previously known binding sequences. 
#   - Could using a higher z-score threshold be too restrictive? We know that lc3b can bind these "new-lirs" (lots of good BLI data) so we would want a pssm to capture that, ideally. It might be beyond a PSSM to do this while also retaining the sequence features that define a good canonical/biological binder
# - Low counts are causing some really weird PSSMs. mostly artifacts. Need to find a way to deal with this that actually makes sense logically and that performs well. Will probably be a custom solution, involving a low count cutoff that defaults to 0 rather than penalizing based on a background distribution.
# - I should consider composite scores from multiple PSSMs.

# %% [markdown]
# Ideas
# - try to find a stringent nonbinder set
# - try to find a non-canonical lir binder set to see if there are any sequence features that are common to them
# - deal with Low counts to generate a more reliable pssm
#
# <br>
#
# Features that I want a PSSM (or any score) to have (ideally).
# - captures the major contributions to binding (the major sequence features that are enriched in the binding sequences)
#   - W at the first position, N-terminal negative charge, etc.
# - penalizes sequence features that are enriched in non-binding sequences? (if there even are any features that stick out in a pssm - this seems unlikely)
# - allows for flexibility at the last position to capture the new motif.

# %% [markdown]
# ---

# %% [markdown]
# TODO:
# - Look for "new-lir" sequence preferences
# - Attempt to define a higher confidence non-binder set

# %%
def normalize_positions(mat: pd.DataFrame) -> pd.DataFrame:
    """
    normalize each position in the matrix so that it sums to 1

    Parameters
    ----------
    mat : pd.DataFrame
        DataFrame in the logomaker format (rows = positions, columns = amino acids)
    
    Returns
    -------
    pd.DataFrame
        DataFrame in the logomaker format (rows = positions, columns = amino acids)
        where each position sums to 1
    """
    freq_df = mat.copy()
    freq_df = freq_df.div(freq_df.sum(axis=1), axis=0)
    return freq_df


# %% [markdown]
# # "non-canonical" [FWY]..[LVIWFMY] but not [FWY]..[LVI]?

# %%
def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches


REGEX = "...[FWY]..[WFY]..."
df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
df["regex_matches"] = df.apply(get_regex_matches, axis=1, regex=REGEX)
df["num_regex_matches"] = df["regex_matches"].apply(lambda x: len(x))
df["num_regex_matches"].value_counts()
df_multi = df[df["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = df[df["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
df = pd.concat([df_multi, df_single])
df[["motif", "motif_start", "motif_end"]] = pd.DataFrame(
    df["regex_matches"].tolist(), index=df.index
)

# %%
df

# %%
temp_fg = df[df['avg_z_score']>0].copy()
# with open('temp.fasta', 'w') as f:
#     for c,i in enumerate(temp_fg.motif.unique()):
#         f.write(f">{c}\n")
#         f.write(f"{i}\n")
print(len(df[df['avg_z_score']>0].motif.unique()))

temp_bg = df[df['Input Count'] > 100].copy()
print(len(temp_bg.motif.unique()))
# with open('temp_bg.fasta', 'w') as f:
#     for c,i in enumerate(temp_bg.motif.unique()):
#         f.write(f">{c}\n")
#         f.write(f"{i}\n")

# %%
fig, ax = plt.subplots(figsize=(10, 2.5))
pssms.plot_logo(pssms.alignment_2_counts(temp_fg.motif.unique()), ax=ax)
ax.set_title("foreground - new-lir motifs with avg z-score > 0")
fig, ax = plt.subplots(figsize=(10, 2.5))
pssms.plot_logo(pssms.alignment_2_counts(temp_bg.motif.unique()), ax=ax)
ax.set_title("background - new-lir motifs in input library with input count > 100")
fig, ax = plt.subplots(figsize=(10, 2.5))
pssms.plot_logo(normalize_positions(pssms.alignment_2_counts(temp_fg.motif.unique())) - normalize_positions(pssms.alignment_2_counts(temp_bg.motif.unique())), ax=ax)
ax.set_title("foreground - background")


# %% [markdown]
# binned by z-score

# %%
# define ranges between min and max avg_z_score
def get_z_score_range(df, min_z_score, max_z_score):
    df_filt = df[(df["avg_z_score"] >= min_z_score) & (df["avg_z_score"] < max_z_score)].copy()
    return df_filt


min_z_score = df["avg_z_score"].min()
max_z_score = df["avg_z_score"].max()
z_score_range = np.linspace(min_z_score, max_z_score, 5)
z_score_range = np.round(z_score_range, 2)
# split range into bins
z_score_bins = []
for i in range(len(z_score_range) - 1):
    z_score_bins.append([z_score_range[i], z_score_range[i + 1]])
z_score_bins[-1][1] += 0.01


for i, z_score_bin in enumerate(z_score_bins):
    min_z_score = z_score_bin[0]
    max_z_score = z_score_bin[1]
    df_filt = get_z_score_range(df, min_z_score, max_z_score)
    ax = pssms.plot_logo(pssms.alignment_2_counts(df_filt["motif"]))
    ax.set_title(f"z-score: {min_z_score} - {max_z_score}")

# %%
df_filt = get_z_score_range(df, 2, 5)
ax = pssms.plot_logo(pssms.alignment_2_counts(df_filt["motif"]))
ax.set_title(f"z-score: {df_filt['avg_z_score'].min()} - {df_filt['avg_z_score'].max()}")

# %% [markdown]
# binned by rank

# %%
df_zs = df[~df["avg_z_score"].isna()].copy()
df_zs = df_zs.sort_values(by = "avg_z_score", ascending = False)
df_zs = df_zs.reset_index(drop=True)
# create 10 ranges from df_zs.index
df_zs["z_score_range"] = pd.cut(df_zs.index, bins=20, labels=False)
n_plots = len(df_zs["z_score_range"].unique())
fig, axes = plt.subplots(nrows = n_plots, ncols = 1, figsize=(10, 2.5*n_plots))
for i, ax in zip(df_zs["z_score_range"].unique(), axes):
    df_filt = df_zs[df_zs["z_score_range"] == i]
    pssms.plot_logo(pssms.alignment_2_counts(df_filt["motif"]), ax=ax)
    ax.set_title(f"Z-score range: {df_filt['avg_z_score'].min()} - {df_filt['avg_z_score'].max()}") 
plt.tight_layout()


# %% [markdown]
# ## non-binders

# %%
def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches


REGEX = "....[FWY]..[LVIWFY]"
df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
df["regex_matches"] = df.apply(get_regex_matches, axis=1, regex=REGEX)
df["num_regex_matches"] = df["regex_matches"].apply(lambda x: len(x))
df["num_regex_matches"].value_counts()
df_multi = df[df["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = df[df["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
df = pd.concat([df_multi, df_single])
df[["motif", "motif_start", "motif_end"]] = pd.DataFrame(
    df["regex_matches"].tolist(), index=df.index
)

# %%
df

# %%
print(len(df))
df = df[df["Count sum"] > 40].copy()
print(len(df))
df=df.reset_index(drop=True)

# %%
# declare the first one as the "WT"
count_cols = [i for i in df.columns if "Count" in i and i != "Count sum"]
wt_counts = df.loc[0, count_cols]
wt_counts

# %%
for i in count_cols:
    df[i.split(" ")[0] + " - WT normalized"] = np.log2(df[i] / wt_counts[i])


# %%
df

# %%
df2 = df.copy()
# drop rows with -inf values
df2 = df2.replace(-np.inf, np.nan)
df2 = df2.dropna(subset=["Input - WT normalized", "1 - WT normalized"])
df2["input to sort1 difference"] =  df2["1 - WT normalized"] - df2["Input - WT normalized"]

# %%
df2["input to sort1 difference"].hist(bins=100)

# %%
df2 = df2.sort_values(by="input to sort1 difference")
df2.head(2000)

# %%
pssms.plot_logo(pssms.alignment_2_counts(df2.head(2000).motif.unique()))

# %%
pssms.plot_logo(pssms.alignment_2_counts(df2.head(200).motif.unique()))

# %%
pssms.plot_logo(pssms.alignment_2_counts(df.sort_values(by="ER 1", ascending=True).head(200).motif.unique()))

# %%
pssms.plot_logo(pssms.alignment_2_counts(df.sort_values(by="ER 1", ascending=True).head(2000).motif.unique()))

# %% [markdown]
# ## I can't find a nonbinder set with distinct sequence features. This makes a lot of sense, because the non-binder sequences are going to be really diverse. Not sure I should've expected to find anything
