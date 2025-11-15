# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%

from lir_proteome_screen_pssm import environment as env
import pandas as pd
import lir_proteome_screen_pssm.sequence_utils as seqtools
import re
import numpy as np
import copy
import lir_proteome_screen_pssm.data_loaders as dl
from pathlib import Path
import lir_proteome_screen_pssm.pssms as pssms
import matplotlib.pyplot as plt
# plt.style.use("lir_proteome_screen_pssm.lir")
plt.style.use("custom_standard")
mm = 1 / 25.4
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
%load_ext autoreload
%autoreload 2


def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches

def import_full_data_table():
    full_data_table = pd.read_csv(env.RAWFILEPATHS.full_screening_table_2, sep=',')
    # full_data_table['ID'] = 'PLR' + full_data_table['ID']
    full_data_table = full_data_table[~full_data_table['ID'].str.contains(r'HPQ', regex=True)]
    full_data_table = full_data_table[full_data_table['ID']!='PLRASQGSDDDWDDEWDDSSTVADEPGALGSGAYPDLDG'] # For this sequence, we know that the actual binding motif is WDDEW from lir_central
    assert full_data_table.duplicated(subset=["ID"], keep=False).sum() == 0, "There should be no duplicate IDs in the full data table"  
    return full_data_table


def import_background(regex, remove_multi_motif_sequences=True):
    full_data_table = import_full_data_table()
    bg_df = full_data_table[full_data_table['Input Count'] >= 50].copy()
    bg_df["regex_matches"] = bg_df.apply(get_regex_matches, axis=1, regex=regex)
    bg_df["num_regex_matches"] = bg_df["regex_matches"].apply(lambda x: len(x))
    df_multi = bg_df[bg_df["num_regex_matches"] > 1].copy()
    df_multi = df_multi.explode("regex_matches")
    df_single = bg_df[bg_df["num_regex_matches"] == 1].copy()
    df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
    bg_df = pd.concat([df_multi, df_single])
    bg_df[["lir_sequence", "motif_start", "motif_end"]] = pd.DataFrame(
        bg_df["regex_matches"].tolist(), index=bg_df.index
    )
    if remove_multi_motif_sequences:
        print(f"removing {len(bg_df[bg_df['num_regex_matches'] > 1])} lirs from sequences with multiple motifs")
        check_l = len(bg_df[bg_df["num_regex_matches"] == 1])
        bg_df = bg_df.drop_duplicates(keep=False, subset="ID")
        assert len(bg_df) == check_l, "deduplication yields different length than number with = 1 motif. something is very wrong"
        assert len(bg_df) == len(df_single), "deduplication yields different length than number with = 1 motif. something is very wrong"
    print(len(bg_df))
    return bg_df
# %%

REGEX = seqtools.regex2overlapping("...[FWY]..[ILVWFY]")
input_library = list(import_background(regex=REGEX, remove_multi_motif_sequences=True)['lir_sequence'].unique())
# bg_df = import_background(regex=REGEX, remove_multi_motif_sequences=True)
# input_library = list(bg_df['lir_sequence'].unique())
version = "v2"
PROCESSED_SEQUENCE_TABLES = dl.get_processed_sequence_tables(version=version)
TEST_SETS = dl.get_test_sets(version=version)
BGFREQS = dl.get_background_frequencies(version=version)
binders = PROCESSED_SEQUENCE_TABLES.screen_binders["7mer"].to_list()
high_z_binders = PROCESSED_SEQUENCE_TABLES.screen_binders[PROCESSED_SEQUENCE_TABLES.screen_binders["avg_z_score"] >= 2.3]["7mer"].to_list()
len(binders), len(high_z_binders), len(input_library)

# %% [markdown]
# # counts and frequency logos

# %%

FSIZE = (9, 4)


def plot_counts(counts_df, ax=None, title='Counts logo'):
    if ax is None:
        fig, ax = plt.subplots(figsize=FSIZE)
    ax = pssms.plot_logo(counts_df, ax=ax)
    ax.set_ylabel('Counts')
    ax.set_xlabel('Position')
    ax.set_title(title)
    ax.set_title(title, pad=20)
    return ax


def plot_frequencies(freqs_df, ax=None, title='Frequencies logo'):
    if ax is None:
        fig, ax = plt.subplots(figsize=FSIZE)
    ax = pssms.plot_logo(freqs_df, ax=ax)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Position')
    ax.set_title(title, pad=20)
    return ax



binder_counts = pssms.seqlist_2_counts_matrix(binders, pseudocount=0)
high_z_binder_counts = pssms.seqlist_2_counts_matrix(high_z_binders, pseudocount=0)
input_counts = pssms.seqlist_2_counts_matrix(input_library, pseudocount=0)
binder_freqs = pssms.normalize_positions(binder_counts)
high_z_binder_freqs = pssms.normalize_positions(high_z_binder_counts)
input_freqs = pssms.normalize_positions(input_counts)

# plot_counts(binder_counts, title='Foreground counts (binders)')
plot_counts(input_counts, title='Background counts (input library)')
plot_counts(high_z_binder_counts, title='Foreground counts (best binders)')
# plot_frequencies(binder_freqs, title='Foreground frequencies (binders)')
plot_frequencies(input_freqs, title='Background frequencies (input library)')
plot_frequencies(high_z_binder_freqs, title='Foreground frequencies (best binders)')

# %%
fig, ax = plt.subplots(figsize=(7,7))
pssms.plot_logo_as_heatmap(high_z_binder_freqs, ax=ax)
# Customize the appearance of the plot
# %%
fig, ax = plt.subplots(figsize=(7,7))
pssms.plot_logo_as_heatmap(binder_counts, ax=ax)
# %%

fig, ax = plt.subplots(figsize=(7,7))
pssms.plot_logo_as_heatmap(binder_counts + 1, ax=ax)

# %%
fig, ax = plt.subplots(figsize=(7,7))
pssms.plot_logo_as_heatmap(pssms.normalize_positions(binder_counts + 1), ax=ax)

# %%
FSIZE = (9, 4)
def make_and_plot_pssm(foreground_seqlist, background_seqlist, pseudocount=1, title='PSSM logo', plot=True):
    fg_counts = pssms.seqlist_2_counts_matrix(foreground_seqlist, pseudocount=pseudocount)
    bg_counts = pssms.seqlist_2_counts_matrix(background_seqlist, pseudocount=pseudocount)
    pssm = pssms.make_pssm(
        df_counts=fg_counts,
        bg=pssms.normalize_positions(bg_counts)
    )
    if plot:
        fig, ax = plt.subplots(figsize=FSIZE)
        ax = pssms.plot_logo(pssm, ax = ax)
        ax.set_title(title)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        pssms.plot_logo_as_heatmap(pssms.normalize_positions(fg_counts), ax=ax1)
        ax1.set_title('Foreground frequency (binders)')
        pssms.plot_logo_as_heatmap(pssms.normalize_positions(bg_counts), ax=ax2)
        ax2.set_title('Background frequency (input library)')
    return pssm


for pcount in [0.1, 1]:
    p = make_and_plot_pssm(
        foreground_seqlist=high_z_binders,
        background_seqlist=input_library,
        pseudocount=pcount,
        title=f'log-odds PSSM; pseudocount={pcount}',
        plot=True
    )
    fig, ax = plt.subplots(figsize=(7,7))
    pssms.plot_logo_as_heatmap(p, ax=ax)
    ax.set_title(f'log-odds PSSM; pseudocount={pcount}')
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(fg_counts, title='Foreground counts (binders)', ax=ax)
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(input_counts, title='Background counts (input library)', ax=ax)
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(fg_freqs, title='Foreground frequencies (binders)', ax=ax)
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(input_freqs, title='Background frequencies (input library)', ax=ax)
# %%
FSIZE = (9, 4)
def make_and_plot_pssm(foreground_seqlist, background_seqlist, title='PSSM logo', plot=True):
    fg_counts = pssms.seqlist_2_counts_matrix(foreground_seqlist, pseudocount=1/len(foreground_seqlist))
    bg_counts = pssms.seqlist_2_counts_matrix(background_seqlist, pseudocount=1/len(background_seqlist))
    pssm = pssms.make_pssm(
        df_counts=fg_counts,
        bg=pssms.normalize_positions(bg_counts)
    )
    if plot:
        fig, ax = plt.subplots(figsize=FSIZE)
        ax = pssms.plot_logo(pssm, ax = ax)
        ax.set_title(title)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        pssms.plot_logo_as_heatmap(pssms.normalize_positions(fg_counts), ax=ax1)
        ax1.set_title('Foreground frequency (binders)')
        pssms.plot_logo_as_heatmap(pssms.normalize_positions(bg_counts), ax=ax2)
        ax2.set_title('Background frequency (input library)')
    return pssm



p = make_and_plot_pssm(
    foreground_seqlist=high_z_binders,
    background_seqlist=input_library,
    title=f'log-odds PSSM; pseudocount=1/n_seqs',
    plot=True
)
fig, ax = plt.subplots(figsize=(7,7))
pssms.plot_logo_as_heatmap(p, ax=ax)
ax.set_title(f'log-odds PSSM; pseudocount=1/n_seqs')
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(fg_counts, title='Foreground counts (binders)', ax=ax)
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(input_counts, title='Background counts (input library)', ax=ax)
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(fg_freqs, title='Foreground frequencies (binders)', ax=ax)
# fig, ax = plt.subplots(figsize=fsize)
# pssms.plot_logo(input_freqs, title='Background frequencies (input library)', ax=ax)





# %%

def select_random_items(items, n, replace=False, seed=None):
    """
    Randomly select n items from a list.

    Args:
        items: Sequence to sample from.
        n: Number of items to select.
        replace: If True, sample with replacement.
        seed: Optional seed for reproducibility.

    Returns:
        List of selected items.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not replace and n > len(items):
        raise ValueError("n cannot exceed the length of items when replace=False")

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if replace:
        indices = rng.integers(0, len(items), size=n)
    else:
        indices = rng.choice(len(items), size=n, replace=False)
    return [items[i] for i in indices]



FSIZE = (9, 4)
def make_and_plot_pssm(foreground_seqlist, background_seqlist, pseudocount=1, title='PSSM logo', plot=True):
    fg_counts = pssms.seqlist_2_counts_matrix(foreground_seqlist, pseudocount=pseudocount)
    bg_counts = pssms.seqlist_2_counts_matrix(background_seqlist, pseudocount=pseudocount)
    pssm = pssms.make_pssm(
        df_counts=fg_counts,
        bg=pssms.normalize_positions(bg_counts)
    )
    if plot:
        fig, ax = plt.subplots(figsize=FSIZE)
        ax = pssms.plot_logo(pssm, ax = ax)
        ax.set_title(title)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        pssms.plot_logo_as_heatmap(pssms.normalize_positions(fg_counts), ax=ax1)
        ax1.set_title('Foreground frequency (binders)')
        pssms.plot_logo_as_heatmap(pssms.normalize_positions(bg_counts), ax=ax2)
        ax2.set_title('Background frequency (input library)')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
        pssms.plot_logo(pssms.normalize_positions(fg_counts), ax=ax1)
        ax1.set_title('Foreground frequency (binders)')
        pssms.plot_logo(pssms.normalize_positions(bg_counts), ax=ax2)
        ax2.set_title('Background frequency (input library)')
        plt.tight_layout()
    return pssm

temp = select_random_items(input_library, n=70, replace=False, seed=42)
for pcount in [0.1, 1]:
    p = make_and_plot_pssm(
        foreground_seqlist=high_z_binders,
        background_seqlist=temp,
        pseudocount=pcount,
        title=f'log-odds PSSM; pseudocount={pcount}',
        plot=True
    )
    fig, ax = plt.subplots(figsize=(7,7))
    pssms.plot_logo_as_heatmap(p, ax=ax)
    ax.set_title(f'log-odds PSSM; pseudocount={pcount}')

# %%

background = BGFREQS.proteome
fg_counts = pssms.seqlist_2_counts_matrix(high_z_binders)
p = pssms.make_pssm(
    df_counts=fg_counts,
    bg=background
)
fig, ax = plt.subplots(figsize=(9, 4))
pssms.plot_logo(p, ax = ax, title='PSSM - no pseudocount; proteome background')
fig, ax = plt.subplots(figsize=(7,7))
pssms.plot_logo_as_heatmap(p, ax=ax)


# %%

import logomaker as lm
infodf = lm.transform_matrix(fg_counts, from_type='counts', to_type='information')

pssms.plot_logo(infodf)



# %%