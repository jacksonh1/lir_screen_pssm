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
plt.style.use('custom_standard')
plt.style.use('custom_small')
import seaborn as sns

from pathlib import Path

# %load_ext autoreload
# %autoreload 2

# %%
env.RAWFILEPATHS

# %% [markdown]
# # screening

# %%
sc_lirs = pd.read_csv(env.RAWFILEPATHS.screening_hits_table)
sc_binders = sc_lirs[sc_lirs['Bind/Nonbind'] == 'Bind'].copy()
sc_nbs = sc_lirs[sc_lirs['Bind/Nonbind'] == 'Nonbind'].copy()

# %%
sc_lirs

# %% [markdown]
# only unique binders

# %%
binders = sc_binders['first_8_residues'].to_list()
assert all([len(b) == 8 for b in binders]), "All binders should be 8 residues long"
c_binders = pssms.alignment_2_counts(list(set(binders)))
ax = pssms.plot_logo(c_binders);
ax.set_title('binding LIRs from screen');
f_binders = c_binders.div(c_binders.sum(axis=1), axis=0)
ax = pssms.plot_logo_as_heatmap(c_binders);
# ax = pssms.plot_logo_as_heatmap(f_binders);

# %% [markdown]
# binders with duplicates

# %%
binders = sc_binders['first_8_residues'].to_list()
assert all([len(b) == 8 for b in binders]), "All binders should be 8 residues long"
c_binders = pssms.alignment_2_counts(list(set(binders)))
ax = pssms.plot_logo(c_binders);
ax.set_title('binding LIRs from screen');
f_binders = c_binders.div(c_binders.sum(axis=1), axis=0)
ax = pssms.plot_logo_as_heatmap(c_binders);
# ax = pssms.plot_logo_as_heatmap(f_binders);

# %%
nonbinders = sc_nbs['first_8_residues'].to_list()
assert all([len(b) == 8 for b in nonbinders]), "All nonbinders should be 8 residues long"
c_nbs = pssms.alignment_2_counts(nonbinders)
ax = pssms.plot_logo(c_nbs)
ax.set_title('nonbinding LIRs from screen');
f_nbs = c_nbs.div(c_nbs.sum(axis=1), axis=0)
ax = pssms.plot_logo_as_heatmap(
    c_nbs.loc[[i for i in c_nbs.index if i != 4 and i!=7], :],
);

# %%
sc_nbs

# %%
undefined_positions = c_nbs.loc[[i for i in c_nbs.index if i != 4 and i!=7], :].copy()
(undefined_positions.sum()/undefined_positions.sum().sum())

# %%
pssms.plot_logo(f_binders - f_nbs, title='nonbinder freq subtracted from binder freq');


# %%
def freq_2_weight_matrix(fg, bg=None):
    """
    Calculate the weight matrix for a given foreground and background frequency matrix.
    The weight matrix is calculated as log2(fg / bg) for each position in the matrix.
    if no bg is provided, it will be set to a flat distribution.
    zero values in fg or bg will be SET TO 0 in the weight matrix. this should mean that 
    pseudocounts are not needed?
    """
    if isinstance(bg, dict):
        assert set(bg.keys()) == set(fg.columns), "Background dictionary keys must match foreground columns"
        bg = pd.DataFrame(bg, index=fg.index)
    elif bg is None:
        bg = pd.DataFrame(1, index=fg.index, columns=fg.columns)
        bg = bg.div(bg.sum(axis=1), axis=0)
    assert isinstance(bg, pd.DataFrame), "Background must be a DataFrame, dictionary, or None"
    assert (fg.columns == bg.columns).all(), "Foreground and background matrices must have the same columns"
    assert (fg.index == bg.index).all(), "Foreground and background matrices must have the same index"
    weight_matrix = fg.copy()
    for col in fg.columns:
        for ind in fg.index:
            # Calculate the weight using log2(fg / bg)
            if fg.loc[ind, col] == 0 or bg.loc[ind, col] == 0:
                weight_matrix.loc[ind, col] = 0
            else:
                weight_matrix.loc[ind, col] = np.log2(fg.loc[ind, col] / bg.loc[ind, col])
    return weight_matrix


def freq_2_information_matix(fg, bg=None):
    """
    Calculate the information matrix for a given foreground and background frequency matrix.
    The information matrix is calculated as fg * sum(fg*log2(fg/bg)) for each position in the matrix.
    if no bg is provided, it will be set to a flat distribution.
    """
    information_matrix = fg.copy()
    weight_matrix = freq_2_weight_matrix(fg, bg)
    col_information = []
    for ind in fg.index:
        i_i = 0
        for col in fg.columns:
            i_i+=fg.loc[ind, col] * weight_matrix.loc[ind, col]
        for col in fg.columns:
            information_matrix.loc[ind, col] = fg.loc[ind, col] * i_i
        col_information.append(i_i)
    return information_matrix, col_information


# df = f_binders.copy()
# for col in f_binders.columns:
#     for ind in f_binders.index:
#         if f_nbs.loc[ind, col] == 0:
#             df.loc[ind, col] = 0
#         else:
#             df.loc[ind, col] = np.log2(f_binders.loc[ind, col] / f_nbs.loc[ind, col])
# df2=freq_2_weight_matrix(f_binders, f_nbs)

# %%
c_binders_2_5 = pssms.alignment_2_counts(sc_binders[sc_binders['avg_z_score']>=2.5]['first_8_residues'].to_list())
f_binders_2_5 = c_binders_2_5.div(c_binders_2_5.sum(axis=1), axis=0)

# %% [markdown]
# ### messing around with some weight matrices

# %%
pssms.plot_logo(freq_2_weight_matrix(f_binders)) # no pseudocounts, flat bg
pssms.plot_logo(freq_2_weight_matrix(f_binders, f_nbs)) # no pseudocounts, nonbinder bg
pssms.plot_logo(lm.transform_matrix(f_binders, from_type='probability', to_type='weight', pseudocount=1, background=f_nbs)) # pseudocounts, nonbinder bg
pssms.plot_logo(freq_2_weight_matrix(f_binders_2_5, f_nbs)) # no pseudocounts, nonbinder bg, avg z score > 2.5
df3, col_info3 = freq_2_information_matix(f_binders)
pssms.plot_logo(df3) # no pseudocounts, flat bg, information matrix
df4, col_info4 = freq_2_information_matix(f_binders, f_nbs)
pssms.plot_logo(df4) # no pseudocounts, nonbinder bg, information matrix
print(col_info3)
print(col_info4)
print(np.sum(col_info3))
print(np.sum(col_info4))
# pssms.plot_logo(freq_2_weight_matrix(f_binders, f_nbs)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
pssms.plot_logo(lm.alignment_to_matrix(binders, to_type='weight', pseudocount=1), ax = ax)
pssms.plot_logo_as_heatmap(lm.alignment_to_matrix(binders, to_type='weight', pseudocount=1))
fig, ax = plt.subplots(figsize=(10, 5))
bg_temp = lm.alignment_to_matrix(nonbinders, to_type='probability', pseudocount=1)
pssms.plot_logo(lm.alignment_to_matrix(binders, to_type='weight', pseudocount=1, background=bg_temp), ax = ax)

# %% [markdown]
# #### A lot of these fancy matrices don't actually give results that make any sense. The use of pseudocounts and background normalization leads to extremely confusing effects and I suspected the matrices will be very misleading.

# %% [markdown]
# ### plot z-score cutoff vs information content

# %%
sc_binders


# %%
def information_at_z_score_cutoff(binder_df = sc_binders, background = f_nbs, z_score_cutoff=1.7, ax = None):
    binders = binder_df[binder_df['avg_z_score']>=z_score_cutoff]['first_8_residues'].to_list()
    c_binders = pssms.alignment_2_counts(binders)
    f_binders = c_binders.div(c_binders.sum(axis=1), axis=0)
    info_matrix, col_info = freq_2_information_matix(f_binders, bg=background)
    if ax is not None:
        pssms.plot_logo(c_binders, ax = ax)
        ax.set_title(f'z-score cutoff: {z_score_cutoff}');
    return np.sum(col_info), ax


cutoffs = np.arange(1.7, 3.2, 0.15)
# plot each cutoff on a subplot
n_plots = len(cutoffs)
information = []
fig, axs = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots))
for axis, cutoff in zip(axs, cutoffs):
    I, _ = information_at_z_score_cutoff(sc_binders, f_nbs, z_score_cutoff=cutoff, ax=axis)
    information.append(I)
plt.tight_layout()
fig, ax = plt.subplots()
ax.plot(cutoffs, information)
ax.set_xlabel('z-score cutoff')
ax.set_ylabel('information content')
ax.set_title('Information content at different z-score cutoffs')
ax.axhline(y=0, color='r', linestyle='--')


# %%
def information_at_z_score_cutoff_flipped(binder_df = sc_binders, background = f_nbs, z_score_cutoff=1.7, ax = None):
    binders = binder_df[binder_df['avg_z_score']<=z_score_cutoff]['first_8_residues'].to_list()
    c_binders = pssms.alignment_2_counts(binders)
    f_binders = c_binders.div(c_binders.sum(axis=1), axis=0)
    info_matrix, col_info = freq_2_information_matix(f_binders, bg=background)
    if ax is not None:
        pssms.plot_logo(c_binders, ax = ax)
        ax.set_title(f'z-score cutoff: {z_score_cutoff}');
    return np.sum(col_info), ax


cutoffs = np.arange(1.7, 3.2, 0.15)
# plot each cutoff on a subplot
n_plots = len(cutoffs)
information = []
fig, axs = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots))
for axis, cutoff in zip(axs, cutoffs):
    I, _ = information_at_z_score_cutoff_flipped(sc_binders, f_nbs, z_score_cutoff=cutoff, ax=axis)
    information.append(I)
plt.tight_layout()
fig, ax = plt.subplots()
ax.plot(cutoffs, information)
ax.set_xlabel('z-score cutoff')
ax.set_ylabel('information content')
ax.set_title('Information content at different z-score cutoffs')
ax.axhline(y=0, color='r', linestyle='--')

# %%
df, col_info = freq_2_information_matix(f_binders)
pssms.plot_logo(df)

# %%
df, col_info = freq_2_information_matix(f_nbs)
pssms.plot_logo(df)


# %% [markdown]
# I think that the other residues are fucking up the information calculation. I can't understand why the 6 peptide logo above is giving such a high information content. My guess is that the other residues being set to 0 is inflating it somehow or maybe it's the fact that there are apparent preferred residues at positions compared to others. This might be where pseudocounts are helpful.

# %% [markdown]
# ### frequencies over different z-score ranges

# %%
def plot_logo_at_z_score_range(binder_df, z_score_range, ax = None):
    binders = binder_df[(binder_df['avg_z_score']>=z_score_range[0]) & (binder_df['avg_z_score']<z_score_range[1])]['first_8_residues'].to_list()
    c_binders = pssms.alignment_2_counts(binders)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    pssms.plot_logo(c_binders, ax = ax)
    ax.set_title(f'z-score range: {z_score_range[0]} - {z_score_range[1]}');
    # increase title font size
    ax.title.set_fontsize(16)
    return ax


# cutoffs = np.arange(1.7, 3.2, 0.15)
# define (start, stop) ranges spanning the z-score cutoffs
cutoffs = [(1.7, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.2)]

# plot each cutoff on a subplot
n_plots = len(cutoffs)
information = []
fig, axs = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots))
for axis, cutoff in zip(axs, cutoffs):
    plot_logo_at_z_score_range(sc_binders, z_score_range=cutoff, ax=axis)
plt.tight_layout()


# %% [markdown]
# ### logomaker standard information matrix with pseudocount = 1 and flat background

# %%
def information_at_z_score_cutoff_logomaker_default(binder_df = sc_binders, z_score_cutoff=1.7, ax = None):
    binders = binder_df[binder_df['avg_z_score']>=z_score_cutoff]['first_8_residues'].to_list()
    i_mat = lm.alignment_to_matrix(binders, to_type='information', pseudocount=1)
    if ax is not None:
        pssms.plot_logo(i_mat, ax = ax)
        ax.set_title(f'z-score cutoff: {z_score_cutoff}');
    return i_mat.sum(axis=1).sum()

cutoffs = np.arange(1.7, 3.2, 0.15)
n_plots = len(cutoffs)
information = []
fig, axs = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots))
for axis, cutoff in zip(axs, cutoffs):
    I = information_at_z_score_cutoff_logomaker_default(sc_binders, z_score_cutoff=cutoff, ax=axis)
    information.append(I)
plt.tight_layout()
fig, ax = plt.subplots()
ax.plot(cutoffs, information)
ax.set_xlabel('z-score cutoff')
ax.set_ylabel('information content')
ax.set_title('Information content at different z-score cutoffs')
ax.axhline(y=0, color='r', linestyle='--')


# %%
def information_at_z_score_cutoff_logomaker_default(binder_df = sc_binders, z_score_cutoff=1.7, background=None, ax = None):
    binders = binder_df[binder_df['avg_z_score']>=z_score_cutoff]['first_8_residues'].to_list()
    i_mat = lm.alignment_to_matrix(binders, to_type='information', pseudocount=1, background=background)
    if ax is not None:
        pssms.plot_logo(i_mat, ax = ax)
        ax.set_title(f'z-score cutoff: {z_score_cutoff}');
    return i_mat.sum(axis=1).sum()


nonbinders = sc_nbs['first_8_residues'].to_list()
temp_bg = lm.alignment_to_matrix(nonbinders, to_type='probability', pseudocount=1)

cutoffs = np.arange(1.7, 3.2, 0.15)
n_plots = len(cutoffs)
information = []
fig, axs = plt.subplots(n_plots, 1, figsize=(10, 2.5*n_plots))
for axis, cutoff in zip(axs, cutoffs):
    I = information_at_z_score_cutoff_logomaker_default(sc_binders, z_score_cutoff=cutoff, ax=axis, background=temp_bg)
    information.append(I)
plt.tight_layout()
fig, ax = plt.subplots()
ax.plot(cutoffs, information)
ax.set_xlabel('z-score cutoff')
ax.set_ylabel('information content')
ax.set_title('Information content at different z-score cutoffs')
ax.axhline(y=0, color='r', linestyle='--')

# %% [markdown]
# ### more strictly defined binders?

# %%
import lir_proteome_screen_pssm.sequence_utils as seqtools

df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')


def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches


df["regex_matches"] = df.apply(get_regex_matches, axis=1, regex="....[FWY]..[LVIWFY]")
df["num_regex_matches"] = df["regex_matches"].apply(lambda x: len(x))
df["num_regex_matches"].value_counts()
df_multi = df[df["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = df[df["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
df = pd.concat([df_multi, df_single])
df[["8mer", "8mer_start", "8mer_end"]] = pd.DataFrame(
    df["regex_matches"].tolist(), index=df.index
)

# %%
print(len(df))
df = df[df['Input Count'] >=10].copy()
print(len(df))
df=df.reset_index(drop=True)
pssms.plot_logo(pssms.alignment_2_counts(df[df['avg_z_score']>=1.7]['8mer'].unique()))

# %% [markdown]
# It looks a little different than the set of binders that jen defined. in other words when I import the `liradj_peptides_250411.csv` file, I get a different set of binders than when I import the `231209_completedata_JK.csv`

# %% [markdown]
# # input library

# %%
import lir_proteome_screen_pssm.sequence_utils as seqtools

df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')


def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches


df["regex_matches"] = df.apply(get_regex_matches, axis=1, regex="....[FWY]..[LVIWFMY]")
df["num_regex_matches"] = df["regex_matches"].apply(lambda x: len(x))
df["num_regex_matches"].value_counts()
df_multi = df[df["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = df[df["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
df = pd.concat([df_multi, df_single])
df[["8mer", "8mer_start", "8mer_end"]] = pd.DataFrame(
    df["regex_matches"].tolist(), index=df.index
)

# %%
print(len(df))
pssms.plot_logo(lm.alignment_to_matrix(df['8mer'].to_list(), to_type='counts'))
df = df[df['Input Count'] >=10]
print(len(df))
pssms.plot_logo(lm.alignment_to_matrix(df['8mer'].to_list(), to_type='counts'))
df = df[df['Input Count'] >=100]
print(len(df))
pssms.plot_logo(lm.alignment_to_matrix(df['8mer'].to_list(), to_type='counts'))

# %%
print(len(df))
df = df[df['Input Count'] >=100]
print(len(df))

# %%
pssms.plot_logo(lm.alignment_to_matrix(df['8mer'].to_list(), to_type='counts'))

# %% [markdown]
# # ilir 

# %%
df = pd.read_csv(env.RAWFILEPATHS.ilir_table)
df

# %%

# %%
pssms.plot_logo(pssms.alignment_2_counts(df['Sequence'].to_list()))

# %% [markdown]
# # Lir central test set

# %%
df = pd.read_csv(env.RAWFILEPATHS.lir_central_table)
df['8mer'] = df['Combined'].str[-8:]
df

# %%
df['True label'].value_counts()

# %%
TP = df[df['True label'] == 1].copy()
TN = df[df['True label'] == 0].copy()
ax = pssms.plot_logo(pssms.alignment_2_counts(TP['8mer'].to_list()))
ax.set_title('test set binders')
ax = pssms.plot_logo(pssms.alignment_2_counts(TN['8mer'].to_list()))
ax.set_title('test set nonbinders')

# %%

# %%

# %%

# %% [markdown]
# # junk

# %%

# %%

# %%

# %%

# %%

# %%
pssms.plot_logo_as_heatmap(f_binders)
pssms.plot_logo_as_heatmap(f_nbs)

# %%
pssms.plot_logo(lm.transform_matrix(f_binders, from_type='probability', to_type='information', background=f_nbs))
pssms.plot_logo(lm.transform_matrix(f_binders, from_type='probability', to_type='information'))
pssms.plot_logo(lm.alignment_to_matrix(binders, to_type='information', background=f_nbs, pseudocount=0))

# %%
assert all(f_binders.columns == f_nbs.columns)
assert all(f_binders.index == f_nbs.index)

# %%
df = f_binders.copy()
fg_vals = f_binders.values
bg_vals = f_nbs.values
temp = fg_vals * (np.log2(fg_vals + np.finfo(float).tiny) - np.log2(bg_vals + np.finfo(float).tiny))
col_info = temp.sum(axis=1)
# temp[:, np.newaxis] * np.ones([8, 8])
col_info[:, np.newaxis] * fg_vals
df.loc[:, :] = col_info[:, np.newaxis] * fg_vals
pssms.plot_logo(df)

# %%
col_info

# %%
t=0
for i,y in zip(f_binders.loc[1,:], f_nbs.loc[1,:]):
    try:
        t += i*np.log2(i/y)
        print(f"{i:.2g}, {y:.2g}, {i*np.log2(i/y):.2g}, {t:.2g}")
    except ZeroDivisionError:
        print(i, y, 'divide by zero')

# %%
for i in binders:
    print(i)

# %%
for i in nonbinders:
    print(i)

# %%
temp

# %%
fg_vals.shape
