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

from pathlib import Path

# %load_ext autoreload
# %autoreload 2

# %%
import lir_proteome_screen_pssm.sequence_utils as seqtools

df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
df = df[df['avg_z_score'] >= 1.7].copy()

def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches


# %%
df["oldlir_regex_matches"] = df.apply(get_regex_matches, axis=1, regex="...[FWY]..[LVI]")
df["oldlir_num_regex_matches"] = df["oldlir_regex_matches"].apply(lambda x: len(x))
df["newlir_regex_matches"] = df.apply(get_regex_matches, axis=1, regex="...[FWYLVI]..[WFY]")
df["newlir_num_regex_matches"] = df["newlir_regex_matches"].apply(lambda x: len(x))
only1newlir = (
    df[(df["newlir_num_regex_matches"] == 1) & (df["oldlir_num_regex_matches"] == 0)]
    .copy()
    .reset_index(drop=True)
)
only1newlir["newlir_regex_matches"] = only1newlir["newlir_regex_matches"].apply(
    lambda x: x[0]
)
only1newlir[["7mer", "7mer_start", "7mer_end"]] = pd.DataFrame(
    only1newlir["newlir_regex_matches"].tolist(), index=only1newlir.index
)
pssms.plot_logo(pssms.seqlist_2_counts_matrix(only1newlir["7mer"].to_list()))

# %%
df["oldlir_regex_matches"] = df.apply(get_regex_matches, axis=1, regex="...[FWY]..[LVI]")
df["oldlir_num_regex_matches"] = df["oldlir_regex_matches"].apply(lambda x: len(x))
df["newlir_regex_matches"] = df.apply(get_regex_matches, axis=1, regex="...[FWY]..[WFY]")
df["newlir_num_regex_matches"] = df["newlir_regex_matches"].apply(lambda x: len(x))
only1newlir = (
    df[(df["newlir_num_regex_matches"] == 1) & (df["oldlir_num_regex_matches"] == 0)]
    .copy()
    .reset_index(drop=True)
)
only1newlir["newlir_regex_matches"] = only1newlir["newlir_regex_matches"].apply(
    lambda x: x[0]
)
only1newlir[["7mer", "7mer_start", "7mer_end"]] = pd.DataFrame(
    only1newlir["newlir_regex_matches"].tolist(), index=only1newlir.index
)
pssms.plot_logo(pssms.seqlist_2_counts_matrix(only1newlir["7mer"].to_list()))

# %%

# %%
print(len(df[df['newlir_num_regex_matches'] > 0]))
print(len(df[df['oldlir_num_regex_matches'] > 0]))

# %%

# %%

# %%

# %%

df_multi = df[df["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = df[df["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
df = pd.concat([df_multi, df_single])
df[["8mer", "8mer_start", "8mer_end"]] = pd.DataFrame(
    df["regex_matches"].tolist(), index=df.index
)

# %%

# %%

# %%

# %%

# %%

# %%

# %%
