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
df["oldlir_regex_matches"] = df.apply(get_regex_matches, axis=1, regex="...[FWY]..[FWYLVI]")
df["oldlir_num_regex_matches"] = df["oldlir_regex_matches"].apply(lambda x: len(x))
df["newlir_regex_matches"] = df.apply(get_regex_matches, axis=1, regex="...[LVI]..[LVI]")
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


# %% [markdown]
# ## binder set regex counting

# %%
rs = [
    "...[FWY]..[LVI]",
    "...[FWY]..[FWY]",
    "...[LVI]..[LVI]",
    "...[LVI]..[FWY]",
    "...[FWY]..[LVIFWY]",
]


# %%
def func1(df_in: pd.DataFrame, regex: str) -> pd.DataFrame:
    df = df_in.copy()
    df['matches'] = df.apply(get_regex_matches, axis=1, regex = regex)
    df["n_matches"] = df["matches"].apply(lambda x: len(x))
    return df[df["n_matches"] == 0].copy(), len(df[df["n_matches"] > 0])


# temp = df.copy()
# total = len(temp)
# for r in rs:
#     temp, n =  func1(temp, r)
#     print(f"{r}: {n} ({n/total:.2%})")

def regex_filter_count(df: pd.DataFrame, regex_list: list[str]) -> None:
    temp = df.copy()
    total = len(df)
    for r in regex_list:
        temp, n =  func1(temp, r)
        print(f"{r}: {n} ({n/total:.2%})")


def add_matches_columns(df_in: pd.DataFrame, regex: str):
    df = df_in.copy()
    df[f'{regex}matches'] = df.apply(get_regex_matches, axis=1, regex=regex)
    df[f"{regex}n_matches"] = df[f"{regex}matches"].apply(lambda x: len(x))
    return df


def seqs_with_match(df_in: pd.DataFrame, regex_list: list[str]) -> None:
    df = df_in.copy()
    n = len(df)
    for r in regex_list:
        df = add_matches_columns(df, r)
        n_s_match = len(df[df[f'{r}n_matches']>0])
        print(f"{r}: {n_s_match} ({n_s_match/n:.2%})")


def regex_match_per_library_member(df_in: pd.DataFrame, regex_list: str) -> None:
    df = df_in.copy()
    n = len(df)
    for r in regex_list:
        df = add_matches_columns(df, r)
        print(f"{r}: {df[f'{r}n_matches'].sum()} - {df[f'{r}n_matches'].sum()/n:.3} ave matches per sequence")


def multi_regex_match_per_library_member(df_in: pd.DataFrame, regex_list: list[str]) -> None:
    df = df_in.copy()
    n = len(df)
    for r in regex_list:
        df = add_matches_columns(df, r)
        print(f"number of seqs with >1 match to {r}: {len(df[df[f'{r}n_matches']>1])} ({len(df[df[f'{r}n_matches']>1])/n:.2%})")

# %%
rs = [
    "...[FWY]..[LVI]",
    "...[FWY]..[FWY]",
    "...[LVI]..[LVI]",
    "...[LVI]..[FWY]",
    "...[FWY]..[LVIFWY]",
]
df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
df = df[df['avg_z_score'] >= 1.7].copy()

print(f"Binder set (avg_z_score >= 1.7) (n={len(df)})")
print("-------------------------------")
print('')
print("non-filtering - n sequences with a match")
seqs_with_match(df, rs)
print('')
print("filtering - n sequences with a match. Sequences with a match are removed from the pool for the next regex")
regex_filter_count(df, rs)
print('')
print("non-filtering - total number of matches (>1 match counted) - total # matches / n sequences")
regex_match_per_library_member(df, rs)
print('')
print("non-filtering - n sequences with >1 match")
multi_regex_match_per_library_member(df, rs)

df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
df = df[df['Input Count'] >=100]
print('')
print('')
print(f"input library (Input Count >= 100) (n={len(df)})")
print("-------------------------------")
print('')
print("non-filtering - n sequences with a match")
seqs_with_match(df, rs)
print('')
print("filtering - n sequences with a match. Sequences with a match are removed from the pool for the next regex")
regex_filter_count(df, rs)
print('')
print("non-filtering - total number of matches (>1 match counted) - total # matches / n sequences")
regex_match_per_library_member(df, rs)
print('')
print("non-filtering - n sequences with >1 match")
multi_regex_match_per_library_member(df, rs)
# %%
df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
df = df[df['avg_z_score'] >= 1.7].copy()

# %%
df
