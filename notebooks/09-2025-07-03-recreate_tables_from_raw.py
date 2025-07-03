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
import pandas as pd
import lir_proteome_screen_pssm.sequence_utils as seqtools
import re
import numpy as np
import copy
import lir_proteome_screen_pssm.data_loaders as dl

output_dir = env.PROCESSED_DATA_DIR
output_dir.mkdir(exist_ok=True, parents=True)
old_regex = "...[FWY]..[LVI]"
new_regex = "...[FWY]..[WFY]"

# %% [markdown]
# # preprocessed sequence table from Jen

# %%
rename_dict = {
    "first_4_residues": "4mer",
    "first_6_residues": "6mer",
    "first_8_residues": "8mer",
    "first_14_residues": "14mer",
    "first_7_residues": "7mer",
    "first_5_residues": "5mer",
}

screen_df = pd.read_csv(env.RAWFILEPATHS.screening_hits_table)
screen_df = screen_df.rename(columns=rename_dict)
assert all(
    [len(s) == 7 for s in screen_df["7mer"].tolist()]
), "All binders should be 7 residues long"
assert all(
    screen_df["7mer"].str.contains('-') == False
), "No 7mers should contain gaps"

screen_binders_df = screen_df[screen_df["Bind/Nonbind"] == "Bind"].copy()
screen_binders_df["true label"] = 1
screen_nonbinders_df = screen_df[screen_df['Bind/Nonbind'] == 'Nonbind'].copy()
screen_nonbinders_df["true label"] = 0
print(len(screen_binders_df))
a = set(screen_binders_df["7mer"].to_list())

# %% [markdown]
# # from all data to preprocessed binders (a double check)

# %%
full_data_table = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
screen_binders_from_raw = full_data_table[full_data_table['avg_z_score'] >= 1.7].copy()
screen_binders_from_raw = screen_binders_from_raw[screen_binders_from_raw['Input Count'] >= 10].copy()
# regex extract 7mer from 7mer column
# only keep 

def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches

print(len(screen_binders_from_raw))
# REGEX = seqtools.regex2overlapping("...[FWY]..[ILVWFY]")
REGEX = "...[FWY]..[ILVWFY]"
# REGEX = "[FWY]..[ILVWFY]"
screen_binders_from_raw["regex_matches"] = screen_binders_from_raw.apply(get_regex_matches, axis=1, regex=REGEX)
screen_binders_from_raw["num_regex_matches"] = screen_binders_from_raw["regex_matches"].apply(lambda x: len(x))
screen_binders_from_raw["num_regex_matches"].value_counts()
df_multi = screen_binders_from_raw[screen_binders_from_raw["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = screen_binders_from_raw[screen_binders_from_raw["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
screen_binders_from_raw = pd.concat([df_multi, df_single])
screen_binders_from_raw[["7mer", "motif_start", "motif_end"]] = pd.DataFrame(
    screen_binders_from_raw["regex_matches"].tolist(), index=screen_binders_from_raw.index
)
print(len(screen_binders_from_raw))
b = set(screen_binders_from_raw["7mer"].to_list())

# %%
print(len(a), len(b))

# %%
print(len(a.difference(b)))
print(len(b.difference(a)))

# %% [markdown]
# # junk

# %%
process_binders = dl.PROCESSED_SEQUENCE_TABLES.screen_binders
process_binders_from_raw = pd.read_csv(env.PROCESSED_DATA_DIR / "screen-binders_from_raw.csv")

# %%
c = set(process_binders["7mer"].to_list())
d = set(process_binders_from_raw["7mer"].to_list())
print(len(c), len(d))
print(len(c.difference(d)))
print(len(d.difference(c)))

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
