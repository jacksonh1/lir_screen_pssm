# %%
from lir_proteome_screen_pssm import environment as env
import pandas as pd
import lir_proteome_screen_pssm.sequence_utils as seqtools
import re
import numpy as np
import copy
import lir_proteome_screen_pssm.data_loaders as dl

version = "v2"
output_dir = env.PROCESSED_DATA_DIR / version
output_dir.mkdir(exist_ok=True, parents=True)
TEST_SETS = dl.get_test_sets(version=version)

# %%
def get_regex_matches(s: pd.Series, regex: str):
    matches = list(seqtools.get_regex_matches(regex, s["ID"]))
    # if len(matches) == 0:
    #     return
    return matches


REGEX = seqtools.regex2overlapping("...[FWY]..[ILVWFY]")
# REGEX = "...[FWY]..[ILVWFY]"
# %%
# ==============================================================================
# // screening data
# ==============================================================================
full_data_table = pd.read_csv(env.RAWFILEPATHS.full_screening_table_2, sep=',')
full_data_table['ID'] = 'PLR' + full_data_table['ID']
full_data_table = full_data_table[~full_data_table['ID'].str.contains(r'HPQ', regex=True)]
full_data_table = full_data_table[full_data_table['ID']!='PLRASQGSDDDWDDEWDDSSTVADEPGALGSGAYPDLDG'] # For this sequence, we know that the actual binding motif is WDDEW
assert full_data_table.duplicated(subset=["ID"], keep=False).sum() == 0, "There should be no duplicate IDs in the full data table"  

# ==============================================================================
# // nonbinders
# ==============================================================================
# sequences with >= 50 input counts and negative ERs in sort 1 and 3, then no counts in sorts 4,5,6
input_count_cutoff = 50
screen_nonbinders_df = full_data_table[
    full_data_table["Input Count"] >= input_count_cutoff
].copy()
screen_nonbinders_df = screen_nonbinders_df[
    (screen_nonbinders_df['ER 1'] < 0) &
    (screen_nonbinders_df['ER 3'] < 0)
].copy()
screen_nonbinders_df = screen_nonbinders_df[
    (screen_nonbinders_df['4 Count'] == 0) &
    (screen_nonbinders_df['5 Count'] == 0) &
    (screen_nonbinders_df['6 Count'] == 0)
].copy()
screen_nonbinders_df["regex_matches"] = screen_nonbinders_df.apply(get_regex_matches, axis=1, regex=REGEX)
screen_nonbinders_df["num_regex_matches"] = screen_nonbinders_df["regex_matches"].apply(lambda x: len(x))
df_multi = screen_nonbinders_df[screen_nonbinders_df["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = screen_nonbinders_df[screen_nonbinders_df["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
screen_nonbinders_df = pd.concat([df_multi, df_single])
screen_nonbinders_df[["7mer", "motif_start", "motif_end"]] = pd.DataFrame(
    screen_nonbinders_df["regex_matches"].tolist(), index=screen_nonbinders_df.index
)
screen_nonbinders_df["true label"] = 0
screen_nonbinders_df = screen_nonbinders_df.drop_duplicates(keep=False, subset="ID")

hide_col_dict = {}
for col in screen_nonbinders_df.columns:
    if col in ["7mer", "true label", "avg_z_score"]:
        continue
    hide_col_dict[col] = "_" + col
screen_nonbinders_df = screen_nonbinders_df.rename(columns=hide_col_dict)


# %%
# ==============================================================================
# // binders
# ==============================================================================
screen_binders_df = full_data_table[full_data_table['avg_z_score'] >= 1.7].copy()
screen_binders_df = screen_binders_df[screen_binders_df['Input Count'] >= 10].copy()
# regex extract 7mer from 7mer column
# REGEX = seqtools.regex2overlapping("...[FWY]..[ILVWFY]")
# REGEX = "[FWY]..[ILVWFY]"
screen_binders_df["regex_matches"] = screen_binders_df.apply(get_regex_matches, axis=1, regex=REGEX)
screen_binders_df["num_regex_matches"] = screen_binders_df["regex_matches"].apply(lambda x: len(x))
screen_binders_df["num_regex_matches"].value_counts()
df_multi = screen_binders_df[screen_binders_df["num_regex_matches"] > 1].copy()
df_multi = df_multi.explode("regex_matches")
df_single = screen_binders_df[screen_binders_df["num_regex_matches"] == 1].copy()
df_single["regex_matches"] = df_single["regex_matches"].apply(lambda x: x[0])
screen_binders_df = pd.concat([df_multi, df_single])
screen_binders_df[["7mer", "motif_start", "motif_end"]] = pd.DataFrame(
    screen_binders_df["regex_matches"].tolist(), index=screen_binders_df.index
)
screen_binders_df["true label"] = 1
screen_binders_df = screen_binders_df.drop_duplicates(keep=False, subset="ID")
print(len(screen_binders_df))
hide_col_dict = {}
for col in screen_binders_df.columns:
    if col in ["7mer", "true label", "avg_z_score"]:
        continue
    hide_col_dict[col] = "_" + col
screen_binders_df = screen_binders_df.rename(columns=hide_col_dict)


# %%
# ==============================================================================
# // ilir
# ==============================================================================

def ids_equal(s: pd.Series):
    original_id = s["UNIPROT ACC"]
    split_id = s["header"].split("|")[1]
    return original_id == split_id


def find_motif_in_sequence(s: pd.Series):
    """
    Find the motif in the sequence
    :param s: Series with 'full_length_seq' and 'first_7_residues'
    :return: start and end of the motif
    """
    seq = s["full_length_seq"]
    motif = s["Sequence"]
    matches = list(seqtools.find_all(seq, motif))
    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches
    else:
        print(f"Multiple matches found for {s['UNIPROT ACC']}: {matches}")
        return matches


def get_7mer_from_full_length(s: pd.Series):
    """
    Get the 7-mer from the full length sequence
    :param s: Series with 'full_length_seq' and 'start_position'
    :return: 7-mer
    """
    seq = s["full_length_seq"]
    start = s["start_position"]
    return seq[start - 1 : start + 6]  # want the n terminal residues so it's n-1 to n+6


ilir_df = pd.read_csv(env.RAWFILEPATHS.ilir_table)
ilir_df[["header", "full_length_seq"]] = (
    ilir_df["UNIPROT ACC"]
    .apply(lambda x: seqtools.download_uniprot_sequence(x)) # type: ignore
    .apply(pd.Series)
)
ilir_df["ids_equal"] = ilir_df.apply(ids_equal, axis=1)
assert ilir_df[
    "ids_equal"
].all(), "input ids do not match downloaded sequence ids should be equal"

ilir_df["start_position"] = ilir_df.apply(find_motif_in_sequence, axis=1) # type: ignore
ilir_df["n_matches"] = ilir_df["start_position"].apply(
    lambda x: len(x) if isinstance(x, list) else 0
)
assert (ilir_df["n_matches"] == 1).all(), "All sequences should have exactly one match"
ilir_df["start_position"] = ilir_df["start_position"].apply(lambda x: x[0])
ilir_df["7mer"] = ilir_df.apply(get_7mer_from_full_length, axis=1)
assert (
    ilir_df["7mer"].str[1:] == ilir_df["Sequence"]
).all(), "7mer[1:] and 6mer don't match"

ilir_df = ilir_df.rename(columns={"Sequence": "6mer"})

hide_col_dict = {}
for col in ilir_df.columns:
    if col in ["7mer", "6mer"]:
        continue
    hide_col_dict[col] = "_" + col
ilir_df = ilir_df.rename(columns=hide_col_dict)
# %%
# ==============================================================================
# // remove duplicate and overlapping 7mers from all sets
# ==============================================================================

# Remove ilir overlap with binders and nonbinders
ilir_7mers = ilir_df["7mer"].to_list()
print("ilir 7mers in binders and nonbinders")
print(screen_binders_df["7mer"].isin(ilir_7mers).sum())
print(screen_nonbinders_df["7mer"].isin(ilir_7mers).sum())
print("dropping ilir 7mers from binders and nonbinders")
screen_binders_df = screen_binders_df[~screen_binders_df["7mer"].isin(ilir_7mers)]
screen_nonbinders_df = screen_nonbinders_df[
    ~screen_nonbinders_df["7mer"].isin(ilir_7mers)
]

# remove duplicates in binders and nonbinders
print("number of binder/nonbinder duplicates")
print("binders ", screen_binders_df["7mer"].duplicated().sum())
print("nonbinders", screen_nonbinders_df["7mer"].duplicated().sum())
print("dropped duplicates")
screen_binders_df = screen_binders_df.drop_duplicates(keep="first", subset=["7mer"])
screen_nonbinders_df = screen_nonbinders_df.drop_duplicates(
    keep="first", subset=["7mer"]
)

# remove nonbinders from binders and vice versa
print("removing 7mers present in both binders and nonbinders")
print("number of binders in nonbinders")
print(screen_binders_df["7mer"].isin(screen_nonbinders_df["7mer"]).sum())
print("number of nonbinders in binders")
print(screen_nonbinders_df["7mer"].isin(screen_binders_df["7mer"]).sum())
blist = copy.deepcopy(screen_binders_df["7mer"].tolist())
nblist = copy.deepcopy(screen_nonbinders_df["7mer"].tolist())
screen_binders_df = screen_binders_df[~screen_binders_df["7mer"].isin(nblist)]
screen_nonbinders_df = screen_nonbinders_df[~screen_nonbinders_df["7mer"].isin(blist)]

# remove 7mers from binders and nonbinders that are in lir central test set
lir_central_test = TEST_SETS.lir_central
lir_central_test_7mers = lir_central_test["7mer"].tolist()
print("number of lir central test 7mers in binders and nonbinders")
print("binders ", screen_binders_df["7mer"].isin(lir_central_test_7mers).sum())
print("nonbinders ", screen_nonbinders_df["7mer"].isin(lir_central_test_7mers).sum())
screen_binders_df = screen_binders_df[
    ~screen_binders_df["7mer"].isin(lir_central_test_7mers)
]
screen_nonbinders_df = screen_nonbinders_df[
    ~screen_nonbinders_df["7mer"].isin(lir_central_test_7mers)
]

# remove 7mers from binders and nonbinders that are in augmented lir central test set (this is basically the same as lir_central_test but with some additional 7mers)
# so I could have just used the augmented lir central test set from the start
lir_central_test_augmented = TEST_SETS.lir_central_augmented
lir_central_test_augmented_7mers = lir_central_test_augmented["7mer"].tolist()
print("number of 7mers in binders and nonbinders that are also in the 7mers added to the augmented lir central test set")
print("binders ", screen_binders_df["7mer"].isin(lir_central_test_augmented_7mers).sum())
print("nonbinders ", screen_nonbinders_df["7mer"].isin(lir_central_test_augmented_7mers).sum())
screen_binders_df = screen_binders_df[
    ~screen_binders_df["7mer"].isin(lir_central_test_augmented_7mers)
]
screen_nonbinders_df = screen_nonbinders_df[
    ~screen_nonbinders_df["7mer"].isin(lir_central_test_augmented_7mers)
]
print(f"number of binders after filtering: {len(screen_binders_df)}")
print(f"Number of nonbinders after filtering: {len(screen_nonbinders_df)}")

# ==============================================================================
# // breaking into lir types
# ==============================================================================

old_regex = "...[FWY]..[LVI]"
new_regex = "...[FWY]..[WFY]"


def match_regex(seq, re_pattern1, re_pattern2):
    if re.fullmatch(re_pattern1, seq):
        return re_pattern1
    elif re.fullmatch(re_pattern2, seq):
        return re_pattern2
    else:
        return np.nan


screen_binders_df["lir_type"] = screen_binders_df["7mer"].apply(
    lambda x: match_regex(x, old_regex, new_regex)
)
screen_nonbinders_df["lir_type"] = screen_nonbinders_df["7mer"].apply(
    lambda x: match_regex(x, old_regex, new_regex)
)
# %%
ilir_df.to_csv(output_dir / "ilir_binders.csv", index=False)
screen_binders_df.to_csv(output_dir / "screen-binders.csv", index=False)
screen_nonbinders_df.to_csv(output_dir / "screen-nonbinders.csv", index=False)

# %%
