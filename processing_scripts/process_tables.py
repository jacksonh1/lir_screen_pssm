from lir_proteome_screen_pssm import environment as env
import pandas as pd
import lir_proteome_screen_pssm.sequence_utils as seqtools
import re
import numpy as np
import copy

output_dir = env.PROCESSED_DATA_DIR
output_dir.mkdir(exist_ok=True, parents=True)
old_regex = "...[FWY]..[LVI]"
new_regex = "...[FWY]..[WFY]"

# ==============================================================================
# // screening data
# ==============================================================================

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

hide_col_dict = {}
for col in screen_df.columns:
    if col in list(rename_dict.values()) + ["avg_z_score"]:
        continue
    hide_col_dict[col] = "_" + col
screen_df = screen_df.rename(columns=hide_col_dict)

screen_binders_df = screen_df[screen_df["_Bind/Nonbind"] == "Bind"].copy()
screen_binders_df["true label"] = 1
screen_nonbinders_df = screen_df[screen_df['_Bind/Nonbind'] == 'Nonbind'].copy()
screen_nonbinders_df["true label"] = 0

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

# ==============================================================================
# // remove duplicate 7mers and 7mers from ilir from screening sets
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
print(screen_binders_df["7mer"].duplicated().sum())
print(screen_nonbinders_df["7mer"].duplicated().sum())
print("dropped duplicates")
screen_binders_df = screen_binders_df.drop_duplicates(keep="first", subset=["7mer"])
screen_nonbinders_df = screen_nonbinders_df.drop_duplicates(
    keep="first", subset=["7mer"]
)

# remove nonbinders from binders and vice versa
print("number of binders in nonbinders")
print(screen_binders_df["7mer"].isin(screen_nonbinders_df["7mer"]).sum())
print("number of nonbinders in binders")
print(screen_nonbinders_df["7mer"].isin(screen_binders_df["7mer"]).sum())
blist = copy.deepcopy(screen_binders_df["7mer"].tolist())
nblist = copy.deepcopy(screen_nonbinders_df["7mer"].tolist())
screen_binders_df = screen_binders_df[~screen_binders_df["7mer"].isin(nblist)]
screen_nonbinders_df = screen_nonbinders_df[~screen_nonbinders_df["7mer"].isin(blist)]


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



ilir_df.to_csv(output_dir / "ilir_binders.csv", index=False)
screen_binders_df.to_csv(output_dir / "screen-binders.csv", index=False)
screen_nonbinders_df.to_csv(output_dir / "screen-nonbinders.csv", index=False)