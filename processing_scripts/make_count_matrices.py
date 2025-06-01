'''
generates count matrices from the screening hits and iLIR data


'''
from lir_proteome_screen_pssm import environment as env
from lir_proteome_screen_pssm import pssms
import pandas as pd
import numpy as np
import lir_proteome_screen_pssm.sequence_utils as seqtools
import copy
from pathlib import Path
import lir_proteome_screen_pssm.data_loaders as dl


output_dir = env.DATA_DIR / "processed" / "count_matrices"
output_dir.mkdir(exist_ok=True, parents=True)


# ==============================================================================
# // data from screen
# ==============================================================================

screen_df = pd.read_csv(env.RAWFILEPATHS.screening_hits_table)
binders_df = screen_df[screen_df["Bind/Nonbind"] == "Bind"].copy()
screen_binders_all = binders_df["first_7_residues"].tolist()
assert all(
    [len(b) == 7 for b in screen_binders_all]
), "All binders should be 7 residues long"

sb_all = pssms.seqlist_2_counts_matrix(screen_binders_all)
sb_all.to_csv(output_dir / "screen-all_binders.csv", index=False)

sb_all_weighted = pssms.seqlist_2_counts_matrix(
    binders_df["first_7_residues"].tolist(), weights=binders_df["avg_z_score"].tolist()
)
sb_all_weighted.to_csv(output_dir / "screen-all_binders-z_score_weighted.csv", index=False)

temp = binders_df[binders_df["avg_z_score"] > 2.4].copy()
sb_2_4 = pssms.seqlist_2_counts_matrix(temp["first_7_residues"].tolist())
sb_2_4.to_csv(output_dir / "screen-z_score_above_2_4.csv", index=False)


# ==============================================================================
# // iLIR
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
).all(), "7mer[1:] and Sequence don't match"
ilir_binders = pssms.seqlist_2_counts_matrix(ilir_df["7mer"].to_list())

ilir_binders.to_csv(output_dir / "ilir-binders.csv", index=False)