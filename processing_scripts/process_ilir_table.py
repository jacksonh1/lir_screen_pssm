'''
import ilir table
use uniprot id to get full length sequence
use full length to extend the 6mer to 7mer
check all sequences have exactly one match, etc.
'''
# %%

from lir_proteome_screen_pssm import environment as env
from lir_proteome_screen_pssm import pssms
import pandas as pd
import lir_proteome_screen_pssm.sequence_utils as seqtools

output_dir = env.PROCESSED_DATA_DIR
output_dir.mkdir(exist_ok=True, parents=True)


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

ilir_df.to_csv(output_dir / "ilir_binders.csv", index=False)

# %%