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
'''
import raw data
define binders and nonbinders
remove overlap on ID column
calculate localCIDER features
'''

import lir_proteome_screen_pssm.environment as env
# import modin.pandas as pd
import pandas as pd
import numpy as np
from localcider.sequenceParameters import SequenceParameters
import multiprocessing

# %%
def get_local_cider_features(sequence_str):
    Seqob = SequenceParameters(sequence_str)
    d = {
        "PPII_propensity": Seqob.get_PPII_propensity(),
        "uversky_hydropathy": Seqob.get_uversky_hydropathy(),
        "mean_hydropathy": Seqob.get_mean_hydropathy(),
        "mean_net_charge": Seqob.get_mean_net_charge(),
        "Omega": Seqob.get_Omega(),
        "kappa": Seqob.get_kappa(),
        "fraction_expanding": Seqob.get_fraction_expanding(),
        "fraction_positive": Seqob.get_fraction_positive(),
        "fraction_negative": Seqob.get_fraction_negative(),
        "countNeut": Seqob.get_countNeut(),
        "n pos residues": Seqob.get_countPos(),
        "n neg residues": Seqob.get_countNeg(),
        "charge (n pos - n neg)": Seqob.get_countPos() - Seqob.get_countNeg(),
        "isoelectric_point": Seqob.get_isoelectric_point(),
        "net charge per residue": Seqob.get_NCPR(),
        "ID": sequence_str,
    }
    return d


def main(n_processes):
    full_data_table = pd.read_csv(env.RAWFILEPATHS.full_screening_table_2, sep=',')
    # full_data_table['ID'] = 'PLR' + full_data_table['ID'] # I don't think we need this because we're not using a regex with sites preceding the match
    full_data_table = full_data_table[~full_data_table['ID'].str.contains(r'HPQ', regex=True)]
    assert full_data_table.duplicated(subset=["ID"], keep=False).sum() == 0, "There should be no duplicate IDs in the full data table"
    regexes = [r'[FWY]..[ILV]',r'[FWY]..[WFY]']
    for regex in regexes:
        full_data_table['contains_' + regex] = full_data_table['ID'].str.contains(regex, regex=True)

    # %%
    # ==============================================================================
    # // nonbinders
    # ==============================================================================
    # sequences with >= 50 input counts and negative z-scores in sort 1 and 3, then no counts in sorts 4,5,6
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
    full_data_table['nonbinder'] = full_data_table['ID'].isin(screen_nonbinders_df['ID'])

    # %%
    # %%
    # ==============================================================================
    # // binders
    # ==============================================================================
    screen_binders_df = full_data_table[full_data_table['avg_z_score'] >= 1.7].copy()
    screen_binders_df = screen_binders_df[screen_binders_df['Input Count'] >= 10].copy()
    full_data_table['binder'] = full_data_table['ID'].isin(screen_binders_df['ID'])

    # %%
    # ==============================================================================
    # // final_data_table
    # ==============================================================================
    df = full_data_table[full_data_table['Input Count'] >= 10].copy()

    # %%
    assert df['nonbinder'].sum() == len(screen_nonbinders_df)
    assert df['binder'].sum() == len(screen_binders_df)

    seqs = list(df['ID'].unique())
    res = []
    with multiprocessing.Pool(n_processes) as p:
        results_iterator = p.imap_unordered(
            get_local_cider_features,
            seqs,
            chunksize=1,
        )
        for result in results_iterator:
            res.append(result)
    localcider_results = pd.DataFrame(res)
    localcider_results.to_csv('./localcider_results.csv', index=False)



if __name__ == "__main__":
    main(60)


# %%
# ==============================================================================
# // add localCIDER features
# ==============================================================================
# df = df.reset_index(drop=True)
# cider_features = df['ID'].apply(get_local_cider_features)
# cider_df = pd.DataFrame(cider_features.tolist())
# df = pd.concat([df, cider_df], axis=1)

# import tqdm
# feature_map = {}
# for s in tqdm.tqdm(df['ID'].unique()):
#     feature_map[s] = get_local_cider_features(s)



# %%

# %%
