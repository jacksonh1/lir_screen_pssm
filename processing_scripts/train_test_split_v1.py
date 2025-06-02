"""
import binders/nonbinders and ilir
import lir central
remove lir central 7mers from binders/nonbinders and ilir
remove nonbinder/binder duplicate 7mers
no longer need ilir
split into train/test sets
test sets:
    "...[FWY]..[LVI]"
        - 50 binders
        - 50 nonbinders
    "...[FWY]..[WFY]"
        - 20 binders
        - 20 nonbinders
remove those sequences from the training sets
how many ilir 7mers are in the binders/nonbinders?
"""

# %%
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import re
import numpy as np
import copy

from lir_proteome_screen_pssm import environment as env
from lir_proteome_screen_pssm import data_loaders as dl

old_regex = "...[FWY]..[LVI]"
old_regex_test_size = 40
new_regex = "...[FWY]..[WFY]"
new_regex_test_size = 30


version = "v1"  # Increment as needed
split_dir = env.PROCESSED_DATA_DIR / "train_test_splits" / version
split_dir.mkdir(parents=True, exist_ok=True)
random_seed = 73  # Set a random seed for reproducibility


# %%

# seqtables = dl.PROCESSED_SEQUENCE_TABLES
ilir_df = dl.PROCESSED_SEQUENCE_TABLES.ilir_binders.copy()
screen_binders_df = dl.PROCESSED_SEQUENCE_TABLES.screen_binders.copy()
screen_binders_df["binding_label"] = 1
screen_nonbinders_df = dl.PROCESSED_SEQUENCE_TABLES.screen_nonbinders.copy()
screen_nonbinders_df["binding_label"] = 0

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


# %%


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

print(screen_binders_df["lir_type"].value_counts())
print(screen_nonbinders_df["lir_type"].value_counts())

# %%

# split into train and test sets with different lir types


def make_test_set(binders_df, nonbinders_df, lir_type, test_size=50, random_seed=42):
    test_binders = (
        binders_df[binders_df["lir_type"] == lir_type]
        .sample(n=test_size, random_state=random_seed, replace=False)
        .copy()
    )
    test_nonbinders = (
        nonbinders_df[nonbinders_df["lir_type"] == lir_type]
        .sample(n=test_size, random_state=random_seed, replace=False)
        .copy()
    )
    return pd.concat([test_binders, test_nonbinders], ignore_index=True)


oldlir_test = make_test_set(
    screen_binders_df,
    screen_nonbinders_df,
    old_regex,
    test_size=old_regex_test_size,
    random_seed=random_seed,
)
newlir_test = make_test_set(
    screen_binders_df,
    screen_nonbinders_df,
    new_regex,
    test_size=new_regex_test_size,
    random_seed=random_seed,
)

test_7mers = oldlir_test["7mer"].tolist() + newlir_test["7mer"].tolist()
# remove test 7mers from binders and nonbinders
binder_training_set = screen_binders_df[~screen_binders_df["7mer"].isin(test_7mers)]
print(len(screen_binders_df), len(binder_training_set))
print(binder_training_set["lir_type"].value_counts())


# %%

oldlir_test.to_csv(
    split_dir / f"{old_regex.replace('.', 'x')}_screen_test_set.csv", index=False
)
newlir_test.to_csv(
    split_dir / f"{new_regex.replace('.', 'x')}_screen_test_set.csv", index=False
)
binder_training_set.to_csv(split_dir / "screen_binders_train_set.csv", index=False)
# Create and save metadata
metadata = {
    "created_at": datetime.now().isoformat(),
    "train_size": len(binder_training_set),
    f"{old_regex.replace('.', 'x')}_screen_test_size-binders": len(
        oldlir_test[oldlir_test["binding_label"] == 1]
    ),
    f"{old_regex.replace('.', 'x')}_screen_test_size-nonbinders": len(
        oldlir_test[oldlir_test["binding_label"] == 0]
    ),
    f"{new_regex.replace('.', 'x')}_screen_test_size-binders": len(
        newlir_test[newlir_test["binding_label"] == 1]
    ),
    f"{new_regex.replace('.', 'x')}_screen_test_size-nonbinders": len(
        newlir_test[newlir_test["binding_label"] == 0]
    ),
    "random_seed": random_seed,  # Replace with your actual seed
    "source_data": [
        str(dl.PROCESSED_SEQUENCE_TABLES.ilir_binders_csv),
        str(dl.PROCESSED_SEQUENCE_TABLES.screen_binders_csv),
        str(dl.PROCESSED_SEQUENCE_TABLES.screen_nonbinders_csv),
    ],
    "version": version,
    "lir_types": {
        old_regex: {
            "test_size": len(oldlir_test),
            "test_binders": len(oldlir_test[oldlir_test["binding_label"] == 1]),
            "test_nonbinders": len(oldlir_test[oldlir_test["binding_label"] == 0]),
        },
        new_regex: {
            "test_size": len(newlir_test),
            "test_binders": len(newlir_test[newlir_test["binding_label"] == 1]),
            "test_nonbinders": len(newlir_test[newlir_test["binding_label"] == 0]),
        },
},
    # "source_data_hash": hashlib.md5(
    #     open(env.RAWFILEPATHS.screening_hits_table, "rb").read()
    # ).hexdigest(),
}

with open(split_dir / "split_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
