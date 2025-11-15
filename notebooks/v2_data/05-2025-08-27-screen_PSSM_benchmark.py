# %%
from lir_proteome_screen_pssm import environment as env
from lir_proteome_screen_pssm import pssms
import pandas as pd
import numpy as np
import logomaker as lm
import matplotlib.pyplot as plt

plt.style.use("custom_standard")
plt.style.use("custom_small")
import seaborn as sns
import lir_proteome_screen_pssm.sequence_utils as seqtools
import copy
from pathlib import Path
import re

# import umap
from sklearn.preprocessing import OneHotEncoder
import lir_proteome_screen_pssm.data_loaders as dl
import lir_proteome_screen_pssm.stats as stats

plt.style.use("lir_proteome_screen_pssm.lir")
from adjustText import adjust_text

mm = 1 / 25.4

# %%


# %% [markdown]
# # scoring ilir and screening PSSMs - test set 1
#
# Test sets: 7mers
# - screening - all binders, high z-score, low z-score
#
#
# PSSM foregrounds:
# - ilir
# - screening data:
#     - all binders (z>=1.7)
#     - high z-score (z>=2.3)
#     - low z-score (1.7<=z<2.3)
#
# PSSM background:
# - proteome
#
# psuedocounts:
# - 0

# %%


def check_tables(processed_sequence_tables, test_sets):
    assert (processed_sequence_tables.screen_nonbinders["7mer"].str.len() == 7).all()
    assert (processed_sequence_tables.screen_binders["7mer"].str.len() == 7).all()
    assert (processed_sequence_tables.ilir_binders["7mer"].str.len() == 7).all()
    assert (test_sets.lir_central["7mer"].str.len() == 7).all()
    assert (test_sets.lir_central_augmented["7mer"].str.len() == 7).all()


def check_for_fg_test_overlap(
    foregrounds: dict[str, list[str]], test_set_dict: dict[str, pd.DataFrame]
):
    for fg_name, fg_seqs in foregrounds.items():
        fg_set = set(fg_seqs)
        for test_name, test_set in test_set_dict.items():
            test_set_seqs = set(test_set["7mer"].to_list())
            overlap = fg_set.intersection(test_set_seqs)
            if len(overlap) > 0:
                if "cheating" in fg_name:
                    continue
                raise ValueError(
                    f"Overlap between foreground '{fg_name}' and test set '{test_name}'"
                )


def score_test_sets_with_pssms(
    test_set_dict: dict[str, pd.DataFrame], pssm_dict: dict[str, pd.DataFrame]
):
    auc_results = []
    for foreground, pssm in pssm_dict.items():
        for test_set_name, test_set in test_set_dict.items():
            temp_df = test_set.copy()
            temp_df["pssm_score"] = temp_df["7mer"].apply(
                pssms.PSSM_score_sequence, PSSM=pssm
            )
            auroc = stats.df_2_roc_auc(temp_df, "true label", "pssm_score")
            auc_results.append(
                {
                    "foreground": foreground,
                    "test set": test_set_name,
                    "auROC": auroc,
                }
            )
    return pd.DataFrame(auc_results)


def get_stratified_sample(dfin, class_col, sample_num_map, random_seed=42):
    df = dfin.copy()
    samples = []
    for k, n in sample_num_map.items():
        samples.append(
            df[df[class_col] == k].sample(n=n, replace=False, random_state=random_seed)
        )
    sample = pd.concat(samples, ignore_index=True)
    return sample


class TTSplitMaker:

    def __init__(self, binders_df, nonbinders_df):
        self.binders_df = binders_df.copy()
        self.nonbinders_df = nonbinders_df.copy()

    def _update_datasets(self, test_7mers):
        self.binders_df = self.binders_df[
            ~self.binders_df["7mer"].isin(test_7mers)
        ].copy()
        self.nonbinders_df = self.nonbinders_df[
            ~self.nonbinders_df["7mer"].isin(test_7mers)
        ].copy()

    def make_test_set(
        self, test_size_dict, stratification_column="z_score_class", random_seed=42
    ):
        test_binders = get_stratified_sample(
            self.binders_df,
            stratification_column,
            test_size_dict,
            random_seed=random_seed,
        )
        test_nonbinders = self.nonbinders_df.sample(
            n=len(test_binders), random_state=random_seed, replace=False
        ).copy()
        z_score_class_list = []
        for k, v in test_size_dict.items():
            z_score_class_list.extend([k] * v)
        test_nonbinders["z_score_class"] = z_score_class_list
        # test_nonbinders["z_score_class"] = "nonbinder"
        test_df = pd.concat([test_binders, test_nonbinders], ignore_index=True)
        test_7mers = test_df["7mer"].tolist()
        assert not test_df['7mer'].duplicated().any(), "Duplicate 7mers in test set"
        self._update_datasets(test_7mers)
        return test_df

    def make_train_set(self, class_col = "z_score_class", random_seed=42):
        '''
        essentially just adds some randomly selected nonbinders to the binder dataframe and exports.
        This should be run AFTER the make_test_set function.
        '''
        train_binders = self.binders_df.copy()
        class_size_dict = {k:len(train_binders[train_binders[class_col]==k]) for k in train_binders[class_col].unique()} 
        z_score_class_list = []
        for k, v in class_size_dict.items():
            z_score_class_list.extend([k] * v)
        train_nonbinders = self.nonbinders_df.sample(
            n=len(train_binders), random_state=random_seed, replace=False
        ).copy()
        train_nonbinders["z_score_class"] = z_score_class_list
        train_df = pd.concat([train_binders, train_nonbinders], ignore_index=True)
        assert not train_df['7mer'].duplicated().any(), "Duplicate 7mers in train set"
        return train_df




def clean_df(df):
    return df[
        [
            "avg_z_score",
            "7mer",
            "true label",
            # "z_score_class",
        ]
    ]


# %%
version = "v2"
PROCESSED_SEQUENCE_TABLES = dl.get_processed_sequence_tables(version)
TEST_SETS = dl.get_test_sets(version)
BGFREQS = dl.get_background_frequencies(version)
PSEUDOCOUNT = 0
ITERATIONS = 100
STARTING_SEED = 42
output_dir = Path("./05-output_temp")
output_dir.mkdir(exist_ok=True, parents=True)
plot_count = 1
ttsplit_output = output_dir / 'train-test-splits'
ttsplit_output.mkdir(exist_ok=True, parents=True)
# all_binders_output = ttsplit_output / 'all_binders'
# all_binders_output.mkdir(exist_ok=True, parents=True)
# low_z_output = ttsplit_output / 'low_z_score'
# low_z_output.mkdir(exist_ok=True, parents=True)
# high_z_output = ttsplit_output / 'high_z_score'
# high_z_output.mkdir(exist_ok=True, parents=True)

check_tables(PROCESSED_SEQUENCE_TABLES, TEST_SETS)

SAMPLE_SIZE_DICT = {"high z-score": 15, "low z-score": 30}

labels = {
    "all binders": "all binders\nPSSM",
    "low z-score": "low z-score\nPSSM",
    "high z-score": "high z-score\nPSSM",
    "ilir": "$\mathregular{iLIR_{27}}$\nPSSM",
}

binder_df = clean_df(PROCESSED_SEQUENCE_TABLES.screen_binders.copy())
nonbinder_df = clean_df(PROCESSED_SEQUENCE_TABLES.screen_nonbinders.copy())
binder_df["z_score_class"] = binder_df["avg_z_score"].apply(
    lambda x: "high z-score" if x >= 2.3 else "low z-score"
)


seed = STARTING_SEED
auc_roc_replicates = []
for i in range(ITERATIONS):
    seed += 1
    test_splitter = TTSplitMaker(binder_df, nonbinder_df)
    test_set = test_splitter.make_test_set(
        SAMPLE_SIZE_DICT, stratification_column="z_score_class", random_seed=seed
    )

    test_set.to_csv(ttsplit_output / f"test_set_{i}.csv", index=False)
    train_set = test_splitter.make_train_set(class_col="z_score_class", random_seed=seed)
    train_set.to_csv(ttsplit_output / f"train_set_{i}.csv", index=False)

    # train_set[train_set["z_score_class"]=='high z-score'].to_csv(high_z_output / f"train_set_{i}.csv", index=False)

    train_seqs = set(train_set["7mer"].tolist())
    test_seqs = set(test_set["7mer"].tolist())
    common_seqs = train_seqs.intersection(test_seqs)
    if len(common_seqs) > 0:
        raise ValueError(f"Common sequences found between train and test sets: {common_seqs}")

    fgs = {
        "all binders": test_splitter.binders_df["7mer"].to_list(),
        "high z-score": test_splitter.binders_df[
            test_splitter.binders_df["z_score_class"] == "high z-score"
        ]["7mer"].to_list(),
        "low z-score": test_splitter.binders_df[
            test_splitter.binders_df["z_score_class"] == "low z-score"
        ]["7mer"].to_list(),
        "ilir": PROCESSED_SEQUENCE_TABLES.ilir_binders["7mer"].to_list(),
    }
    testsets = {
        "all binders": test_set.copy(),
        "high z-score": test_set[test_set["z_score_class"] == "high z-score"].copy(),
        "low z-score": test_set[test_set["z_score_class"] == "low z-score"].copy(),
    }

    check_for_fg_test_overlap(fgs, testsets)
    pssm_dict = {}
    for k, v in fgs.items():
        counts = pssms.seqlist_2_counts_matrix(v, pseudocount=PSEUDOCOUNT)
        pssm = pssms.make_pssm(
            df_counts=counts,
            bg=BGFREQS.proteome,
        )
        pssm_dict[k] = pssm
    auc_results = score_test_sets_with_pssms(testsets, pssm_dict)
    auc_roc_replicates.append(auc_results)
auc_roc_replicates_df = pd.concat(auc_roc_replicates, ignore_index=True)



# %%
auc_roc_mean = (
    auc_roc_replicates_df.groupby(["foreground", "test set"]).mean().reset_index()
)
auc_roc_std = (
    auc_roc_replicates_df.groupby(["foreground", "test set"]).std().reset_index()
)
auc_roc_mean = auc_roc_mean.rename(columns={"auROC": "mean auROC"})
auc_roc_std = auc_roc_std.rename(columns={"auROC": "std auROC"})
auc_roc_results = pd.merge(auc_roc_mean, auc_roc_std, on=["foreground", "test set"])
auc_roc_results.to_csv(output_dir / "screening_performance_summary.csv", index=False)

fig, axes = plt.subplots(3, 1, figsize=(60 * mm, 200 * mm))
test_sets = auc_roc_results["test set"].unique()

for i, test_set in enumerate(test_sets):
    data = auc_roc_results[auc_roc_results["test set"] == test_set].copy()
    order = ["all binders", "high z-score", "low z-score", "ilir"]
    data = data.set_index("foreground").reindex(order).reset_index()

    axes[i].bar(
        data["foreground"], data["mean auROC"], yerr=data["std auROC"], capsize=5
    )
    axes[i].set_title(f"Test Set: {test_set}")
    # axes[i].set_xticklabels(axes[i].get_xticklabels(), ha="right", va="top")
    axes[i].set_xlabel("Foreground")
    axes[i].set_ylabel("Mean auROC")
    # axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim([0.4, 1.05])
    axes[i].set_xlabel("")
    for bar, value in zip(axes[i].patches, data["mean auROC"]):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.09,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    # Specify the order of x-axis values
    current_labels = [tick.get_text() for tick in axes[i].get_xticklabels()]
    new_labels = [labels.get(label, label) for label in current_labels]
    axes[i].tick_params(axis="x", rotation=90)
    axes[i].set_xticklabels(new_labels, ha="center", va="top")
    axes[i].set_xticklabels(
        axes[i].get_xticklabels(),
        ha="right",         # right-align the text
        va="center",        # vertically center the text
        rotation_mode="anchor"  # anchor the rotation to the alignment point
    )


plt.tight_layout()
plt.savefig(output_dir / "screening_test_sets.png", dpi=300, bbox_inches="tight", format="png")

# %%
# g = sns.catplot(
#     data=auc_roc_replicates_df,
#     x="foreground",
#     y="auROC",
#     col="test set",
#     kind="bar",
#     sharey=True,
#     height=4.5,
#     aspect=1.1,
#     # order=order,
# )
# g.set_xticklabels(rotation=90)
# g.set_titles(col_template="{col_name}")
# g.set_axis_labels("Foreground", "auROC")
# for ax in g.axes.flatten():
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#     ax.set_ylabel("auROC")  # Ensure y-axis label on every subplot
# for ax in g.axes.flatten():
#     ax.yaxis.set_tick_params(labelleft=True)


# %%
labels = {
    "screen all binders": "all binders",
    "screen low z-score": "low z-score",
    "screen high z-score": "high z-score",
    "ilir": "$\mathregular{iLIR_{27}}$",
    "lir central augmented": "lir central augmented",
}


# %%

        # test_nonbinder_list = []
        # for k, v in test_size_dict.items():
        #     temp = self.nonbinders_df.sample(
        #         n=v, random_state=random_seed, replace=False
        #     ).copy()
        #     temp["z_score_class"] = k
        #     test_nonbinder_list.append(temp)
        # test_nonbinders = pd.concat(test_nonbinder_list, ignore_index=True)