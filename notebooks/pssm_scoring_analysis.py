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
import lir_proteome_screen_pssm.data_loaders as dl
import lir_proteome_screen_pssm.stats as stats
from localcider.sequenceParameters import SequenceParameters


class PssmTestScores:

    def __init__(
        self,
        test_set_dict: dict[str, pd.DataFrame],
        foregrounds: dict[str, pd.DataFrame],
        pssm_dict: dict[str, pd.DataFrame],
        auc_results: pd.DataFrame,
    ):
        self.test_set_dict = test_set_dict
        self.foregrounds = foregrounds
        self.pssm_dict = pssm_dict
        self.auc_results = auc_results


def check_tables(processed_sequence_tables, test_sets):
    assert (processed_sequence_tables.screen_nonbinders["7mer"].str.len() == 7).all()
    assert (processed_sequence_tables.screen_binders["7mer"].str.len() == 7).all()
    assert (processed_sequence_tables.ilir_binders["7mer"].str.len() == 7).all()
    assert (test_sets.lir_central["7mer"].str.len() == 7).all()
    assert (test_sets.lir_central_augmented["7mer"].str.len() == 7).all()


def _make_test_set(binders_df, nonbinders_df, lir_type, test_size=50, random_seed=42):
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


def make_screen_sets(
    screen_binders_df: pd.DataFrame,
    screen_nonbinders_df: pd.DataFrame,
    old_regex: str = "...[FWY]..[LVI]",
    new_regex: str = "...[FWY]..[WFY]",
    old_regex_test_size: int = 40,
    new_regex_test_size: int = 30,
    random_seed: int = 42,
):
    """
    parameters
    ----------
    screen_binders_df : pd.DataFrame
        DataFrame containing binder sequences with a column "7mer" for the 7-mer sequence.
    screen_nonbinders_df : pd.DataFrame
        DataFrame containing non-binder sequences with a column "7mer" for the 7-mer sequence.
    old_regex : str
        Regular expression for the old LIR type (default: "...[FWY]..[LVI]").
    new_regex : str
        Regular expression for the new LIR type (default: "...[FWY]..[WFY]").
    old_regex_test_size : int
        Number of sequences to sample for the old LIR type test set (default: 40).
    new_regex_test_size : int
        Number of sequences to sample for the new LIR type test set (default: 30).
    random_seed : int
        Random seed for reproducibility (default: 42).

    returns
    -------
    oldlir_test : pd.DataFrame
        DataFrame containing the test set for the old LIR type.
    newlir_test : pd.DataFrame
        DataFrame containing the test set for the new LIR type.
    binder_training_set : pd.DataFrame
        DataFrame containing the training set for binders, excluding the test sequences from both tests.
    """
    oldlir_test = _make_test_set(
        screen_binders_df,
        screen_nonbinders_df,
        old_regex,
        test_size=old_regex_test_size,
        random_seed=random_seed,
    )
    newlir_test = _make_test_set(
        screen_binders_df,
        screen_nonbinders_df,
        new_regex,
        test_size=new_regex_test_size,
        random_seed=random_seed,
    )
    test_7mers = oldlir_test["7mer"].tolist() + newlir_test["7mer"].tolist()
    binder_training_set = screen_binders_df[
        ~screen_binders_df["7mer"].isin(test_7mers)
    ].copy()
    return oldlir_test, newlir_test, binder_training_set


def plot_test_set(test_set_dict: dict[str, pd.DataFrame]):
    n_plots = len(test_set_dict)
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(8, 2 * n_plots),
        sharex=True,
    )
    for ax, (name, testset) in zip(axes, test_set_dict.items()):
        ax.set_title(name)
        pssms.plot_logo(
            pssms.seqlist_2_counts_matrix(
                testset[testset["true label"] == 1]["7mer"].to_list()
            ),
            ax=ax,
        )
        ax.set_ylabel("count")
    axes[-1].set_xlabel("Position")
    plt.tight_layout()
    return fig, axes


def plot_auc_results(auc_results_df: pd.DataFrame):
    order = [
        "screen",
        "screen: 1.7 <= z-score < 2.0",
        "screen: 2.0 <= z-score < 3.0",
        "screen: 3.0 <= z-score < 4.2",
        "screen: xxx[FWY]xx[WFY]",
        "screen: xxx[FWY]xx[LVI]",
        "ilir",
    ]
    g = sns.catplot(
        data=auc_results_df,
        x="foreground",
        y="auROC",
        col="test set",
        kind="bar",
        sharey=True,
        height=4,
        aspect=1.2,
        order=order,
    )
    g.set_xticklabels(rotation=90)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Foreground", "auROC")
    for ax in g.axes.flatten():
        for spine in ax.spines.values():
            spine.set_visible(True)


def plot_auc_results_replicates(
    auc_results_df: pd.DataFrame, order: list[str] | None = None
):
    if order is None:
        order = [
            "screen",
            "screen: 1.7 <= z-score < 2.0",
            "screen: 2.0 <= z-score < 3.0",
            "screen: 3.0 <= z-score < 4.2",
            "screen: xxx[FWY]xx[WFY]",
            "screen: xxx[FWY]xx[LVI]",
            "ilir",
            # "screen: (cheating)",
            # "screen: xxx[FWY]xx[WFY] (cheating)",
        ]
    g = sns.catplot(
        data=auc_results_df,
        x="foreground",
        y="auROC",
        col="test set",
        kind="boxen",
        sharey=True,
        height=4.5,
        aspect=1.1,
        order=order,
    )
    g.set_xticklabels(rotation=90)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Foreground", "auROC")
    for ax in g.axes.flatten():
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_ylabel("auROC")  # Ensure y-axis label on every subplot
    for ax in g.axes.flatten():
        ax.yaxis.set_tick_params(labelleft=True)
    # plt.gcf().set_size_inches(max(6, 1.6 * len(order)), 2.3 * len(g.col_names))
    # plt.tight_layout()


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


def check_for_fg_test_overlap(
    foregrounds: dict[str, list[str]], test_set_dict: dict[str, pd.DataFrame]
):
    for fg_name, fg_seqs in foregrounds.items():
        fg_set = set(fg_seqs)
        for test_name, test_set in test_set_dict.items():
            test_set_seqs = set(test_set["7mer"].to_list())
            overlap = fg_set.intersection(test_set_seqs)
            if len(overlap) > 0:
                if 'cheating' in fg_name:
                    continue
                print(
                    f"Overlap between foreground '{fg_name}' and test set '{test_name}'"
                )


def driver(
    processed_sequence_tables,
    test_sets,
    background,
    pseudo_count=1,
    z_score_ranges=None,
    old_regex_test_size=40,
    new_regex_test_size=30,
    random_seed=55,
):
    if z_score_ranges is None:
        z_score_ranges = [
            [1.7, 2.0],
            [2.0, 3.0],
            [3.0, 4.2],
        ]
    oldlirtest, newlirtest, screen_training = make_screen_sets(
        screen_binders_df=processed_sequence_tables.screen_binders,
        screen_nonbinders_df=processed_sequence_tables.screen_nonbinders,
        old_regex_test_size=old_regex_test_size,
        new_regex_test_size=new_regex_test_size,
        random_seed=random_seed,
    )
    test_set_dict = {
        "lir_central": test_sets.lir_central.copy(),
        "lir_central_augmented": test_sets.lir_central_augmented.copy(),
        "[FWY]xx[LVI]": oldlirtest.copy(),
        "[FWY]xx[WFY]": newlirtest.copy(),
    }
    foregrounds = {
        "ilir": processed_sequence_tables.ilir_binders["7mer"].to_list(),
        "screen": screen_training["7mer"].to_list(),
    }
    for z_score_range in z_score_ranges:
        temp = screen_training[
            (screen_training["avg_z_score"] >= z_score_range[0])
            & (screen_training["avg_z_score"] < z_score_range[1])
        ].copy()
        foregrounds[f"screen: {z_score_range[0]} <= z-score < {z_score_range[1]}"] = (
            temp["7mer"].to_list()
        )
    foregrounds["screen: xxx[FWY]xx[WFY]"] = screen_training[
        screen_training["lir_type"] == "...[FWY]..[WFY]"
    ]["7mer"].to_list()
    foregrounds["screen: xxx[FWY]xx[LVI]"] = screen_training[
        screen_training["lir_type"] == "...[FWY]..[LVI]"
    ]["7mer"].to_list()
    foregrounds["screen: (cheating)"] = processed_sequence_tables.screen_binders.copy()[
        "7mer"
    ].to_list()
    temp = processed_sequence_tables.screen_binders.copy()
    foregrounds["screen: xxx[FWY]xx[WFY] (cheating)"] = temp[
        temp["lir_type"] == "...[FWY]..[WFY]"
    ]["7mer"].to_list()
    check_for_fg_test_overlap(foregrounds, test_set_dict)
    for k, v in foregrounds.items():
        foregrounds[k] = pssms.seqlist_2_counts_matrix(v, pseudocount=pseudo_count)
    pssm_dict = {}
    for name, foreground in foregrounds.items():
        pssm = pssms.make_pssm(
            df_counts=foreground,
            bg=background,
        )
        pssm_dict[name] = pssm
    auc_results = score_test_sets_with_pssms(test_set_dict, pssm_dict)
    return test_set_dict, foregrounds, pssm_dict, auc_results


def _make_test_set_z(binders_df, nonbinders_df, test_size=50, random_seed=42):
    test_binders = binders_df.sample(
        n=test_size, random_state=random_seed, replace=False
    ).copy()
    test_nonbinders = nonbinders_df.sample(
        n=test_size, random_state=random_seed, replace=False
    ).copy()
    return pd.concat([test_binders, test_nonbinders], ignore_index=True)


class TTSplitMaker:

    def __init__(self, binders_df, nonbinders_df):
        self.binders_df = binders_df.copy()
        self.nonbinders_df = nonbinders_df.copy()

    def _update_binders(self, test_7mers):
        self.binders_df = self.binders_df[
            ~self.binders_df["7mer"].isin(test_7mers)
        ].copy()

    def make_test_set(self, test_size=50, random_seed=42):
        df = _make_test_set_z(
            self.binders_df,
            self.nonbinders_df,
            test_size=test_size,
            random_seed=random_seed,
        )
        test_7mers = df["7mer"].tolist()
        self._update_binders(test_7mers)
        return df

    def make_zscore_range_test(
        self, z_score_range: list[float], test_size=50, random_seed=42
    ):
        temp_binders_df = self.binders_df[
            (self.binders_df["avg_z_score"] >= z_score_range[0])
            & (self.binders_df["avg_z_score"] < z_score_range[1])
        ].copy()
        df = _make_test_set_z(
            temp_binders_df,
            self.nonbinders_df,
            test_size=test_size,
            random_seed=random_seed,
        )
        test_7mers = df["7mer"].tolist()
        self._update_binders(test_7mers)
        return df

    def make_lir_type_test(self, lir_type: str, test_size=50, random_seed=42):
        temp_binders_df = self.binders_df[
            self.binders_df["lir_type"] == lir_type
        ].copy()
        df = _make_test_set_z(
            temp_binders_df,
            self.nonbinders_df,
            test_size=test_size,
            random_seed=random_seed,
        )
        test_7mers = df["7mer"].tolist()
        self._update_binders(test_7mers)
        return df

    def make_zscore_range_lir_type_test(
        self, z_score_range: list[float], lir_type: str, test_size=50, random_seed=42
    ):
        temp_binders_df = self.binders_df[
            (self.binders_df["avg_z_score"] >= z_score_range[0])
            & (self.binders_df["avg_z_score"] < z_score_range[1])
            & (self.binders_df["lir_type"] == lir_type)
        ].copy()
        df = _make_test_set_z(
            temp_binders_df,
            self.nonbinders_df,
            test_size=test_size,
            random_seed=random_seed,
        )
        test_7mers = df["7mer"].tolist()
        self._update_binders(test_7mers)
        return df


# %%
def driver2(
    processed_sequence_tables,
    test_sets,
    background,
    pseudo_count=1,
    z_score_ranges=None,
    test_size=40,
    random_seed=55,
    min_seq4pssm=10,
):
    if z_score_ranges is None:
        z_score_ranges = [
            [1.7, 4.2],
            [1.7, 2.0],
            [2.0, 2.5],
            [2.5, 4.2],
        ]
    test_set_dict = {
        "lir_central": test_sets.lir_central.copy(),
        "lir_central_augmented": test_sets.lir_central_augmented.copy(),
    }
    screen_binders = processed_sequence_tables.screen_binders.copy()
    screen_nonbinders = processed_sequence_tables.screen_nonbinders.copy()
    foregrounds = {
        "ilir": processed_sequence_tables.ilir_binders["7mer"].to_list(),
        # "screen": screen_binders["7mer"].to_list(),
    }
    test_splitter = TTSplitMaker(screen_binders, screen_nonbinders)
    for z_score_range in z_score_ranges:
        test_df = test_splitter.make_zscore_range_test(
            z_score_range,
            test_size=test_size,
            random_seed=random_seed,
        )
        test_set_dict[f"screen: {z_score_range[0]} <= z-score < {z_score_range[1]}"] = (
            test_df.copy()
        )
    # high z score canonical LIR
    high_z_score_range = z_score_ranges[-1]
    lir_type = "...[FWY]..[LVI]"
    high_z_old_lir_name = f"screen: {lir_type.replace('.','x')}; {high_z_score_range[0]} <= z-score < {high_z_score_range[1]}"
    # test_set_dict[high_z_old_lir_name] = test_splitter.make_zscore_range_lir_type_test(
    #     z_score_range=high_z_score_range,
    #     lir_type=lir_type,
    #     test_size=test_size,
    #     random_seed=random_seed,
    # )
    screen_training = test_splitter.binders_df.copy()
    print(f"{len(screen_training)} of {len(screen_binders)} training sequences left after test set creation")
    for z_score_range in z_score_ranges:
        temp = screen_training[
            (screen_training["avg_z_score"] >= z_score_range[0])
            & (screen_training["avg_z_score"] < z_score_range[1])
        ].copy()
        if len(temp) <= min_seq4pssm:
            raise ValueError(f"Not enough sequences found in z-score range: {z_score_range}. need at least {min_seq4pssm} sequences.")
        foregrounds[f"screen: {z_score_range[0]} <= z-score < {z_score_range[1]}"] = (
            temp["7mer"].to_list()
        )
    temp = screen_training[screen_training["lir_type"] == lir_type].copy()
    temp = temp[
        (temp["avg_z_score"] >= high_z_score_range[0])
        & (temp["avg_z_score"] < high_z_score_range[1])
    ].copy()
    if len(temp) <= min_seq4pssm:
        raise ValueError(f"Not enough sequences found in z-score range: {z_score_range}. need at least {min_seq4pssm} sequences.")
    foregrounds[high_z_old_lir_name] = temp["7mer"].to_list()
    check_for_fg_test_overlap(foregrounds, test_set_dict)
    for k, v in foregrounds.items():
        foregrounds[k] = pssms.seqlist_2_counts_matrix(v, pseudocount=pseudo_count)
    pssm_dict = {}
    for name, foreground in foregrounds.items():
        pssm = pssms.make_pssm(
            df_counts=foreground,
            bg=background,
        )
        pssm_dict[name] = pssm
    auc_results = score_test_sets_with_pssms(test_set_dict, pssm_dict)
    return test_set_dict, foregrounds, pssm_dict, auc_results


def plot_foregrounds(foregrounds: dict[str, pd.DataFrame]):
    n_plots = len(foregrounds)
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(8, 2 * n_plots),
        sharex=True,
    )
    for ax, (name, foreground) in zip(axes, foregrounds.items()):
        ax.set_title(name)
        pssms.plot_logo(
            foreground,
            ax=ax,
        )
        ax.set_ylabel("count")
    axes[-1].set_xlabel("Position")
    plt.tight_layout()
    return fig, axes


def plot_pssms(pssm_dict: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(
        nrows=len(pssm_dict),
        ncols=1,
        figsize=(5, 2 * len(pssm_dict)),
        sharex=True,
    )
    for ax, (name, pssm) in zip(axes, pssm_dict.items()):
        pssms.plot_logo(pssm, title=name, ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    return fig, axes


def get_local_cider_features(sequence_str):
    Seqob = SequenceParameters(sequence_str)
    d = {
        # "PPII_propensity": Seqob.get_PPII_propensity(),
        "uversky_hydropathy": Seqob.get_uversky_hydropathy(),
        "mean_hydropathy": Seqob.get_mean_hydropathy(),
        # "mean_net_charge": Seqob.get_mean_net_charge(),
        # "Omega": Seqob.get_Omega(),
        # "kappa": Seqob.get_kappa(),
        # "fraction_expanding": Seqob.get_fraction_expanding(),
        # "fraction_positive": Seqob.get_fraction_positive(),
        # "fraction_negative": Seqob.get_fraction_negative(),
        # "countNeut": Seqob.get_countNeut(),
        "n pos residues": Seqob.get_countPos(),
        "n neg residues": Seqob.get_countNeg(),
        "charge (n pos - n neg)": Seqob.get_countPos() - Seqob.get_countNeg(),
        "isoelectric_point": Seqob.get_isoelectric_point(),
        "net charge per residue": Seqob.get_NCPR(),
    }
    return d
