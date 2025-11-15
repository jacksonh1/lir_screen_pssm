from lir_proteome_screen_pssm import environment as env
import lir_proteome_screen_pssm.sequence_utils as seqtools
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import json


# ==============================================================================
# // processed sequence tables
# ==============================================================================

@dataclass
class ProcessedSequenceTables:

    screen_binders_csv: str | Path
    screen_nonbinders_csv: str | Path
    ilir_binders_csv: str | Path

    def __post_init__(self):
        self.screen_binders = pd.read_csv(self.screen_binders_csv)
        self.screen_nonbinders = pd.read_csv(self.screen_nonbinders_csv)
        self.ilir_binders = pd.read_csv(self.ilir_binders_csv)

    def __repr__(self):
        s = "Processed Sequence Tables:\n"
        s += f"screen_binders_csv='{self.screen_binders_csv}'\n"
        c = "\n    ".join(self.screen_binders.columns.tolist())
        s += f" - columns:\n    {c}\n"
        s += f"screen_nonbinders_csv='{self.screen_nonbinders_csv}'\n"
        c = "\n    ".join(self.screen_nonbinders.columns.tolist())
        s += f" - columns:\n    {c}\n"
        s += f"ilir_binders_csv='{self.ilir_binders_csv}'\n"
        c = "\n    ".join(self.ilir_binders.columns.tolist())
        s += f" - columns:\n    {c}\n"
        return s


def get_processed_sequence_tables(version=None):
    """Return ProcessedSequenceTables for a specific processed data version."""
    if version is None:
        version = env.processed_data_version
    processed_dir = env.PROCESSED_DATA_DIR / version
    tables = {
        "screen_binders_csv": processed_dir / "screen-binders.csv",
        "screen_nonbinders_csv": processed_dir / "screen-nonbinders.csv",
        "ilir_binders_csv": processed_dir / "ilir_binders.csv",
    }
    return ProcessedSequenceTables(**tables)


# ==============================================================================
# // background frequencies and count matrices
# ==============================================================================

def _import_background_frequencies(
    background_freq_csv: Path
) -> dict:
    """
    Import background frequencies from a CSV file.
    The CSV file should have a column with 'Residue' and the frequencies for each residue.
    """
    bg_freqs = pd.read_csv(background_freq_csv)
    bg_freqs = bg_freqs.set_index("Residue").to_dict(orient="dict")
    return bg_freqs


# this is just to make the frequency dict easier to use (autocomplete for the attributes instead of memorizing the dict keys)
class BGFrequencies:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return "background frequencies: \n - {}".format(
            "\n - ".join([k for k in self.__dict__.keys()])
        )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def get_background_frequencies(version=None):
    """Get background frequencies for a specific processed data version."""
    if version is None:
        version = env.processed_data_version
    background_freq_csv = env.PROCESSED_DATA_DIR / version / "background_frequencies.csv"
    if not background_freq_csv.exists():
        raise FileNotFoundError(f"Background frequency file not found: {background_freq_csv}")
    bg_freqs = _import_background_frequencies(background_freq_csv)
    return BGFrequencies(**bg_freqs)


class CountMatrices:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if v.exists():
                setattr(self, k, pd.read_csv(v))
            setattr(self, f"{k}-file", v)

    def __repr__(self):
        return "attributes: \n - {}".format(
            "\n - ".join([k for k in self.__dict__.keys()])
        )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def get_count_matrices(version=None):
    """Get count matrices for a specific processed data version."""
    if version is None:
        version = env.processed_data_version
    count_matrix_dir = env.PROCESSED_DATA_DIR / version / "count_matrices"
    count_matrix_files = {
        "screen_all_binders": count_matrix_dir / "screen-all_binders.csv",
        "screen_all_binders_weighted": count_matrix_dir
        / "screen-all_binders-z_score_weighted.csv",
        "screen_z_score_above_2_4": count_matrix_dir / "screen-z_score_above_2_4.csv",
        "ilir_binders": count_matrix_dir / "ilir-binders.csv",
    }
    return CountMatrices(**count_matrix_files)

# ==============================================================================
# // outside test sets (i.e. not part of the training data)
# ==============================================================================


@dataclass
class TestSets:
    """
    A class to represent the test sets in this project.
    Attributes:
        lir_central_augmented (pd.DataFrame): The augmented LIR central test set.
        lir_central_testset_csv (str|Path): Path to the CSV file containing the LIR central test set.
    """

    lir_central_testset_csv: str | Path
    lir_central_augmented_testset_csv: str | Path
    # screening_testset_LVI_csv: str | Path =

    def __post_init__(self):
        self.lir_central = pd.read_csv(self.lir_central_testset_csv)
        self.lir_central_augmented = pd.read_csv(self.lir_central_augmented_testset_csv)

    def __repr__(self):
        s = f"Test set: lir_central_augmented_testset_csv='{self.lir_central_augmented_testset_csv}\n'"
        c = "\n    ".join(self.lir_central_augmented.columns.tolist())
        s += f" - columns: {c}\n    "
        s += f"Test set: lir_central_testset_csv='{self.lir_central_testset_csv}\n'"
        c = "\n    ".join(self.lir_central.columns.tolist())
        s += f"- columns: {c}\n    "
        return s


def get_test_sets(version=None):
    """Return TestSets for a specific processed data version."""
    if version is None:
        version = env.processed_data_version
    test_set_dir = env.PROCESSED_DATA_DIR / version / "test_sets"
    test_sets = {
        "lir_central_testset_csv": test_set_dir / "lir_central_test_set.csv",
        "lir_central_augmented_testset_csv": test_set_dir / "lir_central_augmented_test_set.csv",
    }
    return TestSets(**test_sets)

# ==============================================================================
# // train/test splits - DEPRECATED
# ==============================================================================

_TRAIN_TEST_SETS = {
    "screening_v1": {
        "train": env.PROCESSED_DATA_DIR / "v1"
        / "train_test_splits"
        / "v1"
        / "screen_binders_train_set.csv",
        "test-LVI": env.PROCESSED_DATA_DIR / "v1"
        / "train_test_splits"
        / "v1"
        / "xxx[FWY]xx[LVI]_screen_test_set.csv",
        "test-WFY": env.PROCESSED_DATA_DIR / "v1"
        / "train_test_splits"
        / "v1"
        / "xxx[FWY]xx[WFY]_screen_test_set.csv",
        "metadata": env.PROCESSED_DATA_DIR / "v1"
        / "train_test_splits"
        / "v1"
        / "split_metadata.json",
    },
}


class ScreeningTrainTestSplit:
    """
    A class to represent the train/test split for the screening dataset.
    Attributes:
        train (pd.DataFrame): The training set.
        test_LVI (pd.DataFrame): The test set for the LVI regex.
        test_WFY (pd.DataFrame): The test set for the WFY regex.
        metadata (dict): Metadata about the split.
    """

    def __init__(self, version="screening_v1"):
        if version not in _TRAIN_TEST_SETS:
            raise ValueError(f"Train/test split version {version} not found")

        self.version = version
        self._train_file = _TRAIN_TEST_SETS[version]["train"]
        self.train = pd.read_csv(_TRAIN_TEST_SETS[version]["train"])
        self._test_LVI_file = _TRAIN_TEST_SETS[version]["test-LVI"]
        self.test_LVI = pd.read_csv(_TRAIN_TEST_SETS[version]["test-LVI"])
        self._test_WFY_file = _TRAIN_TEST_SETS[version]["test-WFY"]
        self.test_WFY = pd.read_csv(_TRAIN_TEST_SETS[version]["test-WFY"])

        self._metadata_file = _TRAIN_TEST_SETS[version]["metadata"]
        with open(_TRAIN_TEST_SETS[version]["metadata"], "r") as f:
            self.metadata = json.load(f)


    def __repr__(self):
        s = f"ScreeningTrainTestSplit({self.version})\n"
        s += "\n".join([f"{k} - {v}" for k, v in self.metadata.items() if not k.startswith("_")])
        return s


# def get_train_test_data(version="v1"):
#     """Get training and test data from the specified version of the split."""
#     if version not in _TRAIN_TEST_SETS:
#         raise ValueError(f"Train/test split version {version} not found")

#     train_df = pd.read_csv(_TRAIN_TEST_SETS[version]["train"])
#     test_df = pd.read_csv(_TRAIN_TEST_SETS[version]["test"])

#     return train_df, test_df
