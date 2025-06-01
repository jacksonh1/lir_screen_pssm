import logomaker as lm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import lir_proteome_screen_pssm.environment as env

# AMINO_ACIDS = sorted(
#     [
#         "V",
#         "E",
#         "K",
#         "I",
#         "H",
#         "L",
#         "G",
#         "T",
#         "M",
#         "N",
#         "S",
#         "P",
#         "A",
#         "F",
#         "W",
#         "Y",
#         "Q",
#         "R",
#         "C",
#         "D",
#     ]
# )

def mask_low_counts(
    mat: pd.DataFrame,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Mask low counts in a PSSM matrix.

    Parameters
    ----------
    mat : pd.DataFrame
        PSSM matrix (logomaker format)
    min_count : int
        Minimum count to keep in the matrix

    Returns
    -------
    pd.DataFrame
        Masked PSSM matrix (logomaker format)
    """
    mat = mat.copy()
    mat[mat < min_count] = 0
    return mat


def seqfile2list(seqs_file):
    with open(seqs_file) as handle:
        f = handle.readlines()
        sequence_list = [i.strip() for i in f]
    return sequence_list


def write_seqlist(seqlist, filename):
    with open(filename, "w") as handle:
        for seq in seqlist:
            handle.write(seq + "\n")


def union_2_lists(l1, l2):
    """converts each list into a set. finds the union of the 2 sets and returns it as a list"""
    sl1 = set(l1)
    sl2 = set(l2)
    return list(sl1.union(sl2))


def count_sequences(seq_list):
    seq_counts = {i: seq_list.count(i) for i in set(seq_list)}
    return seq_counts


def alignment_2_counts(sequence_list):
    """
    creates a counts matrix PSSM (logomaker format) from a list of sequences
    if an amino acid is absent at a position in the input sequence list,
     it adds a 0 count for that entry in the matrix

    Parameters
    ----------
    sequence_list : list
        list of sequences (strings) from which to generate PSSM

    Returns
    ------
    PSSM counts matrix (logomaker format)
    """
    counts = lm.alignment_to_matrix(sequences=sequence_list, to_type="counts")
    for AA in env.STANDARD_AMINO_ACIDS:
        if AA not in counts.columns:
            counts[AA] = 0.0
    counts = counts.reindex(sorted(counts.columns), axis=1)
    return counts


def PSSM_score_sequence(sequence, PSSM):
    """
    PSSM is a PSSM dataframe of sequence position (rows) vs. residue (columns)
    """
    score = 0
    for pos, aa in enumerate(sequence):
        score = score + PSSM.loc[pos, aa]
    return score


def plot_logo(
    matrix: pd.DataFrame,
    ax=None,
    title=None,
    **kwargs,
):
    """
    Plot a sequence logo using logomaker.

    Parameters
    ----------
    matrix : pd.DataFrame
        A DataFrame containing the matrix you're trying to plot.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to logomaker.logo function.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=[10, 2.5])
    lm.Logo(
        matrix,
        color_scheme="chemistry",
        ax=ax,
        **kwargs,
    )
    if title is not None:
        ax.set_title(title, fontsize=16)
    return ax


def plot_logo_as_heatmap(mat, ax=None):
    """
    Plots matrix as a heatmap from logomaker style PSSM

    Parameters
    ----------
    mat
        PSSM matrix (logomaker format)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=[6, 5])
    sns.heatmap(
        mat.T, annot=True, fmt=".2g", linewidth=0.5, ax=ax, annot_kws={"size": 10}
    )
    ax.tick_params(axis="both", which="both", length=0, labelsize=12)
    ax.set_xlabel("position")
    return ax


def normalize_positions(mat: pd.DataFrame) -> pd.DataFrame:
    """
    normalize each position in the matrix so that it sums to 1

    Parameters
    ----------
    mat : pd.DataFrame
        DataFrame in the logomaker format (rows = positions, columns = amino acids)

    Returns
    -------
    pd.DataFrame
        DataFrame in the logomaker format (rows = positions, columns = amino acids)
        where each position sums to 1
    """
    freq_df = mat.copy()
    freq_df = freq_df.div(freq_df.sum(axis=1), axis=0)
    return freq_df


def freq_2_weight_matrix(
    fg: pd.DataFrame, bg: pd.DataFrame | dict[str, float] | None = None
):
    """
    Calculate the weight matrix for a given foreground and background frequency matrix.
    The weight matrix is calculated as log2(fg / bg) for each position in the matrix.
    if no bg is provided, it will be set to a flat distribution (residues have equal probability).
    zero values in fg or bg will be SET TO 0 in the weight matrix. Thus pseudo-counts are not needed
    necessarily.

    if you want to use pseudo counts, add them to 'fg'/'bg' before calling this function.
    If you want to weight the matrices, do it before calling this function.

    Parameters
    ----------
    fg : pd.DataFrame
        Foreground frequency matrix (logomaker format)
    bg : pd.DataFrame | dict[str, float] | None
        Background frequency matrix (logomaker format) or a dictionary with background frequencies.
        If None, a flat distribution will be used (default).
        If a dictionary is provided, it should have the same keys as the columns
        of 'fg'. The values are the background frequencies for each residue.
        if a dataframe is provided, it should have the same index and columns as 'fg'.
        The values are the background frequencies for each residue.

    Returns
    -------
    pd.DataFrame
        Weight matrix (logomaker format)
    """
    if isinstance(bg, dict):
        assert set(bg.keys()) == set(
            fg.columns
        ), "Background dictionary keys must match foreground columns"
        bg = pd.DataFrame(bg, index=fg.index)
        bg = bg[list(fg.columns)]
    elif bg is None:
        bg = pd.DataFrame(1, index=fg.index, columns=fg.columns)
        bg = normalize_positions(bg)
    assert (
        fg.sum(axis=1).apply(lambda x: np.isclose(x, 1))
    ).all(), f"Foreground matrix must be normalized, {fg.sum(axis=1)}"
    assert isinstance(fg, pd.DataFrame), "Foreground must be a DataFrame"
    assert isinstance(
        bg, pd.DataFrame
    ), "Background must be a DataFrame, dictionary, or None"
    assert (
        fg.columns == bg.columns
    ).all(), f"Foreground and background matrices must have the same columns: {fg.columns} != {bg.columns}"
    assert (
        fg.index == bg.index
    ).all(), "Foreground and background matrices must have the same index"
    weight_matrix = fg.copy()
    for col in fg.columns:
        for ind in fg.index:
            # Calculate the weight using log2(fg / bg)
            if fg.loc[ind, col] == 0 or bg.loc[ind, col] == 0:
                weight_matrix.loc[ind, col] = 0
            else:
                weight_matrix.loc[ind, col] = np.log2(fg.loc[ind, col] / bg.loc[ind, col])  # type: ignore
    return weight_matrix


def freq_2_information_matix(fg, bg=None):
    """
    Calculate the information matrix for a given foreground and background frequency matrix.
    The information matrix is calculated as fg * sum(fg*log2(fg/bg)) for each position in the matrix.
    if no bg is provided, it will be set to a flat distribution.
    """
    information_matrix = fg.copy()
    weight_matrix = freq_2_weight_matrix(fg, bg)
    col_information = []
    for ind in fg.index:
        i_i = 0
        for col in fg.columns:
            i_i += fg.loc[ind, col] * weight_matrix.loc[ind, col]
        for col in fg.columns:
            information_matrix.loc[ind, col] = fg.loc[ind, col] * i_i
        col_information.append(i_i)
    return information_matrix, col_information


def seqlist_2_counts_matrix(seqlist, weights=None, pseudocount=0):
    """
    Create a counts matrix from a list of sequences.

    Parameters
    ----------
    seqlist : list
        List of sequences (strings) from which to generate PSSM.
    weights : list, optional
        List of weights for each sequence. Weights should be the same length as
        'seqlist'. For each position in each sequence, the weight corresponding
        to that sequence is added to the counts matrix.
        If None, all sequences are counted equally (each position in each
        sequence adds 1 to corresponding matrix), default.
    pseudocount : float, optional
        Pseudocount to add to each entry in the counts matrix. default is 0.

    Returns
    -------
    pd.DataFrame
        Counts matrix (logomaker format).
    """
    assert len(set(map(len, seqlist))) == 1, "All sequences must be the same length"
    seq_length = len(seqlist[0])
    matrix = pd.DataFrame(
        pseudocount,
        index=range(seq_length),
        columns=env.STANDARD_AMINO_ACIDS,
        dtype=float,
    )
    if weights is None:
        weights = np.ones(len(seqlist))
    for seq, weight in zip(seqlist, weights):
        for pos, aa in enumerate(seq):
            matrix.at[pos, aa] += weight
    matrix.index.name = "pos"
    return matrix


def make_pssm(
    df_counts,
    bg=None,
    min_count=0,
    pseudocount=0.0,
    plot=True,
    plot_title=None,
    ax=None,
):
    fg = df_counts.copy()
    fg = fg + pseudocount
    fg = mask_low_counts(fg, min_count=min_count)
    fg = normalize_positions(fg)
    pssm = freq_2_weight_matrix(fg, bg=bg)
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 2))
        plot_logo(pssm, ax=ax)
        ax.set_title(plot_title) # type: ignore
    return pssm
