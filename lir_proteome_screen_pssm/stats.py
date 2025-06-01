import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Optional
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score

def roc_plot(y_true, y_score, ax, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, label=f'{label}\n(AUC = {roc_auc:.4f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.text(0.02, 0.85, f"{y_true.sum()} binder\n{len(y_true) - y_true.sum()} nonbinder", transform=ax.transAxes, fontsize=12)
    return ax, roc_auc


def df_2_roc_auc(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
) -> float:
    """calculates the roc auc from a dataframe with 2 columns. one with the true labels (1 and 0's) and one with measured scores

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with at least 2 columns. one with the true labels (1 and 0's) and one with measured scores
    label_col : str
        column name of the column with the true labels (1 and 0's)
    score_col : str
        column name of the column with the measured scores

    Returns
    -------
    float
        area under the roc curve (roc_auc)
    """
    
    return roc_auc_score(
        df[label_col].astype(int).values, df[score_col].astype(float).values # type: ignore
    )


def precision_recall_curve(
    true_labels: list[int], scores: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """calculates the precision recall curve from 2 lists. one with the true labels (1 and 0's) and one with measured scores

    Parameters
    ----------
    true_labels : list[int]
        The true labels (1 and 0's)
    scores : list[float]
        The measured scores which would be used to predict the labels

    Returns
    -------
    tuple[np.array[float], np.array[float], np.array[float], float]
        a tuple with the `precision`, `recall`, `thresholds` and area under the precision recall curve (`auPRC`)
    """
    assert len(true_labels) == len(
        scores
    ), "true_labels and scores must be of same length"
    assert set(true_labels) == {0, 1}, "true_labels must be 0 or 1"
    precision, recall, thresholds = metrics.precision_recall_curve(
        np.array(true_labels),
        np.array(scores),
    )
    auPRC = metrics.auc(recall, precision)
    return precision, recall, thresholds, float(auPRC)


def df_2_precision_recall_curve(
    df: pd.DataFrame, label_col: str, score_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """calculates the precision recall curve from a dataframe with 2 columns. one with the true labels (1 and 0's) and one with measured scores

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with 2 columns. one with the true labels (1 and 0's) and one with measured scores
    label_col : str
        column name of the column with the true labels (1 and 0's)
    score_col : str
        column name of the column with the measured scores

    Returns
    -------
    tuple[np.array[float], np.array[float], np.array[float], float]
        a tuple with the `precision`, `recall`, `thresholds` and area under the precision recall curve (`auPRC`)
    """
    return precision_recall_curve(
        list(df[label_col].astype(int).values), list(df[score_col].astype(float).values)
    )


def ave_precision(true_labels, scores):
    """calculates the average precision from 2 lists. one with the true labels (1 and 0's) and one with measured scores

    Parameters
    ----------
    true_labels : list[int]
        The true labels (1 and 0's)
    scores : list[float]
        The measured scores which would be used to predict the labels

    Returns
    -------
    float
        average precision
    """
    return float(metrics.average_precision_score(true_labels, scores))


def df_2_ave_precision(df: pd.DataFrame, label_col: str, score_col: str) -> float:
    """calculates the average precision from a dataframe with 2 columns. one with the true labels (1 and 0's) and one with measured scores

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with 2 columns. one with the true labels (1 and 0's) and one with measured scores
    label_col : str
        column name of the column with the true labels (1 and 0's)
    score_col : str
        column name of the column with the measured scores

    Returns
    -------
    float
        average precision
    """
    return ave_precision(
        list(df[label_col].astype(int).values), list(df[score_col].astype(float).values)
    )
