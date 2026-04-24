from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


REPRESENTATION_STATUS = "EXACT"


def representation_per_class_columns() -> list[str]:
    return ["label", "precision", "recall", "f1", "support", "status"]


def embedding_columns(df: pd.DataFrame, *, prefixes: Sequence[str] = ("phi_", "embed_", "embedding_")) -> list[str]:
    cols: list[str] = []
    for prefix in prefixes:
        cols.extend([col for col in df.columns if col.startswith(prefix)])
    return sorted(dict.fromkeys(cols))


def run_linear_probe(
    embeddings: pd.DataFrame,
    *,
    columns: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train a linear probe on strategy embeddings and report classification metrics."""

    if embeddings.empty:
        return _empty_metrics("empty embedding table"), pd.DataFrame(columns=representation_per_class_columns())

    label_col = columns.get("label", "style_label")
    split_col = columns.get("split", "split")
    fund_col = columns.get("fund_id", "fund_id")
    prefixes = tuple(cfg.get("embedding_prefixes", ["phi_", "embed_", "embedding_"]))
    feature_cols = list(cfg.get("embedding_columns", [])) or embedding_columns(embeddings, prefixes=prefixes)
    if not feature_cols:
        return _empty_metrics("no embedding columns found"), pd.DataFrame(columns=representation_per_class_columns())
    if label_col not in embeddings.columns:
        return _empty_metrics(f"missing label column: {label_col}"), pd.DataFrame(columns=representation_per_class_columns())

    df = embeddings.copy()
    if bool(cfg.get("average_embeddings_over_time", False)):
        df = _average_embeddings(df, feature_cols=feature_cols, label_col=label_col, split_col=split_col, fund_col=fund_col)

    df = df.dropna(subset=[label_col]).copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols)
    if df.empty:
        return _empty_metrics("no complete embedding rows after cleaning"), pd.DataFrame(columns=representation_per_class_columns())

    train_df, eval_df, split_source = _split_embedding_frame(df, split_col=split_col, cfg=cfg)
    y_train = train_df[label_col].astype(str).to_numpy()
    y_eval = eval_df[label_col].astype(str).to_numpy()
    if len(np.unique(y_train)) < 2:
        return _empty_metrics("training labels contain fewer than two classes"), pd.DataFrame(columns=representation_per_class_columns())
    if eval_df.empty:
        return _empty_metrics("empty evaluation split"), pd.DataFrame(columns=representation_per_class_columns())

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    x_eval = eval_df[feature_cols].to_numpy(dtype=float)
    classifier_name = str(cfg.get("classifier", "linear_svm"))

    try:
        classifier, classifier_used = _fit_classifier(classifier_name, x_train, y_train, cfg=cfg)
        y_pred = classifier.predict(x_eval)
        metrics, per_class = _classification_report(y_eval, y_pred, classifier_used=classifier_used)
    except Exception as exc:  # pragma: no cover - exercised by graceful integration tests when sklearn is absent/broken
        return _empty_metrics(f"classifier failed: {exc}"), pd.DataFrame(columns=representation_per_class_columns())

    metrics.update(
        {
            "status": REPRESENTATION_STATUS,
            "split_source": split_source,
            "train_rows": int(train_df.shape[0]),
            "eval_rows": int(eval_df.shape[0]),
            "embedding_columns": feature_cols,
        }
    )
    return metrics, per_class


def _average_embeddings(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    label_col: str,
    split_col: str,
    fund_col: str,
) -> pd.DataFrame:
    if fund_col not in df.columns:
        return df
    agg: dict[str, str] = {col: "mean" for col in feature_cols}
    agg[label_col] = "first"
    if split_col in df.columns:
        agg[split_col] = lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]
    return df.groupby(fund_col, dropna=False, as_index=False).agg(agg)


def _split_embedding_frame(df: pd.DataFrame, *, split_col: str, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train_split = str(cfg.get("train_split", "train"))
    eval_split = str(cfg.get("eval_split", cfg.get("test_split", "test")))
    if split_col in df.columns and (df[split_col] == train_split).any() and (df[split_col] == eval_split).any():
        return (
            df.loc[df[split_col] == train_split].copy(),
            df.loc[df[split_col] == eval_split].copy(),
            f"{train_split}_to_{eval_split}",
        )
    if split_col in df.columns and (df[split_col] == train_split).any():
        train_df = df.loc[df[split_col] == train_split].copy()
        eval_df = df.loc[df[split_col] != train_split].copy()
        if not eval_df.empty:
            return train_df, eval_df, f"{train_split}_to_non_train"

    rng = np.random.default_rng(int(cfg.get("random_seed", 42)))
    indices = np.arange(df.shape[0])
    rng.shuffle(indices)
    cutoff = max(1, int(round(df.shape[0] * float(cfg.get("fallback_train_fraction", 0.7)))))
    cutoff = min(cutoff, max(1, df.shape[0] - 1))
    train_idx = indices[:cutoff]
    eval_idx = indices[cutoff:]
    return df.iloc[train_idx].copy(), df.iloc[eval_idx].copy(), "deterministic_fallback_split"


def _fit_classifier(classifier_name: str, x_train: np.ndarray, y_train: np.ndarray, *, cfg: Mapping[str, Any]) -> tuple[Any, str]:
    try:
        if classifier_name == "linear_svm":
            from sklearn.svm import LinearSVC

            classifier = LinearSVC(
                C=float(cfg.get("C", 1.0)),
                max_iter=int(cfg.get("max_iter", 5000)),
                random_state=int(cfg.get("random_seed", 42)),
                dual="auto",
            )
            classifier.fit(x_train, y_train)
            return classifier, "linear_svm"
    except Exception:
        if classifier_name == "linear_svm":
            pass
        else:
            raise

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(
        max_iter=int(cfg.get("max_iter", 5000)),
        random_state=int(cfg.get("random_seed", 42)),
        multi_class="auto",
    )
    classifier.fit(x_train, y_train)
    return classifier, "logistic_regression"


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray, *, classifier_used: str) -> tuple[dict[str, Any], pd.DataFrame]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    labels = np.array(sorted(set(y_true.tolist()) | set(y_pred.tolist())))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
            "status": REPRESENTATION_STATUS,
        },
        columns=representation_per_class_columns(),
    )
    metrics = {
        "classifier": classifier_used,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }
    return metrics, per_class


def _empty_metrics(reason: str) -> dict[str, Any]:
    return {
        "status": "SKIPPED",
        "reason": reason,
        "classifier": "",
        "macro_precision": np.nan,
        "macro_recall": np.nan,
        "macro_f1": np.nan,
        "accuracy": np.nan,
        "labels": [],
        "confusion_matrix": [],
    }
