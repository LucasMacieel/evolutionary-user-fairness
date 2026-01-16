import logging
from utils.rank_metrics import ndcg_at_k


def create_logger(name="result_logger", path="results.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Use mode='w' to overwrite file instead of append
    file_handler = logging.FileHandler(path, mode="w")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def evaluation_methods(df, metrics):
    """
    Generate evaluation scores (vectorized version).
    :param df:
    :param metrics:
    :return:
    """
    evaluations = []
    data_df = df.copy(deep=True)
    data_df["q*s"] = data_df["q"] * data_df["score"]

    # Pre-sort once for all metrics
    tmp_df = data_df.sort_values(by="q*s", ascending=False, ignore_index=True)

    # Add row number within each user group (for top-k selection)
    tmp_df["rank_in_group"] = tmp_df.groupby("uid").cumcount()

    for metric in metrics:
        k = int(metric.split("@")[-1])

        # Filter to top-k per user
        top_k_df = tmp_df[tmp_df["rank_in_group"] < k]

        if metric.startswith("ndcg@"):
            # Vectorized NDCG calculation
            def calc_ndcg(group):
                labels = group["label"].values[:k]
                return ndcg_at_k(labels.tolist(), k=k, method=1)

            ndcg_series = top_k_df.groupby("uid").apply(calc_ndcg, include_groups=False)
            evaluations.append(ndcg_series.mean())

        elif metric.startswith("hit@"):
            # Vectorized hit calculation: any label > 0 in top-k
            hits = top_k_df.groupby("uid")["label"].sum() > 0
            evaluations.append(hits.mean())

        elif metric.startswith("precision@"):
            # Vectorized precision: sum of labels in top-k / k
            precisions = top_k_df.groupby("uid")["label"].sum() / k
            evaluations.append(precisions.mean())

        elif metric.startswith("recall@"):
            # Vectorized recall: need total labels per user
            top_k_labels = top_k_df.groupby("uid")["label"].sum()
            total_labels = tmp_df.groupby("uid")["label"].sum()

            # Filter users with at least one relevant item
            valid_mask = total_labels > 0
            recalls = top_k_labels[valid_mask] / total_labels[valid_mask]
            evaluations.append(recalls.mean())

        elif metric.startswith("f1@"):
            # Vectorized F1: 2 * hits_in_top_k / (total_relevant + k)
            top_k_labels = top_k_df.groupby("uid")["label"].sum()
            total_labels = tmp_df.groupby("uid")["label"].sum()

            # Filter users with at least one relevant item
            valid_mask = total_labels > 0
            f1_scores = 2 * top_k_labels[valid_mask] / (total_labels[valid_mask] + k)
            evaluations.append(f1_scores.mean())

    return evaluations
