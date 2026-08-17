import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import log_loss


class Evaluator:
    """Compute recommendation and CTR metrics."""

    @staticmethod
    def score_recommendation(test_df: pl.DataFrame, predictions: np.ndarray) -> dict[str, float]:
        """Compute ranking metrics from impression-level predictions.

        Args:
            test_df (pl.DataFrame): Evaluation data.
            predictions (np.ndarray): CTR predictions.

        Returns:
            dict[str, float]: Metric dict containing hit, precision, recall, ndcg, and coverage.
        """
        pred_df = test_df.with_columns(pl.Series(predictions).alias("prediction"))
        ranking = (
            pred_df.sort("impression_id", "prediction", descending=[False, True])
            .group_by("impression_id", maintain_order=True)
            .agg([pl.col("clicked"), pl.col("prediction"), pl.col("target_news_id")])
        )
        k_list = [1, 3, 10]
        metrics: dict[str, float] = {}
        for k in k_list:
            metrics[f"hit_{k}"] = ranking.select((pl.col("clicked").list.head(k).list.max()).mean()).item()
            metrics[f"precision_{k}"] = ranking.select((pl.col("clicked").list.head(k).list.sum() / k).mean()).item()
            metrics[f"recall_{k}"] = ranking.select(
                (pl.col("clicked").list.head(k).list.sum() / pl.col("clicked").list.sum()).mean()
            ).item()
            metrics[f"ndcg_{k}"] = ranking.select(
                (
                    pl.col("clicked")
                    .list.head(k)
                    .list.eval(pl.element() / (pl.arange(pl.element().len()) + 2).log(2))
                    .list.sum()
                    / pl.col("clicked")
                    .list.sort(descending=True)
                    .list.head(k)
                    .list.eval(pl.element() / (pl.arange(pl.element().len()) + 2).log(2))
                    .list.sum()
                ).mean()
            ).item()
            metrics[f"coverage_{k}"] = ranking.select(
                pl.col("target_news_id").list.head(k).explode().unique().len()
            ).item()
        return metrics

    @staticmethod
    def ctr_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        """Compute the binary cross-entropy of CTR predictions.

        Args:
            y_true (np.ndarray): Ground-truth labels.
            y_pred (np.ndarray): Predicted probabilities.
            sample_weight (np.ndarray | None): Per-sample weights.

        Returns:
            float: Cross-entropy loss.
        """
        return float(log_loss(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight))

    @staticmethod
    def log_metrics(metrics: dict[str, float], prefix: str, step: int) -> None:
        """Log metrics to MLflow with a common prefix.

        Args:
            metrics (dict[str, float]): Metric dict to log.
            prefix (str): Prefix to prepend to each metric name.
            step (int): MLflow logging step.

        Returns:
            None
        """
        mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()}, step=step)
