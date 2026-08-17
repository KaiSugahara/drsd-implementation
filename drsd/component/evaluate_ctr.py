import numpy as np
import polars as pl

from drsd.evaluation import Evaluator


def evaluate_ctr_component(
    df: pl.DataFrame,
    predictions: np.ndarray,
) -> dict[str, float]:

    metrics = Evaluator.score_recommendation(df, predictions)
    metrics["cross_entropy"] = Evaluator.ctr_cross_entropy(df.get_column("clicked").to_numpy(), predictions)
    return metrics
