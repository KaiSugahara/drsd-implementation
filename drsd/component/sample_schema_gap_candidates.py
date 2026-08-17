import numpy as np
import polars as pl

from drsd.annotation_store import AnnotationStore
from drsd.schema import News, NewsPair


def sample_schema_gap_candidates_component(
    relationship_probabilities: np.ndarray,
    final_predictions: np.ndarray,
    top_n: int,
    train_df: pl.DataFrame,
    news_df: pl.DataFrame,
    annotation_store: AnnotationStore,
) -> list[NewsPair]:

    uncertainty = 1.0 - relationship_probabilities.max(axis=1)

    sample_df = (
        train_df
        # join news info
        .join(
            news_df.rename({"news_id": "query_news_id", "title": "query_title", "category": "query_category"}),
            on="query_news_id",
            how="left",
        )
        .join(
            news_df.rename({"news_id": "target_news_id", "title": "target_title", "category": "target_category"}),
            on="target_news_id",
            how="left",
        )
        # scoring
        .with_columns(
            pl.Series(uncertainty).alias("uncertainty"),
            pl.Series(final_predictions).alias("ctr_pred"),
        )
        .with_columns((pl.col("uncertainty") * (-pl.col("ctr_pred").log())).alias("score"))
        # filter
        .filter(pl.col("clicked") == 1)
        .filter(pl.col("query_category") == pl.col("target_category"))
        .unique(["query_news_id", "target_news_id"], maintain_order=True)
        # sort
        .sort("score", descending=True)
        # extract
        .head(top_n)
    )

    return [
        NewsPair(
            query_news=News(news_id=row["query_news_id"], title=row["query_title"]),
            target_news=News(news_id=row["target_news_id"], title=row["target_title"]),
        )
        for row in sample_df.to_dicts()
    ]
