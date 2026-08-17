import polars as pl

from drsd.schema import News, NewsPair


def sample_random_candidates_component(
    top_n: int,
    train_df: pl.DataFrame,
    news_df: pl.DataFrame,
    seed: int,
) -> list[NewsPair]:

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
        # filter
        .filter(pl.col("clicked") == 1)
        .filter(pl.col("query_category") == pl.col("target_category"))
        .unique(["query_news_id", "target_news_id"], maintain_order=True)
        # sample
        .sample(top_n, seed=seed)
    )

    return [
        NewsPair(
            query_news=News(news_id=row["query_news_id"], title=row["query_title"]),
            target_news=News(news_id=row["target_news_id"], title=row["target_title"]),
        )
        for row in sample_df.to_dicts()
    ]
