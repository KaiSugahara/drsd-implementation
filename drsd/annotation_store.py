import polars as pl

from drsd.schema import NewsPair


class AnnotationStore:
    def __init__(self) -> None:
        self._df = pl.DataFrame(
            schema={
                "query_news_id": pl.String,
                "target_news_id": pl.String,
                "relationship": pl.String,
                "round_id": pl.Int64,
                "source": pl.String,
            }
        )

    def add(self, labeled_pairs: list[NewsPair], round_id: int, source: str) -> None:
        incoming_df = pl.DataFrame(
            [
                {
                    "query_news_id": pair.query_news.news_id,
                    "target_news_id": pair.target_news.news_id,
                    "relationship": pair.relationship,
                    "round_id": round_id,
                    "source": source,
                }
                for pair in labeled_pairs
            ]
        )

        self._df = pl.concat([self._df, incoming_df], how="vertical").unique(
            subset=["query_news_id", "target_news_id"], keep="last", maintain_order=True
        )

    def to_frame(self) -> pl.DataFrame:
        return self._df

    def remove(self, relationship: str) -> None:
        self._df = self._df.filter(pl.col("relationship") != relationship)

    def add_relationship_column_to_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.join(
            self.to_frame().select("query_news_id", "target_news_id", "relationship"),
            on=["query_news_id", "target_news_id"],
            how="left",
        )
