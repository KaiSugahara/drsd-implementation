from typing import Literal

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split


class MINDsmallReader:
    def get_news_df(self, subset: Literal["train", "dev"] = "train") -> pl.DataFrame:
        file_path = f"/workspace/dataset/MINDsmall_{subset}/news.tsv"
        return pl.from_pandas(
            pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                names=[
                    "news_id",
                    "category",
                    "subcategory",
                    "title",
                    "abstract",
                    "url",
                    "title_entities",
                    "abstract_entities",
                ],
            )
        )

    def get_behavior_df(self, subset: Literal["train", "dev"] = "train") -> pl.DataFrame:
        file_path = f"/workspace/dataset/MINDsmall_{subset}/behaviors.tsv"
        return (
            pl.from_pandas(
                pd.read_csv(
                    file_path,
                    sep="\t",
                    header=None,
                    names=["impression_id", "user_id", "time", "history", "impressions"],
                )
            )
            .with_columns(
                pl.col("history").str.split(" "),
                pl.col("impressions").str.split(" "),
            )
            .with_columns(
                pl.col("impressions")
                .list.eval(pl.element().str.replace("-1", "").filter(pl.element().str.ends_with("-1")))
                .alias("clicked"),
                pl.col("impressions")
                .list.eval(pl.element().str.replace("-0", "").filter(pl.element().str.ends_with("-0")))
                .alias("non_clicked"),
            )
        )

    def get_dataset_dfs(
        self,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        feature_df = pl.read_parquet("/workspace/processed/feature.parquet")

        df_by_subset: dict[str, pl.DataFrame] = {}
        df_by_subset["train"], df_by_subset["valid"] = train_test_split(
            self.get_behavior_df("train"), test_size=0.3, random_state=0
        )
        df_by_subset["test"] = self.get_behavior_df("dev")

        for subset in df_by_subset.keys():
            df_by_subset[subset] = df_by_subset[subset].select(
                "impression_id", pl.col("history").list.last().alias("last_news_id"), "clicked", "non_clicked"
            )
            df_by_subset[subset] = (
                pl.concat(
                    [
                        df_by_subset[subset].select(
                            "impression_id",
                            pl.col("last_news_id").alias("query_news_id"),
                            pl.col("clicked").alias("target_news_id"),
                            pl.lit(1).alias("clicked"),
                        ),
                        df_by_subset[subset].select(
                            "impression_id",
                            pl.col("last_news_id").alias("query_news_id"),
                            pl.col("non_clicked").alias("target_news_id"),
                            pl.lit(0).alias("clicked"),
                        ),
                    ]
                )
                .filter(pl.col("query_news_id").is_not_null())
                .explode("target_news_id")
                .join(
                    feature_df,
                    left_on=["query_news_id", "target_news_id"],
                    right_on=["news_id", "candidate_news_id"],
                )
            )

        X_columns = feature_df.drop("news_id", "candidate_news_id").columns

        return (
            df_by_subset["train"].drop(X_columns),
            df_by_subset["train"].select(X_columns),
            df_by_subset["valid"].drop(X_columns),
            df_by_subset["valid"].select(X_columns),
            df_by_subset["test"].drop(X_columns),
            df_by_subset["test"].select(X_columns),
        )
