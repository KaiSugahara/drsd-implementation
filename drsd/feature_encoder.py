from typing import Iterable, Self

import polars as pl


class FeatureEncoder:
    """Encode categorical feature columns to integer ids.

    Example:
        encoder = FeatureEncoder(["category", "subcategory"])
        encoder.fit(train_df)
        train_encoded = encoder.transform(train_df)
        valid_encoded = encoder.transform(valid_df)
    """

    def __init__(self, categorical_columns: Iterable[str]) -> None:
        self.categorical_columns = list(categorical_columns)
        self.category_maps: dict[str, dict[object, int]] = {}

    def fit(self, df: pl.DataFrame) -> Self:
        missing = [c for c in self.categorical_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing categorical columns in DataFrame: {missing}")

        for col in self.categorical_columns:
            categories = df.select(pl.col(col)).drop_nulls().unique(maintain_order=True).get_column(col).to_list()
            self.category_maps[col] = {category: idx + 1 for idx, category in enumerate(categories)}

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.category_maps:
            raise ValueError("FeatureEncoder is not fitted. Call fit() before transform().")

        return df.with_columns(
            pl.col(col).replace_strict(mapping, default=0) for col, mapping in self.category_maps.items()
        )

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)
