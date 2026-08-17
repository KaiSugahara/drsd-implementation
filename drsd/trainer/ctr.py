from typing import Self

import lightgbm as lgb
import numpy as np
import polars as pl

from drsd.feature_encoder import FeatureEncoder


class CTRTrainer:
    def __init__(
        self,
        feature_encoder: FeatureEncoder,
    ) -> None:

        self.feature_encoder = feature_encoder
        self.model: lgb.Booster | None = None

    def _prepare_dataset(
        self,
        X_df: pl.DataFrame,
        y_sr: pl.Series,
        weight_sr: pl.Series,
    ) -> lgb.Dataset:

        return lgb.Dataset(
            self.feature_encoder.transform(X_df).to_numpy(),
            label=y_sr.to_numpy(),
            feature_name=X_df.columns,
            categorical_feature=self.feature_encoder.categorical_columns,
            weight=weight_sr.to_numpy(),
        )

    def train(
        self,
        train_X_df: pl.DataFrame,
        valid_X_df: pl.DataFrame,
        train_y_sr: pl.Series,
        valid_y_sr: pl.Series,
        train_weight_sr: pl.Series,
        valid_weight_sr: pl.Series,
        seed: int,
        num_round: int = 256,
    ) -> Self:

        train_data = self._prepare_dataset(train_X_df, train_y_sr, train_weight_sr)
        valid_data = self._prepare_dataset(valid_X_df, valid_y_sr, valid_weight_sr)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "verbose": -1,
            "early_stopping_round": 8,
            "seed": seed,
        }
        self.model = lgb.train(params, train_data, num_boost_round=num_round, valid_sets=[valid_data])

        return self

    def predict(self, X_df: pl.DataFrame) -> np.ndarray:

        assert self.model is not None
        X = self.feature_encoder.transform(X_df).to_numpy()
        return self.model.predict(X)  # type: ignore
