from typing import Self

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.utils import compute_sample_weight

from drsd.feature_encoder import FeatureEncoder
from drsd.schema import RelationshipSchema


class RelationshipTrainer:
    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        schema: RelationshipSchema,
    ) -> None:

        self.feature_encoder = feature_encoder
        self.schema = schema
        self.model: lgb.Booster | None = None
        self.label_map: dict[str, int] = self.schema.active_label_map

    def _prepare_dataset(
        self,
        X_df: pl.DataFrame,
        y_sr: pl.Series,
    ) -> lgb.Dataset:

        if y_sr.n_unique() > 1:
            weights = compute_sample_weight(class_weight="balanced", y=y_sr.to_numpy())
        else:
            weights = None

        return lgb.Dataset(
            self.feature_encoder.transform(X_df).to_numpy(),
            label=y_sr.replace_strict(self.label_map).to_numpy(),
            feature_name=X_df.columns,
            categorical_feature=self.feature_encoder.categorical_columns,
            weight=weights,
        )

    def train(
        self,
        train_X_df: pl.DataFrame,
        train_y_sr: pl.Series,
        valid_X_df: pl.DataFrame,
        valid_y_sr: pl.Series,
        seed: int,
        num_round: int = 512,
    ) -> Self:

        train_data = self._prepare_dataset(train_X_df, train_y_sr)
        valid_data = self._prepare_dataset(valid_X_df, valid_y_sr)
        params = {
            "objective": "multiclass",
            "num_class": len(self.label_map),
            "num_leaves": 31,
            "verbose": -1,
            "early_stopping_round": 8,
            "seed": seed,
        }
        self.model = lgb.train(params, train_data, num_boost_round=num_round, valid_sets=[valid_data])

        return self

    def predict_proba(self, X_df: pl.DataFrame) -> np.ndarray:

        if self.model is None:
            probs = np.zeros((X_df.shape[0], len(self.label_map)))
            probs[:, 0] = 1
            return probs

        return self.model.predict(self.feature_encoder.transform(X_df).to_numpy())  # type: ignore

    def labels(self) -> list[str]:
        return list(self.label_map.keys())
