import polars as pl

from drsd.feature_encoder import FeatureEncoder
from drsd.trainer.ctr import CTRTrainer


def train_baseline_ctr_component(
    feature_encoder: FeatureEncoder,
    train_X_df: pl.DataFrame,
    valid_X_df: pl.DataFrame,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    seed: int,
) -> CTRTrainer:
    """Train a single CTR model with uniform sample weights as baseline."""

    train_weight_sr = pl.Series("weight", [1.0] * len(train_df))
    valid_weight_sr = pl.Series("weight", [1.0] * len(valid_df))

    return CTRTrainer(feature_encoder=feature_encoder).train(
        train_X_df=train_X_df,
        valid_X_df=valid_X_df,
        train_y_sr=train_df.get_column("clicked"),
        valid_y_sr=valid_df.get_column("clicked"),
        train_weight_sr=train_weight_sr,
        valid_weight_sr=valid_weight_sr,
        seed=seed,
    )
