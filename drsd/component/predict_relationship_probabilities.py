import numpy as np
import polars as pl

from drsd.trainer import RelationshipTrainer


def predict_relationship_probabilities_component(
    relationship_trainer: RelationshipTrainer,
    X_df: pl.DataFrame,
) -> np.ndarray:

    return relationship_trainer.predict_proba(X_df)
