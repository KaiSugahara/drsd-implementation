import numpy as np
import polars as pl

from drsd.trainer.ctr import CTRTrainer
from drsd.trainer.relationship import RelationshipTrainer


def predict_expert_predictions_component(
    relationship_trainer: RelationshipTrainer,
    expert_trainers: dict[str, CTRTrainer],
    X_df: pl.DataFrame,
) -> np.ndarray:

    expert_predictions = np.hstack(
        [expert_trainers[label].predict(X_df).reshape(-1, 1) for label in relationship_trainer.labels()]
    )

    return expert_predictions
