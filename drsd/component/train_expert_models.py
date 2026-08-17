import logging

import polars as pl

from drsd.feature_encoder import FeatureEncoder
from drsd.schema import RelationshipSchema
from drsd.trainer.ctr import CTRTrainer
from drsd.trainer.relationship import RelationshipTrainer
from drsd.util import hard_routing


def train_expert_models_component(
    relationship_trainer: RelationshipTrainer,
    feature_encoder: FeatureEncoder,
    schema: RelationshipSchema,
    train_X_df: pl.DataFrame,
    valid_X_df: pl.DataFrame,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    seed: int,
    logger: logging.Logger,
) -> dict[str, CTRTrainer]:
    """Train one CTR expert per relationship label for MoE inference."""

    train_relationship_probabilities = relationship_trainer.predict_proba(train_X_df)
    valid_relationship_probabilities = relationship_trainer.predict_proba(valid_X_df)

    train_relationship_probabilities = hard_routing(train_relationship_probabilities)
    valid_relationship_probabilities = hard_routing(valid_relationship_probabilities)

    expert_trainers: dict[str, CTRTrainer] = {}

    for label, idx in schema.active_label_map.items():
        trainer = CTRTrainer(
            feature_encoder=feature_encoder,
        ).train(
            train_X_df=train_X_df,
            valid_X_df=valid_X_df,
            train_y_sr=train_df.get_column("clicked"),
            valid_y_sr=valid_df.get_column("clicked"),
            train_weight_sr=pl.Series(train_relationship_probabilities[:, idx]),
            valid_weight_sr=pl.Series(valid_relationship_probabilities[:, idx]),
            seed=seed,
        )

        assert trainer.model is not None
        logger.debug(f"CTRTrainer / {label} / best_iteration: {trainer.model.best_iteration}")

        expert_trainers[label] = trainer

    return expert_trainers
