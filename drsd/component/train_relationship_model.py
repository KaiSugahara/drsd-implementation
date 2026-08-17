import logging

import polars as pl
from sklearn.model_selection import train_test_split

from drsd.annotation_store import AnnotationStore
from drsd.feature_encoder import FeatureEncoder
from drsd.schema import RelationshipSchema
from drsd.trainer import RelationshipTrainer


def train_relationship_model_component(
    annotation_store: AnnotationStore,
    train_df: pl.DataFrame,
    feature_encoder: FeatureEncoder,
    schema: RelationshipSchema,
    train_X_df: pl.DataFrame,
    seed: int,
    logger: logging.Logger,
) -> RelationshipTrainer:

    labeled_train_df = annotation_store.add_relationship_column_to_frame(train_df)

    target_relationships = (
        labeled_train_df.get_column("relationship")
        .drop_nulls()
        .value_counts()
        .filter(pl.col("count") > 1)
        .get_column("relationship")
    )  # only relationships with 2+ samples (a single sample cannot be split)
    mask = labeled_train_df.get_column("relationship").is_in(target_relationships)

    trainer = RelationshipTrainer(feature_encoder=feature_encoder, schema=schema)

    if mask.sum() == 0:
        logger.warning("No relationships with more than 1 sample found. Skipping training.")
        return trainer

    train_train_X_df, train_valid_X_df, train_train_y_sr, train_valid_y_sr = train_test_split(
        train_X_df.filter(mask),
        labeled_train_df.filter(mask).get_column("relationship"),
        test_size=0.3,
        random_state=seed,
        stratify=labeled_train_df.filter(mask).get_column("relationship"),
    )

    trainer.train(
        train_X_df=train_train_X_df,
        valid_X_df=train_valid_X_df,
        train_y_sr=train_train_y_sr,
        valid_y_sr=train_valid_y_sr,
        seed=seed,
    )

    assert trainer.model is not None
    logger.debug(f"RelationshipTrainer / best_iteration: {trainer.model.best_iteration}")

    return trainer
