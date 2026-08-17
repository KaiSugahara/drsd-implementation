import numpy as np
import polars as pl

from drsd.evaluation import Evaluator
from drsd.trainer.relationship import RelationshipTrainer
from drsd.util import hard_routing


def select_prunable_label_component(
    df: pl.DataFrame,
    relationship_trainer: RelationshipTrainer,
    relationship_probabilities: np.ndarray,
    expert_predictions: np.ndarray,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Select the label whose removal yields the smallest ablated cross-entropy."""

    y_true = df.get_column("clicked").to_numpy()

    y_pred = (relationship_probabilities * expert_predictions).sum(axis=1)

    ce_diff_by_label = {}

    for i, label in enumerate(relationship_trainer.labels()):
        sample_weight = hard_routing(relationship_probabilities)[:, i]
        if sum(sample_weight) == 0:
            ce_diff_by_label[label] = 0
            continue

        ce = Evaluator.ctr_cross_entropy(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

        removed_relationship_probabilities = np.hstack(
            [relationship_probabilities[:, :i], relationship_probabilities[:, i + 1 :]]
        )
        removed_relationship_probabilities = (
            removed_relationship_probabilities.T / removed_relationship_probabilities.sum(axis=1)
        ).T
        removed_expert_predictions = np.hstack([expert_predictions[:, :i], expert_predictions[:, i + 1 :]])
        removed_y_pred = (removed_relationship_probabilities * removed_expert_predictions).sum(axis=1)
        removed_ce = Evaluator.ctr_cross_entropy(y_true=y_true, y_pred=removed_y_pred, sample_weight=sample_weight)

        ce_diff_by_label[label] = removed_ce - ce

    removed_labels = [k for k, v in ce_diff_by_label.items() if (v == 0) and (k not in ["Unknown"])]
    rejected_labels = [k for k, v in ce_diff_by_label.items() if (v < 0) and (k not in ["Unknown"])]

    return removed_labels, rejected_labels, ce_diff_by_label
