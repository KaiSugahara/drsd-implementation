import numpy as np


def predict_final_predictions_component(
    relationship_probabilities: np.ndarray,
    expert_predictions: np.ndarray,
) -> np.ndarray:

    assert relationship_probabilities.shape == expert_predictions.shape
    return (relationship_probabilities * expert_predictions).sum(axis=1)
