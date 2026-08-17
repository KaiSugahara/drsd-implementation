import logging
from dataclasses import dataclass

import mlflow
import numpy as np

from drsd.annotation_store import AnnotationStore
from drsd.component.annotate_pairs import annotate_pairs_component
from drsd.component.evaluate_ctr import evaluate_ctr_component
from drsd.component.predict_expert_predictions import predict_expert_predictions_component
from drsd.component.predict_final_predictions import predict_final_predictions_component
from drsd.component.predict_relationship_probabilities import predict_relationship_probabilities_component
from drsd.component.propose_hypothesis import propose_hypothesis_component
from drsd.component.sample_random_candidates import sample_random_candidates_component
from drsd.component.sample_schema_gap_candidates import sample_schema_gap_candidates_component
from drsd.component.select_prunable_label import select_prunable_label_component
from drsd.component.train_baseline_ctr import train_baseline_ctr_component
from drsd.component.train_expert_models import train_expert_models_component
from drsd.component.train_relationship_model import train_relationship_model_component
from drsd.feature_encoder import FeatureEncoder
from drsd.llm import LLMClient
from drsd.pruning import ConsecutivePruningManager
from drsd.reader import MINDsmallReader
from drsd.schema import RelationshipSchema
from drsd.trainer.ctr import CTRTrainer
from drsd.trainer.relationship import RelationshipTrainer

CATEGORICAL_FEATURES = [
    "category",
    "subcategory",
    "category_candidate",
    "subcategory_candidate",
    "category_concat",
    "subcategory_concat",
    "category_match",
    "subcategory_match",
]


@dataclass
class SchemaDiscoveryResult:
    round_id: int
    hypothesis_name: str | None
    hypothesis_definition: str | None
    valid_metrics: dict[str, float] | None
    test_metrics: dict[str, float] | None
    relationship_trainer: RelationshipTrainer
    expert_trainers: dict[str, CTRTrainer]


@dataclass
class BaselineResult:
    valid_metrics: dict[str, float]
    test_metrics: dict[str, float]
    trainer: CTRTrainer


class BasePipeline:
    def __init__(self, seed: int) -> None:
        # Initialize SeedSequence
        mlflow.log_param("seed", seed)
        self.seed_seq = np.random.SeedSequence(seed)

        # Set Dataset
        reader = MINDsmallReader()
        self.news_df = reader.get_news_df("train").select("news_id", "title", "category")
        self.train_df, self.train_X_df, self.valid_df, self.valid_X_df, self.test_df, self.test_X_df = (
            reader.get_dataset_dfs()
        )

        # Initialize Feature Encoder
        categorical_columns = [
            "category",
            "subcategory",
            "category_candidate",
            "subcategory_candidate",
            "category_concat",
            "subcategory_concat",
            "category_match",
            "subcategory_match",
        ]
        self.feature_encoder = FeatureEncoder(categorical_columns=categorical_columns)
        self.feature_encoder.fit(self.train_X_df)

        # Initialize Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def generate_seed(self) -> int:
        return self.seed_seq.spawn(1)[0].generate_state(1)[0] % 2**10


class SchemaDiscoveryPipeline(BasePipeline):
    def __init__(self, seed: int, candidate_num: int) -> None:
        super().__init__(seed=seed)

        # Set Hyper Parameters
        self.candidate_num = candidate_num
        mlflow.log_param("candidate_num", candidate_num)

        # Initialize Managers
        self.schema = RelationshipSchema(
            {
                "Background": "The Target News offers historical precedents or underlying social conditions that help explain the current breaking events reported in the Query News.",
                "Update": "The Target News reports on new developments, chronological progressions, or the aftermath of the incident originally covered in the Query News.",
            }
        )
        self.llm = LLMClient()
        self.annotation_store = AnnotationStore()
        self.pruning_manager = ConsecutivePruningManager(threshold=3)

    def run_round(
        self,
        round_id: int,
        relationship_probabilities_by_subset: dict[str, np.ndarray] | None,
        final_predictions_by_subset: dict[str, np.ndarray] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

        if relationship_probabilities_by_subset is not None and final_predictions_by_subset is not None:
            # Step1: Schema-gap Sampling
            self.logger.info("Step1: Schema-gap Sampling")
            candidates = sample_schema_gap_candidates_component(
                relationship_probabilities=relationship_probabilities_by_subset["train"],
                final_predictions=final_predictions_by_subset["train"],
                top_n=self.candidate_num,
                train_df=self.train_df,
                news_df=self.news_df,
                annotation_store=self.annotation_store,
            )

            # Step2-1: Hypothesis proposal by LLM
            self.logger.info("Step2-1: Hypothesis proposal by LLM")
            hypothesis, proposal_prompt = propose_hypothesis_component(
                llm=self.llm, schema=self.schema, news_pairs=candidates, seed=self.generate_seed()
            )
            for hypothesis_name, hypothesis_definition in hypothesis.items():
                self.logger.debug(f"Add: {hypothesis_name} ({hypothesis_definition})")
                self.schema.add(hypothesis_name, hypothesis_definition)
            mlflow.log_text(proposal_prompt, f"proposal_prompt_{round_id}.txt")

        else:
            candidates = sample_random_candidates_component(
                top_n=self.candidate_num,
                train_df=self.train_df,
                news_df=self.news_df,
                seed=self.generate_seed(),
            )

        # Step2-2: Additional annotation
        self.logger.info("Step2-2: Additional annotation")
        annotated_pairs, annotation_prompt = annotate_pairs_component(
            llm=self.llm,
            schema=self.schema,
            news_pairs=candidates,
            seed=self.generate_seed(),
        )
        self.annotation_store.add(annotated_pairs, round_id=round_id, source="annotation")
        mlflow.log_text(annotation_prompt, f"annotation_prompt_{round_id}.txt")
        mlflow.log_table(self.annotation_store.to_frame().to_pandas(), f"annotation_result_{round_id}.json")
        mlflow.log_table(
            self.annotation_store.to_frame().group_by("relationship").len().sort("len", descending=True).to_pandas(),
            f"annotation_count_{round_id}.json",
        )

        # Step3: Extend and retrain models
        self.logger.info("Step3: Extend and retrain models")
        relationship_trainer = train_relationship_model_component(
            annotation_store=self.annotation_store,
            train_df=self.train_df,
            feature_encoder=self.feature_encoder,
            schema=self.schema,
            train_X_df=self.train_X_df,
            seed=self.generate_seed(),
            logger=self.logger,
        )
        expert_trainers = train_expert_models_component(
            relationship_trainer=relationship_trainer,
            feature_encoder=self.feature_encoder,
            schema=self.schema,
            train_X_df=self.train_X_df,
            valid_X_df=self.valid_X_df,
            train_df=self.train_df,
            valid_df=self.valid_df,
            seed=self.generate_seed(),
            logger=self.logger,
        )

        # Step4: Inference and evaluation
        self.logger.info("Step4: Inference and evaluation")
        relationship_probabilities_by_subset: dict[str, np.ndarray] = {}
        expert_predictions_by_subset: dict[str, np.ndarray] = {}
        final_predictions_by_subset: dict[str, np.ndarray] = {}
        metrics_by_subset = {}
        for subset_name, X_df, df in [
            ("train", self.train_X_df, self.train_df),
            ("valid", self.valid_X_df, self.valid_df),
            ("test", self.test_X_df, self.test_df),
        ]:
            relationship_probabilities_by_subset[subset_name] = predict_relationship_probabilities_component(
                relationship_trainer=relationship_trainer,
                X_df=X_df,
            )
            expert_predictions_by_subset[subset_name] = predict_expert_predictions_component(
                relationship_trainer=relationship_trainer,
                expert_trainers=expert_trainers,
                X_df=X_df,
            )
            final_predictions_by_subset[subset_name] = predict_final_predictions_component(
                relationship_probabilities=relationship_probabilities_by_subset[subset_name],
                expert_predictions=expert_predictions_by_subset[subset_name],
            )
            metrics_by_subset[subset_name] = evaluate_ctr_component(
                df=df,
                predictions=final_predictions_by_subset[subset_name],
            )

        # Step5: Ablation and pruning of variable labels (Greedy Pruning)
        self.logger.info("Step5: Ablation and pruning of variable labels (Greedy Pruning)")
        removed_labels, rejected_labels, ablation_ces = select_prunable_label_component(
            df=self.valid_df,
            relationship_trainer=relationship_trainer,
            relationship_probabilities=relationship_probabilities_by_subset["valid"],
            expert_predictions=expert_predictions_by_subset["valid"],
        )
        mlflow.log_dict(ablation_ces, f"ablation_ces_{round_id}.json")

        confirmed_removed_labels, confirmed_rejected_labels, streak_snapshot = self.pruning_manager.apply_round(
            removed_labels=removed_labels,
            rejected_labels=rejected_labels,
        )
        mlflow.log_dict(streak_snapshot, f"pruning_streaks_{round_id}.json")

        for removed_label in confirmed_removed_labels:
            self.logger.debug(f"Remove: {removed_label}")
            self.annotation_store.remove(removed_label)
            self.schema.remove(removed_label)

        for rejected_label in confirmed_rejected_labels:
            self.logger.debug(f"Reject: {rejected_label}")
            self.annotation_store.remove(rejected_label)
            self.schema.reject(rejected_label)

        mlflow.log_dict(self.schema.hypothesis_memory, f"hypothesis_memory_{round_id}.json")

        # Logging to MLflow
        self.logger.info("Logging to MLflow")
        for subset_name, metrics in metrics_by_subset.items():
            mlflow.log_metrics({f"{subset_name}_{k}": v for k, v in metrics.items()}, step=round_id)

        return relationship_probabilities_by_subset, final_predictions_by_subset


class BaselinePipeline(BasePipeline):
    def run(self) -> None:
        self.logger.info("Train CTR model")
        trainer = train_baseline_ctr_component(
            feature_encoder=self.feature_encoder,
            train_X_df=self.train_X_df,
            valid_X_df=self.valid_X_df,
            train_df=self.train_df,
            valid_df=self.valid_df,
            seed=self.generate_seed(),
        )

        self.logger.info("Predict with baseline CTR model")
        valid_predictions = trainer.predict(self.valid_X_df)
        test_predictions = trainer.predict(self.test_X_df)

        self.logger.info("Evaluate CTR model")
        valid_metrics = evaluate_ctr_component(df=self.valid_df, predictions=valid_predictions)
        test_metrics = evaluate_ctr_component(df=self.test_df, predictions=test_predictions)

        # Logging to MLflow
        self.logger.info("Logging to MLflow")
        mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
