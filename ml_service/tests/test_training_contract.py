"""Tests for Training Contract, Evaluation Metrics, and Checkpoint Validation.

All test fixtures use synthetic, clearly labelled mock data.
NO real model weights or student data are used.
"""

import json
from pathlib import Path
import pytest

from ml_service.model_loader import HuggingFaceModelLoader
from ml_service.training_contract import (
    ID2LABEL,
    LABEL2ID,
    MODEL_FAMILY,
    NUM_LABELS,
    ClassMetrics,
    EvaluationReport,
    TrainingHyperparameters,
    TrainingMetadata,
    validate_checkpoint_directory,
)


class TestTrainingContractConstants:
    """Test suite for canonical mapping consistency across components."""

    def test_label_mappings_bidirectional_consistency(self) -> None:
        """Verify LABEL2ID and ID2LABEL are strictly consistent mathematical inverses."""
        assert len(LABEL2ID) == 3
        assert len(ID2LABEL) == 3
        assert NUM_LABELS == 3

        for lbl, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == lbl

        for idx, lbl in ID2LABEL.items():
            assert LABEL2ID[lbl] == idx

        assert LABEL2ID["NO"] == 0
        assert LABEL2ID["YES"] == 1
        assert LABEL2ID["REVIEW"] == 2

    def test_contract_labels_match_model_loader_expectations(self) -> None:
        """Verify the contract labels match HuggingFaceModelLoader.REQUIRED_TASK_LABELS."""
        loader_labels = HuggingFaceModelLoader.REQUIRED_TASK_LABELS
        contract_labels = set(LABEL2ID.keys())
        assert contract_labels == loader_labels


class TestTrainingHyperparameters:
    """Test suite for training hyperparameter configuration."""

    def test_default_hyperparameters(self) -> None:
        hp = TrainingHyperparameters()
        assert hp.base_model == MODEL_FAMILY
        assert hp.learning_rate == 2e-5
        assert hp.batch_size == 16
        assert hp.num_epochs == 4
        assert hp.random_seed == 42
        assert hp.fp16 is True
        assert hp.metric_for_best_model == "macro_f1"

    def test_invalid_learning_rate_rejected(self) -> None:
        with pytest.raises(ValueError):
            TrainingHyperparameters(learning_rate=-0.001)

    def test_invalid_epochs_rejected(self) -> None:
        with pytest.raises(ValueError):
            TrainingHyperparameters(num_epochs=0)


class TestEvaluationReportSchema:
    """Test suite for evaluation report schema and metric validation."""

    @pytest.fixture
    def mock_eval_report(self) -> EvaluationReport:
        """Synthetic mock evaluation report fixture."""
        return EvaluationReport(
            model_name="distilbert-student-support",
            model_version="1.0.0",
            dataset_version="1.0.0",
            evaluated_at="2026-10-22T00:00:00Z",
            test_samples_count=300,
            macro_f1=0.88,
            weighted_f1=0.89,
            accuracy=0.89,
            per_class={
                "NO": ClassMetrics(precision=0.88, recall=0.86, f1_score=0.87, support=90),
                "YES": ClassMetrics(precision=0.92, recall=0.90, f1_score=0.91, support=130),
                "REVIEW": ClassMetrics(precision=0.85, recall=0.89, f1_score=0.87, support=80),
            },
            confusion_matrix=[
                [77, 3, 10],   # True NO
                [4, 117, 9],   # True YES
                [6, 3, 71],    # True REVIEW
            ],
            false_yes_count=6,
            false_no_count=10,
            review_precision=0.85,
            notes="Synthetic mock evaluation report for schema test verification.",
        )

    def test_valid_evaluation_report(self, mock_eval_report: EvaluationReport) -> None:
        assert mock_eval_report.macro_f1 == 0.88
        assert mock_eval_report.per_class["YES"].precision == 0.92
        assert mock_eval_report.confusion_matrix[1][1] == 117
        assert mock_eval_report.false_yes_count == 6

    def test_invalid_confusion_matrix_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a 3x3 grid"):
            EvaluationReport(
                model_name="distilbert-student-support",
                model_version="1.0.0",
                dataset_version="1.0.0",
                evaluated_at="2026-10-22T00:00:00Z",
                test_samples_count=100,
                macro_f1=0.8,
                weighted_f1=0.8,
                accuracy=0.8,
                per_class={
                    "NO": ClassMetrics(precision=0.8, recall=0.8, f1_score=0.8, support=30),
                    "YES": ClassMetrics(precision=0.8, recall=0.8, f1_score=0.8, support=40),
                    "REVIEW": ClassMetrics(precision=0.8, recall=0.8, f1_score=0.8, support=30),
                },
                confusion_matrix=[[10, 20], [30, 40]],  # 2x2 instead of 3x3!
                false_yes_count=5,
                false_no_count=5,
                review_precision=0.8,
            )

    def test_missing_required_class_metrics_rejected(self) -> None:
        with pytest.raises(ValueError, match="per_class metrics must include all classes"):
            EvaluationReport(
                model_name="distilbert-student-support",
                model_version="1.0.0",
                dataset_version="1.0.0",
                evaluated_at="2026-10-22T00:00:00Z",
                test_samples_count=100,
                macro_f1=0.8,
                weighted_f1=0.8,
                accuracy=0.8,
                per_class={
                    "NO": ClassMetrics(precision=0.8, recall=0.8, f1_score=0.8, support=50),
                    "YES": ClassMetrics(precision=0.8, recall=0.8, f1_score=0.8, support=50),
                    # Missing "REVIEW"!
                },
                confusion_matrix=[[25, 5, 0], [5, 25, 0], [0, 0, 40]],
                false_yes_count=5,
                false_no_count=5,
                review_precision=0.8,
            )


class TestCheckpointDirectoryValidator:
    """Test suite for checkpoint validation utility."""

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "does_not_exist"
        is_valid, errors = validate_checkpoint_directory(missing_dir)
        assert is_valid is False
        assert any("does not exist" in e for e in errors)

    def test_missing_config_file(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_checkpoint"
        empty_dir.mkdir()
        is_valid, errors = validate_checkpoint_directory(empty_dir)
        assert is_valid is False
        assert any("config.json" in e for e in errors)

    def test_generic_labels_in_config_rejected(self, tmp_path: Path) -> None:
        cp_dir = tmp_path / "generic_checkpoint"
        cp_dir.mkdir()
        config_path = cp_dir / "config.json"
        config_path.write_text(json.dumps({"id2label": {"0": "LABEL_0", "1": "LABEL_1"}}))
        is_valid, errors = validate_checkpoint_directory(cp_dir, require_metadata=False)
        assert is_valid is False
        assert any("Invalid id2label in config.json" in e for e in errors)

    def test_valid_mock_checkpoint_passes(self, tmp_path: Path) -> None:
        cp_dir = tmp_path / "valid_checkpoint"
        cp_dir.mkdir()

        # 1. Write config.json with proper labels
        config_path = cp_dir / "config.json"
        config_path.write_text(json.dumps({
            "id2label": {"0": "NO", "1": "YES", "2": "REVIEW"},
            "label2id": {"NO": 0, "YES": 1, "REVIEW": 2},
        }))

        # 2. Touch mock weight file
        (cp_dir / "model.safetensors").touch()

        # 3. Touch mock tokenizer file
        (cp_dir / "tokenizer.json").touch()

        # 4. Write valid training_metadata.json
        meta = TrainingMetadata(
            model_name="distilbert-student-support",
            model_version="1.0.0",
            base_model="distilbert-base-uncased",
            trained_at="2026-10-22T00:00:00Z",
            git_commit_hash="synth_commit_12345",
            hyperparameters=TrainingHyperparameters(),
            dataset_manifest_id="synth_manifest_001",
            dataset_version="1.0.0",
            train_samples_count=1000,
            val_samples_count=200,
            test_samples_count=200,
            evaluation_summary=EvaluationReport(
                model_name="distilbert-student-support",
                model_version="1.0.0",
                dataset_version="1.0.0",
                evaluated_at="2026-10-22T00:00:00Z",
                test_samples_count=200,
                macro_f1=0.88,
                weighted_f1=0.89,
                accuracy=0.89,
                per_class={
                    "NO": ClassMetrics(precision=0.88, recall=0.86, f1_score=0.87, support=60),
                    "YES": ClassMetrics(precision=0.92, recall=0.90, f1_score=0.91, support=90),
                    "REVIEW": ClassMetrics(precision=0.85, recall=0.89, f1_score=0.87, support=50),
                },
                confusion_matrix=[
                    [52, 3, 5],
                    [3, 81, 6],
                    [4, 2, 44],
                ],
                false_yes_count=5,
                false_no_count=7,
                review_precision=0.85,
            ),
        )
        (cp_dir / "training_metadata.json").write_text(meta.model_dump_json(indent=2))

        # Validate checkpoint directory
        is_valid, errors = validate_checkpoint_directory(cp_dir, require_metadata=True)
        assert is_valid is True
        assert errors == []


class TestSafetyGuardrailSeparation:
    """Test suite ensuring learned model predictions never bypass safety guardrails."""

    def test_classifier_yes_prediction_overridden_on_conflict(self) -> None:
        """Verify that when a model predicts YES, but ticket has explicit resolution,
        the decision engine flags conflict, forces REVIEW, and suppresses outbound calls.
        """
        from unittest.mock import MagicMock
        from ml_service.predictor import StudentSupportPredictor
        from ml_service.schemas import FollowUpRequired, PriorityCategory, TicketInput

        # Create a mock classifier predicting YES with 0.99 confidence
        mock_classifier = MagicMock()
        mock_classifier.is_loaded = True
        mock_classifier.is_fine_tuned = True
        mock_classifier.status = "READY"
        mock_classifier.status_reason = "Mock fine-tuned classifier"
        mock_classifier.classify_text.return_value = {
            "status": "READY",
            "label": "YES",
            "score": 0.99,
            "probabilities": {"NO": 0.005, "YES": 0.99, "REVIEW": 0.005},
        }

        pipeline = StudentSupportPredictor.create_default_pipeline(classifier=mock_classifier)

        # Ticket with closed status and explicit resolution in text
        ticket = TicketInput(
            ticket_id="TICK-CONFLICT-01",
            issue_description="The issue has been resolved successfully, everything is sorted now.",
            status="closed",
        )
        prediction = pipeline.predict(ticket)

        # Guardrail invariant: conflict between model YES and text resolution forces REVIEW!
        assert prediction.follow_up_required == FollowUpRequired.REVIEW
        assert prediction.priority == PriorityCategory.NOT_APPLICABLE
        assert prediction.priority_score is None
        assert "call" not in prediction.recommended_action.lower()

    def test_classifier_yes_prediction_on_empty_text_routed_to_review(self) -> None:
        """Verify empty text bypasses classifier and cannot receive a call or score."""
        from unittest.mock import MagicMock
        from ml_service.predictor import StudentSupportPredictor
        from ml_service.schemas import FollowUpRequired, ModelStatus, PriorityCategory, TicketInput

        mock_classifier = MagicMock()
        mock_classifier.is_loaded = True
        mock_classifier.is_fine_tuned = True
        mock_classifier.classify_text.return_value = {
            "status": "READY",
            "label": "YES",
            "score": 0.99,
        }

        pipeline = StudentSupportPredictor.create_default_pipeline(classifier=mock_classifier)
        prediction = pipeline.predict(TicketInput(ticket_id="TICK-EMPTY-01", issue_description="   "))

        assert prediction.follow_up_required == FollowUpRequired.REVIEW
        assert prediction.priority == PriorityCategory.NOT_APPLICABLE
        assert prediction.priority_score is None
        assert prediction.model_status == ModelStatus.EMPTY_DESCRIPTION_REVIEW.value

