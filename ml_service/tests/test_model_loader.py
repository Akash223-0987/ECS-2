"""Unit tests for HuggingFaceModelLoader and safe model integration.

Uses dependency injection and mocks to test:
- Lazy importing and missing dependency handling.
- Prevention of network calls / downloads.
- Distinction between generic pretrained models and task-specific fine-tuned models.
- Rejection of generic pretrained labels (e.g. LABEL_0, POSITIVE).
- Graceful REVIEW fallback when model or dependencies are unready.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from ml_service.config import MLServiceConfig
from ml_service.model_loader import HuggingFaceModelLoader, check_dependencies
from ml_service.predictor import StudentSupportPredictor
from ml_service.schemas import (
    FollowUpRequired,
    ModelStatus,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
)


class MockModelOutput:
    """Mock logits output from a transformer sequence classification model."""

    def __init__(self, logits: Any) -> None:
        self.logits = logits


class MockTensor:
    """Mock PyTorch tensor supporting basic reduction and indexing."""

    def __init__(self, data: list) -> None:
        self.data = data

    def to(self, device: str) -> "MockTensor":
        return self

    def squeeze(self) -> "MockTensor":
        return self

    def tolist(self) -> list:
        return self.data

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class MockTorchModule:
    """Mock torch namespace for isolated unit testing without requiring real torch."""

    @staticmethod
    def softmax(tensor: Any, dim: int = -1) -> MockTensor:
        # Simple simulated softmax on data
        raw = tensor.data if hasattr(tensor, "data") else tensor
        total = sum(raw) if sum(raw) != 0 else 1.0
        return MockTensor([x / total for x in raw])

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False


class TestHuggingFaceModelLoader:
    """Test suite for HuggingFaceModelLoader without network access or heavy weights."""

    def test_default_initialization(self):
        """Verify loader initializes without loading weights, accessing network, or requiring GPU."""
        loader = HuggingFaceModelLoader()
        assert loader.device == "cpu"
        assert loader.local_files_only is True
        assert loader.is_loaded is False
        assert loader.is_fine_tuned is False
        assert loader.status == ModelStatus.NOT_INITIALIZED
        assert loader.model is None
        assert loader.tokenizer is None

    def test_custom_device_and_path(self):
        """Verify configurable model paths and explicit device selection."""
        loader = HuggingFaceModelLoader(
            model_name_or_path="./models/custom_distilbert",
            device="cuda:0",
            local_files_only=True,
        )
        assert loader.model_name_or_path == "./models/custom_distilbert"
        assert loader.device == "cuda:0"
        assert loader.local_files_only is True
        assert loader.is_loaded is False

    def test_check_dependencies_missing_simulation(self):
        """Verify check_dependencies returns clear message when packages are missing."""
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            with patch("builtins.__import__", side_effect=ImportError("Mocked missing module")):
                available, msg = check_dependencies()
                assert available is False
                assert msg is not None
                assert "Missing optional ML dependencies" in msg

    def test_load_model_when_dependencies_unavailable(self):
        """Verify load_model fails gracefully with DEPENDENCIES_UNAVAILABLE status."""
        loader = HuggingFaceModelLoader()
        with patch("ml_service.model_loader.check_dependencies", return_value=(False, "Mock missing dependencies")):
            success = loader.load_model()
            assert success is False
            assert loader.status == ModelStatus.DEPENDENCIES_UNAVAILABLE
            assert loader.is_loaded is False
            assert "Mock missing dependencies" in loader.status_reason

    def test_load_model_when_local_path_not_found(self):
        """Verify load_model fails gracefully with MODEL_NOT_FOUND when local directory does not exist."""
        loader = HuggingFaceModelLoader(model_name_or_path="./nonexistent_model_dir_xyz")
        with patch("ml_service.model_loader.check_dependencies", return_value=(True, None)):
            success = loader.load_model()
            assert success is False
            assert loader.status == ModelStatus.MODEL_NOT_FOUND
            assert loader.is_loaded is False
            assert "Local model directory not found" in loader.status_reason

    def test_classify_text_when_uninitialized(self):
        """Verify classify_text returns uninitialized status if called before loading."""
        loader = HuggingFaceModelLoader()
        res = loader.classify_text("Sample student inquiry")
        assert res["status"] == ModelStatus.NOT_INITIALIZED
        assert res["label"] is None
        assert res["score"] is None

    def test_generic_pretrained_model_rejection_label_numeric(self):
        """Task 6 & 7: Verify models with generic 'LABEL_0', 'LABEL_1' are identified as generic and rejected."""
        loader = HuggingFaceModelLoader()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        generic_id2label = {0: "LABEL_0", 1: "LABEL_1"}
        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label=generic_id2label,
        )

        assert loader.is_loaded is True
        assert loader.is_fine_tuned is False
        assert loader.status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
        assert "NOT fine-tuned" in loader.status_reason

        # classify_text must refuse to return generic labels
        res = loader.classify_text("I need urgent help with hostel fee refund")
        assert res["status"] == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
        assert res["label"] is None
        assert res["score"] is None

    def test_generic_sentiment_labels_rejection(self):
        """Task 7: Verify models with generic POSITIVE/NEGATIVE sentiment labels are not mapped to decisions."""
        loader = HuggingFaceModelLoader()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        sentiment_id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label=sentiment_id2label,
        )

        assert loader.is_loaded is True
        assert loader.is_fine_tuned is False
        assert loader.status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED

        res = loader.classify_text("Student query")
        assert res["status"] == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
        assert res["label"] is None

    def test_task_specific_fine_tuned_model_injection(self):
        """Task 6: Verify model with task-specific labels (YES, NO, REVIEW) is recognized as fine-tuned."""
        loader = HuggingFaceModelLoader()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        task_id2label = {0: "NO", 1: "YES", 2: "REVIEW"}
        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label=task_id2label,
        )

        assert loader.is_loaded is True
        assert loader.is_fine_tuned is True
        assert loader.status == ModelStatus.READY
        assert "Task-specific fine-tuned model ready" in loader.status_reason

    def test_classify_text_with_injected_fine_tuned_mock(self):
        """Verify successful classification using mocked model and tokenizer outputs."""
        loader = HuggingFaceModelLoader()
        mock_tokenizer = MagicMock(return_value={"input_ids": MockTensor([101, 2054, 102])})
        # Simulate logits highest for index 1 ('YES')
        mock_model = MagicMock(return_value=MockModelOutput(logits=MockTensor([1.0, 9.0, 0.5])))

        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label={0: "NO", 1: "YES", 2: "REVIEW"},
        )

        with patch.dict("sys.modules", {"torch": MockTorchModule()}):
            res = loader.classify_text("Urgent issue requiring advisor assistance")
            assert res["status"] == ModelStatus.READY
            assert res["label"] == "YES"
            assert res["score"] > 0.8
            assert "YES" in res["probabilities"]


class TestPredictorWithModelLoader:
    """Test suite verifying StudentSupportPredictor integration with model loader states."""

    def test_predictor_with_uninitialized_loader_returns_review(self):
        """Task 8: Predictor returns REVIEW and explicit NOT_INITIALIZED status when loader is not loaded."""
        loader = HuggingFaceModelLoader()
        predictor = StudentSupportPredictor(model_loader=loader)

        ticket = TicketInput(
            ticket_id="T101",
            issue_description="Inquiry about scholarship eligibility.",
        )
        pred: TicketPrediction = predictor.predict(ticket)

        assert pred.ticket_id == "T101"
        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.model_status == ModelStatus.NOT_INITIALIZED.value
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert "not initialized" in pred.follow_up_reason.lower()

    def test_predictor_with_dependencies_unavailable_returns_review(self):
        """Task 8: Predictor returns REVIEW and DEPENDENCIES_UNAVAILABLE when dependencies missing."""
        loader = HuggingFaceModelLoader()
        loader.status = ModelStatus.DEPENDENCIES_UNAVAILABLE
        loader.status_reason = "Missing optional ML packages: torch, transformers."
        predictor = StudentSupportPredictor(model_loader=loader)

        ticket = TicketInput(
            ticket_id="T102",
            issue_description="Question regarding hostel allotment.",
        )
        pred = predictor.predict(ticket)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.model_status == ModelStatus.DEPENDENCIES_UNAVAILABLE.value
        assert pred.priority_score is None
        assert "dependencies missing" in pred.follow_up_reason.lower()

    def test_predictor_with_missing_model_checkpoint_returns_review(self):
        """Task 8: Predictor returns REVIEW and MODEL_NOT_FOUND when model path does not exist."""
        loader = HuggingFaceModelLoader(model_name_or_path="./missing_weights")
        loader.status = ModelStatus.MODEL_NOT_FOUND
        loader.status_reason = "Local model directory not found: './missing_weights'"
        predictor = StudentSupportPredictor(model_loader=loader)

        ticket = TicketInput(
            ticket_id="T103",
            issue_description="Library fine dispute.",
        )
        pred = predictor.predict(ticket)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.model_status == ModelStatus.MODEL_NOT_FOUND.value
        assert pred.priority_score is None

    def test_predictor_with_generic_model_returns_review_and_no_generic_interpretation(self):
        """Task 7 & 8: Predictor with generic model NEVER interprets generic labels as YES/NO/REVIEW."""
        loader = HuggingFaceModelLoader()
        loader.inject_model_and_tokenizer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            id2label={0: "LABEL_0", 1: "LABEL_1"},
        )
        predictor = StudentSupportPredictor(model_loader=loader)

        ticket = TicketInput(
            ticket_id="T104",
            issue_description="Need extension on course registration.",
        )
        pred = predictor.predict(ticket)

        # Must NOT interpret LABEL_1 as YES or LABEL_0 as NO
        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.model_status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED.value
        assert pred.priority_score is None
        assert "generic pretrained model detected" in pred.follow_up_reason.lower()

    def test_predictor_with_ready_fine_tuned_model_produces_prediction(self):
        """Verify fine-tuned model produces expected decision while keeping priority_score null."""
        loader = HuggingFaceModelLoader()
        mock_tokenizer = MagicMock(return_value={"input_ids": MockTensor([1, 2, 3])})
        mock_model = MagicMock(return_value=MockModelOutput(logits=MockTensor([0.1, 0.9, 0.0])))

        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label={0: "NO", 1: "YES", 2: "REVIEW"},
        )
        predictor = StudentSupportPredictor(model_loader=loader)

        ticket = TicketInput(
            ticket_id="T105",
            issue_description="Medical emergency preventing me from attending the mid-term exam.",
        )
        with patch.dict("sys.modules", {"torch": MockTorchModule()}):
            pred = predictor.predict(ticket)
            assert pred.ticket_id == "T105"
            assert pred.follow_up_required == FollowUpRequired.YES
            assert pred.model_status == ModelStatus.READY.value
            # Constraint: priority_score remains null (scoring rules not implemented yet)
            assert pred.priority_score is None
            assert pred.priority == PriorityCategory.NOT_APPLICABLE
            assert "Classified by fine-tuned DistilBERT model" in pred.follow_up_reason
