"""Unit Tests for Simplified Priority Predictor and Priority Model Loader.

Tests:
1. Simplified schemas: PriorityInferenceInput and PriorityInferenceOutput.
2. Inference interface: PriorityPredictor.predict_priority() with string, dict, and schema.
3. PriorityModelLoader label validation and guardrails:
   - Rejection of follow-up triage labels {"YES", "NO", "REVIEW"}.
   - Rejection of auxiliary intent labels {"payment_issue", ...}.
   - Rejection of generic labels {"LABEL_0", "LABEL_1"}.
   - Acceptance of genuine priority labels {"LOW", "MEDIUM", "HIGH"}.
4. Fallback heuristic priority prediction when GPU-trained model is pending.
5. Invariance and error handling on empty or corrupted transcripts.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from ml_service.config import MLServiceConfig
from ml_service.predictor import PriorityPredictor
from ml_service.priority_contract import (
    ALLOWED_PRIORITY_LABELS,
    PRIORITY_ID2LABEL,
    PRIORITY_LABEL2ID,
    normalize_priority_label,
    validate_priority_checkpoint_directory,
)
from ml_service.priority_model import PriorityModelLoader
from ml_service.schemas import (
    ModelStatus,
    PriorityInferenceInput,
    PriorityInferenceOutput,
    PriorityLevel,
)


class TestPrioritySchemas:
    """Validate simplified priority schemas."""

    def test_priority_input_with_issue_transcription(self):
        inp = PriorityInferenceInput(issue_transcription="Cannot access portal", ticket_id="T1")
        assert inp.issue_transcription == "Cannot access portal"
        assert inp.ticket_id == "T1"

    def test_priority_input_with_alias_issue_description(self):
        inp = PriorityInferenceInput(issue_description="Fee was deducted twice")
        assert inp.issue_transcription == "Fee was deducted twice"

    def test_priority_input_coercion(self):
        inp = PriorityInferenceInput(issue_transcription=None)
        assert inp.issue_transcription == ""

        inp2 = PriorityInferenceInput(issue_transcription="  whitespace  ")
        assert inp2.issue_transcription == "whitespace"

    def test_priority_output_structure(self):
        out = PriorityInferenceOutput(
            priority_level="HIGH",
            confidence=0.95,
            model_status="READY",
            is_trained_model=True,
        )
        assert out.priority_level == "HIGH"
        assert out.confidence == 0.95
        assert out.is_trained_model is True


class TestPriorityModelLoader:
    """Validate model loading, mock injection, and label compatibility guardrails."""

    def test_model_not_found_sets_diagnostic_status(self, tmp_path):
        loader = PriorityModelLoader(model_dir=str(tmp_path / "nonexistent"))
        loaded = loader.load_model()
        assert loaded is False
        assert loader.status == ModelStatus.MODEL_NOT_FOUND
        assert loader.is_fine_tuned is False
        assert "No priority model checkpoint found" in loader.status_reason

    def test_rejects_follow_up_triage_labels(self):
        """Task guardrail: Hugging Face triage model {NO, YES, REVIEW} is incompatible."""
        loader = PriorityModelLoader()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Follow-up triage labels
        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label={0: "NO", 1: "YES", 2: "REVIEW"},
        )
        assert loader.is_fine_tuned is False
        assert loader.status == ModelStatus.INCOMPATIBLE_LABELS
        assert "follow-up triage labels" in loader.status_reason

    def test_rejects_auxiliary_intent_labels(self):
        """Task guardrail: Auxiliary intent model is incompatible with priority."""
        loader = PriorityModelLoader()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label={0: "payment_issue", 1: "get_refund", 2: "contact_customer_service"},
        )
        assert loader.is_fine_tuned is False
        assert loader.status == ModelStatus.INCOMPATIBLE_LABELS

    def test_rejects_generic_labels(self):
        """Task guardrail: Generic pretrained labels (LABEL_0, LABEL_1) are rejected."""
        loader = PriorityModelLoader()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label={0: "LABEL_0", 1: "LABEL_1"},
        )
        assert loader.is_fine_tuned is False
        assert loader.status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED

    def test_accepts_genuine_priority_labels(self):
        """Valid priority labels {LOW, MEDIUM, HIGH} are accepted."""
        loader = PriorityModelLoader()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        loader.inject_model_and_tokenizer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            id2label={0: "LOW", 1: "MEDIUM", 2: "HIGH"},
        )
        assert loader.is_fine_tuned is True
        assert loader.status == ModelStatus.READY

    def test_fallback_heuristic_detection(self):
        """When trained model is not yet loaded, heuristic fallback provides estimates."""
        loader = PriorityModelLoader()
        # High urgency inquiry
        res_high = loader.classify_priority("Tuition fee was deducted twice, today is the deadline to pay!")
        assert res_high["priority_level"] == "HIGH"
        assert res_high["is_trained_model"] is False

        # Medium urgency inquiry
        res_med = loader.classify_priority("Hostel room change application status verification")
        assert res_med["priority_level"] == "MEDIUM"
        assert res_med["is_trained_model"] is False

        # Routine inquiry
        res_low = loader.classify_priority("Could you share the academic calendar?")
        assert res_low["priority_level"] == "LOW"
        assert res_low["is_trained_model"] is False

    def test_empty_transcription_handling(self):
        loader = PriorityModelLoader()
        res_empty = loader.classify_priority("")
        assert res_empty["priority_level"] == "LOW"
        assert res_empty["status"] == ModelStatus.EMPTY_TRANSCRIPTION.value
        assert res_empty["is_trained_model"] is False

        res_none = loader.classify_priority(None)
        assert res_none["priority_level"] == "LOW"


class TestPriorityPredictor:
    """Validate PriorityPredictor inference interface."""

    def test_predict_priority_with_string(self):
        predictor = PriorityPredictor()
        output = predictor.predict_priority("Fee debited twice, deadline today!")
        assert isinstance(output, PriorityInferenceOutput)
        assert output.priority_level == "HIGH"
        assert output.is_trained_model is False

    def test_predict_priority_with_schema(self):
        predictor = PriorityPredictor()
        inp = PriorityInferenceInput(
            issue_transcription="General inquiry regarding hostel fee schedule",
            ticket_id="T101",
        )
        output = predictor.predict_priority(inp)
        assert isinstance(output, PriorityInferenceOutput)
        assert output.priority_level in ("LOW", "MEDIUM", "HIGH")

    def test_predict_priority_with_dict(self):
        predictor = PriorityPredictor()
        output = predictor.predict_priority({"issue_description": "Portal error when downloading hall ticket"})
        assert output.priority_level == "HIGH"

    def test_predict_batch(self):
        predictor = PriorityPredictor()
        batch = [
            "Emergency deadline today",
            "General campus inquiry",
        ]
        results = predictor.predict_batch(batch)
        assert len(results) == 2
        assert results[0].priority_level == "HIGH"
        assert results[1].priority_level == "LOW"

    def test_ticket_id_invariance_does_not_alter_prediction(self):
        """Changing ticket_id while keeping transcription constant produces identical predictions."""
        predictor = PriorityPredictor()
        transcription = "Fee was debited twice from my account and today is the deadline to pay."

        ticket_ids = ["TICK-001", "TICK-999", "John123", "", None]
        predictions = []

        for tid in ticket_ids:
            inp = PriorityInferenceInput(issue_transcription=transcription, ticket_id=tid)
            out = predictor.predict_priority(inp)
            predictions.append(out)

        first = predictions[0]
        assert first.priority_level == "HIGH"
        for other in predictions[1:]:
            assert other.priority_level == first.priority_level
            assert other.confidence == first.confidence
            assert other.model_status == first.model_status
            assert other.is_trained_model == first.is_trained_model



class TestPriorityContractHelpers:
    """Validate helper functions and checkpoint validator."""

    def test_normalize_priority_label(self):
        assert normalize_priority_label("high") == "HIGH"
        assert normalize_priority_label("HIGH") == "HIGH"
        assert normalize_priority_label("high", title_case=True) == "High"
        assert normalize_priority_label("medium", title_case=True) == "Medium"
        assert normalize_priority_label("critical") == "HIGH"
        assert normalize_priority_label("unknown") == "LOW"

    def test_checkpoint_validator(self, tmp_path):
        # Empty dir
        valid, errors, _ = validate_priority_checkpoint_directory(tmp_path)
        assert valid is False
        assert any("config.json" in e for e in errors)

        # Create valid mock checkpoint files
        import json
        config_data = {
            "id2label": {"0": "LOW", "1": "MEDIUM", "2": "HIGH"},
            "label2id": {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)
        (tmp_path / "model.safetensors").write_text("fake-weights")
        (tmp_path / "tokenizer.json").write_text("fake-tokenizer")

        valid, errors, meta = validate_priority_checkpoint_directory(tmp_path)
        assert valid is True
        assert len(errors) == 0
        assert meta["labels"] == ["HIGH", "LOW", "MEDIUM"] or set(meta["labels"]) == {"HIGH", "MEDIUM", "LOW"}
