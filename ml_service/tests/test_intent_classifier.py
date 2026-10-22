"""Tests for Auxiliary Support Intent Classifier and Dataset Preparation.

Validates:
1. Dataset preparation manifest, 4-class filtering, and zero-leakage cross-split partitioning.
2. AuxiliaryIntentClassifier loading, missing artifact safety, and baseline predictions.
3. Integration with StudentSupportPredictor and TicketPrediction.
4. Absolute isolation: intent predictions CANNOT alter follow_up_required,
   priority_score, priority tier, or outbound call recommendations.
"""

import json
from pathlib import Path
import pytest

from ml_service.intent_classifier import (
    AuxiliaryIntentClassifier,
    IntentModelStatus,
    IntentPrediction,
)
from ml_service.predictor import StudentSupportPredictor
from ml_service.schemas import (
    FollowUpRequired,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "auxiliary_intent"
BASELINE_DIR = Path(__file__).resolve().parent.parent / "models" / "auxiliary-intent-baseline"
DISTILBERT_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert-intent-classifier"


class TestIntentDatasetPreparation:
    """Test suite verifying dataset preparation, manifest integrity, and split leakage."""

    def test_dataset_manifest_exists_and_valid(self) -> None:
        manifest_file = DATA_DIR / "intent_dataset_manifest.json"
        assert manifest_file.exists(), "Manifest file not found!"

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["dataset_source"] == "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        assert manifest["num_classes"] == 4
        assert manifest["selected_intents"] == [
            "payment_issue",
            "get_refund",
            "contact_customer_service",
            "recover_password",
        ]
        assert manifest["leakage_check_passed"] is True
        assert manifest["statistics"]["conflicting_duplicate_instructions"] == 0
        assert manifest["split_counts"]["total"] == 3879
        assert manifest["split_counts"]["train"] == 2715
        assert manifest["split_counts"]["val"] == 582
        assert manifest["split_counts"]["test"] == 582

    def test_splits_files_exist_and_non_empty(self) -> None:
        for split in ["train", "val", "test"]:
            file_path = DATA_DIR / f"{split}.jsonl"
            assert file_path.exists(), f"Split file {split}.jsonl missing"
            lines = [l for l in file_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) > 0

    def test_zero_leakage_between_train_val_test(self) -> None:
        """Verify that normalized instruction texts NEVER appear across multiple splits."""
        def get_normalized_texts(split_name: str) -> set:
            texts = set()
            file_path = DATA_DIR / f"{split_name}.jsonl"
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        texts.add(item["normalized_text"])
            return texts

        train_texts = get_normalized_texts("train")
        val_texts = get_normalized_texts("val")
        test_texts = get_normalized_texts("test")

        assert len(train_texts.intersection(val_texts)) == 0, "Leakage between train and val!"
        assert len(train_texts.intersection(test_texts)) == 0, "Leakage between train and test!"
        assert len(val_texts.intersection(test_texts)) == 0, "Leakage between val and test!"

    def test_target_intents_adherence(self) -> None:
        """Verify all examples belong strictly to the 4 approved intent classes."""
        allowed = {"payment_issue", "get_refund", "contact_customer_service", "recover_password"}
        for split in ["train", "val", "test"]:
            file_path = DATA_DIR / f"{split}.jsonl"
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        assert item["intent"] in allowed, f"Unexpected intent: {item['intent']}"
                        assert item["label_id"] in {0, 1, 2, 3}


class TestAuxiliaryIntentClassifierInterface:
    """Test suite for AuxiliaryIntentClassifier lifecycle and baseline inference."""

    def test_uninitialized_classifier_status(self) -> None:
        classifier = AuxiliaryIntentClassifier()
        assert classifier.status == IntentModelStatus.NOT_INITIALIZED
        assert classifier.is_loaded is False

        pred = classifier.predict_intent("I have an issue with fee payment")
        assert pred.status == IntentModelStatus.NOT_INITIALIZED
        assert pred.predicted_intent is None
        assert pred.intent_confidence is None

    def test_missing_model_directory_status(self, tmp_path: Path) -> None:
        classifier = AuxiliaryIntentClassifier(model_dir=tmp_path / "nonexistent")
        loaded = classifier.load_model()
        assert loaded is False
        assert classifier.status == IntentModelStatus.MODEL_NOT_FOUND
        assert classifier.is_loaded is False

        pred = classifier.predict_intent("Any inquiry")
        assert pred.status == IntentModelStatus.MODEL_NOT_FOUND
        assert pred.predicted_intent is None

    def test_load_baseline_model(self) -> None:
        classifier = AuxiliaryIntentClassifier(model_dir=BASELINE_DIR)
        loaded = classifier.load_model()
        assert loaded is True
        assert classifier.status == IntentModelStatus.READY
        assert classifier.is_loaded is True
        assert classifier.model_type == "baseline_tfidf_logistic_regression"

    def test_baseline_predictions_across_intents(self) -> None:
        classifier = AuxiliaryIntentClassifier(model_dir=BASELINE_DIR)
        classifier.load_model()

        # 1. Payment issue query
        p1 = classifier.predict_intent("There is an error with my payment transaction and fee was deducted")
        assert p1.status == IntentModelStatus.READY
        assert p1.predicted_intent == "payment_issue"
        assert p1.intent_confidence is not None and p1.intent_confidence > 0.5
        assert p1.model_type == "baseline_tfidf_logistic_regression"

        # 2. Refund query
        p2 = classifier.predict_intent("I want to get a refund of my deposit money")
        assert p2.status == IntentModelStatus.READY
        assert p2.predicted_intent == "get_refund"
        assert p2.intent_confidence is not None and p2.intent_confidence > 0.5

        # 3. Contact customer service query
        p3 = classifier.predict_intent("I want to speak with customer support agent directly")
        assert p3.status == IntentModelStatus.READY
        assert p3.predicted_intent == "contact_customer_service"
        assert p3.intent_confidence is not None and p3.intent_confidence > 0.5

        # 4. Recover password query
        p4 = classifier.predict_intent("Forgot my portal password and cannot reset my user PIN")
        assert p4.status == IntentModelStatus.READY
        assert p4.predicted_intent == "recover_password"
        assert p4.intent_confidence is not None and p4.intent_confidence > 0.5

    def test_empty_and_whitespace_text_handling(self) -> None:
        classifier = AuxiliaryIntentClassifier(model_dir=BASELINE_DIR)
        classifier.load_model()

        pred_empty = classifier.predict_intent("")
        assert pred_empty.predicted_intent is None
        assert pred_empty.intent_confidence is None

        pred_ws = classifier.predict_intent("   ")
        assert pred_ws.predicted_intent is None
        assert pred_ws.intent_confidence is None

    def test_distilbert_model_loading_and_inference_if_present(self) -> None:
        if not (DISTILBERT_DIR / "config.json").exists():
            pytest.skip("DistilBERT checkpoint not present.")
        classifier = AuxiliaryIntentClassifier(model_dir=DISTILBERT_DIR)
        loaded = classifier.load_model()
        assert loaded is True
        assert classifier.status == IntentModelStatus.READY
        assert classifier.model_type == "distilbert"

        pred = classifier.predict_intent("I want to get a refund for my tuition fee")
        assert pred.status == IntentModelStatus.READY
        assert pred.predicted_intent in classifier.ALLOWED_INTENTS
        assert pred.intent_confidence is not None
        assert pred.model_type == "distilbert"


class TestPredictorIntentIntegrationAndSafetyGuardrails:
    """Test suite verifying intent integration and strict guardrail non-interference."""

    def test_predictor_without_intent_classifier_has_none_fields(self) -> None:
        pipeline = StudentSupportPredictor.create_default_pipeline()
        prediction = pipeline.predict(TicketInput(ticket_id="TICK-001", issue_description="fees payment issue"))
        assert prediction.predicted_intent is None
        assert prediction.intent_confidence is None

    def test_predictor_with_loaded_intent_classifier_enriches_prediction(self) -> None:
        intent_clf = AuxiliaryIntentClassifier(model_dir=BASELINE_DIR)
        intent_clf.load_model()

        pipeline = StudentSupportPredictor.create_default_pipeline(intent_classifier=intent_clf)
        prediction = pipeline.predict(TicketInput(
            ticket_id="TICK-INTENT-01",
            issue_description="I want to speak to customer support about a problem.",
        ))

        assert prediction.predicted_intent == "contact_customer_service"
        assert prediction.intent_confidence is not None and prediction.intent_confidence > 0.5

    def test_intent_prediction_never_alters_follow_up_decision_on_resolved_ticket(self) -> None:
        """Critical Invariant: An explicitly resolved ticket with closed status MUST
        receive NO follow-up and NOT_APPLICABLE priority, even if intent is payment_issue.
        """
        intent_clf = AuxiliaryIntentClassifier(model_dir=BASELINE_DIR)
        intent_clf.load_model()

        pipeline = StudentSupportPredictor.create_default_pipeline(intent_classifier=intent_clf)
        ticket = TicketInput(
            ticket_id="TICK-RESOLVED-01",
            issue_description="The payment problem has been resolved successfully, thank you.",
            status="closed",
        )
        prediction = pipeline.predict(ticket)

        # Safety rule: confirmed resolution with closed status yields NO follow-up
        assert prediction.follow_up_required == FollowUpRequired.NO
        assert prediction.priority == PriorityCategory.NOT_APPLICABLE
        assert prediction.priority_score is None
        assert "call" not in prediction.recommended_action.lower()
        # Intent is populated as auxiliary info only
        assert prediction.predicted_intent == "payment_issue"

    def test_intent_prediction_never_fabricates_priority_score(self) -> None:
        """Critical Invariant: An inquiry with insufficient context returns null priority_score,
        regardless of whether intent classification succeeded.
        """
        intent_clf = AuxiliaryIntentClassifier(model_dir=BASELINE_DIR)
        intent_clf.load_model()

        pipeline = StudentSupportPredictor.create_default_pipeline(intent_classifier=intent_clf)
        ticket = TicketInput(
            ticket_id="TICK-VAGUE-01",
            issue_description="payment",  # Ambiguous single keyword
            status="open",
        )
        prediction = pipeline.predict(ticket)

        assert prediction.follow_up_required == FollowUpRequired.REVIEW
        assert prediction.priority == PriorityCategory.NOT_APPLICABLE
        assert prediction.priority_score is None
