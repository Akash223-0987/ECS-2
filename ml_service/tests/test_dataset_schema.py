"""Tests for Dataset Schema, Validation, and Split Management.

All test fixtures use synthetic, clearly labelled mock data.
NO real student records or personal data are used.
"""

from datetime import datetime
from pathlib import Path
import pytest

from ml_service.dataset_schema import (
    AdjudicationStatus,
    DatasetManifest,
    LabelProvenance,
    SplitType,
    TrainingExample,
    detect_potential_pii,
    sanitize_text_for_training,
)
from ml_service.schemas import FollowUpRequired


# ==============================================================================
# TrainingExample Unit Tests
# ==============================================================================

class TestTrainingExampleSchema:
    """Test suite for TrainingExample schema validation and safety rules."""

    def test_valid_training_example_creation(self) -> None:
        """Verify valid training example instantiation with all required fields."""
        example = TrainingExample(
            example_id="SYNTH-EX-001",
            text="Unable to download course syllabus from portal due to 500 internal server error.",
            label=FollowUpRequired.YES,
            label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            adjudication_status=AdjudicationStatus.APPROVED,
            notes="Technical issue requiring web admin check.",
            group_id="synth_case_101",
            split=SplitType.TRAIN,
            created_at=datetime.utcnow(),
            metadata={"reviewer_id": "rev_04", "source_ticket_id": "TICK-MOCK-001"},
        )
        assert example.example_id == "SYNTH-EX-001"
        assert example.label == "YES"
        assert example.label_provenance == "HUMAN_SINGLE_REVIEW"
        assert example.adjudication_status == "APPROVED"
        assert example.split == "TRAIN"
        assert example.notes == "Technical issue requiring web admin check."
        assert "500 internal server error" in example.text

    def test_empty_text_rejected(self) -> None:
        """Verify empty string or whitespace text is rejected."""
        with pytest.raises(ValueError, match="must not be empty or whitespace"):
            TrainingExample(
                example_id="SYNTH-EX-002",
                text="   ",
                label=FollowUpRequired.YES,
                label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            )

    def test_empty_example_id_rejected(self) -> None:
        """Verify empty example ID is rejected."""
        with pytest.raises(ValueError, match="example_id must not be empty"):
            TrainingExample(
                example_id="",
                text="Valid issue description without PII.",
                label=FollowUpRequired.NO,
                label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            )

    def test_pii_detection_phone_number_rejected(self) -> None:
        """Verify text containing raw phone numbers is rejected by validator."""
        with pytest.raises(ValueError, match="unredacted PII"):
            TrainingExample(
                example_id="SYNTH-EX-003",
                text="Please call me back at +91 9876543210 regarding my registration.",
                label=FollowUpRequired.YES,
                label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            )

    def test_pii_detection_email_rejected(self) -> None:
        """Verify text containing raw email addresses is rejected by validator."""
        with pytest.raises(ValueError, match="unredacted PII"):
            TrainingExample(
                example_id="SYNTH-EX-004",
                text="Send fee receipt to student.mock@example.edu as soon as possible.",
                label=FollowUpRequired.YES,
                label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            )

    def test_pii_detection_application_number_rejected(self) -> None:
        """Verify text containing raw application/registration numbers is rejected."""
        with pytest.raises(ValueError, match="unredacted PII"):
            TrainingExample(
                example_id="SYNTH-EX-005",
                text="Check status for application 2026BTECH12345 in university database.",
                label=FollowUpRequired.YES,
                label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            )

    def test_notes_leakage_into_text_rejected(self) -> None:
        """Verify that reviewer notes accidentally copied into model text are rejected."""
        notes_text = "INTERNAL_NOTE: Student called 3 times today."
        with pytest.raises(ValueError, match="Reviewer notes must not be leaked into the model input text"):
            TrainingExample(
                example_id="SYNTH-EX-006",
                text=f"Portal error occurred. {notes_text}",
                label=FollowUpRequired.REVIEW,
                label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
                notes=notes_text,
            )

    def test_invalid_label_provenance_rejected(self) -> None:
        """Verify that non-human or unapproved provenance strings are rejected."""
        with pytest.raises(ValueError, match="HUMAN_SINGLE_REVIEW"):
            TrainingExample.model_validate({
                "example_id": "SYNTH-EX-007",
                "text": "Valid non-pii text describing payment error.",
                "label": "YES",
                "label_provenance": "AUTO_HEURISTIC_RULE_ENGINE",  # Disallowed provenance!
            })


# ==============================================================================
# PII Sanitization Helper Unit Tests
# ==============================================================================

class TestPIISanitization:
    """Test suite for PII detection and text scrubbing utilities."""

    def test_detect_potential_pii_clean_text(self) -> None:
        findings = detect_potential_pii("Unable to register for winter semester courses due to prerequisite error.")
        assert findings == []

    def test_detect_potential_pii_with_identifiers(self) -> None:
        text = "Contact 9876543210 or user@test.com regarding 2024BTECH99999."
        findings = detect_potential_pii(text)
        assert "PHONE_NUMBER" in findings
        assert "EMAIL_ADDRESS" in findings
        assert "APPLICATION_OR_REG_NUMBER" in findings

    def test_sanitize_text_replaces_all_identifiers(self) -> None:
        raw = "My name is John Doe, phone 9876543210, email john@test.com, app APP123456."
        sanitized = sanitize_text_for_training(
            raw,
            student_name="John Doe",
            phone_number="9876543210",
            application_number="APP123456",
        )
        assert "[STUDENT_NAME]" in sanitized
        assert "[PHONE_NUMBER]" in sanitized
        assert "[EMAIL]" in sanitized
        assert "[APPLICATION_NUMBER]" in sanitized
        assert "9876543210" not in sanitized
        assert "john@test.com" not in sanitized
        assert "APP123456" not in sanitized


# ==============================================================================
# DatasetManifest Unit Tests & Leakage Prevention
# ==============================================================================

class TestDatasetManifest:
    """Test suite for DatasetManifest and split integrity validation."""

    @pytest.fixture
    def mock_manifest(self) -> DatasetManifest:
        """Create a synthetic dataset manifest for testing."""
        ex1 = TrainingExample(
            example_id="EX-01",
            text="Tuition fee was deducted twice from bank account.",
            label=FollowUpRequired.YES,
            label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            group_id="case_group_A",
            split=SplitType.TRAIN,
        )
        ex2 = TrainingExample(
            example_id="EX-02",
            text="Fee issue is resolved now, downloaded receipt successfully.",
            label=FollowUpRequired.NO,
            label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            group_id="case_group_B",
            split=SplitType.VALIDATION,
        )
        ex3 = TrainingExample(
            example_id="EX-03",
            text="hostel inquiry",
            label=FollowUpRequired.REVIEW,
            label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            group_id="case_group_C",
            split=SplitType.TEST,
        )
        return DatasetManifest(
            dataset_id="synth_manifest_v1",
            name="Synthetic Test Dataset",
            description="Synthetic dataset fixtures for unit test execution.",
            created_at=datetime.utcnow(),
            examples=[ex1, ex2, ex3],
        )

    def test_get_split_filtering(self, mock_manifest: DatasetManifest) -> None:
        train_set = mock_manifest.get_split(SplitType.TRAIN)
        assert len(train_set) == 1
        assert train_set[0].example_id == "EX-01"

        val_set = mock_manifest.get_split(SplitType.VALIDATION)
        assert len(val_set) == 1
        assert val_set[0].example_id == "EX-02"

        test_set = mock_manifest.get_split(SplitType.TEST)
        assert len(test_set) == 1
        assert test_set[0].example_id == "EX-03"

    def test_get_label_distribution(self, mock_manifest: DatasetManifest) -> None:
        dist = mock_manifest.get_label_distribution()
        assert dist["YES"] == 1
        assert dist["NO"] == 1
        assert dist["REVIEW"] == 1

    def test_validate_split_leakage_clean(self, mock_manifest: DatasetManifest) -> None:
        """Verify clean splits return no leakage violations."""
        is_clean, violations = mock_manifest.validate_split_leakage()
        assert is_clean is True
        assert len(violations) == 0

    def test_validate_split_leakage_detected(self) -> None:
        """Verify that sharing a group_id across TRAIN and TEST is detected."""
        ex_train = TrainingExample(
            example_id="EX-T1",
            text="Initial inquiry regarding hostel booking.",
            label=FollowUpRequired.YES,
            label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            group_id="shared_case_999",  # Same group_id
            split=SplitType.TRAIN,
        )
        ex_test = TrainingExample(
            example_id="EX-T2",
            text="Follow-up inquiry regarding same hostel booking.",
            label=FollowUpRequired.YES,
            label_provenance=LabelProvenance.HUMAN_SINGLE_REVIEW,
            group_id="shared_case_999",  # Same group_id in TEST split!
            split=SplitType.TEST,
        )
        manifest = DatasetManifest(
            dataset_id="leakage_manifest",
            name="Leaked Dataset",
            description="Manifest with intentional group leakage.",
            created_at=datetime.utcnow(),
            examples=[ex_train, ex_test],
        )
        is_clean, violations = manifest.validate_split_leakage()
        assert is_clean is False
        assert len(violations) >= 1
        assert any("shared_case_999" in v for v in violations)

    def test_to_hf_dict_mapping(self, mock_manifest: DatasetManifest) -> None:
        """Verify formatting into Hugging Face datasets dictionary structure."""
        hf_dict = mock_manifest.to_hf_dict()
        assert "text" in hf_dict
        assert "label" in hf_dict
        assert "label_id" in hf_dict
        assert len(hf_dict["text"]) == 3
        # In canonical mapping: NO=0, YES=1, REVIEW=2
        assert hf_dict["label"] == ["YES", "NO", "REVIEW"]
        assert hf_dict["label_id"] == [1, 0, 2]

    def test_jsonl_roundtrip(self, mock_manifest: DatasetManifest, tmp_path: Path) -> None:
        """Verify export to and loading from JSONL file."""
        jsonl_file = tmp_path / "test_dataset.jsonl"
        count = mock_manifest.to_jsonl(jsonl_file)
        assert count == 3
        assert jsonl_file.exists()

        loaded_manifest = DatasetManifest.from_jsonl(
            filepath=jsonl_file,
            dataset_id="reloaded_manifest",
            name="Reloaded Manifest",
            description="Loaded from JSONL file.",
        )
        assert len(loaded_manifest.examples) == 3
        assert loaded_manifest.examples[0].example_id == "EX-01"
        assert loaded_manifest.examples[1].example_id == "EX-02"
        assert loaded_manifest.examples[2].example_id == "EX-03"
