"""Model Loader and Inference Engine for Priority Classification.

Provides safe, isolated model loading, label validation, and inference
for predicting `priority_level` from student issue transcription.

CRITICAL INVARIANTS:
1. Isolated Model Loading: The loader is decoupled from the Google Sheets watcher.
2. Label Guardrail: Explicitly verifies that the loaded model predicts priority labels
   {'HIGH', 'MEDIUM', 'LOW'}. Models with labels like {'YES', 'NO', 'REVIEW'} or
   customer query intent topics are marked INCOMPATIBLE_LABELS and rejected.
3. Fallback Test Double: If no trained model is available yet, provides a transparent
   fallback estimator with `is_trained_model=False` so the service can run end-to-end.
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .config import MLServiceConfig
from .priority_contract import (
    ALLOWED_PRIORITY_LABELS,
    DEFAULT_PRIORITY_MODEL_DIR,
    MINIMAL_PRIORITY_LABELS,
    PRIORITY_ID2LABEL,
    PRIORITY_LABEL2ID,
    normalize_priority_label,
)
from .schemas import ModelStatus, PriorityLevel

logger = logging.getLogger(__name__)


def check_torch_transformers_dependencies() -> Tuple[bool, Optional[str]]:
    """Verify if torch and transformers are available in the runtime environment."""
    missing: List[str] = []
    try:
        import torch  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        missing.append("torch")

    try:
        import transformers  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        missing.append("transformers")

    if missing:
        return (
            False,
            f"Missing optional ML dependencies: {', '.join(missing)}. "
            "Install with 'pip install torch transformers' to run fine-tuned models.",
        )
    return True, None


class PriorityModelLoader:
    """Loads, validates, and runs inference for priority classification models."""

    REQUIRED_LABELS: Set[str] = ALLOWED_PRIORITY_LABELS
    MINIMAL_LABELS: Set[str] = MINIMAL_PRIORITY_LABELS

    def __init__(
        self,
        config: Optional[MLServiceConfig] = None,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        self.config = config or MLServiceConfig()
        self.model_dir = (
            model_dir
            or os.getenv("PRIORITY_MODEL_PATH")
            or os.getenv("PRIORITY_MODEL_DIR")
            or DEFAULT_PRIORITY_MODEL_DIR
        )
        self.device = device or self.config.device or "cpu"
        self.local_files_only = local_files_only

        self.status: ModelStatus = ModelStatus.NOT_INITIALIZED
        self.status_reason: str = "Priority model loader initialized; weights not yet loaded."
        self.is_loaded: bool = False
        self.is_fine_tuned: bool = False

        self.model: Any = None
        self.tokenizer: Any = None
        self.id2label: Dict[int, str] = {}
        self.pipeline: Any = None  # For scikit-learn baseline pipelines

    def load_model(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> bool:
        """Attempt to load a fine-tuned priority model checkpoint.
        
        Supports Hugging Face Transformers directories or joblib pipelines.
        Validates target labels to guarantee priority compatibility.
        """
        target = model_path or self.model_dir
        if device:
            self.device = device

        if not target or not os.path.exists(target):
            self.status = ModelStatus.MODEL_NOT_FOUND
            self.status_reason = (
                f"No priority model checkpoint found at '{target}'. "
                "GPU fine-tuning pending; service operating in fallback mode."
            )
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

        target_path = Path(target)

        # 1. Check for classical joblib baseline pipeline
        joblib_file = target_path / "baseline_pipeline.joblib" if target_path.is_dir() else None
        if target_path.suffix == ".joblib" or (joblib_file and joblib_file.is_file()):
            return self._load_joblib_pipeline(joblib_file or target_path)

        # 2. Check for Hugging Face transformer checkpoint
        return self._load_huggingface_checkpoint(target_path)

    def _load_joblib_pipeline(self, pipeline_path: Path) -> bool:
        """Load and validate Scikit-Learn pipeline."""
        try:
            import joblib
            pipeline = joblib.load(pipeline_path)
            classes = list(getattr(pipeline, "classes_", []))
            labels = {str(c).strip().upper() for c in classes}

            if not (self.MINIMAL_LABELS.issubset(labels) or labels == self.REQUIRED_LABELS):
                self.status = ModelStatus.INCOMPATIBLE_LABELS
                self.status_reason = (
                    f"Joblib pipeline classes {sorted(list(labels))} do not match "
                    f"priority labels {sorted(list(self.REQUIRED_LABELS))}."
                )
                self.is_loaded = False
                self.is_fine_tuned = False
                return False

            self.pipeline = pipeline
            self.is_loaded = True
            self.is_fine_tuned = True
            self.status = ModelStatus.READY
            self.status_reason = f"Loaded Scikit-Learn priority pipeline from {pipeline_path}"
            return True
        except Exception as e:
            self.status = ModelStatus.MODEL_NOT_FOUND
            self.status_reason = f"Failed to load joblib pipeline: {e}"
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

    def _load_huggingface_checkpoint(self, path: Path) -> bool:
        """Load and validate Hugging Face DistilBERT checkpoint."""
        deps_ok, dep_err = check_torch_transformers_dependencies()
        if not deps_ok:
            self.status = ModelStatus.DEPENDENCIES_UNAVAILABLE
            self.status_reason = dep_err or "Dependencies torch and transformers required."
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=self.local_files_only)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(path), local_files_only=self.local_files_only
            )
            model.to(self.device)
            model.eval()

            raw_id2label = getattr(model.config, "id2label", {}) or {}
            self.inject_model_and_tokenizer(model=model, tokenizer=tokenizer, id2label=raw_id2label)
            return self.is_fine_tuned
        except Exception as e:
            self.status = ModelStatus.MODEL_NOT_FOUND
            self.status_reason = f"Failed to load Hugging Face model from '{path}': {e}"
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

    def inject_model_and_tokenizer(
        self,
        model: Any,
        tokenizer: Any,
        id2label: Optional[Dict[Any, str]] = None,
    ) -> None:
        """Inject pre-instantiated model and tokenizer (supports mocks and unit tests)."""
        self.model = model
        self.tokenizer = tokenizer
        self.is_loaded = True

        if id2label is not None:
            raw_labels = id2label
        elif hasattr(model, "config") and hasattr(model.config, "id2label"):
            raw_labels = model.config.id2label or {}
        else:
            raw_labels = {}

        self.id2label = {int(k): str(v).strip().upper() for k, v in raw_labels.items()}
        labels_set = set(self.id2label.values())

        # Check for genuine priority labels
        is_incompatible = False
        if not labels_set:
            is_incompatible = True
        elif any(l.startswith("LABEL_") for l in labels_set):
            is_incompatible = True
        elif labels_set.issubset({"YES", "NO", "REVIEW"}):
            # Follow-up triage model detected, not priority
            is_incompatible = True
            self.status = ModelStatus.INCOMPATIBLE_LABELS
            self.status_reason = (
                f"Model loaded with follow-up triage labels {sorted(list(labels_set))}. "
                "Incompatible with priority classification. Priority fine-tuning required."
            )
        elif not (self.MINIMAL_LABELS.issubset(labels_set) or labels_set == self.REQUIRED_LABELS):
            is_incompatible = True
            self.status = ModelStatus.INCOMPATIBLE_LABELS
            self.status_reason = (
                f"Model loaded with labels {sorted(list(labels_set))}. "
                f"Expected priority labels {sorted(list(self.REQUIRED_LABELS))}."
            )

        if is_incompatible:
            self.is_fine_tuned = False
            if self.status not in (ModelStatus.INCOMPATIBLE_LABELS, ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED):
                self.status = ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
                self.status_reason = f"Incompatible labels: {sorted(list(labels_set))}"
        else:
            self.is_fine_tuned = True
            self.status = ModelStatus.READY
            self.status_reason = f"Fine-tuned priority model validated with labels {sorted(list(labels_set))}."

    def _fallback_heuristic_priority(self, text: str) -> str:
        """Transparent heuristic priority estimate when trained weights are pending."""
        lower = text.lower()

        # High priority indicators: imminent deadlines, double deductions, blocked portals
        high_patterns = [
            r"\bdeadline\b",
            r"\btoday\b",
            r"\btomorrow\b",
            r"\burgent\b",
            r"\bemergency\b",
            r"\bdebited\b",
            r"\bdeducted\s+twice\b",
            r"\bportal\s+error\b",
            r"\bcannot\s+(?:download|access|pay|submit)\b",
            r"\bblocked\b",
            r"\bcancelled\b",
            r"\bexam\b",
        ]
        for pat in high_patterns:
            if re.search(pat, lower):
                return "HIGH"

        # Medium priority indicators: administrative inquiries, scholarships, changes
        med_patterns = [
            r"\bscholarship\b",
            r"\brefund\b",
            r"\bhostel\b",
            r"\broom\b",
            r"\bdocument\b",
            r"\bverification\b",
            r"\bgrade\b",
            r"\bbranch\s+change\b",
            r"\bappeal\b",
        ]
        for pat in med_patterns:
            if re.search(pat, lower):
                return "MEDIUM"

        return "LOW"

    def classify_priority(self, text: Optional[str]) -> Dict[str, Any]:
        """Perform priority classification on input transcription.
        
        Returns:
            Dict containing priority_level, confidence, status, is_trained_model, status_reason.
        """
        # 1. Check for empty transcription
        if text is None or not str(text).strip():
            return {
                "priority_level": "LOW",
                "confidence": None,
                "probabilities": None,
                "status": ModelStatus.EMPTY_TRANSCRIPTION.value,
                "is_trained_model": False,
                "status_reason": "Empty or missing transcription text. Defaulted to LOW.",
            }

        clean_text = str(text).strip()

        # 2. Inference via fine-tuned Transformer model
        if self.is_fine_tuned and self.model is not None and self.tokenizer is not None:
            try:
                import torch

                inputs = self.tokenizer(
                    clean_text,
                    max_length=self.config.max_seq_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                device_inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**device_inputs)

                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                best_idx = int(probs.argmax())
                best_label = self.id2label.get(best_idx, "MEDIUM")
                confidence = float(probs[best_idx])
                prob_dict = {
                    self.id2label.get(idx, f"CLASS_{idx}"): float(p)
                    for idx, p in enumerate(probs)
                }

                return {
                    "priority_level": normalize_priority_label(best_label),
                    "confidence": round(confidence, 4),
                    "probabilities": prob_dict,
                    "status": ModelStatus.READY.value,
                    "is_trained_model": True,
                    "status_reason": "Fine-tuned DistilBERT priority model inference.",
                }
            except Exception as exc:
                logger.warning("Transformer inference failed: %s; using fallback.", exc)

        # 3. Inference via fine-tuned Scikit-Learn pipeline
        if self.is_fine_tuned and self.pipeline is not None:
            try:
                probs = self.pipeline.predict_proba([clean_text])[0]
                classes = list(self.pipeline.classes_)
                best_idx = int(probs.argmax())
                best_label = str(classes[best_idx])
                confidence = float(probs[best_idx])
                prob_dict = {str(c).upper(): float(p) for c, p in zip(classes, probs)}

                return {
                    "priority_level": normalize_priority_label(best_label),
                    "confidence": round(confidence, 4),
                    "probabilities": prob_dict,
                    "status": ModelStatus.READY.value,
                    "is_trained_model": True,
                    "status_reason": "Scikit-Learn baseline priority pipeline inference.",
                }
            except Exception as exc:
                logger.warning("Joblib pipeline inference failed: %s; using fallback.", exc)

        # 4. Fallback when trained model is not yet deployed
        fallback_priority = self._fallback_heuristic_priority(clean_text)
        return {
            "priority_level": fallback_priority,
            "confidence": None,
            "probabilities": None,
            "status": self.status.value if self.status != ModelStatus.NOT_INITIALIZED else ModelStatus.PLACEHOLDER_PENDING_MODEL.value,
            "is_trained_model": False,
            "status_reason": f"Fallback priority evaluation: {self.status_reason}",
        }
