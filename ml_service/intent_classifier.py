"""Auxiliary Support Intent Classifier Interface.

Provides an independent intent-classification interface capable of loading
either a lightweight classical baseline (TF-IDF + Logistic Regression) or a
fine-tuned DistilBERT transformer checkpoint.

CRITICAL ARCHITECTURAL CONSTRAINTS & BOUNDARIES:
1. Intent Classification is strictly auxiliary topic tagging.
2. Intent predictions MUST NEVER be mapped to follow-up triage labels (YES, NO, REVIEW).
3. Intent predictions MUST NEVER alter priority scores, urgency, or outbound call decisions.
4. Missing model artifacts return an explicit diagnostic status (`MODEL_NOT_FOUND` /
   `NOT_INITIALIZED`), never fabricated or hallucinated predictions.
5. Confidence scores are uncalibrated softmax/decision probabilities and are explicitly
   documented as such.
6. Baseline model predictions are clearly identified as classical baseline, never
   disguised as DistilBERT predictions.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class IntentModelStatus(str, Enum):
    """Operational status of the auxiliary intent classifier."""
    NOT_INITIALIZED = "NOT_INITIALIZED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    DEPENDENCIES_UNAVAILABLE = "DEPENDENCIES_UNAVAILABLE"
    INVALID_ARTIFACTS = "INVALID_ARTIFACTS"
    READY = "READY"


@dataclass
class IntentPrediction:
    """Structured result of auxiliary intent classification.
    
    Attributes:
        status: Operational diagnostic status of the intent classifier.
        predicted_intent: Target intent label if successfully predicted, else None.
        intent_confidence: Uncalibrated confidence score [0.0 - 1.0], or None.
        model_type: Explicit identifier of model architecture ('baseline_tfidf_logistic_regression' or 'distilbert').
        probabilities: Dictionary of uncalibrated probability scores per class.
        status_reason: Human-readable diagnostic rationale.
    """
    status: IntentModelStatus
    predicted_intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    model_type: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    status_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, IntentModelStatus) else str(self.status),
            "predicted_intent": self.predicted_intent,
            "intent_confidence": self.intent_confidence,
            "model_type": self.model_type,
            "probabilities": self.probabilities,
            "status_reason": self.status_reason,
        }


class AuxiliaryIntentClassifier:
    """Manages loading and inference for auxiliary support intent classification.
    
    Supports:
    1. Classical CPU Baseline (TF-IDF + Logistic Regression pipeline)
    2. Future fine-tuned DistilBERT checkpoint (with task-specific intent labels)
    """

    ALLOWED_INTENTS: Set[str] = {
        "payment_issue",
        "get_refund",
        "contact_customer_service",
        "recover_password",
    }

    DEFAULT_BASELINE_DIR = Path(__file__).resolve().parent / "models" / "auxiliary-intent-baseline"

    def __init__(self, model_dir: Optional[Union[str, Path]] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_BASELINE_DIR
        self.status: IntentModelStatus = IntentModelStatus.NOT_INITIALIZED
        self.status_reason: str = "Classifier instantiated; model weights not loaded."
        self.model_type: Optional[str] = None
        self.pipeline: Any = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self.is_loaded: bool = False

    def load_model(self, model_dir: Optional[Union[str, Path]] = None) -> bool:
        """Load intent classification model from local directory.
        
        Inspects directory contents to distinguish between:
        - Baseline joblib pipeline
        - DistilBERT PyTorch checkpoint
        
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        target_dir = Path(model_dir) if model_dir else self.model_dir

        if not target_dir.exists() or not target_dir.is_dir():
            self.status = IntentModelStatus.MODEL_NOT_FOUND
            self.status_reason = f"Intent model directory does not exist: '{target_dir}'"
            self.is_loaded = False
            return False

        # 1. Check for baseline pipeline (joblib)
        baseline_file = target_dir / "baseline_pipeline.joblib"
        if baseline_file.exists():
            try:
                import joblib
                self.pipeline = joblib.load(baseline_file)
                self.model_type = "baseline_tfidf_logistic_regression"

                # Load label mapping if present
                mapping_file = target_dir / "label_mapping.json"
                if mapping_file.exists():
                    with open(mapping_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    raw_id2label = meta.get("id2label", {})
                    self.id2label = {int(k): str(v) for k, v in raw_id2label.items()}
                    self.label2id = {str(k): int(v) for k, v in meta.get("label2id", {}).items()}

                self.status = IntentModelStatus.READY
                self.status_reason = (
                    "Loaded TF-IDF + Logistic Regression CPU baseline. "
                    "Predictions represent auxiliary topic tags only."
                )
                self.is_loaded = True
                return True
            except Exception as exc:
                self.status = IntentModelStatus.INVALID_ARTIFACTS
                self.status_reason = f"Failed to load baseline joblib pipeline: {str(exc)}"
                self.is_loaded = False
                return False

        # 2. Check for DistilBERT transformer checkpoint
        config_file = target_dir / "config.json"
        has_weights = any((target_dir / w).exists() for w in ["model.safetensors", "pytorch_model.bin"])
        if config_file.exists() and has_weights:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except (ImportError, ModuleNotFoundError) as exc:
                self.status = IntentModelStatus.DEPENDENCIES_UNAVAILABLE
                self.status_reason = f"PyTorch/Transformers not available to load DistilBERT: {str(exc)}"
                self.is_loaded = False
                return False

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(str(target_dir), local_files_only=True)
                self.model.eval()

                raw_id2label = getattr(self.model.config, "id2label", {}) or {}
                self.id2label = {int(k): str(v) for k, v in raw_id2label.items()}
                self.label2id = {str(v): int(k) for k, v in self.id2label.items()}

                # Verify labels match allowed intents
                found_labels = set(self.id2label.values())
                if not found_labels.issubset(self.ALLOWED_INTENTS):
                    self.status = IntentModelStatus.INVALID_ARTIFACTS
                    self.status_reason = (
                        f"Loaded transformer has labels {sorted(list(found_labels))} which do not match "
                        f"expected intent labels {sorted(list(self.ALLOWED_INTENTS))}."
                    )
                    self.is_loaded = False
                    return False

                self.model_type = "distilbert"
                self.status = IntentModelStatus.READY
                self.status_reason = f"Loaded fine-tuned DistilBERT intent classifier with labels: {sorted(list(found_labels))}."
                self.is_loaded = True
                return True
            except Exception as exc:
                self.status = IntentModelStatus.INVALID_ARTIFACTS
                self.status_reason = f"Failed to load DistilBERT checkpoint: {str(exc)}"
                self.is_loaded = False
                return False

        self.status = IntentModelStatus.MODEL_NOT_FOUND
        self.status_reason = f"No supported model artifacts found in '{target_dir}'"
        self.is_loaded = False
        return False

    def predict_intent(self, text: Optional[str]) -> IntentPrediction:
        """Predict support intent for an incoming text query.
        
        Returns:
            IntentPrediction with explicit status, predicted_intent, and uncalibrated confidence.
        """
        # If not loaded, return explicit status without fabricating
        if not self.is_loaded:
            return IntentPrediction(
                status=self.status,
                predicted_intent=None,
                intent_confidence=None,
                model_type=self.model_type,
                status_reason=self.status_reason,
            )

        if not text or not text.strip():
            return IntentPrediction(
                status=IntentModelStatus.READY,
                predicted_intent=None,
                intent_confidence=None,
                model_type=self.model_type,
                status_reason="Empty or whitespace text; intent prediction skipped.",
            )

        clean_text = text.strip()

        # 1. Classical baseline inference
        if self.model_type == "baseline_tfidf_logistic_regression" and self.pipeline is not None:
            try:
                probs = self.pipeline.predict_proba([clean_text])[0]
                classes = list(self.pipeline.classes_)

                best_idx = int(probs.argmax())
                best_label = str(classes[best_idx])
                best_conf = float(probs[best_idx])

                prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}

                return IntentPrediction(
                    status=IntentModelStatus.READY,
                    predicted_intent=best_label,
                    intent_confidence=round(best_conf, 4),
                    model_type=self.model_type,
                    probabilities=prob_dict,
                    status_reason="Baseline TF-IDF + Logistic Regression inference (uncalibrated probability).",
                )
            except Exception as exc:
                return IntentPrediction(
                    status=IntentModelStatus.INVALID_ARTIFACTS,
                    predicted_intent=None,
                    intent_confidence=None,
                    model_type=self.model_type,
                    status_reason=f"Baseline inference failed: {str(exc)}",
                )

        # 2. DistilBERT transformer inference
        if self.model_type == "distilbert" and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                inputs = self.tokenizer(clean_text, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                if not isinstance(probs, list):
                    probs = [float(probs)]

                best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                best_label = self.id2label.get(best_idx, "UNKNOWN")
                best_conf = float(probs[best_idx])

                prob_dict = {self.id2label.get(i, f"CLASS_{i}"): float(p) for i, p in enumerate(probs)}

                return IntentPrediction(
                    status=IntentModelStatus.READY,
                    predicted_intent=best_label,
                    intent_confidence=round(best_conf, 4),
                    model_type=self.model_type,
                    probabilities=prob_dict,
                    status_reason="DistilBERT sequence classification (uncalibrated softmax).",
                )
            except Exception as exc:
                return IntentPrediction(
                    status=IntentModelStatus.INVALID_ARTIFACTS,
                    predicted_intent=None,
                    intent_confidence=None,
                    model_type=self.model_type,
                    status_reason=f"DistilBERT inference failed: {str(exc)}",
                )

        return IntentPrediction(
            status=IntentModelStatus.INVALID_ARTIFACTS,
            predicted_intent=None,
            intent_confidence=None,
            model_type=self.model_type,
            status_reason="Unrecognized model state.",
        )
