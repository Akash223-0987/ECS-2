"""Hugging Face Model Loader for Student Support Ticket Triage.

Designed for DistilBERT sequence classification with strict safety guardrails:
1. Lazy Imports: `transformers` and `torch` are imported on-demand inside methods,
   ensuring the module can be imported even when optional ML dependencies are absent.
2. Offline / No Automatic Download: Does not access Hugging Face Hub or download
   weights during ordinary imports or tests (`local_files_only=True` enforced by default).
3. Explicit Initialization: Model loading is separated from instantiation.
4. CPU Default: CPU execution is default; GPU can be selected explicitly.
5. Task-Specific Label Validation: Strictly distinguishes between generic pretrained
   models (e.g. `['LABEL_0', 'LABEL_1']`, `['POSITIVE', 'NEGATIVE']`) and models
   fine-tuned for student support follow-up labels (`['YES', 'NO', 'REVIEW']`).
   Generic labels are NEVER mapped to business decisions.
6. Safe Fallback: If dependencies are missing or model files are not found,
   sets explicit diagnostic statuses enabling safe REVIEW routing.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .config import MLServiceConfig
from .schemas import FollowUpRequired, ModelStatus


def check_dependencies() -> Tuple[bool, Optional[str]]:
    """Check if torch and transformers are installed and importable.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_available, error_message_if_missing)
    """
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
            "Install with 'pip install torch transformers' on inference machines.",
        )
    return True, None


class HuggingFaceModelLoader:
    """Manages loading, label validation, and inference for DistilBERT ticket models.
    
    Attributes:
        config: Configuration containing model parameters and paths.
        model_name_or_path: Path or identifier for the target model.
        device: Target execution device ('cpu' by default).
        local_files_only: If True, blocks network downloads from Hugging Face Hub.
        status: Diagnostic ModelStatus.
        status_reason: Human-readable diagnostic explanation.
        is_loaded: True if model and tokenizer weights are instantiated in memory.
        is_fine_tuned: True if model contains validated student support task labels.
    """

    REQUIRED_TASK_LABELS: Set[str] = {"YES", "NO", "REVIEW"}
    MINIMAL_TASK_LABELS: Set[str] = {"YES", "NO"}

    def __init__(
        self,
        config: Optional[MLServiceConfig] = None,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        self.config = config or MLServiceConfig()
        self.model_name_or_path = (
            model_name_or_path
            or self.config.model_dir
            or self.config.model_name
        )
        self.device = device or self.config.device or "cpu"
        self.local_files_only = local_files_only

        self.status: ModelStatus = ModelStatus.NOT_INITIALIZED
        self.status_reason: str = "Model loader initialized; weights not yet loaded."
        self.is_loaded: bool = False
        self.is_fine_tuned: bool = False

        self.model: Any = None
        self.tokenizer: Any = None
        self.id2label: Dict[int, str] = {}

    def load_model(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ) -> bool:
        """Explicitly load model and tokenizer from local checkpoint or path.
        
        Args:
            model_name_or_path: Optional override for model directory/name.
            device: Optional override for computation device.
            local_files_only: When True, prevents network calls to Hugging Face Hub.
            
        Returns:
            bool: True if task-specific model was successfully loaded, False otherwise.
        """
        target_path = model_name_or_path or self.model_name_or_path
        target_device = device or self.device
        only_local = (
            self.local_files_only if local_files_only is None else local_files_only
        )

        # 1. Verify dependencies lazily
        deps_ok, dep_err = check_dependencies()
        if not deps_ok:
            self.status = ModelStatus.DEPENDENCIES_UNAVAILABLE
            self.status_reason = dep_err or "Dependencies torch and transformers required."
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

        # 2. Check local path existence if it appears to be a file path
        is_path_like = (
            os.path.sep in target_path
            or "/" in target_path
            or Path(target_path).exists()
            or target_path.startswith(".")
        )
        if is_path_like and not os.path.exists(target_path):
            self.status = ModelStatus.MODEL_NOT_FOUND
            self.status_reason = f"Local model directory not found: '{target_path}'"
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

        # 3. Lazy imports of transformers and torch
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except (ImportError, ModuleNotFoundError) as exc:
            self.status = ModelStatus.DEPENDENCIES_UNAVAILABLE
            self.status_reason = f"Failed to import ML packages: {str(exc)}"
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

        # 4. Attempt model loading with local_files_only guardrail
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                target_path,
                local_files_only=only_local,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                target_path,
                local_files_only=only_local,
            )
            model.to(target_device)
            model.eval()
        except Exception as exc:
            self.status = ModelStatus.MODEL_NOT_FOUND
            self.status_reason = (
                f"Failed to load model from '{target_path}' (local_files_only={only_local}): {str(exc)}"
            )
            self.is_loaded = False
            self.is_fine_tuned = False
            return False

        # 5. Validate label structure
        raw_id2label = getattr(model.config, "id2label", {}) or {}
        self.inject_model_and_tokenizer(model=model, tokenizer=tokenizer, id2label=raw_id2label)
        return self.is_fine_tuned

    def inject_model_and_tokenizer(
        self,
        model: Any,
        tokenizer: Any,
        id2label: Optional[Dict[Any, str]] = None,
    ) -> None:
        """Inject pre-instantiated model and tokenizer (supports mocks and unit testing).
        
        Performs label analysis to strictly distinguish between generic pretrained
        models and task-specific fine-tuned models.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.is_loaded = True

        # Extract and normalize id2label mapping
        if id2label is not None:
            raw_labels = id2label
        elif hasattr(model, "config") and hasattr(model.config, "id2label"):
            raw_labels = model.config.id2label or {}
        else:
            raw_labels = {}

        self.id2label = {int(k): str(v).strip().upper() for k, v in raw_labels.items()}
        labels_set = set(self.id2label.values())

        # Tasks 6 & 7: Check if model has genuine task-specific labels
        # Generic models typically have {"LABEL_0", "LABEL_1"} or {"NEGATIVE", "POSITIVE"}
        is_generic = False
        if not labels_set:
            is_generic = True
        elif any(l.startswith("LABEL_") for l in labels_set):
            is_generic = True
        elif labels_set.issubset({"POSITIVE", "NEGATIVE"}):
            is_generic = True
        elif not (self.MINIMAL_TASK_LABELS.issubset(labels_set) or labels_set == self.REQUIRED_TASK_LABELS):
            is_generic = True

        if is_generic:
            self.is_fine_tuned = False
            self.status = ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
            self.status_reason = (
                f"Model loaded with generic labels {sorted(list(labels_set)) or '[]'}. "
                "It is NOT fine-tuned for student support follow-up triage. "
                "Generic labels must never be mapped to YES, NO, or REVIEW."
            )
        else:
            self.is_fine_tuned = True
            self.status = ModelStatus.READY
            self.status_reason = (
                f"Task-specific fine-tuned model ready with labels: {sorted(list(labels_set))}."
            )

    def classify_text(self, text: str) -> Dict[str, Any]:
        """Perform inference on ticket text with strict safety checks.
        
        Returns:
            Dict containing:
                - status: ModelStatus
                - status_reason: str
                - label: Optional[str] (e.g. 'YES', 'NO', 'REVIEW', or None)
                - score: Optional[float] (confidence score, or None)
                - probabilities: Optional[Dict[str, float]]
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return {
                "status": self.status,
                "status_reason": self.status_reason,
                "label": None,
                "score": None,
                "probabilities": None,
            }

        # Task 7: Never interpret generic pretrained labels as YES, NO, or REVIEW
        if not self.is_fine_tuned:
            return {
                "status": ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED,
                "status_reason": self.status_reason,
                "label": None,
                "score": None,
                "probabilities": None,
            }

        # Lazy import of torch for tensor execution
        try:
            import torch
        except (ImportError, ModuleNotFoundError):
            return {
                "status": ModelStatus.DEPENDENCIES_UNAVAILABLE,
                "status_reason": "torch is required for tensor operations during inference.",
                "label": None,
                "score": None,
                "probabilities": None,
            }

        max_len = self.config.max_seq_length if self.config else 512
        inputs = self.tokenizer(
            text,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Move tensors to device safely if supported
        if isinstance(inputs, dict):
            device_inputs = {}
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    device_inputs[k] = v.to(self.device)
                else:
                    device_inputs[k] = v
        else:
            device_inputs = inputs

        # Execute forward pass without gradients
        if hasattr(torch, "no_grad"):
            with torch.no_grad():
                outputs = self.model(**device_inputs)
        else:
            outputs = self.model(**device_inputs)

        # Extract logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Calculate softmax probabilities
        if hasattr(torch, "softmax"):
            probs = torch.softmax(logits, dim=-1)
            if hasattr(probs, "squeeze"):
                probs = probs.squeeze()
            if hasattr(probs, "tolist"):
                probs_list = probs.tolist()
            else:
                probs_list = list(probs)
        else:
            probs_list = list(logits)

        if not isinstance(probs_list, list):
            probs_list = [float(probs_list)]

        best_idx = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))
        best_label = self.id2label.get(best_idx, "REVIEW")
        best_score = float(probs_list[best_idx])

        prob_dict = {
            self.id2label.get(idx, f"CLASS_{idx}"): float(p)
            for idx, p in enumerate(probs_list)
        }

        return {
            "status": ModelStatus.READY,
            "status_reason": self.status_reason,
            "label": best_label,
            "score": best_score,
            "probabilities": prob_dict,
        }
