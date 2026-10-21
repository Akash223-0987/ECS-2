"""Configuration for the Student Support ML Module.

This configuration specifies parameters for future Hugging Face Transformer model
integration (e.g., DistilBERT). It is strictly declarative and does NOT trigger any
network downloads or weight loading at this foundation stage.
"""

from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class MLServiceConfig:
    """Configuration settings for student support NLP triage module.
    
    Attributes:
        model_name: Future Hugging Face model identifier (e.g. 'distilbert-base-uncased').
                    Configurable via ML_MODEL_NAME environment variable.
        model_type: Architecture type identifier (e.g. 'distilbert').
        tokenizer_name: Optional override for tokenizer name/path. Defaults to model_name.
        max_seq_length: Maximum token length for transformer tokenizer truncation/padding.
        batch_size: Default batch size for batch inference.
        confidence_threshold: Minimum confidence score required to auto-classify without review.
        device: Execution device ('cpu' by default; no GPU dependency required).
        placeholder_mode: When True, predictor runs in placeholder/REVIEW mode without model weights.
        local_files_only: When True, prevents network calls or downloading from Hugging Face Hub.
        model_dir: Optional explicit directory path where fine-tuned model artifacts are stored.
    """
    model_name: str = field(
        default_factory=lambda: os.getenv("ML_MODEL_NAME", "distilbert-base-uncased")
    )
    model_type: str = "distilbert"
    tokenizer_name: Optional[str] = None
    max_seq_length: int = 512
    batch_size: int = 16
    confidence_threshold: float = 0.70
    device: str = "cpu"
    placeholder_mode: bool = True
    local_files_only: bool = True
    model_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
