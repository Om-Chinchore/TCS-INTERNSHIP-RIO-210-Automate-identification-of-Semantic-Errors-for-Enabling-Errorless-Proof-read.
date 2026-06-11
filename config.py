"""
config.py
=========
Central configuration for the Automated Identification and Correction of
Semantic and Grammatical Errors pipeline.

To swap to multilingual support, change TRANSFORMER_MODEL to a multilingual
counterpart (e.g., "bert-base-multilingual-cased" or "Helsinki-NLP/opus-mt-*")
and set LANGUAGE to the target locale.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Language & Transformer Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """
    Configuration for the transformer-based semantic correction model.

    Multilingual swap: change `model_name` to any HuggingFace multilingual
    checkpoint without touching downstream pipeline code.

    Examples
    --------
    English (default)      : "vennify/t5-base-grammar-correction"
    Multilingual (mT5)     : "google/mt5-base"
    Multilingual BERT      : "bert-base-multilingual-cased"
    Translation (opus-mt)  : "Helsinki-NLP/opus-mt-en-ROMANCE"
    """
    model_name: str = "vennify/t5-base-grammar-correction"
    task_prefix: str = "grammar: "          # T5-style task prefix
    max_input_length: int = 512
    max_output_length: int = 512
    num_beams: int = 4
    num_return_sequences: int = 3           # Candidate corrections
    device: str = "cpu"                     # "cuda" for GPU acceleration
    language: str = "en"                    # ISO 639-1 language code


# ---------------------------------------------------------------------------
# ML Classifier Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """Configuration for the traditional ML grammatical-error classifier."""
    model_type: str = "random_forest"       # "random_forest" | "svm"
    n_estimators: int = 200                 # Random Forest trees
    max_depth: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2
    model_save_path: str = "artifacts/classifier.pkl"
    vectorizer_save_path: str = "artifacts/vectorizer.pkl"


# ---------------------------------------------------------------------------
# NLP Preprocessing Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Configuration for the NLP preprocessing engine."""
    language: str = "english"               # NLTK language for stopwords
    lemmatize: bool = True
    remove_stopwords: bool = False          # Keep stopwords for grammar tasks
    pos_tag_backend: str = "nltk"           # "nltk" | "spacy"
    spacy_model: str = "en_core_web_sm"


# ---------------------------------------------------------------------------
# Error Detection Rules Configuration
# ---------------------------------------------------------------------------

@dataclass
class RulesConfig:
    """Heuristic rule sets for pattern-based error detection."""
    # Common grammar patterns (regex-ready)
    double_space_pattern: str = r" {2,}"
    repeated_word_pattern: str = r"\b(\w+)\s+\1\b"
    # Threshold for ML confidence to flag an error
    error_confidence_threshold: float = 0.55


# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Streamlit dashboard configuration."""
    app_title: str = "AI Proofreader — Semantic & Grammar Error Correction"
    page_icon: str = "✍️"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    max_text_length: int = 5000             # Characters allowed in text area
    demo_texts: List[str] = field(default_factory=lambda: [
        "She don't knows how to plays the guitar since yesterday.",
        "The team have went to the store and buyed many item.",
        "He runned fastly to catched the buss before it leave.",
        "Their going to there house over they're for the holidays.",
        "I has been working in this company since three years.",
    ])


# ---------------------------------------------------------------------------
# Master Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    app: AppConfig = field(default_factory=AppConfig)
    verbose: bool = False
    log_level: str = "INFO"


# Singleton instance used across the project
CONFIG = PipelineConfig()
