"""
models.py
=========
Dual-Model Architecture for Automated Grammar & Semantic Error Correction.

Components
----------
1. GrammarClassifier
   - Traditional ML (Random Forest / SVM) trained on NLP features.
   - Predicts whether a sentence contains a grammatical error and
     categorises the error type.

2. SemanticCorrector
   - HuggingFace transformer pipeline (T5 / mT5 / BERT variants).
   - Generates contextually corrected alternatives for erroneous text.

3. ErrorCorrectionPipeline
   - Orchestrates preprocessing → rule detection → ML classification
     → transformer correction into a single callable.

Multilingual swap
-----------------
Change CONFIG.transformer.model_name (in config.py) to any multilingual
checkpoint (e.g. "google/mt5-base", "bert-base-multilingual-cased") and
the SemanticCorrector will load it transparently.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Deferred heavy imports to keep module importable without GPU / network
_transformers_available = False
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    _transformers_available = True
except ImportError:
    logger.warning(
        "transformers not installed. SemanticCorrector will be unavailable."
    )

from config import CONFIG, ClassifierConfig, TransformerConfig
from preprocessing import TextPreprocessor, ErrorPatternDetector


# ---------------------------------------------------------------------------
# 1. GrammarClassifier
# ---------------------------------------------------------------------------

class GrammarClassifier:
    """
    Binary / multi-class grammatical error classifier backed by a
    scikit-learn estimator.

    Labels used in synthetic training data
    ---------------------------------------
    - "NO_ERROR"            : sentence is grammatically acceptable
    - "VERB_AGREEMENT"      : subject–verb number mismatch
    - "TENSE_ERROR"         : incorrect verb tense
    - "ARTICLE_ERROR"       : wrong indefinite article (a/an)
    - "WORD_ORDER"          : constituent order violation
    - "PUNCTUATION"         : missing or extra punctuation
    - "REPEATED_WORD"       : word duplicated consecutively
    - "SPELLING_OOV"        : out-of-vocabulary / misspelled word

    Usage
    -----
    >>> clf = GrammarClassifier()
    >>> clf.train()           # trains on built-in synthetic data
    >>> clf.predict("She don't knows how to plays the guitar.")
    {'label': 'VERB_AGREEMENT', 'confidence': 0.87, 'has_error': True}
    """

    ERROR_LABELS = [
        "NO_ERROR",
        "VERB_AGREEMENT",
        "TENSE_ERROR",
        "ARTICLE_ERROR",
        "WORD_ORDER",
        "PUNCTUATION",
        "REPEATED_WORD",
        "SPELLING_OOV",
    ]

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        self.cfg = config or CONFIG.classifier
        self._preprocessor = TextPreprocessor()
        self._label_encoder = LabelEncoder()
        self._model: Optional[RandomForestClassifier | SVC] = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_synthetic_dataset(self) -> pd.DataFrame:
        """
        Generate a labelled synthetic training corpus.

        In production replace this with real annotated data (e.g. CoNLL-2014,
        BEA-2019 GEC shared task corpora).
        """
        records = [
            # ---- NO_ERROR ----
            ("The cat sat on the mat.", "NO_ERROR"),
            ("She has been working here for three years.", "NO_ERROR"),
            ("They went to the market yesterday.", "NO_ERROR"),
            ("He runs every morning before breakfast.", "NO_ERROR"),
            ("We have finished all the assignments on time.", "NO_ERROR"),
            ("The students submitted their projects last week.", "NO_ERROR"),
            ("I will travel to Paris next month.", "NO_ERROR"),
            ("The manager approved the proposal quickly.", "NO_ERROR"),
            ("Children learn languages faster than adults.", "NO_ERROR"),
            ("My brother and sister are both doctors.", "NO_ERROR"),
            ("The report was submitted before the deadline.", "NO_ERROR"),
            ("She enjoys reading novels on weekends.", "NO_ERROR"),
            ("The team celebrated their victory with a party.", "NO_ERROR"),
            ("He has never visited the museum before.", "NO_ERROR"),
            ("They are planning to launch the product next quarter.", "NO_ERROR"),
            # ---- VERB_AGREEMENT ----
            ("She don't knows how to plays the guitar.", "VERB_AGREEMENT"),
            ("He don't like the new policy.", "VERB_AGREEMENT"),
            ("The team have went to the store.", "VERB_AGREEMENT"),
            ("They is going to the park.", "VERB_AGREEMENT"),
            ("I are ready to leave now.", "VERB_AGREEMENT"),
            ("She have finished the project.", "VERB_AGREEMENT"),
            ("He were present at the meeting.", "VERB_AGREEMENT"),
            ("The children was playing in the yard.", "VERB_AGREEMENT"),
            ("We is planning a trip next week.", "VERB_AGREEMENT"),
            ("The manager don't agree with the proposal.", "VERB_AGREEMENT"),
            ("She don't knows what to do.", "VERB_AGREEMENT"),
            ("They doesn't want to participate.", "VERB_AGREEMENT"),
            ("He have many friends in the city.", "VERB_AGREEMENT"),
            ("The news are very shocking.", "VERB_AGREEMENT"),
            ("My family are arriving tomorrow.", "VERB_AGREEMENT"),
            # ---- TENSE_ERROR ----
            ("He runned fastly to catched the bus.", "TENSE_ERROR"),
            ("She buyed a new car last week.", "TENSE_ERROR"),
            ("They have went to Paris last year.", "TENSE_ERROR"),
            ("I has been working here since 2020.", "TENSE_ERROR"),
            ("We seen the movie yesterday.", "TENSE_ERROR"),
            ("He goed to school every day.", "TENSE_ERROR"),
            ("She thinked about the problem all night.", "TENSE_ERROR"),
            ("They growed the plants in the garden.", "TENSE_ERROR"),
            ("I knowed the answer immediately.", "TENSE_ERROR"),
            ("He leaved the office early.", "TENSE_ERROR"),
            ("She speaked to the manager about it.", "TENSE_ERROR"),
            ("They bringed the report to the meeting.", "TENSE_ERROR"),
            ("We buyed tickets for the concert.", "TENSE_ERROR"),
            ("He telled me the truth.", "TENSE_ERROR"),
            ("She catched the ball perfectly.", "TENSE_ERROR"),
            # ---- ARTICLE_ERROR ----
            ("I saw a elephant at the zoo.", "ARTICLE_ERROR"),
            ("She is a honest person.", "ARTICLE_ERROR"),
            ("He is an useful member of the team.", "ARTICLE_ERROR"),
            ("We need an better solution.", "ARTICLE_ERROR"),
            ("This is a important decision.", "ARTICLE_ERROR"),
            ("She adopted a orphan last year.", "ARTICLE_ERROR"),
            ("He has a umbrella in his bag.", "ARTICLE_ERROR"),
            ("This is an unique opportunity.", "ARTICLE_ERROR"),
            ("She wore a orange dress to the party.", "ARTICLE_ERROR"),
            ("He is an European citizen.", "ARTICLE_ERROR"),
            ("I need an new laptop for work.", "ARTICLE_ERROR"),
            ("She has a idea for the project.", "ARTICLE_ERROR"),
            ("We visited an historical monument.", "ARTICLE_ERROR"),
            ("He is a expert in cybersecurity.", "ARTICLE_ERROR"),
            ("She gave an brief explanation.", "ARTICLE_ERROR"),
            # ---- WORD_ORDER ----
            ("She always is late to work.", "WORD_ORDER"),
            ("He yesterday went to the office.", "WORD_ORDER"),
            ("They often are confused by the instructions.", "WORD_ORDER"),
            ("Never she misses a deadline.", "WORD_ORDER"),
            ("He quickly very spoke.", "WORD_ORDER"),
            ("She beautifully sings songs Hindi.", "WORD_ORDER"),
            ("They to the store went.", "WORD_ORDER"),
            ("He the ball kicked hard.", "WORD_ORDER"),
            ("She the project finished early.", "WORD_ORDER"),
            ("They together always work.", "WORD_ORDER"),
            # ---- PUNCTUATION ----
            ("Its a nice day isnt it", "PUNCTUATION"),
            ("She said hello and waved goodbye", "PUNCTUATION"),
            ("However the results were not conclusive", "PUNCTUATION"),
            ("In conclusion the project was a success", "PUNCTUATION"),
            ("Wait let me check the details first", "PUNCTUATION"),
            ("He asked where is the nearest station", "PUNCTUATION"),
            ("The meeting which was scheduled for Monday was cancelled", "PUNCTUATION"),
            ("She replied yes I will attend the event", "PUNCTUATION"),
            ("Dont forget to submit the form before Friday", "PUNCTUATION"),
            ("Its not just about the money its about the principle", "PUNCTUATION"),
            # ---- REPEATED_WORD ----
            ("The the cat sat on the mat.", "REPEATED_WORD"),
            ("She went to to the store.", "REPEATED_WORD"),
            ("He is is going to the park.", "REPEATED_WORD"),
            ("They have have finished the task.", "REPEATED_WORD"),
            ("I think think we should leave now.", "REPEATED_WORD"),
            ("Please be be careful on the stairs.", "REPEATED_WORD"),
            ("The team will will present tomorrow.", "REPEATED_WORD"),
            ("He said that that he would come.", "REPEATED_WORD"),
            ("She asked if if there was time.", "REPEATED_WORD"),
            ("The project will be be completed soon.", "REPEATED_WORD"),
            # ---- SPELLING_OOV ----
            ("She recieved an emial from the managr.", "SPELLING_OOV"),
            ("He submited the reprot on time.", "SPELLING_OOV"),
            ("The accomodation was very comfortble.", "SPELLING_OOV"),
            ("She achived excelent resultss this quarter.", "SPELLING_OOV"),
            ("The commitee approvd the propsal.", "SPELLING_OOV"),
            ("He acknowleged the errror in his calcualtion.", "SPELLING_OOV"),
            ("The occurance of the incedent was unexpcted.", "SPELLING_OOV"),
            ("She maintians a positve atttitude at work.", "SPELLING_OOV"),
            ("The assessement reveald segnificant improvments.", "SPELLING_OOV"),
            ("He reccommended a diferent approch.", "SPELLING_OOV"),
        ]
        return pd.DataFrame(records, columns=["text", "label"])

    def _featurize(self, texts: List[str]) -> np.ndarray:
        """Convert raw text strings into a numeric feature matrix."""
        rows = []
        for text in texts:
            result = self._preprocessor.preprocess(text)
            f = result["features"]
            rows.append([
                f["num_tokens"],
                f["num_words"],
                f["type_token_ratio"],
                f["avg_word_length"],
                f["avg_sent_length"],
                f["third_sg_verb_count"],
                f["base_verb_count"],
                f["modal_count"],
                f["past_tense_count"],
                f["aux_count"],
                f["punct_count"],
                f["oov_count"],
                f["repeated_bigrams"],
                f["sentence_count"],
            ])
        return np.array(rows, dtype=float)

    def train(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the classifier.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame with columns ["text", "label"].
            Defaults to the built-in synthetic dataset.

        Returns
        -------
        dict : classification_report and train/test split sizes.
        """
        if df is None:
            df = self._build_synthetic_dataset()

        logger.info("Featurizing %d training samples…", len(df))
        X = self._featurize(df["text"].tolist())
        y = self._label_encoder.fit_transform(df["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )

        if self.cfg.model_type == "svm":
            self._model = SVC(
                kernel="rbf",
                probability=True,
                random_state=self.cfg.random_state,
            )
        else:
            self._model = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                random_state=self.cfg.random_state,
                n_jobs=-1,
            )

        self._model.fit(X_train, y_train)
        self._is_trained = True

        y_pred = self._model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=self._label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        logger.info("Training complete. Test accuracy: %.3f", report["accuracy"])
        return {
            "report": report,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "classes": self._label_encoder.classes_.tolist(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict:
        """
        Classify a single sentence.

        Returns
        -------
        dict with:
            label       : predicted error category (str)
            confidence  : probability of the predicted class (float)
            has_error   : True if label != "NO_ERROR"
            all_probs   : probability distribution over all classes
        """
        if not self._is_trained:
            self.train()

        X = self._featurize([text])
        class_idx = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        label = self._label_encoder.inverse_transform([class_idx])[0]
        confidence = float(proba[class_idx])

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "has_error": label != "NO_ERROR",
            "all_probs": {
                cls: round(float(p), 4)
                for cls, p in zip(self._label_encoder.classes_, proba)
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a list of sentences."""
        if not self._is_trained:
            self.train()
        return [self.predict(t) for t in texts]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist model and label encoder to disk."""
        os.makedirs(os.path.dirname(self.cfg.model_save_path), exist_ok=True)
        with open(self.cfg.model_save_path, "wb") as f:
            pickle.dump(
                {"model": self._model, "label_encoder": self._label_encoder},
                f,
            )
        logger.info("Model saved to %s", self.cfg.model_save_path)

    def load(self) -> bool:
        """Load model from disk. Returns True on success."""
        if not os.path.exists(self.cfg.model_save_path):
            return False
        with open(self.cfg.model_save_path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._label_encoder = payload["label_encoder"]
        self._is_trained = True
        return True


# ---------------------------------------------------------------------------
# 2. SemanticCorrector
# ---------------------------------------------------------------------------

class SemanticCorrector:
    """
    Transformer-based semantic & grammatical correction model.

    Wraps a HuggingFace Seq2Seq or Text2Text generation pipeline.
    Default model: vennify/t5-base-grammar-correction (fine-tuned T5).

    Multilingual swap
    -----------------
    Set CONFIG.transformer.model_name to any multilingual checkpoint; the
    rest of the class is model-agnostic.

    Usage
    -----
    >>> corrector = SemanticCorrector()
    >>> corrector.load_model()
    >>> corrector.correct("She don't knows how to plays the guitar.")
    {
        'original': "She don't knows how to plays the guitar.",
        'corrections': ["She doesn't know how to play the guitar."],
        'top_correction': "She doesn't know how to play the guitar.",
    }
    """

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        self.cfg = config or CONFIG.transformer
        self._pipe = None
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def load_model(self) -> None:
        """
        Lazily load the transformer model and tokenizer.

        Called automatically on first correction request. Safe to call
        multiple times (idempotent).
        """
        if self._loaded:
            return

        if not _transformers_available:
            raise RuntimeError(
                "The `transformers` library is required for SemanticCorrector. "
                "Install it with: pip install transformers torch"
            )

        logger.info("Loading transformer model: %s …", self.cfg.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name)
        self._loaded = True
        logger.info("Transformer model loaded successfully.")

    def correct(self, text: str) -> Dict:
        """
        Generate correction candidates for a single sentence using direct generation.

        Parameters
        ----------
        text : str
            Raw (possibly erroneous) sentence.

        Returns
        -------
        dict with:
            original        : input text
            corrections     : list of candidate corrected strings
            top_correction  : best candidate (first beam)
            model_name      : model used for correction
        """
        if not self._loaded:
            self.load_model()

        prefixed = f"{self.cfg.task_prefix}{text}"
        inputs = self._tokenizer(prefixed, return_tensors="pt")
        
        # Use torch.no_grad() for efficient inference
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                max_length=self.cfg.max_output_length,
                num_beams=self.cfg.num_beams,
                num_return_sequences=self.cfg.num_return_sequences,
                early_stopping=True,
            )

        decoded_outputs = [
            self._tokenizer.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

        corrections = [
            c for c in decoded_outputs
            if c != text.strip()
        ]
        # De-duplicate while preserving order
        seen = set()
        unique_corrections = []
        for c in corrections:
            if c not in seen:
                seen.add(c)
                unique_corrections.append(c)

        return {
            "original": text,
            "corrections": unique_corrections or [text],
            "top_correction": unique_corrections[0] if unique_corrections else text,
            "model_name": self.cfg.model_name,
        }

    def correct_batch(self, texts: List[str]) -> List[Dict]:
        """Correct a list of sentences."""
        return [self.correct(t) for t in texts]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# 3. ErrorCorrectionPipeline
# ---------------------------------------------------------------------------

class ErrorCorrectionPipeline:
    """
    End-to-end orchestrator that chains:

    Raw Text
      → TextPreprocessor   (tokenize, POS, lemmatize, features)
      → ErrorPatternDetector (rule-based findings)
      → GrammarClassifier  (ML error category + confidence)
      → SemanticCorrector  (transformer corrected text)
      → Aggregated Report  (dict)

    Usage
    -----
    >>> ep = ErrorCorrectionPipeline()
    >>> ep.initialize()
    >>> report = ep.analyze("She don't knows how to plays the guitar.")
    >>> report["top_correction"]
    "She doesn't know how to play the guitar."
    """

    def __init__(self) -> None:
        self._preprocessor = TextPreprocessor()
        self._rule_detector = ErrorPatternDetector()
        self._classifier = GrammarClassifier()
        self._corrector = SemanticCorrector()

    def initialize(self, load_transformer: bool = True) -> None:
        """
        Warm up all components.

        Parameters
        ----------
        load_transformer : bool
            Set False to skip the heavy transformer download (useful for
            unit tests or when only the ML pipeline is needed).
        """
        logger.info("Initialising GrammarClassifier…")
        if not self._classifier.load():
            self._classifier.train()
            self._classifier.save()

        if load_transformer:
            logger.info("Initialising SemanticCorrector…")
            self._corrector.load_model()

    def analyze(self, text: str, use_transformer: bool = True) -> Dict:
        """
        Full analysis of a single input text.

        Returns
        -------
        dict with:
            original            : raw input
            preprocessed        : dict from TextPreprocessor.preprocess()
            rule_findings       : list from ErrorPatternDetector.detect_all()
            ml_classification   : dict from GrammarClassifier.predict()
            semantic_correction : dict from SemanticCorrector.correct()
            top_correction      : str — best corrected text
            has_any_error       : bool
            error_summary       : human-readable summary str
        """
        # Step 1 – Preprocessing
        preprocessed = self._preprocessor.preprocess(text)

        # Step 2 – Rule-based detection
        rule_findings = self._rule_detector.detect_all(text)

        # Step 3 – ML classification
        ml_result = self._classifier.predict(text)

        # Step 4 – Transformer correction (optional)
        semantic_result: Dict = {}
        top_correction = text
        if use_transformer and self._corrector.is_loaded:
            semantic_result = self._corrector.correct(text)
            top_correction = semantic_result.get("top_correction", text)

        has_any_error = (
            ml_result["has_error"]
            or bool(rule_findings)
        )

        error_summary = self._build_summary(
            ml_result, rule_findings, top_correction, text
        )

        return {
            "original": text,
            "preprocessed": preprocessed,
            "rule_findings": rule_findings,
            "ml_classification": ml_result,
            "semantic_correction": semantic_result,
            "top_correction": top_correction,
            "has_any_error": has_any_error,
            "error_summary": error_summary,
        }

    def analyze_batch(self, texts: List[str], use_transformer: bool = True) -> List[Dict]:
        """Analyze a list of texts."""
        return [self.analyze(t, use_transformer=use_transformer) for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        ml_result: Dict,
        rule_findings: List[Dict],
        top_correction: str,
        original: str,
    ) -> str:
        parts = []
        if ml_result["has_error"]:
            parts.append(
                f"ML Classifier detected: {ml_result['label']} "
                f"(confidence {ml_result['confidence']:.0%})."
            )
        if rule_findings:
            types = list({f["error_type"] for f in rule_findings})
            parts.append(f"Rule-based findings: {', '.join(types)}.")
        if top_correction and top_correction.strip() != original.strip():
            parts.append(f"Suggested correction: \"{top_correction}\"")
        if not parts:
            parts.append("No errors detected. Text appears grammatically correct.")
        return " ".join(parts)
