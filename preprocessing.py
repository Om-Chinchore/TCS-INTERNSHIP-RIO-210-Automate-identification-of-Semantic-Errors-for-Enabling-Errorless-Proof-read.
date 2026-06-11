"""
preprocessing.py
================
NLP Preprocessing Engine for the Automated Grammar & Semantic Correction
pipeline.

Provides:
  - TextPreprocessor  : tokenization, lemmatization, POS tagging, NLP features
  - ErrorPatternDetector : rule-based heuristic checks on raw text

Design note
-----------
All public methods accept plain Python strings and return serialisable Python
objects (lists, dicts) so that this module has no Streamlit or model dependency
and can be imported cleanly in training scripts, unit tests, or notebooks.
"""

import re
import string
import logging
from typing import Dict, List, Optional, Tuple

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from config import CONFIG, PreprocessingConfig, RulesConfig

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (idempotent, safe to call multiple times)
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "stopwords",
    "wordnet",
    "omw-1.4",
]


def _ensure_nltk_resources() -> None:
    """Download missing NLTK data packages silently."""
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                try:
                    nltk.data.find(f"corpora/{resource}")
                except LookupError:
                    nltk.download(resource, quiet=True)


_ensure_nltk_resources()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: map NLTK POS tags → WordNet POS constants
# ---------------------------------------------------------------------------

def _nltk_pos_to_wordnet(nltk_tag: str) -> str:
    """Convert NLTK POS tag prefix to WordNet POS tag."""
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    if nltk_tag.startswith("V"):
        return wordnet.VERB
    if nltk_tag.startswith("N"):
        return wordnet.NOUN
    if nltk_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # Default to noun


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Stateless NLP preprocessing engine.

    Parameters
    ----------
    config : PreprocessingConfig, optional
        Falls back to the global CONFIG.preprocessing if not provided.

    Usage
    -----
    >>> tp = TextPreprocessor()
    >>> result = tp.preprocess("She don't knows how to plays the guitar.")
    >>> result["tokens"]
    ['She', "don't", 'knows', 'how', 'to', 'plays', 'the', 'guitar', '.']
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.cfg = config or CONFIG.preprocessing
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words(self.cfg.language))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> Dict:
        """
        Full preprocessing pipeline for a single input string.

        Returns
        -------
        dict with keys:
            original        : original text
            cleaned         : lightly cleaned text (whitespace, encoding)
            tokens          : word-level tokens (preserves punctuation)
            pos_tags        : list of (token, POS-tag) tuples
            lemmas          : lemmatised token list
            features        : flat feature dict for the ML classifier
        """
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        pos_tags = self._pos_tag(tokens)
        lemmas = self._lemmatize(tokens, pos_tags)

        return {
            "original": text,
            "cleaned": cleaned,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "lemmas": lemmas,
            "features": self.extract_features(text, tokens, pos_tags, lemmas),
        }

    def extract_features(
        self,
        text: str,
        tokens: List[str],
        pos_tags: List[Tuple[str, str]],
        lemmas: List[str],
    ) -> Dict:
        """
        Derive a flat feature dictionary for the ML classifier.

        Features are deliberately language-agnostic statistics so the same
        extractor works after a multilingual model swap.
        """
        words = [t for t in tokens if t not in string.punctuation]
        pos_map = dict(pos_tags)

        # --- Lexical diversity ---
        ttr = len(set(words)) / max(len(words), 1)

        # --- Verb agreement indicators ---
        third_sg_verb_count = sum(
            1 for tok, tag in pos_tags if tag in ("VBZ",)
        )
        base_verb_count = sum(
            1 for tok, tag in pos_tags if tag in ("VB", "VBP")
        )
        modal_count = sum(1 for tok, tag in pos_tags if tag == "MD")
        past_tense_count = sum(
            1 for tok, tag in pos_tags if tag in ("VBD", "VBN")
        )
        aux_count = sum(
            1 for tok in words if tok.lower() in {
                "is", "are", "was", "were", "has", "have", "had",
                "do", "does", "did", "will", "would", "shall", "should",
                "may", "might", "can", "could",
            }
        )

        # --- Punctuation & structure ---
        punct_count = sum(1 for t in tokens if t in string.punctuation)
        avg_word_length = (
            sum(len(w) for w in words) / max(len(words), 1)
        )
        sentence_count = max(text.count(".") + text.count("!") + text.count("?"), 1)
        avg_sent_length = len(words) / sentence_count

        # --- Rare / OOV words (not in WordNet) ---
        oov_count = sum(
            1 for w in words
            if not wordnet.synsets(w.lower()) and w.lower() not in self._stop_words
        )

        # --- Repetition ---
        bigrams = list(zip(words[:-1], words[1:]))
        repeated_bigrams = sum(
            1 for a, b in bigrams if a.lower() == b.lower()
        )

        return {
            "num_tokens": len(tokens),
            "num_words": len(words),
            "type_token_ratio": round(ttr, 4),
            "avg_word_length": round(avg_word_length, 4),
            "avg_sent_length": round(avg_sent_length, 4),
            "third_sg_verb_count": third_sg_verb_count,
            "base_verb_count": base_verb_count,
            "modal_count": modal_count,
            "past_tense_count": past_tense_count,
            "aux_count": aux_count,
            "punct_count": punct_count,
            "oov_count": oov_count,
            "repeated_bigrams": repeated_bigrams,
            "sentence_count": sentence_count,
        }

    def batch_preprocess(self, texts: List[str]) -> List[Dict]:
        """Process a list of texts and return a list of result dicts."""
        return [self.preprocess(t) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Light normalisation: fix encoding artifacts, collapse whitespace."""
        text = text.encode("ascii", errors="ignore").decode()   # Strip non-ASCII
        text = re.sub(r"\s+", " ", text).strip()               # Collapse whitespace
        text = re.sub(r"http\S+|www\.\S+", "[URL]", text)       # Mask URLs
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Word-level tokenisation using NLTK punkt."""
        return word_tokenize(text)

    def _pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Assign Penn-Treebank POS tags to each token."""
        return pos_tag(tokens)

    def _lemmatize(
        self, tokens: List[str], pos_tags: List[Tuple[str, str]]
    ) -> List[str]:
        """Lemmatise tokens using WordNet, guided by POS tags."""
        if not self.cfg.lemmatize:
            return tokens
        return [
            self._lemmatizer.lemmatize(tok, _nltk_pos_to_wordnet(tag))
            for tok, tag in pos_tags
        ]


# ---------------------------------------------------------------------------
# ErrorPatternDetector
# ---------------------------------------------------------------------------

class ErrorPatternDetector:
    """
    Rule-based heuristic error detector.

    Complements the ML classifier by catching surface-level patterns that
    are trivially identifiable with regex or simple token inspection.

    Each method returns a list of finding dicts:
        {
            "error_type"  : str,
            "description" : str,
            "span"        : (start_char, end_char) | None,
            "suggestion"  : str | None,
        }
    """

    def __init__(self, config: Optional[RulesConfig] = None) -> None:
        self.cfg = config or CONFIG.rules
        self._repeated_word_re = re.compile(
            self.cfg.repeated_word_pattern, re.IGNORECASE
        )
        self._double_space_re = re.compile(self.cfg.double_space_pattern)

    def detect_all(self, text: str) -> List[Dict]:
        """Run every heuristic and aggregate findings."""
        findings: List[Dict] = []
        findings.extend(self.detect_repeated_words(text))
        findings.extend(self.detect_double_spaces(text))
        findings.extend(self.detect_homophones(text))
        findings.extend(self.detect_subject_verb_disagreement(text))
        findings.extend(self.detect_article_errors(text))
        return findings

    def detect_repeated_words(self, text: str) -> List[Dict]:
        findings = []
        for m in self._repeated_word_re.finditer(text):
            word = m.group(1)
            findings.append({
                "error_type": "Repeated Word",
                "description": f'Word "{word}" appears consecutively.',
                "span": (m.start(), m.end()),
                "suggestion": word,
            })
        return findings

    def detect_double_spaces(self, text: str) -> List[Dict]:
        findings = []
        for m in self._double_space_re.finditer(text):
            findings.append({
                "error_type": "Extra Whitespace",
                "description": "Multiple consecutive spaces detected.",
                "span": (m.start(), m.end()),
                "suggestion": " ",
            })
        return findings

    def detect_homophones(self, text: str) -> List[Dict]:
        """
        Flag common homophone confusions (their/there/they're, etc.).
        Extend this dict to add more language-specific pairs.
        """
        _HOMOPHONE_RULES = [
            (r"\btheir\b", "their/there/they're", "Check: 'their' (possessive), 'there' (place), 'they're' (they are)"),
            (r"\bthere\b", "their/there/they're", "Check: 'there' (place), 'their' (possessive), 'they're' (they are)"),
            (r"\bthey're\b", "their/there/they're", "Check: 'they're' = they are"),
            (r"\byour\b", "your/you're", "Check: 'your' (possessive), 'you're' (you are)"),
            (r"\byou're\b", "your/you're", "Check: 'you're' = you are"),
            (r"\bits\b", "its/it's", "Check: 'its' (possessive), 'it's' (it is)"),
            (r"\bit's\b", "its/it's", "Check: 'it's' = it is"),
            (r"\bthen\b", "then/than", "Check: 'then' (time), 'than' (comparison)"),
            (r"\bthan\b", "then/than", "Check: 'than' (comparison), 'then' (time)"),
            (r"\baffect\b", "affect/effect", "Check: 'affect' (verb), 'effect' (noun)"),
            (r"\beffect\b", "affect/effect", "Check: 'effect' (noun), 'affect' (verb)"),
        ]
        findings = []
        for pattern, error_type, description in _HOMOPHONE_RULES:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                findings.append({
                    "error_type": f"Possible Homophone Confusion ({error_type})",
                    "description": description,
                    "span": (m.start(), m.end()),
                    "suggestion": None,
                })
        return findings

    def detect_subject_verb_disagreement(self, text: str) -> List[Dict]:
        """
        Detect common subject–verb agreement patterns heuristically.
        """
        _DISAGREEMENT_PATTERNS = [
            (r"\bhe don't\b",    "Subject-Verb Disagreement", "'he don't' → 'he doesn't'", "he doesn't"),
            (r"\bshe don't\b",   "Subject-Verb Disagreement", "'she don't' → 'she doesn't'", "she doesn't"),
            (r"\bit don't\b",    "Subject-Verb Disagreement", "'it don't' → 'it doesn't'", "it doesn't"),
            (r"\bhe don't\b",    "Subject-Verb Disagreement", "'he don't' → 'he doesn't'", "he doesn't"),
            (r"\bI are\b",       "Subject-Verb Disagreement", "'I are' → 'I am'", "I am"),
            (r"\bI is\b",        "Subject-Verb Disagreement", "'I is' → 'I am'", "I am"),
            (r"\bthey was\b",    "Subject-Verb Disagreement", "'they was' → 'they were'", "they were"),
            (r"\bwe was\b",      "Subject-Verb Disagreement", "'we was' → 'we were'", "we were"),
            (r"\byou was\b",     "Subject-Verb Disagreement", "'you was' → 'you were'", "you were"),
            (r"\bhe were\b",     "Subject-Verb Disagreement", "'he were' → 'he was'", "he was"),
            (r"\bshe were\b",    "Subject-Verb Disagreement", "'she were' → 'she was'", "she was"),
            (r"\bit were\b",     "Subject-Verb Disagreement", "'it were' → 'it was'", "it was"),
        ]
        findings = []
        for pattern, error_type, description, suggestion in _DISAGREEMENT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                findings.append({
                    "error_type": error_type,
                    "description": description,
                    "span": (m.start(), m.end()),
                    "suggestion": suggestion,
                })
        return findings

    def detect_article_errors(self, text: str) -> List[Dict]:
        """Detect 'a' used before vowel sounds and 'an' before consonants."""
        _ARTICLE_PATTERNS = [
            (r"\ba ([aeiou]\w*)\b", "Article Error", "Use 'an' before vowel sounds."),
            (r"\ban ([^aeiou\s]\w*)\b", "Article Error", "Use 'a' before consonant sounds."),
        ]
        findings = []
        # Common exceptions (acronyms, etc.)
        _EXCEPTIONS = {"hour", "honor", "honest", "heir", "herb",
                       "universal", "uniform", "unicorn", "university",
                       "unique", "useful", "user", "unit"}
        for pattern, error_type, description in _ARTICLE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                word = m.group(1).lower()
                if word not in _EXCEPTIONS:
                    findings.append({
                        "error_type": error_type,
                        "description": f"{description} (found near '{m.group(0)}')",
                        "span": (m.start(), m.end()),
                        "suggestion": None,
                    })
        return findings
