import os
import re
import string
import logging
import joblib

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------------------------- #
#                           Logging set-up                              #
# --------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
#                       TaskPriorityPredictor                           #
# --------------------------------------------------------------------- #
class TaskPriorityPredictor:
    """
    Loads the trained Random-Forest model and TF-IDF vectorizer, applies the
    same pre-processing steps that were run during training, and exposes a
    simple `predict_priority()` API.
    """

    # ● Compiled regexes reused across calls (faster than rebuilding each time)
    _URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
    _EMAIL_RE = re.compile(r"\S+@\S+")
    _NUM_RE   = re.compile(r"\b\d+\b")

    def __init__(self) -> None:
        here = os.path.dirname(__file__)
        self.model_path      = os.path.join(here, "models", "priority_classifier_final.pkl")
        self.vectorizer_path = os.path.join(here, "models", "tfidf_vectorizer_final.pkl")

        self.model:      "RandomForestClassifier | None" = None
        self.vectorizer: "TfidfVectorizer | None"       = None

        self._load_artifacts()

    # ----------------------------------------------------------------- #
    #                        Pre-processing logic                        #
    # ----------------------------------------------------------------- #
    @classmethod
    def _clean_text(cls, text: str) -> str:
        """
        Reproduces the *exact* cleaning steps performed in your Colab notebook
        (see `ai_task_deployement_version.py`).

        1. Lower-case everything
        2. Strip URLs
        3. Strip e-mail addresses
        4. Remove stand-alone numbers
        5. Remove punctuation
        6. Collapse multiple spaces
        """
        text = text.lower()
        text = cls._URL_RE.sub(" ", text)
        text = cls._EMAIL_RE.sub(" ", text)
        text = cls._NUM_RE.sub(" ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    @classmethod
    def _batch_clean(cls, texts: List[str]) -> List[str]:
        """Vectorised convenience method (keeps API symmetrical)."""
        return [cls._clean_text(t) for t in texts]

    # ----------------------------------------------------------------- #
    #                        Artifact loading                            #
    # ----------------------------------------------------------------- #
    def _load_artifacts(self) -> None:
        """Load model + vectorizer from disk; raise fast if anything is missing."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found → {self.model_path}")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found → {self.vectorizer_path}")

        logger.info("Loading Random-Forest model …")
        self.model = joblib.load(self.model_path)

        logger.info("Loading TF-IDF vectorizer …")
        self.vectorizer = joblib.load(self.vectorizer_path)

        logger.info("Artifacts loaded successfully.")

    # ----------------------------------------------------------------- #
    #                            Inference                               #
    # ----------------------------------------------------------------- #
    def predict_priority(self, task_text: str) -> str:
        """
        Parameters
        ----------
        task_text : str
            Raw task description (no manual cleaning required).

        Returns
        -------
        str
            Human-readable prediction, e.g. `"Predicted Priority: High"`.
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Artifacts not loaded correctly.")

        cleaned = self._clean_text(task_text)
        features = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(features)[0]

        return f"Predicted Priority: {prediction}"


# --------------------------------------------------------------------- #
#                     Singleton instance for import-reuse               #
# --------------------------------------------------------------------- #
predictor = TaskPriorityPredictor()