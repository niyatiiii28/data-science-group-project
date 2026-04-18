"""Text preprocessing utilities: tokenization, lemmatization, stopwords, Hinglish."""

import re
import logging
from functools import lru_cache

import nltk

logger = logging.getLogger("semantic_search")

# Download NLTK data on first import
_NLTK_READY = False


def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    for resource in ["punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_READY = True


def preprocess_query(text: str) -> str:
    """Full preprocessing pipeline for search queries."""
    _ensure_nltk()
    text = text.lower().strip()
    text = normalize_hinglish(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text.strip()


def preprocess_product_text(text: str) -> str:
    """Lighter preprocessing for product descriptions (preserve more info)."""
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)       # HTML
    text = re.sub(r"https?://\S+", "", text)    # URLs
    text = re.sub(r"[^\w\s.,;:!?₹$%-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_special_chars(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    _ensure_nltk()
    from nltk.corpus import stopwords
    from config.constants import STOPWORDS_EXTRA

    stop_words = set(stopwords.words("english")) | STOPWORDS_EXTRA
    words = text.split()
    return " ".join(w for w in words if w not in stop_words)


def lemmatize(text: str) -> str:
    _ensure_nltk()
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


# Common Hinglish → English mappings
_HINGLISH_MAP = {
    "accha": "good",
    "sasta": "cheap",
    "mehenga": "expensive",
    "kapde": "clothes",
    "joote": "shoes",
    "phone": "phone",
    "ghar": "home",
    "khana": "food",
    "paani": "water",
    "sundar": "beautiful",
    "mazboot": "strong durable",
    "aaram": "comfortable",
    "naya": "new",
    "purana": "old",
    "bada": "big large",
    "chhota": "small",
    "safed": "white",
    "kala": "black",
    "lal": "red",
    "neela": "blue",
    "peela": "yellow",
    "hara": "green",
    "badhiya": "excellent quality",
    "tikau": "durable",
    "halka": "lightweight",
    "kamra": "room",
    "rasoi": "kitchen",
    "bistar": "bed bedding",
    "kurta": "kurta ethnic wear",
    "saree": "saree ethnic wear",
    "jeans": "jeans denim",
    "tshirt": "t-shirt",
}


def normalize_hinglish(text: str) -> str:
    """Replace common Hinglish words with English equivalents."""
    words = text.lower().split()
    result = []
    for w in words:
        replacement = _HINGLISH_MAP.get(w)
        if replacement:
            result.append(replacement)
        else:
            result.append(w)
    return " ".join(result)


def extract_combined_text(row: dict) -> str:
    """Combine product fields into a single searchable text."""
    parts = [
        f"Product: {row.get('product_name', '')}",
        f"Category: {row.get('department', '')} {row.get('category', '')} {row.get('subcategory', '')}",
        f"Type: {row.get('product_type', '')}",
        f"Brand: {row.get('brand', '')}",
        f"Features: {row.get('features', '')}",
        f"Description: {row.get('description', '')}",
        f"Tags: {row.get('tags', '')}",
    ]
    return " ".join(p for p in parts if p.split(": ", 1)[-1].strip())
