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


# Flipkart boilerplate patterns that appear in nearly every description and
# dominate embeddings if not removed. Order matters: longer phrases first.
_BOILERPLATE_PATTERNS = [
    r"buy\s+.*?\s+(?:online|for\s+rs\.?\s*[\d,]+(?:\.\d+)?)",
    r"only\s+for\s+rs\.?\s*[\d,]+(?:\.\d+)?",
    r"price\s*[:\-]?\s*rs\.?\s*[\d,]+(?:\.\d+)?",
    r"at\s+best\s+price[s]?",
    r"from\s+flipkart\.com",
    r"on\s+flipkart\.com",
    r"flipkart\.com",
    r"only\s+genuine\s+products?\.?",
    r"30\s*day\s+replacement\s+guarantee\.?",
    r"free\s+shipping\.?",
    r"cash\s+on\s+delivery\.?",
    r"\bcod\b",
    r"key\s+features\s+of\s+[^:.,]*[:.,]",
    r"specifications?\s+of\s+[^:.,]*[:.,]",
    r"general\s+in\s+the\s+box\s+[^:.,]*[:.,]?",
    r"\(pack\s+of\s+\d+\)",
    r"\b[a-z0-9]{10,}\b",  # long alphanumeric SKU-like tokens
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)


def strip_boilerplate(text: str) -> str:
    """Remove Flipkart marketing/boilerplate phrases from product text."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = _BOILERPLATE_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", cleaned)
    return cleaned.strip(" ,.;:-")


def extract_combined_text(row: dict) -> str:
    """Combine product fields into a single searchable text (boilerplate-stripped)."""
    description = strip_boilerplate(str(row.get("description", "")))
    features = strip_boilerplate(str(row.get("features", "")))
    parts = [
        f"Product: {row.get('product_name', '')}",
        f"Category: {row.get('department', '')} {row.get('category', '')} {row.get('subcategory', '')}",
        f"Type: {row.get('product_type', '')}",
        f"Brand: {row.get('brand', '')}",
        f"Features: {features}",
        f"Description: {description}",
        f"Tags: {row.get('tags', '')}",
    ]
    return " ".join(p for p in parts if p.split(": ", 1)[-1].strip())
