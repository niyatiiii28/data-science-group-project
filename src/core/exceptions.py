"""Custom exception hierarchy for the search engine."""


class SemanticSearchError(Exception):
    """Base exception for all semantic search errors."""


class DataValidationError(SemanticSearchError):
    """Raised when input data fails validation."""


class DataLoadError(SemanticSearchError):
    """Raised when dataset cannot be loaded."""


class ModelNotLoadedError(SemanticSearchError):
    """Raised when a search model is not initialized."""


class IndexNotBuiltError(SemanticSearchError):
    """Raised when FAISS index has not been built yet."""


class EmbeddingError(SemanticSearchError):
    """Raised when embedding generation fails."""


class LLMError(SemanticSearchError):
    """Raised when LLM API call fails."""


class QueryTooLongError(SemanticSearchError):
    """Raised when search query exceeds max length."""


class InvalidModelError(SemanticSearchError):
    """Raised when an unknown model ID is requested."""
