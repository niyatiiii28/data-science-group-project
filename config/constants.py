"""Enums, constants, and default values."""

from enum import Enum


class ModelID(str, Enum):
    HIERARCHICAL = "model1_hierarchical"
    FLAT_SEMANTIC = "model2_flat_semantic"
    ENHANCED_SEMANTIC = "model3_enhanced"
    LLM_ENHANCED = "model4_llm"


class Department(str, Enum):
    FASHION = "Fashion"
    ELECTRONICS = "Electronics"
    HOME_GARDEN = "Home & Garden"
    SPORTS = "Sports"
    BOOKS = "Books"
    AUTOMOTIVE = "Automotive"
    BABY = "Baby"
    BEAUTY = "Beauty"
    OTHER = "Other"


class StockStatus(str, Enum):
    IN_STOCK = "In Stock"
    OUT_OF_STOCK = "Out of Stock"
    LIMITED = "Limited Stock"


# Flipkart category tree separator
CATEGORY_TREE_SEPARATOR = " >> "

# Maximum hierarchy depth
MAX_HIERARCHY_DEPTH = 5

# Embedding batch size for SentenceTransformer
EMBEDDING_BATCH_SIZE = 256

# FAISS training threshold — use IVF only above this count. Flat IP is exact
# and fast enough at this size; IVF with nprobe<<nlist loses recall.
FAISS_IVF_THRESHOLD = 50000

# Score decay factor for hierarchical search
HIERARCHY_DECAY_FACTOR = 0.85

# Re-ranking weights for enhanced search (Model 3). Semantic dominates because
# rating/reviews/stock are largely empty or uniform on the Flipkart corpus.
RERANK_WEIGHTS = {
    "semantic_similarity": 0.85,
    "rating_score": 0.05,
    "reviews_score": 0.02,
    "price_relevance": 0.03,
    "stock_bonus": 0.05,
}

# Supported file extensions for dataset loading
SUPPORTED_EXTENSIONS = {".csv", ".json", ".parquet"}

# Text preprocessing
STOPWORDS_EXTRA = {
    "product", "item", "buy", "best", "good", "great",
    "need", "want", "looking", "find", "search",
}
