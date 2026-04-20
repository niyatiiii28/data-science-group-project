"""Curated test query sets with CATEGORY-based ground truth.

Each query is (query_string, relevant_terms) where relevant_terms is a set of
lower-case keywords matched (as substrings) against a result's department,
category, subcategory, product_type, or product_name. A result counts as
relevant if ANY keyword in the set matches ANY of those fields.

This is dataset-agnostic: it works on the full Flipkart corpus (~12k products)
without requiring hand-labelled product-ID ground truth.
"""

# Exact queries - user names a clear product type / brand / attribute.
EXACT_QUERIES = [
    ("Nike running shoes for men", {"running", "sports shoe", "athletic", "sneaker"}),
    ("cotton formal shirt for office", {"formal shirt", "men's shirt", "formal wear"}),
    ("leather wallet for men", {"wallet"}),
    ("Samsung LED smart television", {"television", "tv", "led tv", "smart tv"}),
    ("front load washing machine", {"washing machine"}),
    ("Kindle e-reader tablet", {"ebook reader", "tablet", "ereader", "kindle"}),
    ("Adidas track pants for men", {"track pant", "sportswear", "men's clothing"}),
    ("stainless steel kitchen knife", {"knife", "kitchen tool", "cutlery"}),
    ("yoga mat for exercise", {"yoga", "exercise mat", "fitness accessor"}),
    ("Canon DSLR camera", {"camera", "dslr", "point & shoot", "mirrorless"}),
    ("bluetooth wireless headphones", {"headphone", "headset", "earphone"}),
    ("cotton king size bed sheet", {"bed sheet", "bedsheet", "bed linen"}),
    ("baby boy clothing set", {"baby", "infant wear", "kids' clothing"}),
    ("mobile phone back cover case", {"mobile case", "mobile cover", "back cover"}),
    ("gold plated necklace for women", {"necklace", "jewellery", "pendant"}),
]

# Semantic queries - user describes attributes / use-cases with different words.
SEMANTIC_QUERIES = [
    ("comfortable athletic footwear for marathon training", {"running", "sports shoe", "athletic"}),
    ("professional business attire for office meetings", {"formal shirt", "formal wear", "blazer", "suit"}),
    ("large screen for home entertainment", {"television", "tv", "home theatre", "projector"}),
    ("device for reading digital books", {"ebook reader", "tablet", "kindle"}),
    ("cookware for preparing meals", {"cookware", "pan", "pot", "kitchen"}),
    ("equipment for professional photography", {"camera", "dslr", "lens", "tripod"}),
    ("accessory to protect mobile phone screen", {"screen guard", "tempered glass", "screen protector"}),
    ("audio gear for private music listening", {"headphone", "earphone", "headset"}),
    ("bed linen for double bed", {"bed sheet", "bedsheet", "bed linen"}),
    ("ethnic wear for wedding occasions", {"saree", "kurta", "lehenga", "ethnic"}),
    ("fitness gear for home workouts", {"yoga", "gym", "fitness", "dumbbell", "exercise"}),
    ("portable backup charger for phones", {"power bank", "charger"}),
    ("skincare product for daily routine", {"skin care", "cream", "lotion", "moistur"}),
]

# Vague queries - conversational intent, no product words.
VAGUE_QUERIES = [
    ("something to watch movies on", {"television", "tv", "home theatre", "projector", "monitor"}),
    ("something to wear on my feet for exercise", {"running", "sports shoe", "athletic", "sneaker"}),
    ("something formal to wear to work", {"formal shirt", "formal wear", "blazer", "trouser"}),
    ("something to keep my money organized", {"wallet", "purse", "card holder"}),
    ("want something big to watch movies", {"television", "tv", "home theatre", "projector"}),
    ("need help with household laundry", {"washing machine", "detergent", "iron"}),
    ("want to read digitally", {"ebook reader", "tablet", "kindle"}),
    ("comfortable clothes for working out", {"sportswear", "track pant", "t-shirt", "sports"}),
    ("tools for preparing food", {"knife", "cookware", "kitchen", "utensil"}),
    ("something for relaxation and stretching", {"yoga", "exercise mat", "fitness"}),
    ("want to take better pictures", {"camera", "dslr", "lens", "mirrorless"}),
    ("how do I charge my phone on the go", {"power bank", "charger", "cable"}),
    ("what should I gift a 5-year-old girl", {"kids", "girls", "toy", "doll", "children"}),
    ("i need to listen to music privately", {"headphone", "earphone", "headset"}),
]

ALL_QUERY_SETS = {
    "exact": EXACT_QUERIES,
    "semantic": SEMANTIC_QUERIES,
    "vague": VAGUE_QUERIES,
}


def is_relevant(result_fields: dict, relevant_terms: set) -> bool:
    """True if any relevant term is a substring of any text field of the result."""
    haystack = " ".join(
        str(result_fields.get(f, "")).lower()
        for f in ("department", "category", "subcategory", "product_type", "product_name")
    )
    return any(term.lower() in haystack for term in relevant_terms)
