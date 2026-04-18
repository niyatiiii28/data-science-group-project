"""Curated test query sets with ground truth product IDs."""

# Format: list of (query_string, set_of_relevant_product_ids)
# These match the 10-product sample from the proposal.
# For a full dataset, expand proportionally.

EXACT_QUERIES = [
    ("Nike Running Shoes Black", {"P001"}),
    ("Blue Cotton Office Shirt formal wear", {"P002"}),
    ("Brown Leather Wallet RFID", {"P003"}),
    ("Samsung 65 inch QLED TV smart", {"P004"}),
    ("Bosch Washing Machine 7kg automatic", {"P005"}),
    ("Kindle E-Reader White waterproof", {"P006"}),
    ("Adidas Running Track Pants Black", {"P007"}),
    ("Stainless Steel Kitchen Knife Set", {"P008"}),
    ("Yoga Mat Non-Slip Brown", {"P009"}),
    ("Canon DSLR Camera 24MP professional", {"P010"}),
]

SEMANTIC_QUERIES = [
    ("Best athletic footwear for marathon training", {"P001"}),
    ("Formal business attire for office meetings", {"P002"}),
    ("Wallet with security features for daily use", {"P003"}),
    ("Large television for home entertainment with smart features", {"P004"}),
    ("Energy-efficient laundry solution for household cleaning", {"P005"}),
    ("Device for reading digital books with extended battery", {"P006"}),
    ("Comfortable athletic wear for fitness activities", {"P007"}),
    ("Professional cooking utensils for kitchen preparation", {"P008"}),
    ("Exercise mat for meditation and wellness", {"P009"}),
    ("Professional equipment for photography and videography", {"P010"}),
]

VAGUE_QUERIES = [
    ("I need something to wear on my feet for exercise", {"P001"}),
    ("Looking for something formal to wear to work", {"P002"}),
    ("Need something to keep my money organized", {"P003"}),
    ("Want something big to watch movies on", {"P004"}),
    ("Need help with household laundry tasks", {"P005"}),
    ("Want to read books digitally", {"P006"}),
    ("Comfortable clothes for working out", {"P007"}),
    ("Tools for preparing food", {"P008"}),
    ("Something for relaxation and exercise", {"P009"}),
    ("Want to take better pictures and videos", {"P010"}),
]

ALL_QUERY_SETS = {
    "exact": EXACT_QUERIES,
    "semantic": SEMANTIC_QUERIES,
    "vague": VAGUE_QUERIES,
}
