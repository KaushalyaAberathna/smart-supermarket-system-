"""
category_mapping.py

MODULE 6 (pipeline stage 6 of 10) -- CATEGORY MAPPING

Maps each of the 25 fine-grained Freiburg Groceries class labels produced by
classification.py into a smaller set of supermarket-style categories, the
way a real checkout/inventory system groups SKUs into aisles/departments.

WHY these 6 categories (not the 5 in the original project brief):
The brief's example categories (Fruits, Dairy, Beverage, Snacks, Household)
were illustrative, not derived from this specific dataset. Freiburg
Groceries contains zero fruit and zero household/cleaning products -- it is
a pantry/grocery-item dataset (canned goods, dry goods, drinks, snacks).
Mapping to the brief's literal 5 categories would leave "Fruits" and
"Household" permanently empty, which teaches nothing and looks like a bug
in a demo. Instead, the categories below were chosen to (a) meaningfully
group all 25 actual classes, and (b) each be non-empty and populated by
several classes, so counts/statistics/charts in later modules are
meaningful on real data.

The mapping is a single dictionary so it is trivial to maintain: adding a
new product class only requires adding one line here.
"""

import config


# Freiburg class label -> supermarket category. Every class in
# config.CLASS_NAMES MUST appear here exactly once (enforced below).
CATEGORY_MAP = {
    # Bakery & Grains
    "CAKE": "Bakery & Grains",
    "CEREAL": "Bakery & Grains",
    "CORN": "Bakery & Grains",
    "FLOUR": "Bakery & Grains",
    "PASTA": "Bakery & Grains",
    "RICE": "Bakery & Grains",

    # Pantry & Condiments
    "BEANS": "Pantry & Condiments",
    "HONEY": "Pantry & Condiments",
    "JAM": "Pantry & Condiments",
    "OIL": "Pantry & Condiments",
    "SPICES": "Pantry & Condiments",
    "SUGAR": "Pantry & Condiments",
    "TOMATO_SAUCE": "Pantry & Condiments",
    "VINEGAR": "Pantry & Condiments",

    # Dairy
    "MILK": "Dairy",

    # Beverages
    "COFFEE": "Beverages",
    "JUICE": "Beverages",
    "SODA": "Beverages",
    "TEA": "Beverages",
    "WATER": "Beverages",

    # Snacks & Sweets
    "CANDY": "Snacks & Sweets",
    "CHIPS": "Snacks & Sweets",
    "CHOCOLATE": "Snacks & Sweets",
    "NUTS": "Snacks & Sweets",

    # Protein
    "FISH": "Protein",
}

CATEGORY_NAMES = sorted(set(CATEGORY_MAP.values()))

UNKNOWN_CATEGORY = "Unknown"  # for classifier labels below the confidence threshold


def _validate_mapping():
    """Fail fast at import time if CATEGORY_MAP and config.CLASS_NAMES ever
    drift out of sync (e.g. someone adds a new class folder to the dataset
    but forgets to map it here). Cheaper to catch this now than to silently
    lose products into a mis-count downstream.
    """
    mapped = set(CATEGORY_MAP.keys())
    expected = set(config.CLASS_NAMES)
    missing = expected - mapped
    extra = mapped - expected
    if missing:
        raise ValueError(f"category_mapping.py: no category mapping for classes: {sorted(missing)}")
    if extra:
        raise ValueError(f"category_mapping.py: mapping references unknown classes: {sorted(extra)}")


_validate_mapping()


def get_category(product_label):
    """Map a single classifier label (e.g. "MILK") to its supermarket
    category (e.g. "Dairy"). Unrecognised/"Unknown" labels map to
    UNKNOWN_CATEGORY rather than raising, since low-confidence predictions
    are an expected runtime occurrence, not a programming error.
    """
    return CATEGORY_MAP.get(product_label, UNKNOWN_CATEGORY)


def map_products(crops):
    """Add a "category" key to every crop dict (which must already have a
    "label" key from classification.py). Returns the same list.
    """
    for c in crops:
        c["category"] = get_category(c.get("label", UNKNOWN_CATEGORY))
    return crops


# --------------------------------------------------------------------------
# DEMO / SELF-TEST
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[category_mapping] {len(CATEGORY_MAP)} classes mapped into "
          f"{len(CATEGORY_NAMES)} categories:\n")
    for category in CATEGORY_NAMES:
        members = sorted(k for k, v in CATEGORY_MAP.items() if v == category)
        print(f"  {category}: {members}")

    # Simulate a few classified crops to show map_products() in action.
    fake_crops = [
        {"id": 1, "label": "MILK", "confidence": 0.92},
        {"id": 2, "label": "CANDY", "confidence": 0.81},
        {"id": 3, "label": "Unknown", "confidence": 0.31},
    ]
    print("\n[category_mapping] Example:")
    for c in map_products(fake_crops):
        print(f"  Product {c['id']}: {c['label']} -> {c['category']}")
