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

