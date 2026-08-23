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