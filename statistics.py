import category_mapping

class StatisticsAnalyzer:
    """Computes counting/percentage statistics for one image's detections."""

    def __init__(self, crops):
        """
        Parameters
        ----------
        crops : list[dict]
            Product crops that have already passed through classification.py
            ("label" key) and category_mapping.py ("category" key).
        """
        self.crops = crops
        self.total_products = len(crops)
        self.category_counts = self._compute_category_counts()
        self.category_percentages = self._compute_category_percentages()
        self.label_counts = self._compute_label_counts()

    def _compute_category_counts(self):
        """Zero-fill every known category up front (rather than only
        including categories that appear in this particular image) so
        downstream charts/reports always show the full, consistent set of
        categories -- a category with 0 items this run should show as 0%,
        not vanish from the report entirely.
        """
        counts = {cat: 0 for cat in category_mapping.CATEGORY_NAMES}
        for c in self.crops:
            cat = c.get("category", category_mapping.UNKNOWN_CATEGORY)
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _compute_category_percentages(self):
        """Percentage = (Category Count / Total Products) x 100.

        If no products were detected at all, every category is reported at
        0% rather than raising a division-by-zero error -- an empty basket
        photo is a valid (if unusual) input, not a programming error.
        """
        if self.total_products == 0:
            return {cat: 0.0 for cat in self.category_counts}
        return {
            cat: (count / self.total_products) * 100
            for cat, count in self.category_counts.items()
        }

    def _compute_label_counts(self):
        """Finer-grained counts per exact product label (e.g. "MILK": 2),
        not just per category. Not required by the brief's report format,
        but essentially free to compute here and genuinely useful for an
        itemised "smart checkout" style receipt in main.py.
        """
        counts = {}
        for c in self.crops:
            label = c.get("label", "Unknown")
            counts[label] = counts.get(label, 0) + 1
        return counts

    def summary(self):
        """Return every computed statistic as one plain dict -- convenient
        for passing to visualization.py, saving as JSON, or feeding into a
        report table.
        """
        return {
            "total_products": self.total_products,
            "category_counts": self.category_counts,
            "category_percentages": self.category_percentages,
            "label_counts": self.label_counts,
        }

    