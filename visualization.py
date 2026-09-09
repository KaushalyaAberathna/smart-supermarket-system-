import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
import category_mapping


# --------------------------------------------------------------------------
# CATEGORY COLOR PALETTE
# --------------------------------------------------------------------------
# Fixed hue per category (validated categorical palette, assigned in a
# stable order -- see category_mapping.CATEGORY_NAMES, which is alphabetical
# and has exactly 6 entries). Hex strings are used for matplotlib; BGR
# tuples (derived below) are used for OpenCV drawing.
CATEGORY_COLORS_HEX = {
    "Bakery & Grains":     "#2a78d6",  # blue
    "Beverages":           "#008300",  # green
    "Dairy":               "#e87ba4",  # magenta
    "Pantry & Condiments": "#eda100",  # yellow
    "Protein":             "#1baf7a",  # aqua
    "Snacks & Sweets":     "#eb6834",  # orange
}
UNKNOWN_COLOR_HEX = "#898781"  # neutral gray -- not a real category, so no categorical hue


def _hex_to_bgr(hex_color):
    """Convert '#rrggbb' to an OpenCV-style (B, G, R) integer tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)

def _category_color_hex(category):
    return CATEGORY_COLORS_HEX.get(category, UNKNOWN_COLOR_HEX)

def _readable_text_color(bgr_color):
    """Pick black or white text for readability against a given BGR
    background, using standard relative-luminance weighting.
    """
    b, g, r = bgr_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


# --------------------------------------------------------------------------
# FINAL ANNOTATED IMAGE
# --------------------------------------------------------------------------
def draw_final_annotations(image, crops):
    """Draw a bounding box plus a two-line label chip (product label on top,
    category below) for every classified+mapped crop, e.g.:

        +----------------------+
        | MILK                 |
        | Dairy                |
        +----------------------+

    The chip is filled with the product's category color, so categories are
    visually distinguishable at a glance without reading text.
    """
    annotated = image.copy()
    font = getattr(cv2, config.LABEL_FONT)

    for c in crops:
        x, y, w, h = c["bbox"]
        category = c.get("category", category_mapping.UNKNOWN_CATEGORY)
        color_bgr = _hex_to_bgr(_category_color_hex(category))
        text_color = _readable_text_color(color_bgr)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, config.BOUNDING_BOX_THICKNESS)

        line1 = str(c.get("label", "Unknown"))
        line2 = str(category)
        (w1, h1), base1 = cv2.getTextSize(line1, font, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS)
        (w2, h2), base2 = cv2.getTextSize(line2, font, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS)

        chip_w = max(w1, w2) + 10
        chip_h = h1 + h2 + base1 + base2 + 10
        chip_top = max(0, y - chip_h)

        cv2.rectangle(annotated, (x, chip_top), (x + chip_w, y), color_bgr, -1)
        cv2.putText(
            annotated, line1, (x + 5, chip_top + h1 + 4), font,
            config.LABEL_FONT_SCALE, text_color, config.LABEL_THICKNESS, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, line2, (x + 5, chip_top + h1 + h2 + base1 + 6), font,
            config.LABEL_FONT_SCALE, text_color, config.LABEL_THICKNESS, cv2.LINE_AA,
        )

    return annotated


def show_final_image(annotated_image, save_path=None, show=config.SHOW_PLOTS):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Smart Checkout -- Final Detection Result")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved final annotated image to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------------
# CONSOLE REPORT
# ----------------------------------------------------------------------------
def print_console_report(stats):
    """Print the "SMART CHECKOUT REPORT" console summary.

    All 6 real categories are always listed (even at 0), matching
    statistics.py's zero-filled counts, so the report is consistent across
    runs. "Unknown" (low-confidence classifications) is only shown if it
    actually occurred this run -- it isn't a real product category.
    """
    total = stats["total_products"]
    counts = stats["category_counts"]
    percentages = stats["category_percentages"]

    categories = list(category_mapping.CATEGORY_NAMES)
    if counts.get(category_mapping.UNKNOWN_CATEGORY, 0) > 0:
        categories.append(category_mapping.UNKNOWN_CATEGORY)

    name_width = max(len(c) for c in categories) + 2

    print("=" * 28)
    print("SMART CHECKOUT REPORT")
    print("=" * 28)
    print(f"Total Products : {total}\n")

    for cat in categories:
        print(f"{cat:<{name_width}}: {counts.get(cat, 0)}")

    print("\nDistribution")
    for cat in categories:
        pct = percentages.get(cat, 0.0)
        print(f"{cat:<{name_width}}{round(pct)}%")

    print("=" * 28)

# --------------------------------------------------------------------------
# GRAPHS
# --------------------------------------------------------------------------
def plot_bar_chart(stats, save_path=None, show=config.SHOW_PLOTS):
    """Bar chart of product count per category. Every category is shown
    (including zero-count ones) so the chart's category set is stable
    across runs; each bar is colored by its category and value-labeled
    directly above it (selective direct labeling, no separate legend needed
    since the x-axis already names each category).
    """
    categories = list(category_mapping.CATEGORY_NAMES)
    if stats["category_counts"].get(category_mapping.UNKNOWN_CATEGORY, 0) > 0:
        categories.append(category_mapping.UNKNOWN_CATEGORY)
    counts = [stats["category_counts"].get(c, 0) for c in categories]
    colors = [_category_color_hex(c) for c in categories]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(categories, counts, color=colors, width=0.6)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
            str(count), ha="center", va="bottom", fontsize=10, color="#0b0b0b",
        )

    ax.set_title("Product Count by Category", fontsize=13, color="#0b0b0b")
    ax.set_ylabel("Count", color="#52514e")
    ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved bar chart to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_pie_chart(stats, save_path=None, show=config.SHOW_PLOTS):
    """Pie chart of category distribution. Zero-count categories are
    excluded (an invisible 0% wedge would only clutter the legend); a
    legend is included since pie slices have no axis to label identity by.
    """
    categories = [
        c for c in list(category_mapping.CATEGORY_NAMES) + [category_mapping.UNKNOWN_CATEGORY]
        if stats["category_counts"].get(c, 0) > 0
    ]
    counts = [stats["category_counts"][c] for c in categories]
    colors = [_category_color_hex(c) for c in categories]

    if not counts:
        print("[visualization] No products detected -- skipping pie chart.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        counts, colors=colors, autopct="%1.0f%%", pctdistance=0.75,
        startangle=90, wedgeprops={"edgecolor": "#fcfcfb", "linewidth": 2},
    )
    for t in autotexts:
        t.set_color("#0b0b0b")
        t.set_fontsize(10)

    ax.set_title("Category Distribution", fontsize=13, color="#0b0b0b")
    ax.legend(wedges, categories, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.axis("equal")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved pie chart to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# --------------------------------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------------------------------
def visualize_final_output(image, crops, stats, output_dir=config.OUTPUT_DIR, display=config.SHOW_PLOTS):
    """Run every visualization step and save all four artifacts to
    output_dir: final_annotated.png, bar_chart.png, pie_chart.png, plus the
    console report printed to stdout.
    """
    annotated = draw_final_annotations(image, crops)
    show_final_image(annotated, save_path=os.path.join(output_dir, "final_annotated.png"), show=display)

    print_console_report(stats)

    plot_bar_chart(stats, save_path=os.path.join(output_dir, "bar_chart.png"), show=display)
    plot_pie_chart(stats, save_path=os.path.join(output_dir, "pie_chart.png"), show=display)

    return annotated