"""
Pipeline order and WHY:
    1. Resize        -> normalises working resolution so kernel sizes / area
                         thresholds behave consistently across input photos.
    2. Gaussian Blur  -> suppresses sensor noise and small texture/print
                         details on product packaging BEFORE thresholding,
                         so we threshold the object's shape, not its label art.
    3. Grayscale      -> thresholding and edge detection operate on single-
                         channel intensity, not 3-channel color.
    4. Threshold      -> converts intensity into a binary foreground/
                         background mask (the actual "find the objects" step).
    5. Canny (optional) -> an independent edge map, useful as a sanity check
                         or a fallback cue when thresholding is unreliable
                         (e.g. very uneven lighting).
    6. Morphology     -> cleans the binary mask: removes speckle noise,
                         closes small gaps/holes inside a single product's
                         silhouette (e.g. a light reflection splitting it into
                         two blobs), so each product becomes one solid contour.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config


class ImagePreprocessor:
    """Encapsulates the classical CV preprocessing pipeline and its tunables.

    Parameters are read from config.py.
    """

    

    def __init__(
        self,
        resize_width=config.PREPROCESS_RESIZE_WIDTH,
        gaussian_kernel=config.GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma=config.GAUSSIAN_SIGMA,
        threshold_method=config.THRESHOLD_METHOD,
        adaptive_block_size=config.ADAPTIVE_THRESH_BLOCK_SIZE,
        adaptive_c=config.ADAPTIVE_THRESH_C,
        binary_thresh_value=config.BINARY_THRESH_VALUE,
        invert=config.THRESHOLD_INVERT,
        canny_low=config.CANNY_LOW_THRESHOLD,
        canny_high=config.CANNY_HIGH_THRESHOLD,
        morph_kernel_size=config.MORPH_KERNEL_SIZE,
        morph_iterations=config.MORPH_ITERATIONS,
    ):
        self.resize_width = resize_width
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.threshold_method = threshold_method
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.binary_thresh_value = binary_thresh_value
        self.invert = invert
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel = np.ones(morph_kernel_size, dtype=np.uint8)
        self.morph_iterations = morph_iterations

    # STEP 1: RESIZE
    def resize(self, image):
        """Resize to a fixed working width, preserving aspect ratio.

        WHY: contour-area thresholds, blur kernel sizes and morphology kernel
        sizes are all expressed in pixels. If input photos come in wildly
        different resolutions (a phone photo vs. a scanned image), the same
        pixel-based thresholds would behave inconsistently. Normalising the
        width first makes every later step's constants meaningful.
        """
        
        h, w = image.shape[:2]
        if w == self.resize_width:
            return image.copy()
        scale = self.resize_width / float(w)
        new_size = (self.resize_width, int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # STEP 2: GAUSSIAN BLUR
    def gaussian_blur(self, image):
        """Smooth the image with a Gaussian kernel to remove high-frequency
        noise (sensor grain, JPEG artefacts, printed text/logos on packaging)
        BEFORE thresholding. Without this, thresholding would pick up every
        letter of a product label as a separate blob instead of one solid
        product silhouette.
        """
        return cv2.GaussianBlur(image, self.gaussian_kernel, self.gaussian_sigma)

    # STEP 3: GRAYSCALE  
    def to_grayscale(self, image):
        """Convert BGR -> single-channel grayscale.

        WHY: thresholding, Canny edge detection and morphology all operate on
        a scalar "intensity" value per pixel; they are not colour-aware.
        Reducing 3 channels to 1 also cuts computation for later steps.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # STEP 4: THRESHOLDING  
    def threshold(self, gray):
        """Binarize the grayscale image into foreground (product) vs.
        background (surface/table).

        Three interchangeable strategies (selected via config.THRESHOLD_METHOD):

        - "adaptive": threshold value is computed per local neighbourhood
          (adaptive_block_size). WHY: real photos rarely have uniform
          lighting across the whole frame (shadows, uneven overhead light);
          a single global cutoff would misclassify pixels in darker corners.
        - "otsu": automatically picks ONE global threshold that best
          separates the image's intensity histogram into two classes. WHY:
          fast and effective when lighting is fairly even.
        - "binary": a fixed manual cutoff. WHY: only reliable in a tightly
          controlled capture setup, kept here mainly for comparison/teaching.
        """

        invert_flag = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY

        if self.threshold_method == "adaptive":
            block_size = self.adaptive_block_size
            if block_size % 2 == 0:
                block_size += 1  # cv2 requires an odd block size
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                invert_flag, block_size, self.adaptive_c,
            )
        elif self.threshold_method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, invert_flag + cv2.THRESH_OTSU
            )
        elif self.threshold_method == "binary":
            _, binary = cv2.threshold(
                gray, self.binary_thresh_value, 255, invert_flag
            )
        else:
            raise ValueError(f"Unknown threshold_method: {self.threshold_method}")

        return binary

        # STEP 5: CANNY EDGE DETECTION (optional diagnostic)
    def canny_edges(self, gray):
        """Detect edges via the Canny operator.

        WHY: this is not used to build the detection mask directly (a binary
        threshold mask is more robust for solid product silhouettes), but it
        is a useful independent visualization for the report/demo, and a
        diagnostic to spot-check whether product boundaries are contrasty
        enough for thresholding to work well.
        """
        return cv2.Canny(gray, self.canny_low, self.canny_high)

    
    # STEP 6: MORPHOLOGICAL OPERATIONS
    
    def erode(self, binary):
        """Shrinks white regions. WHY: strips away thin noise speckles and
        separates two products that are touching by a thin bridge of pixels.
        """
        return cv2.erode(binary, self.morph_kernel, iterations=self.morph_iterations)

    def dilate(self, binary):
        """Grows white regions. WHY: re-thickens object silhouettes after
        erosion, and on its own can bridge small gaps within one product.
        """
        return cv2.dilate(binary, self.morph_kernel, iterations=self.morph_iterations)

    def opening(self, binary):
        """Erosion followed by dilation. WHY: removes small white noise
        specks (salt noise) from the background while restoring the size of
        genuine (larger) product blobs -- noise disappears during erosion
        and never comes back during dilation because it was fully erased.
        """
        return cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.morph_iterations
        )

    def closing(self, binary):
        """Dilation followed by erosion. WHY: fills small black holes INSIDE
        a product's silhouette (e.g. a specular highlight or printed text
        creating a dark gap) without growing the object's outer boundary,
        so cv2.findContours later sees one solid blob per product.
        """
        return cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.morph_iterations
        )

    def clean_mask(self, binary):
        """Recommended cleanup chain fed into Module 3 (detection.py):
        closing first (fill internal holes) then opening (strip stray
        background speckles). This order is deliberate -- opening first
        could erase a thin but genuine part of a product before closing
        gets a chance to repair it.
        """
        closed = self.closing(binary)
        cleaned = self.opening(closed)
        return cleaned

       
    # FULL PIPELINE
    
    def run(self, image):
        """Run every preprocessing step in order and return all intermediate
        results plus the final cleaned binary mask, keyed by step name.

        Returns
        -------
        dict[str, np.ndarray] with keys (in pipeline order):
            original, resized, blurred, gray, threshold, canny,
            eroded, dilated, opened, closed, final_mask
        """
        if image is None:
            raise ValueError("ImagePreprocessor.run() received a None image.")

        resized = self.resize(image)
        blurred = self.gaussian_blur(resized)
        gray = self.to_grayscale(blurred)
        binary = self.threshold(gray)
        edges = self.canny_edges(gray)

        eroded = self.erode(binary)
        dilated = self.dilate(binary)
        opened = self.opening(binary)
        closed = self.closing(binary)
        final_mask = self.clean_mask(binary)

        return {
            "original": image,
            "resized": resized,
            "blurred": blurred,
            "gray": gray,
            "threshold": binary,
            "canny": edges,
            "eroded": eroded,
            "dilated": dilated,
            "opened": opened,
            "closed": closed,
            "final_mask": final_mask,
        }
    
# VISUALIZATION HELPER

def visualize_steps(steps, save_path=None, show=config.SHOW_PLOTS):
    """Display every preprocessing step in one figure grid, and optionally
    save it to disk. Satisfies the "display intermediate results for every
    preprocessing step" requirement.
    """
    titles = [
        "1. Resized", "2. Gaussian Blur", "3. Grayscale", "4. Threshold",
        "5. Canny Edges", "6a. Eroded", "6b. Dilated", "6c. Opened",
        "6d. Closed", "Final Mask (Closing->Opening)",
    ]
    keys = [
        "resized", "blurred", "gray", "threshold", "canny",
        "eroded", "dilated", "opened", "closed", "final_mask",
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    for ax, title, key in zip(axes, titles, keys):
        img = steps[key]
        if img.ndim == 3:  # BGR color image -> convert for correct matplotlib colors
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:  # single-channel binary/grayscale image
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[preprocessing] Saved preprocessing steps figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def preprocess_image(image, display=False, save_path=None):
    """Convenience function: run the full pipeline with default config
    settings and optionally display/save the intermediate steps. This is the
    function other modules (detection.py) should import and call.
    """
    preprocessor = ImagePreprocessor()
    steps = preprocessor.run(image)
    if display or save_path:
        visualize_steps(steps, save_path=save_path, show=display)
    return steps

# DEMO / SELF-TEST

if __name__ == "__main__":
    # Prefer a user-supplied basket/table photo from images/, since that is
    # what this pipeline is actually designed for. If none has been added
    # yet, fall back to a single Freiburg dataset sample purely so this
    
    demo_image_path = None
    if os.path.isdir(config.TEST_IMAGES_DIR):
        candidates = [
            f for f in os.listdir(config.TEST_IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if candidates:
            demo_image_path = os.path.join(config.TEST_IMAGES_DIR, candidates[0])

    if demo_image_path is None:
        fallback_class = config.CLASS_NAMES[0]  # "BEANS"
        fallback_dir = os.path.join(config.DATASET_DIR, fallback_class)
        fallback_file = sorted(os.listdir(fallback_dir))[0]
        demo_image_path = os.path.join(fallback_dir, fallback_file)
        print(
            "[preprocessing] No image found in images/. Using a Freiburg "
            f"dataset sample instead for this demo: {demo_image_path}\n"
            "[preprocessing] Add a real basket/table photo to images/ to "
            "test the pipeline on its intended input."
        )

    demo_image = cv2.imread(demo_image_path)
    if demo_image is None:
        raise FileNotFoundError(f"Could not read demo image: {demo_image_path}")

    output_path = os.path.join(config.OUTPUT_DIR, "preprocessing_steps_demo.png")
    preprocess_image(demo_image, display=config.SHOW_PLOTS, save_path=output_path)




    

    