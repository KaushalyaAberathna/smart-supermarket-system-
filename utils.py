import os
import cv2

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def validate_image_path(image_path):
    """Check that image_path points to an existing file with a supported
    extension. Raises a specific, descriptive exception for each failure
    mode rather than a generic error, so the caller (main.py) can report
    exactly what was wrong with the input.
    """
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    if not os.path.isfile(image_path):
        raise ValueError(f"Image path is not a file: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}' for {image_path}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return True

def load_image(image_path):
    """Load a supermarket basket / product-layout image from disk.

    Validates the path first, then decodes it with OpenCV and validates the
    DECODED result too -- a truncated or corrupted file can have a valid
    path and extension yet still fail to decode (cv2.imread silently
    returns None in that case instead of raising), so both checks matter.
    """
    validate_image_path(image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Failed to decode image at '{image_path}' -- the file may be "
            "corrupted, empty, or not actually a valid image despite its extension."
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Image at '{image_path}' has zero width or height.")

    return image