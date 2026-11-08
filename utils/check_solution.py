from PIL import Image
from pathlib import Path
import numpy as np



def is_resulting_image_correct(image: np.array, xlim, ylim) -> bool:
    """
    Compares if resulting image and reference image are the same. (Checks if the results are correct)
    Returns:
        True:  if the images match.
        False: if the images don't match.
        None:  if there's no reference image.
    """

    # Get reference image filename
    image_size_x, image_size_y = image.shape
    filename = Path(f"reference_outputs/{xlim[0]}:{xlim[1]}_{ylim[0]}:{ylim[1]}/{image_size_x}x{image_size_y}.png")

    # Check if such file exists
    if filename.exists() == False:
        return None

    else:
        # Upload reference image
        reference_image = np.array(Image.open(filename), dtype=np.uint8)

        # Check if both images are the same
        if np.array_equal(image, reference_image):
            return True
            
        else:
            return False
