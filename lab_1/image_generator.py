"""Contains logic for image generation."""
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.draw import line
from PIL import Image

from lab_1.consts import (
    DEFAULT_LINE_WIDTH, DEFAULT_LINE_LENGTH, IMAGE_SIZE,
    BASE_AMOUNT_OF_LINES, INTERSECTION_PROB, OUTPUT_NAME,
    MAX_NOISE_LEVEL
)


def generate_binary_image_with_border_crossing_lines(
        image_size: tuple[int, int] = IMAGE_SIZE,
        num_lines: int = BASE_AMOUNT_OF_LINES,
        line_length: int = DEFAULT_LINE_LENGTH,
        line_width: float | int = DEFAULT_LINE_WIDTH,
        intersect_prob: float = INTERSECTION_PROB
) -> np.ndarray:
    """Generates binary image without noise.

        1) Defines "region of interest" for controlling the intersections.
        2) Randomly selects start and end of each line, probably moving them into the "region".
        3) Draws lines using scikit-image.
        4) Fills closest pixels for controlling the width of each line.

    Args:
        image_size (tuple[int, int]): Image size. Defaults to 256X256.
        num_lines: (int) Number of lines. Defaults to 7.
        line_length (int): The length of each line. Defaults to 100.
        line_width (int | float): The width of each line. Defaults to 1.5.
        intersect_prob (float): Probability of line being sent to the "region of interest".

    Returns:
        np.ndarray: Binary image without noise.
    """
    image = np.zeros(image_size, dtype=np.uint8)

    center_x, center_y = image_size[1] // 2, image_size[0] // 2
    region_size = min(image_size) // 4

    for _ in range(num_lines):
        if random.random() < intersect_prob:
            x1 = random.randint(center_x - region_size, center_x + region_size)
            y1 = random.randint(center_y - region_size, center_y + region_size)
        else:
            x1, y1 = random.randint(0, image_size[1] - 1), random.randint(0, image_size[0] - 1)

        angle = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + line_length * np.cos(angle))
        y2 = int(y1 + line_length * np.sin(angle))

        rr, cc = line(y1, x1, y2, x2)

        half_width = line_width / 2.0

        for i in np.arange(-half_width, half_width + 0.1, 0.1):
            for j in np.arange(-half_width, half_width + 0.1, 0.1):
                rr_offset = rr + i
                cc_offset = cc + j

                valid_mask = (
                        (rr_offset >= 0) &
                        (rr_offset < image_size[0]) &
                        (cc_offset >= 0) &
                        (cc_offset < image_size[1])
                )
                image[rr_offset[valid_mask].astype(int), cc_offset[valid_mask].astype(int)] = 1

    return image


def add_varying_intensity_noise(image: np.ndarray, max_noise_level: float = MAX_NOISE_LEVEL) -> np.ndarray:
    """Add varying intensity binary noise to the image.

    Args:
        image (np.ndarray): The original binary image.
        max_noise_level (float): Maximum proportion of pixels to flip (between 0 and 1).
                                    Defaults to 35%.

    Returns:
        np.ndarray: The noisy binary image.
    """
    image_copy = image.copy()

    noise_intensity_map = np.random.rand(*image.shape) * max_noise_level

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.random() < noise_intensity_map[i, j]:
                image_copy[i, j] = 1 - image_copy[i, j]

    return image_copy


if __name__ == '__main__':
    binary_image = generate_binary_image_with_border_crossing_lines()
    noisy_image = add_varying_intensity_noise(binary_image)
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')
    plt.show()

    output_image = Image.fromarray(noisy_image * 255)
    output_image.save(OUTPUT_NAME)
