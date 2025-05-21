import os
import cv2
import numpy as np
from pathlib import Path


def split_image(image_path, output_dir, chunk_size=512, counter_dict=None):
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    # Get image dimensions
    height, width = img.shape[:2]

    # Calculate number of chunks in each dimension
    chunks_h = height // chunk_size
    chunks_w = width // chunk_size

    # Get base filename without extension and determine if it's pre or post
    base_name = image_path.stem
    is_pre = "pre" in base_name.lower()
    prefix = "pre" if is_pre else "post"

    # Initialize counter for this image if not exists
    if counter_dict is None:
        counter_dict = {"pre": 0, "post": 0}

    # Split image into chunks
    for i in range(chunks_h):
        for j in range(chunks_w):
            # Calculate chunk coordinates
            y1 = i * chunk_size
            y2 = (i + 1) * chunk_size
            x1 = j * chunk_size
            x2 = (j + 1) * chunk_size

            # Extract chunk
            chunk = img[y1:y2, x1:x2]

            # Create output filename with sequential numbering
            chunk_name = f"{prefix}_{counter_dict[prefix]}.png"
            output_path = output_dir / chunk_name

            # Save chunk
            cv2.imwrite(str(output_path), chunk)
            counter_dict[prefix] += 1

    return counter_dict


def process_directory(input_dir):
    input_path = Path(input_dir)
    base_output_dir = Path("Archive/test/images_chunks")

    # Process each disaster event folder
    for event_dir in input_path.iterdir():
        if not event_dir.is_dir():
            continue

        print(f"Processing {event_dir.name}...")

        # Create output directory for this event
        event_output_dir = base_output_dir / event_dir.name
        event_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize counters for this event
        counter_dict = {"pre": 0, "post": 0}

        # Process all PNG files in the directory
        for img_file in event_dir.glob("*.png"):
            if img_file.name.endswith(".png"):
                print(f"Splitting {img_file.name}...")
                counter_dict = split_image(
                    img_file, event_output_dir, counter_dict=counter_dict
                )

                # Remove original file after splitting
                img_file.unlink()


if __name__ == "__main__":
    input_directory = "Archive/test/images"
    process_directory(input_directory)
