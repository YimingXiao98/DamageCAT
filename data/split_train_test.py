import os
import random
import shutil
from pathlib import Path


def get_image_ids(image_dir):
    # Get all pre-disaster image IDs
    pre_images = [f for f in os.listdir(image_dir) if f.startswith("pre_")]
    # Extract IDs from filenames (e.g., 'pre_123.png' -> '123')
    return [img.split("_")[1].split(".")[0] for img in pre_images]


def move_files(image_id, train_dir, test_dir):
    # Define file paths
    pre_train = os.path.join(train_dir, "images", f"pre_{image_id}.png")
    post_train = os.path.join(train_dir, "images", f"post_{image_id}.png")
    mask_train = os.path.join(train_dir, "masks", f"pre_{image_id}.png")

    pre_test = os.path.join(test_dir, "images", f"pre_{image_id}.png")
    post_test = os.path.join(test_dir, "images", f"post_{image_id}.png")
    mask_test = os.path.join(test_dir, "masks", f"pre_{image_id}.png")

    # Create test directories if they don't exist
    os.makedirs(os.path.dirname(pre_test), exist_ok=True)
    os.makedirs(os.path.dirname(mask_test), exist_ok=True)

    # Move files
    shutil.move(pre_train, pre_test)
    shutil.move(post_train, post_test)
    shutil.move(mask_train, mask_test)


def main():
    # Define paths
    base_dir = Path("damagecat")
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"

    # Get all image IDs
    image_ids = get_image_ids(train_dir / "images")

    # Calculate number of images to move (1/10 of total)
    num_test = len(image_ids) // 10

    # Randomly select images for test set
    random.seed(42)  # For reproducibility
    test_ids = random.sample(image_ids, num_test)

    print(f"Moving {num_test} images to test set...")

    # Move selected images to test set
    for image_id in test_ids:
        move_files(image_id, train_dir, test_dir)

    print("Done! Dataset split complete.")


if __name__ == "__main__":
    main()
