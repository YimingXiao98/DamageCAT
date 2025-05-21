import os
import random
import shutil
from pathlib import Path


def get_image_ids(image_dir):
    # Get all pre-disaster image IDs
    pre_images = [f for f in os.listdir(image_dir) if f.startswith("pre_")]
    # Extract IDs from filenames (e.g., 'pre_123.png' -> '123')
    return [img.split("_")[1].split(".")[0] for img in pre_images]


def copy_files(image_id, source_dir, dest_dir):
    # Define source file paths
    pre_source = os.path.join(source_dir, "images", f"pre_{image_id}.png")
    post_source = os.path.join(source_dir, "images", f"post_{image_id}.png")
    mask_source = os.path.join(source_dir, "masks", f"post_{image_id}.png")

    # Define destination file paths
    pre_dest = os.path.join(dest_dir, "images", f"pre_{image_id}.png")
    post_dest = os.path.join(dest_dir, "images", f"post_{image_id}.png")
    mask_dest = os.path.join(dest_dir, "masks", f"post_{image_id}.png")

    # Create destination directories if they don't exist
    os.makedirs(os.path.dirname(pre_dest), exist_ok=True)
    os.makedirs(os.path.dirname(mask_dest), exist_ok=True)

    # Copy files
    shutil.copy2(pre_source, pre_dest)
    shutil.copy2(post_source, post_dest)
    shutil.copy2(mask_source, mask_dest)


def main():
    # Define paths
    source_dir = Path("BD_TypoSAT")
    dest_base_dir = Path("damagecat")
    train_dest_dir = dest_base_dir / "train"
    test_dest_dir = dest_base_dir / "test"

    # Create destination directories
    train_dest_dir.mkdir(parents=True, exist_ok=True)
    test_dest_dir.mkdir(parents=True, exist_ok=True)

    # Get all image IDs from source directory
    image_ids = get_image_ids(source_dir / "images")
    total_images = len(image_ids)

    # Calculate number of images for test set (1/10 of total)
    num_test = total_images // 10

    # Randomly select images for test set
    random.seed(2026)  # For reproducibility
    test_ids = set(random.sample(image_ids, num_test))
    train_ids = set(image_ids) - test_ids

    print(f"Copying {len(train_ids)} images to train set...")
    for image_id in train_ids:
        copy_files(image_id, source_dir, train_dest_dir)

    print(f"Copying {len(test_ids)} images to test set...")
    for image_id in test_ids:
        copy_files(image_id, source_dir, test_dest_dir)

    print("Done! Dataset split complete.")
    print(f"Train set: {len(train_ids)} images")
    print(f"Test set: {len(test_ids)} images")


if __name__ == "__main__":
    main()
