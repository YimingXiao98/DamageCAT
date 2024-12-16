import os
import shutil

root_dir = "./data/xbd/all/"  # adjust this to your path
images_dir = os.path.join(root_dir, "images")
masks_dir = os.path.join(root_dir, "masks")

A_dir = os.path.join(root_dir, "A")
B_dir = os.path.join(root_dir, "B")
label_dir = os.path.join(root_dir, "label")
list_dir = os.path.join(root_dir, "list")

# Create target directories if they don't exist
os.makedirs(A_dir, exist_ok=True)
os.makedirs(B_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]

# Process each image file
for img_file in image_files:
    # Example filename: pre_0.png or post_0.png
    if img_file.startswith("pre_"):
        # Move to A/
        src = os.path.join(images_dir, img_file)
        dst = os.path.join(A_dir, img_file)
        shutil.move(src, dst)

        # Corresponding mask: pre_XXX.png in masks should become label_XXX.png
        mask_file = img_file  # same name as the image
        mask_src = os.path.join(masks_dir, mask_file)
        if os.path.exists(mask_src):
            # Convert 'pre_XXX.png' -> 'label_XXX.png'
            label_name = mask_file.replace("pre_", "label_")
            mask_dst = os.path.join(label_dir, label_name)
            shutil.move(mask_src, mask_dst)

    elif img_file.startswith("post_"):
        # Move to B/
        src = os.path.join(images_dir, img_file)
        dst = os.path.join(B_dir, img_file)
        shutil.move(src, dst)

        # For the label, we rely on the 'pre_XXX.png' masks, so we do NOT handle
        # 'post_XXX.png' masks here. If you prefer using post masks, comment out
        # the pre-mask block and use post masks here.
        #
        # If you do need to handle post_ masks because the pre_ masks do not exist:
        # mask_file = img_file
        # mask_src = os.path.join(masks_dir, mask_file)
        # if os.path.exists(mask_src):
        #     label_name = mask_file.replace('post_', 'label_')
        #     mask_dst = os.path.join(label_dir, label_name)
        #     shutil.move(mask_src, mask_dst)

# If there are any leftover mask files (e.g., post_ masks you decided not to use),
# you can remove them or move them elsewhere as needed.

print("Reorganization complete!")
