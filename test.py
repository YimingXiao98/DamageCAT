# test if the images can be read correctly

import cv2
import numpy as np
import os

# read the image

img_path = "c:\\Users\\Yiming\\OneDrive - Ming\\TAMU\\Research\\Ida_Categorical\\Dahitra_main\\DAHiTra\\data\\xbd\\train\\images\\post_683.png"

if not os.path.exists(img_path):
    raise FileNotFoundError(f"The file at path {img_path} does not exist.")

img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# check if the image was read successfully
if img is None:
    raise ValueError(f"Failed to read the image from {img_path}")

# display the image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
