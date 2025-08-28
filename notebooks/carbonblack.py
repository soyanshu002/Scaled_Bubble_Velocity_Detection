import cv2
import numpy as np

# Path to the image
img_path = r"D:\Bubble Vel Project\data\test\test_3img\test_img1.jpg"

def carbon_black_medium(img):
    """Carbon Black filter with medium noise cancellation."""
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Medium smoothing to reduce paper texture
    gray_blur = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

    # Step 3: Medium contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    contrast = clahe.apply(gray_blur)

    # Step 4: Adaptive threshold for crisp text
    carbon = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,  # block size
        12   # C - balanced value between 8 (mild) and 12 (strong)
    )

    # Step 5: Morphological open to remove small specks
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(carbon, cv2.MORPH_OPEN, kernel)

    # Step 6: Light close to fill tiny gaps in lines/text
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed

# Read image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

# Apply filter
result = carbon_black_medium(img)

# Show result
cv2.imshow("Original", img)
cv2.imshow("Carbon Black - Medium", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
