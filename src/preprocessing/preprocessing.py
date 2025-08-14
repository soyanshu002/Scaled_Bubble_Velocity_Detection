import os
import cv2
import numpy as np
from skimage import measure, morphology

# -------------------------------
# Helper: Remove small objects
# -------------------------------
def remove_small_objects(binary_image, min_size):
    labels = measure.label(binary_image, connectivity=2)
    output_image = np.zeros_like(binary_image)
    for region in measure.regionprops(labels):
        if region.area >= min_size:
            for coordinates in region.coords:
                output_image[coordinates[0], coordinates[1]] = 255
    return output_image

# -------------------------------
# Carbon Black medium filter
# -------------------------------
def carbon_black_medium(img):
    """Carbon Black filter with medium noise cancellation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    contrast = clahe.apply(gray_blur)

    carbon = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        12
    )

    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(carbon, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed  # single-channel binary (0/255)

# -------------------------------
# Process one image (merged pipeline)
# -------------------------------
def process_image(image_path, circles_output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # ---- Step 1: Carbon Black ----
    cb_img = carbon_black_medium(image)  # 0/255 single-channel

    # ---- Step 2: Further smoothing/sharpening & produce a "white_background" ----
    blurred = cv2.GaussianBlur(cb_img, (5, 5), 0)

    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

    # Remove gray pixels in range 85–180
    mask = (sharpened >= 85) & (sharpened <= 180)
    sharpened[mask] = 0

    # White background for pixels < 85
    white_background = np.ones_like(sharpened) * 255
    low_gray_mask = sharpened < 85
    white_background[low_gray_mask] = sharpened[low_gray_mask]

    # Save intermediate result
    #cv2.imwrite(output_path, white_background)

    # ---- Binary & cleaning to obtain foreground blobs ----
    _, binary_image = cv2.threshold(white_background, 1, 255, cv2.THRESH_BINARY_INV)
    binary_image_bool = binary_image.astype(bool)

    # Fill small holes
    filled_image_bool = morphology.remove_small_holes(binary_image_bool, area_threshold=200)
    filled_image = filled_image_bool.astype(np.uint8) * 255  # white objects on black bg

    # Remove small objects (keeps only sufficiently large blobs)
    filtered_image_preinv = remove_small_objects(filled_image, min_size=45)  # white objects on black bg

    # ---- NEW STEP: contour -> draw equivalent circles on white canvas ----
    # Robust findContours for different OpenCV versions
    contours_data = cv2.findContours(filtered_image_preinv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

    # Make a white RGB canvas (same size as original image)
    white_canvas = np.ones_like(image, dtype=np.uint8) * 255
    # If original was grayscale (unlikely with cv2.imread default), ensure 3-channels
    if white_canvas.ndim == 2:
        white_canvas = cv2.cvtColor(white_canvas, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        radius = int(np.sqrt(area / np.pi))

        M = cv2.moments(contour)
        if M.get("m00", 0) != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # fallback: use bounding box center
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

        # draw filled black circle on white canvas
        cv2.circle(white_canvas, (cx, cy), max(radius, 1), (0, 0, 0), -1)

    # Save circles output
    cv2.imwrite(circles_output_path, white_canvas)

    # ---- Final step: invert pre-inv filtered image to match previous behavior and save ----
    #filtered_image = cv2.bitwise_not(filtered_image_preinv)
    #cv2.imwrite(filtered_output_path, filtered_image)

# -------------------------------
# Loop over dataset and call process_image
# -------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_PARENT = os.path.join(PROJECT_ROOT, "data", "processed")
    CLEANED_ROOT = os.path.join(PROJECT_ROOT, "data", "preprocessed")
    os.makedirs(CLEANED_ROOT, exist_ok=True)

    if not os.path.isdir(INPUT_PARENT):
        raise SystemExit(f"[ERROR] Input folder does not exist: {INPUT_PARENT}")

    for folder in sorted(os.listdir(INPUT_PARENT)):
        folder_path = os.path.join(INPUT_PARENT, folder)
        if not os.path.isdir(folder_path):
            continue

        for zone in sorted(os.listdir(folder_path)):
            zone_path = os.path.join(folder_path, zone)
            if not os.path.isdir(zone_path):
                continue

            output_zone_path = os.path.join(CLEANED_ROOT, folder, zone)
            os.makedirs(output_zone_path, exist_ok=True)

            for img_file in sorted(os.listdir(zone_path)):
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(zone_path, img_file)
                base_name, _ = os.path.splitext(img_file)

                # Only circles output
                circles_path = os.path.join(output_zone_path, f"{base_name}_cb_circles.png")

                process_image(img_path, circles_path)  # Pass only what you need
                print(f"[INFO] Processed: {img_path}")

    print(f"[INFO] Carbon Black + Bubble-circles complete. Outputs saved in: {CLEANED_ROOT}")

