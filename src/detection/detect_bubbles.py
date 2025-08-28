import os
import cv2
import numpy as np
from pathlib import Path

# =========================
# Utility Functions
# =========================
def detect_filled_black_circles(preprocess_frame):
    """Detect filled black circles in a binary-inverted image."""
    _, thresholded = cv2.threshold(preprocess_frame, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for contour in contours:
        if len(contour) < 5:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        area = cv2.contourArea(contour)
        circle_area = np.pi * (radius ** 2)

        if 0.7 * circle_area < area < 1.3 * circle_area:
            circles.append((center[0], center[1], radius))
    return circles


def classify_bubbles(circles):
    """Classify detected circles into bubble size categories."""
    small_bubbles, medium_bubbles, large_bubbles = [], [], []
    for circle in circles:
        radius = circle[2]
        if radius < 3 :
            small_bubbles.append(circle)
        elif radius < 5:
            medium_bubbles.append(circle)
        elif radius < 7:
            large_bubbles.append(circle)
    return small_bubbles, medium_bubbles, large_bubbles

#===================================
# Bubble Detection in a Zone
#===================================
def detect_bubbles_in_zone(zone_path):
    """
    Detect bubbles in all PNG images inside a zone folder.
    Returns the average small, medium, and large bubble counts.
    """
    all_small_counts, all_medium_counts, all_large_counts = [], [], []

    for image_file in sorted(Path(zone_path).glob("*.png")):
        frame = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue

        circles = detect_filled_black_circles(frame)
        small_bubbles, medium_bubbles, large_bubbles = classify_bubbles(circles)

        all_small_counts.append(len(small_bubbles))
        all_medium_counts.append(len(medium_bubbles))
        all_large_counts.append(len(large_bubbles))

    avg_small_count = float(np.mean(all_small_counts)) if all_small_counts else 0.0
    avg_medium_count = float(np.mean(all_medium_counts)) if all_medium_counts else 0.0
    avg_large_count = float(np.mean(all_large_counts)) if all_large_counts else 0.0

    return avg_small_count, avg_medium_count, avg_large_count


#def process_run_folder(run_folder_path):
    """
    Process a full run folder with multiple zones.
    """
    run_folder_name = os.path.basename(run_folder_path)  # e.g., 'Folder_001'
    run_id = insert_run(run_folder_name)  # store run in DB

    for zone_folder in sorted(os.listdir(run_folder_path)):
        zone_path = os.path.join(run_folder_path, zone_folder)
        if os.path.isdir(zone_path):
            print(f"Processing {zone_folder} in run {run_folder_name}...")
            detect_bubbles_in_zone(zone_path, run_id, run_folder_name, zone_folder)



