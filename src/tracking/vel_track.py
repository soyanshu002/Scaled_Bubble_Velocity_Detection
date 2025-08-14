import cv2
import numpy as np
import os

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_filled_black_circles(preprocessed_frame):
    _, thresholded = cv2.threshold(preprocessed_frame, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for contour in contours:
        if len(contour) < 5:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        area = cv2.contourArea(contour)
        circle_area = np.pi * (radius ** 2)
        if 0.7 * circle_area < area < 1.3 * circle_area:
            circles.append((int(x), int(y), radius))
    return circles

def classify_bubbles(circles):
    small, medium, large = [], [], []
    for c in circles:
        r = c[2]
        if r < 6:
            small.append(c)
        elif r < 8:
            medium.append(c)
        elif r < 10:
            large.append(c)
    return small, medium, large

def calculate_centroids(circles):
    return [(int(c[0]), int(c[1])) for c in circles]

def calculate_velocity(centroids, prev_centroids, fps, px_per_mm):
    velocities = []
    for (x1, y1), (x2, y2) in zip(centroids, prev_centroids):
        distance_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # pixels
        distance_mm = distance_px / px_per_mm                # mm
        distance_m = distance_mm / 1000                      # m
        velocity_mps = distance_m * fps                      # m/s
        velocities.append(velocity_mps)
    return velocities

def average_velocity(velocities):
    return sum(velocities) / len(velocities) if velocities else 0

def calculate_avg_velocities_from_folder(folder_path, fps, px_per_mm):
    """
    Returns: avg_small_vel, avg_med_vel, avg_large_vel
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"[ERROR] Folder not found: {folder_path}")
    
    frame_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    prev_small, prev_medium, prev_large = [], [], []
    
    total_small_vel, total_medium_vel, total_large_vel = 0, 0, 0
    frame_count = 0

    for fname in frame_files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is None:
            continue
        
        preprocessed = preprocess_frame(frame)
        circles = detect_filled_black_circles(preprocessed)
        small, medium, large = classify_bubbles(circles)

        cent_small = calculate_centroids(small)
        cent_medium = calculate_centroids(medium)
        cent_large = calculate_centroids(large)
        
        if prev_small:
            small_vels = calculate_velocity(cent_small, prev_small, fps, px_per_mm)
            medium_vels = calculate_velocity(cent_medium, prev_medium, fps, px_per_mm)
            large_vels = calculate_velocity(cent_large, prev_large, fps, px_per_mm)
            
            total_small_vel += average_velocity(small_vels)
            total_medium_vel += average_velocity(medium_vels)
            total_large_vel += average_velocity(large_vels)
            frame_count += 1
        
        prev_small, prev_medium, prev_large = cent_small, cent_medium, cent_large

    avg_small = total_small_vel / frame_count if frame_count else 0
    avg_medium = total_medium_vel / frame_count if frame_count else 0
    avg_large = total_large_vel / frame_count if frame_count else 0

    return avg_small, avg_medium, avg_large

# =========================
# Database Update
# =========================
#def update_velocity_in_db(db_path, run_name, zone, avg_small, avg_medium, avg_large):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure velocity columns exist
    cur.execute("""
        ALTER TABLE bubble_info ADD COLUMN avg_small_velocity REAL
    """)
    cur.execute("""
        ALTER TABLE bubble_info ADD COLUMN avg_medium_velocity REAL
    """)
    cur.execute("""
        ALTER TABLE bubble_info ADD COLUMN avg_large_velocity REAL
    """)
    conn.commit()

    # Update row for the specific run and zone
    cur.execute("""
        UPDATE bubble_info
        SET avg_small_velocity = ?, avg_medium_velocity = ?, avg_large_velocity = ?
        WHERE run_name = ? AND zone = ?
    """, (avg_small, avg_medium, avg_large, run_name, zone))

    conn.commit()
    conn.close()

# =========================
# Process All Preprocessed Folders
# =========================
#def process_all_velocity(preprocessed_base, db_path, fps, px_per_mm):
    for run_folder in sorted(os.listdir(preprocessed_base)):
        run_path = os.path.join(preprocessed_base, run_folder)
        if not os.path.isdir(run_path):
            continue

        for zone_folder in sorted(os.listdir(run_path)):
            zone_path = os.path.join(run_path, zone_folder)
            if not os.path.isdir(zone_path):
                continue

            print(f"Processing velocities for {run_folder} - {zone_folder}")
            avg_small, avg_medium, avg_large = calculate_avg_velocities_from_folder(zone_path, fps, px_per_mm)
            update_velocity_in_db(db_path, run_folder, zone_folder, avg_small, avg_medium, avg_large)

# =========================
# Example Usage
# =========================
#if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocessed_base = os.path.join(base_dir, "data", "preprocessed")
    database_path = os.path.join(base_dir, "bubble_info.db")
    fps = 100
    px_per_mm = 4.58

    process_all_velocity(preprocessed_base, database_path, fps, px_per_mm)
