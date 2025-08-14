import os
from concurrent.futures import ProcessPoolExecutor

# === Import functions from each stage ===
from src.ingestion.ingest_folders import process_one_input_folder
from src.preprocessing.preprocessing import process_image
from src.database.db_utils import create_tables, insert_run, insert_zone_metrics
from src.detection.detect_bubbles import detect_bubbles_in_zone
from src.tracking.vel_track import calculate_avg_velocities_from_folder


# ---------- Stage 3: Detection + Tracking ----------
def process_zone(run_id, run_name, zone_path, fps, px_per_mm):
    zone_name = os.path.basename(zone_path)

    with ProcessPoolExecutor(max_workers=2) as executor:
        # Detection: pass the correct parameters
        future_detection = executor.submit(detect_bubbles_in_zone, zone_path)


        # Velocity tracking
        future_tracking = executor.submit(
            calculate_avg_velocities_from_folder,
            zone_path,
            fps,
            px_per_mm
        )

        avg_small_count, avg_medium_count, avg_large_count = future_detection.result()
        avg_small_vel, avg_medium_vel, avg_large_vel = future_tracking.result()

    # Store results
    insert_zone_metrics(
        run_id, run_name, zone_name,
        avg_small_count, avg_medium_count, avg_large_count,
        avg_small_vel, avg_medium_vel, avg_large_vel
    )

    print(f"✅ Stored results for {run_name} - {zone_name}")



def process_run(run_folder_path, fps, px_per_mm):
    run_name = os.path.basename(run_folder_path)
    run_id = insert_run(run_name)

    zone_folders = sorted(os.listdir(run_folder_path))
    for zone_folder in zone_folders:
        zone_path = os.path.join(run_folder_path, zone_folder)
        if os.path.isdir(zone_path):
            process_zone(run_id, run_name, zone_path, fps, px_per_mm)


# ---------- Stage 1 + Stage 2: Ingestion & Preprocessing ----------
def run_ingestion_and_preprocessing(project_root):
    # Stage 1: Ingestion
    input_parent = os.path.join(project_root, "data", "test")
    processed_root = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_root, exist_ok=True)

    crop_coords = (390, 1700, 120, 960)
    final_resize_dim = (1000, 600)

    if not os.path.isdir(input_parent):
        raise SystemExit(f"[ERROR] Input parent folder does not exist: {input_parent}")

    for child in sorted(os.listdir(input_parent)):
        child_path = os.path.join(input_parent, child)
        if not os.path.isdir(child_path):
            continue
        if child.lower().startswith("processed") or child.lower().endswith("_preprocessed"):
            continue

        print(f"\n[INFO] Ingestion: {child}")
        process_one_input_folder(child_path, processed_root, crop_coords, final_resize_dim)

    # Stage 2: Preprocessing
    cleaned_root = os.path.join(project_root, "data", "preprocessed")
    os.makedirs(cleaned_root, exist_ok=True)

    for folder in sorted(os.listdir(processed_root)):
        folder_path = os.path.join(processed_root, folder)
        if not os.path.isdir(folder_path):
            continue

        for zone in sorted(os.listdir(folder_path)):
            zone_path = os.path.join(folder_path, zone)
            if not os.path.isdir(zone_path):
                continue

            output_zone_path = os.path.join(cleaned_root, folder, zone)
            os.makedirs(output_zone_path, exist_ok=True)

            for img_file in sorted(os.listdir(zone_path)):
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(zone_path, img_file)
                base_name, _ = os.path.splitext(img_file)
                circles_path = os.path.join(output_zone_path, f"{base_name}_cb_circles.png")

                process_image(img_path, circles_path)
                print(f"[INFO] Preprocessed: {img_path}")

    return cleaned_root


# ---------- Main Orchestration ----------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fps = 100
    px_per_mm = 4.58

    create_tables()
    print("===== Starting Full Orchestration Pipeline =====")

    # Step 1 & 2: Ingestion + Preprocessing
    preprocessed_base = run_ingestion_and_preprocessing(project_root)

    # Step 3: Detection + Tracking
    for run_folder in sorted(os.listdir(preprocessed_base)):
        run_folder_path = os.path.join(preprocessed_base, run_folder)
        if os.path.isdir(run_folder_path):
            process_run(run_folder_path, fps, px_per_mm)

    print("===== Pipeline Completed =====")
