import os
import cv2

# ZONES are defined on the resized image (width=1000, height=600)
ZONES = {
    'SU': (0, 0, 250, 300),
    'SL': (0, 300, 250, 600),
    'TM': (250, 0, 750, 200),
    'UR': (750, 0, 1000, 150)
}

ALLOWED_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

def process_one_input_folder(input_folder_path, processed_root, crop_coords, final_resize_dim):
    """
    Process all images under input_folder_path (walks subfolders).
    Creates: processed_root/<input_folder_name>_preprocessed/{SU,SL,TM,UR}/
    Saves zone images with names: 00001_relpathfilename.jpg
    """
    x1, x2, y1, y2 = crop_coords
    folder_name = os.path.basename(input_folder_path.rstrip(os.sep))
    out_base = os.path.join(processed_root, f"{folder_name}_preprocessed")

    # create zone subfolders
    for z in ZONES:
        os.makedirs(os.path.join(out_base, z), exist_ok=True)

    counter = 1
    # Walk the folder so images inside nested subfolders are also processed
    for root, _, files in sorted(os.walk(input_folder_path)):
        for fname in sorted(files):
            if not fname.lower().endswith(ALLOWED_EXTS):
                continue

            in_path = os.path.join(root, fname)
            # build a relative-name-safe base for output filename
            rel = os.path.relpath(in_path, input_folder_path)                 # e.g. "sub1/frame001.jpg"
            rel_base = os.path.splitext(rel)[0].replace(os.sep, '__')        # e.g. "sub1__frame001"

            img = cv2.imread(in_path)
            if img is None:
                print(f"[WARN] Could not read image: {in_path}. Skipping.")
                continue

            h, w = img.shape[:2]
            # clamp crop coordinates to image bounds
            x1c = max(0, min(w, x1))
            x2c = max(0, min(w, x2))
            y1c = max(0, min(h, y1))
            y2c = max(0, min(h, y2))

            if x1c >= x2c or y1c >= y2c:
                print(f"[WARN] Invalid crop for {in_path} after clamping -> skipping.")
                continue

            cropped = img[y1c:y2c, x1c:x2c]
            if cropped.size == 0:
                print(f"[WARN] Empty crop for {in_path} -> skipping.")
                continue

            try:
                final_img = cv2.resize(cropped, final_resize_dim, interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"[WARN] Resize failed for {in_path}: {e}. Skipping.")
                continue

            # save each zone
            for zone_name, (zx1, zy1, zx2, zy2) in ZONES.items():
                # clamp zone coords (shouldn't be necessary if final_resize_dim matches expectations)
                fw, fh = final_resize_dim
                zx1c = max(0, min(fw, zx1))
                zx2c = max(0, min(fw, zx2))
                zy1c = max(0, min(fh, zy1))
                zy2c = max(0, min(fh, zy2))

                if zx1c >= zx2c or zy1c >= zy2c:
                    print(f"[WARN] Invalid zone {zone_name} for {in_path} -> skipping this zone.")
                    continue

                zone_img = final_img[zy1c:zy2c, zx1c:zx2c]
                if zone_img.size == 0:
                    print(f"[WARN] Empty zone {zone_name} for {in_path} -> skipping zone.")
                    continue

                out_name = f"{counter:05d}_{rel_base}.jpg"
                out_path = os.path.join(out_base, zone_name, out_name)
                cv2.imwrite(out_path, zone_img)

            counter += 1

    print(f"[INFO] Finished processing '{folder_name}'. Saved zones to: {out_base}")

if __name__ == "__main__":
    # detect project root (assumes this file sits in src/... so go 3 levels up)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # INPUT: the parent folder that contains multiple folders having images
    # According to your setup, images are in subfolders under data/test/
    INPUT_PARENT = os.path.join(PROJECT_ROOT, "data", "test")
    PROCESSED_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(PROCESSED_ROOT, exist_ok=True)

    # crop coordinates from original images (as you gave earlier)
    crop_coords = (390, 1700, 120, 960)   # x1, x2, y1, y2

    # resized size before zone split
    final_resize_dim = (1000, 600)        # (width, height)

    # iterate every direct child folder of data/test
    if not os.path.isdir(INPUT_PARENT):
        raise SystemExit(f"[ERROR] Input parent folder does not exist: {INPUT_PARENT}")

    for child in sorted(os.listdir(INPUT_PARENT)):
        child_path = os.path.join(INPUT_PARENT, child)
        if not os.path.isdir(child_path):
            continue
        # skip a processed folder if present under data/test by name
        if child.lower().startswith("processed") or child.lower().endswith("_preprocessed"):
            continue

        print(f"\n[INFO] Starting folder: {child}")
        process_one_input_folder(child_path, PROCESSED_ROOT, crop_coords, final_resize_dim)
