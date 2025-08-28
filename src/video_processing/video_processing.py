import os
import cv2


def create_video_from_images(image_folder, video_output_path, fps=100.0):
    """
    Create a video from images in a folder.

    Args:
        image_folder (str): Path to folder containing preprocessed images.
        video_output_path (str): Path to save the video file.
        fps (float): Frames per second for the output video.
    """
    # Collect images in sorted order
    image_files = [
        f for f in sorted(os.listdir(image_folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"[WARNING] No images found in {image_folder}")
        return

    # Read first frame to get dimensions
    first_image_path = os.path.join(image_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"[ERROR] Could not read first image: {first_image_path}")
        return

    height, width, _ = frame.shape

    # Initialize video writer (XVID → .avi format)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Write frames sequentially
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[WARNING] Skipping unreadable image: {image_file}")
            continue
        video_writer.write(frame)

    # Release video file
    video_writer.release()
    print(f"[INFO] Video created: {video_output_path}")
