import cv2
import os
import time

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
IMAGES_COUNT = 500        # how many images to capture per letter
IMG_SIZE     = 64

# ROI — same green box as realtime.py
ROI_TOP, ROI_LEFT     = 100, 400
ROI_BOTTOM, ROI_RIGHT = 400, 700

# Special labels that should stay lowercase
LOWERCASE_LABELS = ['del', 'nothing', 'space']

def format_label(letter: str) -> str:
    """Keep special labels lowercase, uppercase single letters."""
    if letter.lower() in LOWERCASE_LABELS:
        return letter.lower()
    return letter.upper()

def collect(letter: str):
    label    = format_label(letter)
    save_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images so we don't overwrite
    existing = len(os.listdir(save_dir))

    cap = cv2.VideoCapture(0)
    print(f"\n📸 Collecting data for: {label}")
    print(f"   Saving to folder  : {save_dir}")
    print(f"   Existing images   : {existing}")
    print(f"   Will capture      : {IMAGES_COUNT} more")
    print(f"\n   Get your hand ready inside the GREEN BOX")
    print(f"   Press  S  to start capturing")
    print(f"   Press  Q  to quit\n")

    capturing = False
    count     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Draw ROI box
        cv2.rectangle(frame,
                      (ROI_LEFT, ROI_TOP),
                      (ROI_RIGHT, ROI_BOTTOM),
                      (0, 255, 0), 2)

        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

        # Status overlay
        status = f"CAPTURING {count}/{IMAGES_COUNT}" if capturing else "Press S to Start"
        color  = (0, 0, 255) if capturing else (255, 255, 0)
        cv2.putText(frame, f"Label : {label}",
                    (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, status,
                    (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "S=start  Q=quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("MySign – Data Collection", frame)

        # Save ROI image
        if capturing:
            img_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            filename    = os.path.join(save_dir, f"{label}_{existing + count}.jpg")
            cv2.imwrite(filename, img_resized)
            count += 1
            time.sleep(0.05)

            if count >= IMAGES_COUNT:
                print(f"✅ Done! {IMAGES_COUNT} images saved to {save_dir}")
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            capturing = True
            print("   📸 Capturing started …")
        elif key == ord('q'):
            break

        if cv2.getWindowProperty("MySign – Data Collection",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("✋ MySign – Data Collection Tool")
    print("================================")
    print("Special labels: del, nothing, space")
    print("Regular letters: A, B, C ... Z")
    print()
    letter = input("Enter the letter to collect: ").strip()
    if letter:
        collect(letter)