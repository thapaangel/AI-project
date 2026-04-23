import os
import sys
import numpy as np
import cv2
import tensorflow as tf

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import preprocess_frame, MODEL_PATH, DATASET_DIR


def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run  python src/train.py  first."
        )
    model      = tf.keras.models.load_model(MODEL_PATH)
    class_names = sorted(os.listdir(DATASET_DIR))
    return model, class_names


def run_realtime():
    model, class_names = load_assets()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("✋ MySign – Real-time ASL Interpreter")
    print("   Press  Q  to quit,  C  to clear sentence,  SPACE  to add space")

    sentence      = ""
    prev_letter   = ""
    hold_counter  = 0
    HOLD_FRAMES   = 20          # frames a letter must be stable before appending

    # ROI box (right hand area)
    ROI_TOP, ROI_LEFT   = 100, 400
    ROI_BOTTOM, ROI_RIGHT = 400, 700

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror

        # Draw ROI
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP),
                      (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)

        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

        # Predict
        img   = preprocess_frame(roi)
        preds = model.predict(img, verbose=0)[0]
        idx   = np.argmax(preds)
        conf  = preds[idx]
        letter = class_names[idx]

        # Stable-letter logic
        if letter == prev_letter and conf > 0.85:
            hold_counter += 1
        else:
            hold_counter = 0
        prev_letter = letter

        if hold_counter == HOLD_FRAMES:
            if letter == "SPACE":
                sentence += " "
            elif letter not in ("NOTHING", "DELETE"):
                sentence += letter
            elif letter == "DELETE" and sentence:
                sentence = sentence[:-1]
            hold_counter = 0

        # Overlay
        cv2.putText(frame, f"Letter : {letter} ({conf*100:.0f}%)",
                    (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Sentence: {sentence}",
                    (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, "Q=quit  C=clear",
                    (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        cv2.imshow("MySign – ASL Interpreter", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):  break
        elif key == ord('c'):  sentence = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()