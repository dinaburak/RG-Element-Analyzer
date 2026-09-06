import cv2
import math
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Path to this project folder
project_folder = Path(__file__).resolve().parent
model_path = project_folder / "pose_landmarker_full.task"


# MediaPipe aliases
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode


# Create the pose detector
options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=VisionRunningMode.VIDEO
)

detector = PoseLandmarker.create_from_options(options)


def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) -
        math.atan2(ay - by, ax - bx)
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO
)

detector = PoseLandmarker.create_from_options(options)
cap = cv2.VideoCapture("test.mp4")
frame_number = 1

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Run detection
    result = detector.detect_for_video(mp_image, frame_number)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # RIGHT landmarks (index-based)
        right_shoulder = landmarks[12]
        right_hip = landmarks[24]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]

        # Convert to (x, y)
        shoulder = (right_shoulder.x, right_shoulder.y)
        hip = (right_hip.x, right_hip.y)
        knee = (right_knee.x, right_knee.y)
        ankle = (right_ankle.x, right_ankle.y)

        # Calculate angle
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle  = calculate_angle(shoulder, hip, knee)

        print(
                f"Frame: {frame_number}, "
                f"Knee angle: {knee_angle}, "
                f"Hip angle: {hip_angle}")

        # Display
        cv2.putText(
            frame,
            f"right knee angle: {int(knee_angle)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"right hip angle: {int(hip_angle)}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Pose", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()