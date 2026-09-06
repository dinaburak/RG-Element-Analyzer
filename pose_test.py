MYVIDEO = "test.mp4"
import cv2
import math
import mediapipe as mp
from pathlib import Path
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from element_identifier import identify_element


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


cap = cv2.VideoCapture(MYVIDEO)
frame_number = 1


# -------------------------
# HOLD DETECTION VARIABLES
# -------------------------

state = "waiting"

movement_window = deque(maxlen=5)
stability_window = deque(maxlen=10)

MOVEMENT_THRESHOLD = 6
STABILITY_THRESHOLD = 5

pose_frames = []


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
    result = detector.detect_for_video(
        mp_image,
        frame_number
    )

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # RIGHT landmarks
        right_shoulder = landmarks[12]
        right_hip = landmarks[24]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]

        # Convert to (x, y)
        shoulder = (
            right_shoulder.x,
            right_shoulder.y
        )

        hip = (
            right_hip.x,
            right_hip.y
        )

        knee = (
            right_knee.x,
            right_knee.y
        )

        ankle = (
            right_ankle.x,
            right_ankle.y
        )

        # Calculate angles
        knee_angle = calculate_angle(
            hip,
            knee,
            ankle
        )

        hip_angle = calculate_angle(
            shoulder,
            hip,
            knee
        )

        # ---------------------------------
        # Convert landmarks to pixel coordinates
        # ---------------------------------

        height, width, _ = frame.shape

        shoulder_px = (
            int(right_shoulder.x * width),
            int(right_shoulder.y * height)
        )

        hip_px = (
            int(right_hip.x * width),
            int(right_hip.y * height)
        )

        knee_px = (
            int(right_knee.x * width),
            int(right_knee.y * height)
        )

        ankle_px = (
            int(right_ankle.x * width),
            int(right_ankle.y * height)
        )


        # ---------------------------------
        # Draw lines between the joints
        # ---------------------------------

        cv2.line(frame, shoulder_px, hip_px, (0, 255, 255), 3)
        cv2.line(frame, hip_px, knee_px, (0, 255, 255), 3)
        cv2.line(frame, knee_px, ankle_px, (0, 255, 255), 3)


        # ---------------------------------
        # Draw circles on the joints
        # ---------------------------------

        for point in [shoulder_px, hip_px, knee_px, ankle_px]:
            cv2.circle(
                frame,
                point,
                8,
                (0, 0, 255),
                -1
            )

        # =====================================
        # MOVEMENT / HOLD DETECTION
        # =====================================

        movement_window.append(
            (knee_angle, hip_angle)
        )

        stability_window.append(
            (knee_angle, hip_angle)
        )

        # -------------------------
        # 1. WAITING FOR MOVEMENT
        # -------------------------

        if (
            state == "waiting"
            and len(movement_window) == movement_window.maxlen
        ):

            first_knee, first_hip = movement_window[0]
            last_knee, last_hip = movement_window[-1]

            knee_change = last_knee - first_knee
            hip_change = last_hip - first_hip

            combined_change = math.sqrt(
                knee_change ** 2 +
                hip_change ** 2
            )

            if combined_change > MOVEMENT_THRESHOLD:

                state = "moving"


        # -------------------------
        # 2. LOOKING FOR HOLD
        # -------------------------

        elif (
            state == "moving"
            and len(stability_window)
            == stability_window.maxlen
        ):

            knee_angles = [
                x[0] for x in stability_window
            ]

            hip_angles = [
                x[1] for x in stability_window
            ]

            knee_range = (
                max(knee_angles) -
                min(knee_angles)
            )

            hip_range = (
                max(hip_angles) -
                min(hip_angles)
            )

            combined_stability = math.sqrt(
                knee_range ** 2 +
                hip_range ** 2
            )

            if combined_stability < STABILITY_THRESHOLD:

                state = "holding"


        # -------------------------
        # 3. HOLDING
        # -------------------------

        elif state == "holding":

            first_knee, first_hip = movement_window[0]
            last_knee, last_hip = movement_window[-1]

            knee_change = last_knee - first_knee
            hip_change = last_hip - first_hip

            combined_change = math.sqrt(
                knee_change ** 2 +
                hip_change ** 2
            )

            # If movement starts again,
            # gymnast is leaving the pose
            if combined_change > MOVEMENT_THRESHOLD:

                state = "finished"

            else:

                pose_frames.append({
                    "frame": frame_number,
                    "knee": knee_angle,
                    "hip": hip_angle
                })


        # Display angles
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

        # Show current detection state
        cv2.putText(
            frame,
            f"state: {state}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


    cv2.imshow("Pose", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

    frame_number += 1


cap.release()
cv2.destroyAllWindows()


# -------------------------
# CALCULATE POSE AVERAGES
# -------------------------

if pose_frames:

    average_knee = sum(
        frame["knee"]
        for frame in pose_frames
    ) / len(pose_frames)

    average_hip = sum(
        frame["hip"]
        for frame in pose_frames
    ) / len(pose_frames)

    print()

else:

    print("No held pose detected")


element = identify_element(
    average_knee,
    average_hip
)

if element:
    print("Identified element:", element.element_name)
    print("Element value:", element.base_value)
else:
    print("No matching element found")