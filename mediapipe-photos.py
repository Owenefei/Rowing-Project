import cv2 # OpenCV library for image processing.
import mediapipe as mp # MediaPipe library for various computer vision tasks.
import numpy as np # Numerical computing library.

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 1)

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils # Utility functions for drawing landmarks and connections.
mp_drawing_styles = mp.solutions.drawing_styles # Drawing styles for landmarks.
mp_pose = mp.solutions.pose # MediaPipe's pose estimation solution.

# Define constants and load the image:
file =  "//Users//owenbrue//Pictures//Python Coding//mediapipe//happyguy.png" # Path to the input image file.
BG_COLOR = (192, 192, 192) # Background color for the segmented area. Here we've chosen gray.

with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:
    image = cv2.imread(file) # Load the input image using `cv2.imread()`.
    image_height, image_width, _ = image.shape

    # Convert the BGR image to RGB before processing.
    # Use the pose estimation model to process the image and obtain results containing pose landmarks and segmentation mask.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # the coordinates of the nose landmark in the format: "Nose coordinates: (x, y)"
    # where x is the horizontal position (scaled by image width) and y is the vertical position (scaled by image height)
    # x and y: These landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )
    print(
        f'Right_Elbow coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height}, '
        f'Left_Elbow coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height}, '
    )
    print(
        calculate_angle(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST])

    )
    # Create a copy of the input image that we're going to draw on.
    annotated_image = image.copy()

    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    # Use the segmentation mask to conditionally overlay the detected pose on the image.
    annotated_image = np.where(condition, annotated_image, bg_image)
    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Save the annotated image to a file.
    cv2.imwrite('happyguyannotated.png', annotated_image)