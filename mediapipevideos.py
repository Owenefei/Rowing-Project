import cv2 # OpenCV library for image processing.
import mediapipe as mp # MediaPipe library for various computer vision tasks.
import numpy as np # Numerical computing library.
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 1)

mpPose = mp.solutions.pose # MediaPipe's pose estimation solution.
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils # Utility functions for drawing landmarks and connections.

# Open video capture
# If using a webcam, use "cap = cv2.VideoCapture(0)" instead
cap = cv2.VideoCapture('/Users/owenbrue/Pictures/Python Coding/mediapipe/videoplayback.mp4')  # Use video file
pTime = 0

if (cap.isOpened() == False):
    print("Error opening the video file")
else:
    # Get frame rate information
    fps = int(cap.get(5))
    print("Frame Rate : ",fps,"frames per second")	

    # Get frame count
    frame_count = cap.get(7)
    print("Frame count : ", frame_count)
    
# Obtain frame size information using get() method
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print("Frame Width: ", frame_width)
print("Frame Height: ", frame_height)

frame_size = (frame_width,frame_height)

# Initialize video writer object to output a vidoe as .avi
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
previouship = [(0,0)]
previousangle = 0
previoushipdifference = 0
previoushand = [(0,0)]
previoushanddifference = 0
phase = ""
while(cap.isOpened()):
    # vid_capture.read() method returns a tuple
    #   first element is a bool - if True, you can do processing. Otherwise you should stop.
    #   second element is the frame. If the first was False, don't read this.
    ret, frame = cap.read()
    if ret == True:
        # Converts the BGR image (from OpenCV) to RGB format (required for MediaPipe).
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processes the RGB image using the Pose solution to detect pose landmarks.
        results = pose.process(imgRGB)

        # Prints the list of detected pose landmarks, including their (x, y, z) coordinates and visibility.
      

        if results.pose_landmarks:
            hip = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]
            ankle = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ANKLE]
            knee = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE]
            kneeangle = calculate_angle ((hip.x,hip.y), (knee.x,knee.y), (ankle.x,ankle.y))
            if len(previouship)> 4:
                hipdifference = hip.x - previouship[-4][0]
                if previoushipdifference>0 and hipdifference<0:
                    phase = "catch"
                    print ("catch",cap.get(cv2.CAP_PROP_POS_MSEC))
                
                previoushipdifference = hipdifference
                # Draw the pose landmarks and connections on the image
            previouship.append((hip.x,hip.y))
                  
            previousangle = kneeangle
            hand = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
            left_elbow = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_ELBOW]
            if len(previoushand)> 4:
                handdifference = hand.x - previoushand[-4][0]
                if previoushanddifference<0 and handdifference>0:
                    phase = "finish"
                    print ("finish",cap.get(cv2.CAP_PROP_POS_MSEC))
                
                previoushanddifference = handdifference
                # Draw the pose landmarks and connections on the image
            previoushand.append((hand.x,hand.y))


            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # Draw circles at the vertices of each pose landmark
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
             
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
        # Write the frame to the output files
        output.write(frame)
    else:
        print("Stream disconnected")
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()