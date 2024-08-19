import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import csv
import re


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image




img = cv2.imread("level_1(normal)/David_Goggins.jpg")
cv2.imshow("",img)
cv2.waitKey(0)
# STEP 1: Import the necessary modules.



# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='level_1(normal)/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.


image = mp.Image.create_from_file("level_1(normal)/David_Goggins.jpg")


# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)
DEFAULT_SIZE_WHITE_CHANNEL = (640, 640,3)
Black_channel_image = np.zeros(DEFAULT_SIZE_WHITE_CHANNEL, dtype = "uint8") 
# white_channel_image[:]=[50,50,50,0]


overlay_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=Black_channel_image)
# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
annotated_image = cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)

cv2.imshow('',annotated_image)
  
cv2.waitKey(0)

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imshow('',visualized_mask)
cv2.waitKey(0)


    