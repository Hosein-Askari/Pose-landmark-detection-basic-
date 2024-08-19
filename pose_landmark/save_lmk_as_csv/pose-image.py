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




img = cv2.imread("save_lmk_as_csv/David_Goggins.jpg")
cv2.imshow("",img)
# cv2.imshow("image", imgg)

cv2.waitKey(0)
# STEP 1: Import the necessary modules.



# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='save_lmk_as_csv/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.

    
  

image = mp.Image.create_from_file("save_lmk_as_csv/David_Goggins.jpg")


# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)


# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
annotated_image = cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)

cv2.imshow('',annotated_image)
  
cv2.waitKey(0)

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imshow('',visualized_mask)
cv2.waitKey(0)


    
pose_landmarks_list=detection_result.pose_landmarks

  
  
pose_landmarks_points= [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks_list[0]
]


with open('save_lmk_as_csv/details.csv', 'w', newline = '') as file: 
    writer = csv.writer(file) 
      
    writer.writerow(["Landmark.No", "X", "Y","Z"])

for i,p in enumerate(pose_landmarks_points):
    

    
  points = re.sub(r"[^:.0-9]","",str(p))

  
  point=points.split(":")
  
  with open('save_lmk_as_csv/details.csv', 'a', newline = '') as file: 
    
    writer = csv.writer(file) 
    writer.writerow([i+1,point[1], point[2],point[3]]) 

file.close() 

# for printing in the one line
# pointt=[]
# for i,p in enumerate(pose_landmarks_points):
    

    
#     points = re.sub(r"[^:.0-9]","",str(p))


#     point = points.split(":")
#     pointt += point[1:]

# with open('pose-land-mark/details.csv', 'a', newline = '') as file: 
  
#   writer = csv.writer(file) 
#   writer.writerows([pointt]) 
# file.close() 