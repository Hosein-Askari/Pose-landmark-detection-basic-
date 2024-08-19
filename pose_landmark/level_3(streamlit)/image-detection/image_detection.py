import cv2
import numpy as np
from PIL import Image
import streamlit as st
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2



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

st.title("Detect pose-landmark")

upload_file = st.file_uploader("upload image",type=["jpg","png","jpeg"])
base_options = python.BaseOptions(model_asset_path='level_3(streamlit)/image-detection/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

if upload_file is not None :
    img = Image.open(upload_file)
    st.success("file upload successfully")
    st.image(upload_file)
    
    



    image=np.array(img)  
    image=mp.Image(image_format=mp.ImageFormat.SRGB, data=image)



    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)
    DEFAULT_SIZE_WHITE_CHANNEL = (640, 640,3)
    Black_channel_image = np.zeros(DEFAULT_SIZE_WHITE_CHANNEL, dtype = "uint8") 
    # white_channel_image[:]=[50,50,50,0]


    overlay_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=Black_channel_image)
    
    image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    


    st.image(image)


else:
    st.error("nothing has uploaded yet")










  

