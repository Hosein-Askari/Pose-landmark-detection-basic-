import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from PIL import Image

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





# STEP 1: Import the necessary modules.



# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='level_1(normal)/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.





  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
  # Capture the video frame 
  # by frame 
  ret, frame = vid.read() 

  # Display the resulting frame 
  # cv2.imshow('frame', frame) 

  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)




  image = mp_image


  # STEP 4: Detect pose landmarks from the input image.
  detection_result = detector.detect(image)

  DEFAULT_SIZE_WHITE_CHANNEL = (640, 640,3)
  Black_channel_image = np.zeros(DEFAULT_SIZE_WHITE_CHANNEL, dtype = "uint8") 
  # Black_channel_image[:]=[0,0,0]
  overlay_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=Black_channel_image)
  # STEP 5: Process the detection result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  cv2.imshow('pose-landmark',annotated_image)


  # segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
  # visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
  # cv2.imshow('',visualized_mask)
  # cv2.waitKey(0)
  # the 'q' button is set as the 
  # quitting button you may use any 
  # desired button of your choice 
  if cv2.waitKey(25) & 0xFF == ord('q'): 
      break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()