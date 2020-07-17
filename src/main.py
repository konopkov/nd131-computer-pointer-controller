from face_detection import ModelFaceDetection
from facial_landmarks_detection import ModelFacialLandmarksDetection
from head_pose_estimation import ModelHeadPoseEstimation
from gaze_estimation import ModelGazeEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder
import argparse
import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys

def main(args):
    model_face_detection = args.model_face_detection
    model_facial_landmarks = args.model_facial_landmarks
    device = args.device
    video_file = args.video
    threshold = args.threshold
    extensions = args.extensions

    face_detection = ModelFaceDetection(model_face_detection, device, extensions)
    face_detection.load_model()

    facial_landmarks = ModelFacialLandmarksDetection(model_facial_landmarks, device, extensions)
    facial_landmarks.load_model()

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            
            face_coords = face_detection.predict(frame)
            print(face_coords)

            eyes_coords = facial_landmarks.predict(frame)
            print(eyes_coords)

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_face_detection', required=True)
    parser.add_argument('--model_facial_landmarks', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--extensions', default=None)
    
    args=parser.parse_args()

    main(args)