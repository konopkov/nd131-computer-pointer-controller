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

def draw_circle(frame, x, y):
    cv2.circle(
            img=frame, 
            center=(int(x), int(y)), 
            radius=10, 
            color=(0,0,255), 
            thickness=5
            )
    return frame

def draw_rectangle(frame, x1, y1, x2, y2):
    cv2.rectangle(
                    frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    (0,0,255), 
                    3
                    )

def main(args):
    model_face_detection = args.model_face_detection
    model_facial_landmarks = args.model_facial_landmarks
    model_head_pose_estimation = args.model_head_pose_estimation 
    model_gaze_estimation = args.model_gaze_estimation 
    device = args.device
    video_file = args.video
    threshold = args.threshold
    extensions = args.extensions
    is_streaming = bool(args.stream)
    out_file_path = args.out

    face_detection = ModelFaceDetection(model_face_detection, device, extensions)
    face_detection.load_model()

    facial_landmarks = ModelFacialLandmarksDetection(model_facial_landmarks, device, extensions)
    facial_landmarks.load_model()

    head_pose = ModelHeadPoseEstimation(model_head_pose_estimation, device, extensions)
    head_pose.load_model()

    gaze_estimation = ModelGazeEstimation(model_gaze_estimation, device, extensions)
    gaze_estimation.load_model()

    if (out_file_path):
        if sys.platform == "linux" or sys.platform == "linux2":
            CODEC = 0x7634706d
        elif sys.platform == "darwin":
            CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
        else:
            print("Unsupported OS.")
            exit(1)

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    width = int(cap.get(3))
    height = int(cap.get(4))

    if (out_file_path):
        out = cv2.VideoWriter(out_file_path, CODEC, 30, (width, height))

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            
            # Detect face
            face_coords = face_detection.predict(frame)
            face_coords_0 = face_coords[0]
            face_image = frame[face_coords_0[1]:face_coords_0[3], face_coords_0[0]:face_coords_0[2]]

            # Detect eyes on face
            eyes_coords = facial_landmarks.predict(face_image)

            left_eye_x1 = int(eyes_coords["left"]["x"]) - 10
            left_eye_x2 = int(eyes_coords["left"]["x"]) + 10
            left_eye_y1 = int(eyes_coords["left"]["y"]) - 10
            left_eye_y2 = int(eyes_coords["left"]["y"]) + 10

            right_eye_x1 = int(eyes_coords["right"]["x"]) - 10
            right_eye_x2 = int(eyes_coords["right"]["x"]) + 10
            right_eye_y1 = int(eyes_coords["right"]["y"]) - 10
            right_eye_y2 = int(eyes_coords["right"]["y"]) + 10

            left_eye_image = face_image[left_eye_x1:left_eye_x2, left_eye_y1:left_eye_y2]
            right_eye_image = face_image[right_eye_x1:right_eye_x2, right_eye_y1:right_eye_y2]

            # Detect head pose angles
            head_pose_angles = head_pose.predict(face_image)

            # Detect gaze vector
            head_pose_angles_normalized = [head_pose_angles["angle_y_fc"], head_pose_angles["angle_p_fc"], head_pose_angles["angle_r_fc"]]
            gaze_vector = gaze_estimation.predict(left_eye_image, right_eye_image, head_pose_angles_normalized)
            print(gaze_vector)

            # Draw face bounds
            draw_rectangle(frame, face_coords_0[0], face_coords_0[1], face_coords_0[2], face_coords_0[3])

            # Draw eyes bounds
            draw_circle(frame, eyes_coords["left"]["x"] + face_coords_0[0], eyes_coords["left"]["y"] + face_coords_0[1])
            draw_circle(frame, eyes_coords["right"]["x"] + face_coords_0[0], eyes_coords["right"]["y"] + face_coords_0[1])

            # File output
            if (out_file_path):
                out.write(frame)

            # Stdout output
            if (is_streaming):
                sys.stdout.buffer.write(frame)
                sys.stdout.flush()

        cap.release()
        if (out_file_path):
            out.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_face_detection', required=True)
    parser.add_argument('--model_facial_landmarks', required=True)
    parser.add_argument('--model_head_pose_estimation', required=True)
    parser.add_argument('--model_gaze_estimation', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--extensions', default=None)
    parser.add_argument('--stream', default=False)
    parser.add_argument('--out', default=None)
    
    args=parser.parse_args()

    main(args)
