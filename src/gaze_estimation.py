from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import numpy as np

class ModelGazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initializing class instance
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold = 0.5
        self.extensions = extensions

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            self.network = IENetwork(self.model_structure, self.model_weights)
            self.plugin = IECore()

            if self.extensions and "CPU" in self.device:
                self.plugin.add_extension(self.extensions, self.device)

            self.check_model()
            self.exec_network = self.plugin.load_network(self.network,self.device)

            self.input_name=next(iter(self.network.inputs))
            self.input_shape=self.network.inputs[self.input_name].shape
            self.output_name=next(iter(self.network.outputs))
            self.output_shape=self.network.outputs[self.output_name].shape
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        left_eye_image: Blob with the name left_eye_image and the shape [1x3x60x60].
        right_eye_image: Blob with the name right_eye_image and the shape [1x3x60x60].
        head_pose_angles: Blob with the name head_pose_angles and the shape [1x3].
        '''

        left_eye_image = self.preprocess_input(left_eye_image)
        right_eye_image = self.preprocess_input(right_eye_image)
        input_dict = {
            "left_eye_image": left_eye_image,
            "right_eye_image": right_eye_image,
            "head_pose_angles": head_pose_angles
            }

        self.exec_network.start_async(request_id=0, inputs=input_dict)
        status = self.exec_network.requests[0].wait(-1)
        if status==0:
            result = self.exec_network.requests[0].outputs

        coords = self.preprocess_output(result)
        
        return coords

    def check_model(self):
        '''
        Checking for unsupported layers
        '''
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            raise ValueError("Unsupported layers: {0}".format(unsupported_layers))

    def preprocess_input(self, image):
        '''
        left_eye_image: Blob with the name left_eye_image and the shape [1x3x60x60].
        right_eye_image: Blob with the name right_eye_image and the shape [1x3x60x60].
        head_pose_angles: Blob with the name head_pose_angles and the shape [1x3].
        '''
        image = cv2.resize(image, (60, 60))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        '''
        gaze_vector
        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates 
        of gaze direction vector. Please note that the output vector is not normalizes 
        and has non-unit length.
        '''

        return outputs["gaze_vector"]
