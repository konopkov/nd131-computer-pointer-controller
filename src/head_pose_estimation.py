from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import numpy as np

class ModelHeadPoseEstimation:
    '''
    Class for the Head Pose Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.5, extensions=None):
        '''
        Initializing class instance
        '''
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.threshold = threshold
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

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.width = int(image.shape[1]) 
        self.height = int(image.shape[0])
        frame = self.preprocess_input(image)
        input_dict = {self.input_name: frame}

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
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        '''
        Output layer names in Inference Engine format:

        name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

        Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll).
        '''

        result = {
            "angle_y_fc": outputs['angle_y_fc'][0][0],
            "angle_p_fc": outputs['angle_p_fc'][0][0],
            "angle_r_fc": outputs['angle_r_fc'][0][0]
        }

        return result
