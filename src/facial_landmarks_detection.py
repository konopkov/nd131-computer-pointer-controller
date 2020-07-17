from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import numpy as np

class ModelFacialLandmarksDetection:
    '''
    Class for the Face Detection Model.
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
            result = self.exec_network.requests[0].outputs[self.output_name]

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
        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point 
        values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). All the 
        coordinates are normalized to be in range [0,1].
        '''
        coords = {
            "left": {
                "x": outputs[0][0][0] * self.width,
                "y": outputs[0][1][0] * self.height
            },
            "right": {
                "x": outputs[0][2][0] * self.width,
                "y": outputs[0][3][0] * self.height
            }
        }
        
        return coords
