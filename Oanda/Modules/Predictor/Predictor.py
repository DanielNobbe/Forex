""" In this file, the base predictor class is defined, including functions 
    for making predictions. 
"""
# import Training.Models.markov_kernel_1n
import Modules.Training.Models.markov_kernel_1n as mk
import torch
import torch.nn as nn
import numpy as np

#TODO: Modify default values
class Predictor():
    def __init__(self, model, prediction_time = 12, pretrained_path=None):
        self.prediction_time = prediction_time # Hrs in the future
        self.model_type = str(model)
        self.model = model

        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path)) # Loads pretrained model
        
    def predict(self): # Input is model dependent
        input = self.gather_input()
        prediction = self.model.forward(input).detach().numpy()
        return prediction
    
    def gather_input(self):
        # Must be implemented in inherited class
        raise NotImplementedError
    

class MarkovPredictor(Predictor):
    def __init__(self, pretrained_path=None):
        model = mk.MarkovKernel(hidden_size=[16]) # Default value, TODO
        if pretrained_path is None:
            pretrained_path = "Pre-trained Models/markov_kernel_n1.pt"
        super(MarkovPredictor, self).__init__(model=model, pretrained_path=pretrained_path)
    
    def gather_input(self):
        # Call information module to supply latest candlestick value
        # Specifically needs to be close value
         # Can't be implemented yet, no Information Module yet
        input = torch.tensor([1.008]) 
        print("Not functional yet, returning dummy values!")
        return input

class TestMarkovPredictor(Predictor):
    def __init__(self, pretrained_path=None):
        model = mk.MarkovKernel(hidden_size=[16]) # Default value, TODO
        if pretrained_path is None:
            pretrained_path = "Pre-trained Models/markov_kernel_n1.pt"
        super(TestMarkovPredictor, self).__init__(model=model, pretrained_path=pretrained_path)
    
    def gather_input(self):
        # Call information module to supply latest candlestick value
        # Specifically needs to be close value
         # Can't be implemented yet, no Information Module yet
        input = torch.tensor([1.008]) 
        print("Not functional yet, returning dummy values!")
        return input

    def predict(self):
        input = self.gather_input()
        prediction = self.model.forward(input).detach().numpy()
        prediction = np.array([1.008, 0.02])
        return prediction

