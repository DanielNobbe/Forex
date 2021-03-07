""" In this file, the base predictor class is defined, including functions 
    for making predictions. 
"""
# import training.models.markov_kernel_1n
import modules.training.models.markov_kernel_1n as mk
import torch
import torch.nn as nn
import numpy as np

#TODO: Modify default values
class Predictor():
    # TODO: Modify this to deal with higher numbers of input, by defining 
    # object-specific input retrieval
    def __init__(self, model, prediction_time = 12, pretrained_path=None):
        self.prediction_time = prediction_time # Hrs in the future
        self.model_type = str(model)
        self.model = model
        model.eval()

        if pretrained_path is not None:
            loaded_dict = torch.load(pretrained_path)
            model.load_state_dict(loaded_dict['state_dict']) # Loads pretrained model
            model.dt_settings = loaded_dict['dt_settings']
            model.notes = loaded_dict['notes']
        
    def predict_with_input(self, input): # Input is model dependent
        if not isinstance(input, torch.Tensor):
            if not isinstance(input, list):
                input = [float(input)]
            # input should now be a list of one input value
            input = torch.tensor(input)

        prediction = self.model.infer(test_data = input).detach().numpy()
        return prediction
    
    def predict(self):
        # Automatically retrieves relevant samples from remote
        prediction = self.model.infer().detach().numpy()
        return prediction
    
    def __call__(self, *args):
        return self.predict(*args)



    

# class MarkovPredictor(Predictor):
#     def __init__(self, pretrained_path=None):
#         model = mk.MarkovKernel(hidden_size=[16]) # Default value, TODO
#         if pretrained_path is None:
#             pretrained_path = "pre-trained models/markov_kernel_n1.pt"
#         super(MarkovPredictor, self).__init__(model=model, pretrained_path=pretrained_path)
    
#     def gather_input(self):
#         # Call information module to supply latest candlestick value
#         # Specifically needs to be close value
#          # Can't be implemented yet, no Information Module yet
#         input = torch.tensor([1.008]) 
#         print("Not functional yet, returning dummy values!")
#         return input

# class MarkovPredictor(Predictor):
#     def __init__(self, pretrained_path=None):
#         model = mk.MarkovKernel(hidden_size=[16]) # Default value, 
#         if pretrained_path is None:
#             pretrained_path = "pre-trained models/markov_kernel_n1.pt"
#         super(TestMarkovPredictor, self).__init__(model=model, pretrained_path=pretrained_path)
    
#     def gather_input(self):
#         # Call information module to supply latest candlestick value
#         # Specifically needs to be close value
#          # Can't be implemented yet, no Information Module yet
#         input = torch.tensor([1.008]) 
#         print("Not functional yet, returning dummy values!")
#         return input

#     def predict(self):
#         input = self.gather_input()
#         prediction = self.model.forward(input).detach().numpy()
#         return prediction
    
#     def predict(self, input):
#         prediction = self.model.forward(input).detach().numpy()
#         return prediction
