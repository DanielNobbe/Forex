""" In this file, the base predictor class is defined, including functions 
    for making predictions. 
"""
# import training.models.markov_kernel_1n
import modules.training.models.markov_kernel_1n as mk
from modules.training.models import *
from modules.training.models import ARCHITECTURES as ARCHITECTURES
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from modules.info.retrieval import gran_to_sec, build_dt
from pdb import set_trace


#TODO: Modify default values
class Predictor():
    # TODO: Modify this to deal with higher numbers of input, by defining 
    # object-specific input retrieval
    def __init__(self, model, pretrained_path, soft_margin): #prediction_time = 3600):
        # self.prediction_time = prediction_time # seconds in the future, does not seem necessary
        self.model_type = str(model)
        self.model = model
        self.soft_margin = soft_margin
        model.eval()

        # if pretrained_path is not None:
        # Can not use a predictor that has not been trained
        loaded_dict = torch.load(pretrained_path)
        model.load_state_dict(loaded_dict['state_dict']) # Loads pretrained model
        model.dt_settings = loaded_dict['dt_settings']
        model.notes = loaded_dict['notes']

    @classmethod
    def build_from_cfg(cls, cfg):

        this_path = os.path.relpath(__file__+'/../')
        model_cfg_path = '../../../configs/models/'
        # mod_config_file = cfg
        model_cfg_file = cfg['model']['config_file']
        relative_path = os.path.join(this_path, model_cfg_path, model_cfg_file)
        abspath = os.path.abspath(relative_path)
        print(abspath)
        with open(relative_path) as file:
            mcfg = yaml.full_load(file)
        
        # Model:
        # model_cfg_file = cfg['model']['config_file']

        # instrument = cfg["instrument"]

        # model_cfg_rel = os.path.join(this_path, model_cfg_path, model_cfg_file)
        # with open(model_cfg_rel) as file:
        #     mcfg = yaml.full_load(file)

        model = cls.build_model(mcfg)
        pt_path = cfg['model']['pt_path']
        pt_models_folder = "pre-trained-models"
        pt_path = os.path.join(pt_models_folder, pt_path)
        # assert model.instrument == instrument, (
        #     "Model is trained for different currency than chosen for trading."
        # )

        predictor = Predictor(model, pretrained_path=pt_path, soft_margin=0.2*gran_to_sec['D'])
        #  # Example
        # pt_path = "pre-trained-models/markov2n_[8]_M1_i0.pt"
        # # model = build_model(cfg)
        # soft_gran = gran_to_sec[cfg['retrieval']['soft_gran']]
        # soft_margin = cfg['retrieval']['soft_margin'] * soft_gran

        # # TODO: deal with soft margin in a smarter way

        return predictor

    

    @classmethod
    def build_model(cls, mcfg):

        instrument = mcfg['instrument']

        dt_descr = mcfg['dt_settings']
        dt = build_dt(dt_descr)

        # hidden_sizes = [8]
        # model = markov_kernel.MarkovKernel(2, hidden_sizes, 1, dt_settings=dt, instrument=instrument)
        
        arch = mcfg['architecture']
        model_type = ARCHITECTURES[arch['model_type']]
        model_args = arch['args']


        model = model_type(**model_args, dt_settings=dt, instrument=instrument)

        return model


    def predict_with_input(self, input_): # Input is model dependent
        if not isinstance(input_, torch.Tensor):
            if not isinstance(input_, list):
                input_ = [float(input_)]
            # input should now be a list of one input value
            input = torch.tensor(input_)

        prediction = self.model.infer(test_data = input).detach().numpy()
        return prediction, input_
    
    def predict(self):
        # Automatically retrieves relevant samples from remote
        prediction, current_value = (thing.detach().numpy() for thing in self.model.infer())
        return prediction, current_value
    
    def __call__(self, *args):
        return self.predict(*args)



    

# class MarkovPredictor(Predictor):
#     def __init__(self, pretrained_path=None):
#         model = mk.MarkovKernel(hidden_size=[16]) # Default value, TODO
#         if pretrained_path is None:
#             pretrained_path = "pre-trained-models/markov_kernel_n1.pt"
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
#             pretrained_path = "pre-trained-models/markov_kernel_n1.pt"
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

