""" In this file, the base predictor class is defined, including functions 
    for making predictions. 
"""
# import training.models.markov_kernel_1n
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
    """
    Predictor object. Contains a model and methods to run this model
    for inference / prediction. To be used in trading.
    When calling the predictor object after initialisation, it calls the
    `predict` method, see the definition of `__call__`.
    Attributes:
        `model_type`: Type of model used (str).
        `model`: The model, to be used for prediction.
        `soft_margin`: soft_margin rules for retrieval when doing inference.
            See info/retrieval/history.py
    """
    def __init__(self, model, pretrained_path, soft_margin): #prediction_time = 3600):
        """
        Creates a Predictor object with a pretrained model.
        Args:
            `model`: PyTorch model to use.
            `pretrained_path`: str for path to pretrained weights.
            `soft_margin`: see `soft_margin` in attributes.
        """
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
        """
        Used to initialise a Predictor from a config dict.
        Will extract the necessary arguments from the `cfg` dict and
        use these to initialise a Predictor through the `__init__`
        method.
        Args:
            `cfg`: cfg dict, from configs/trading/_.yaml
        Returns:
            Predictor object
        """
        this_path = os.path.relpath(__file__+'/../')
        model_cfg_path = '../../../configs/models/'
        model_cfg_file = cfg['model']['config_file']
        relative_path = os.path.join(this_path, model_cfg_path, model_cfg_file)
        abspath = os.path.abspath(relative_path)
        print(abspath)
        with open(relative_path) as file:
            mcfg = yaml.full_load(file)

        soft_gran = cfg['retrieval']['soft_gran']
        if isinstance(soft_gran, str): soft_gran = gran_to_sec[soft_gran]
        soft_margin = cfg['retrieval']['soft_margin']*soft_gran

        model = cls.build_model(mcfg, soft_margin)
        pt_path = cfg['model']['pt_path']
        pt_models_folder = "pre-trained-models"
        pt_path = os.path.join(pt_models_folder, pt_path)


        predictor = Predictor(model, pretrained_path=pt_path, soft_margin=soft_margin)

        return predictor

    @classmethod
    def build_model(cls, mcfg, soft_margin):
        """
        Builds a model using a model config dict. Extracts the necessary
        arguments from the `mcfg` dict, and uses these to initialise a
        model.
        Args:
            `mcfg`: Model config dict, from configs/models/_.yaml.
        Returns:
            Initialised model.
        """

        instrument = mcfg['instrument']

        dt_descr = mcfg['dt_settings']
        dt = build_dt(dt_descr)

        arch = mcfg['architecture']
        model_type = ARCHITECTURES[arch['model_type']]
        model_args = arch['args']

        model = model_type(**model_args, dt_settings=dt, instrument=instrument, soft_margin=soft_margin)

        return model


    def predict_with_input(self, input_): # Input is model dependent
        """
        Used to predict a target value from an input, using the model.
        This method differs from the `predict` method in that it's possible
        to feed any input into the prediction. This allows this method 
        to be used for back-testing, should we implement that.
        Args:
            `input_`: the model input that should be used for prediction.
        Returns:
            tuple:
                [0]: predicted target value
                [1]: `input_`
        """
        if not isinstance(input_, torch.Tensor):
            if not isinstance(input_, list):
                input_ = [float(input_)]
            # input should now be a list of one input value
            input = torch.tensor(input_)

        prediction = self.model.infer(test_data = input).detach().numpy()
        return prediction, input_
    
    def predict(self):
        """
        Retrieves latest required samples from Oanda back-end, and makes
        a prediction based on that.
        Returns:
            [0]: predicted target value
            [1]: the model input that was retrieved
        """
        prediction, current_value = (thing.detach().numpy() for thing in self.model.infer())
        return prediction, current_value
    
    def __call__(self, *args):
        return self.predict(*args)
