"""
    This file is a test script for the predictor class. Use for reference.
"""

import sys, os
import pathlib
# Specify path to look for Modules folder, based on current folder structure
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)
from modules.trader.predictor import *
from modules.training.models import markov_kernel
import modules.info.retrieval as retrieval
import torch

def Test(inputt):
    """
    Initialises a predictor and calls it as a test.
    WARNING: Has not been tested with latest version of the Predictor class.
    """
    # predictor = TestMarkovPredictor()
    # print(predictor.model_type)
    # print(predictor.predict())

    # Data settings:
    dt = [2*retrieval.gran_to_sec['D'], retrieval.gran_to_sec['D']]

    hidden_sizes = [8]
    model = markov_kernel.MarkovKernel(2, hidden_sizes, 1, dt_settings = None, instrument="EUR_USD") # Example
    
    # Now we can find the dt values in model.dt_settings
    
    granularity = "D"

    # pt_path = f"pre-trained-models/markov1n_{hidden_sizes}_{granularity}_i1.pt"
    pt_path = "pre-trained-models/markov2n_[8]_M1_i0.pt"
    predictor = Predictor(model, prediction_time=24, pretrained_path=pt_path)

    print("dt: ", predictor.model.dt_settings)
    print("model notes: ", predictor.model.notes)
    # input = torch.tensor()
    # prediction = predictor() # Usually would use this, uses current value
    test_data = [1.09, 1.08]
    prediction = predictor.predict_with_input(test_data)
    print(f"Input: {test_data} - prediction: {prediction}")

if __name__=='__main__':
    Test(0.01)
