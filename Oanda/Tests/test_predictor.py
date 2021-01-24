"""
    This file is a test script for the predictor class. Use for reference.
"""

import sys, os
import pathlib
# Specify path to look for Modules folder, based on current folder structure
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)
from Modules.Predictor.Predictor import *
from Modules.Training.Models import markov_kernel
import Modules.Training.Retrieval as retrieval
import torch
# from Modules.Predictor.Predictor import *

def main():
    # predictor = TestMarkovPredictor()
    # print(predictor.model_type)
    # print(predictor.predict())

    # Data settings:
    dt = [2*retrieval.gran_to_sec['D'], retrieval.gran_to_sec['D']]

    hidden_sizes = [8]
    model = markov_kernel.MarkovKernel(2, hidden_sizes, 1, dt_settings = None) # Example
    
    # Now we can find the dt values in model.dt_settings
    
    granularity = "D"

    # pt_path = f"Pre-trained Models/markov1n_{hidden_sizes}_{granularity}_i1.pt"
    pt_path = "Pre-trained Models/markov2n_[8]_M1_i0.pt"
    predictor = Predictor(model, prediction_time=24, pretrained_path=pt_path)

    print("dt: ", predictor.model.dt_settings)
    print("model notes: ", predictor.model.notes)
    input = torch.tensor([1.08, 1.09])
    prediction = predictor(input)
    print(f"Input: {input} - prediction: {prediction}")

if __name__=='__main__':
    main()
