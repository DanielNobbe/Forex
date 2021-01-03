"""
    This file is a test script for the predictor class. Use for reference.
"""

import sys, os
import pathlib
# Specify path to look for Modules folder, based on current folder structure
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)
from Modules.Predictor.Predictor import *
from Modules.Training.Models import markov_kernel_1n
import torch
# from Modules.Predictor.Predictor import *

def main():
    # predictor = TestMarkovPredictor()
    # print(predictor.model_type)
    # print(predictor.predict())
    hidden_sizes = [8]
    model = markov_kernel_1n.MarkovKernel(hidden_sizes, 1) # Example
    granularity = "D"

    pt_path = f"Pre-trained Models/markov1n_{hidden_sizes}_{granularity}_i1.pt"
    predictor = Predictor(model, prediction_time=24, pretrained_path=pt_path)

    input = torch.tensor([1.08])
    prediction = predictor(input)

    print(f"Input: {input} - prediction: {prediction}")

if __name__=='__main__':
    main()
