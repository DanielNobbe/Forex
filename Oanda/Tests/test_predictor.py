"""
    This file is a test script for the predictor class. Use for reference.
"""

import sys, os
import pathlib
# Specify path to look for Modules folder, based on current folder structure
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)
from Modules.Predictor.Predictor import *
# from Modules.Predictor.Predictor import *

def main():
    predictor = TestMarkovPredictor()
    print(predictor.model_type)
    print(predictor.predict())

if __name__=='__main__':
    main()
