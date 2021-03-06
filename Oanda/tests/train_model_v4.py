"""
This file contains a training loop for markov kernel models. Most necessary 
functions are in the "trainer" folder, this script only calls them.
 """

import sys, os
# import pathlib
# Specify path to look for Modules folder, based on current folder structure
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)
from modules.training.trainer.trainer import *

if __name__=='__main__':
    trainer()