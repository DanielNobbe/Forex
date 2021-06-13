"""
This file contains a training loop for an RNN model. Most necessary functions 
are in the "trainer" folder, this script is only to call those functions.
 """

import sys, os
# import pathlib
# Specify path to look for Modules folder, based on current folder structure
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)
from modules.training.trainer.ketchum import *

if __name__=='__main__':
    main_rnn()