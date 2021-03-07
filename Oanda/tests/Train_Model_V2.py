"""
This file contains a training loop. Most necessary functions are in the
 "trainer" folder.
 """

import sys, os
import pathlib
# Specify path to look for Modules folder, based on current folder structure
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)
from modules.training.trainer.misty import *

if __name__=='__main__':
    misty()