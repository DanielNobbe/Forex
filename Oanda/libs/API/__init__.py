import yaml
import os #.path as path

root_path = os.path.relpath(__file__+'/../../../')
credentials_path = os.path.relpath(root_path + '/configs/credentials/credentials.yaml')
with open(credentials_path, "r") as cfgfile:
    API_CONFIG = yaml.safe_load(cfgfile)
from .oanda import *
# from .order_position_book import *
# from .orders import *
from .working_functions import readable_output