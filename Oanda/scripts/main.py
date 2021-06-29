'''
Main function for testing
'''
# Modify path to run from root folder:
import sys, os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)

from libs.API import *
from modules.trader.interpreter import Interpreter
from modules.info.retrieval import gran_to_sec
from modules.trader.predictor import Predictor
from modules.training.models import markov_kernel
import modules.info.retrieval as retrieval
# from libs.API.working_functions import ReadableOutput

# This does not work, because the args are not updated between runs
from apscheduler.triggers.date import DateTrigger # Keyword 'date': use when you want to run the job just once at a certain point of time
from apscheduler.triggers.interval import IntervalTrigger # Keyword 'interval': use when you want to run the job at fixed intervals of time
from apscheduler.triggers.cron import CronTrigger # Keyword 'cron': use when you want to run the job periodically at certain time(s) of day

from apscheduler.jobstores.memory import MemoryJobStore # Stores jobs in an array in RAM. Provides no persistence support.
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore # Stores jobs in a database table using SQLAlchemy. The table will be created if it doesn’t exist in the database. Provides persistency

from apscheduler.executors.pool import ThreadPoolExecutor # This is default
from apscheduler.executors.pool import ProcessPoolExecutor # You can make use of multiple CPU cores

from apscheduler.schedulers.base import STATE_STOPPED
from apscheduler.schedulers.blocking import BlockingScheduler # use when the scheduler is the only thing running in your process
from apscheduler.schedulers.background import BackgroundScheduler # use when you’re not using any of the frameworks below, and want the scheduler to run in the background inside your application. Doesn't show in the terminal

from apscheduler.events import EVENT_JOB_ERROR

from functools import partial

import yaml



trade_config_file = 'trade.yaml'
this_path = os.path.relpath(__file__+'/../')
trading_cfg_path = '../configs/trading/'
model_cfg_path = '../configs/models/'
relative_path = os.path.join(this_path, trading_cfg_path, trade_config_file)
with open(relative_path) as file:
    cfg = yaml.full_load(file)

# =============================================================================
# Function
# =============================================================================
# def retrieve_current_price(access_token, accountID):
#     value = (float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutAsk'])
#                        + float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutBid']))/2
#     return value



# def build_model(cfg):
#     model_cfg_file = cfg['model']['config_file']
#     pt_path = cfg['model']['pt_path']
#     model_cfg_rel = os.path.join(this_path, model_cfg_path, model_cfg_file)
#     with open(model_cfg_rel) as file:
#         mcfg = yaml.full_load(file)
#     # TODO: Implement model building from cfg file
#     return model

# def build_predictor(cfg):
#     # TODO: Make this based on model config file
#     # This is an example function, should modify arguments here
#     instrument = cfg["instrument"]
#     dt = [2*gran_to_sec['D'], gran_to_sec['D']]
#     hidden_sizes = [8]
#     model = markov_kernel.MarkovKernel(2, hidden_sizes, 1, dt_settings=dt, instrument=instrument) # Example
#     pt_path = "pre-trained-models/markov2n_[8]_M1_i0.pt"
#     # model = build_model(cfg)
#     soft_gran = gran_to_sec[cfg['retrieval']['soft_gran']]
#     soft_margin = cfg['retrieval']['soft_margin'] * soft_gran

#     predictor = Predictor(model, pretrained_path=pt_path, soft_margin=0.2*gran_to_sec['D'])
#     # TODO: deal with soft margin in a smarter way

#     return predictor

# TODO: Add something that checks whether the whole process has not taken too long
# and perhaps something that checks for errors?
# TODO: Make scheduler stop on error

def handle_job_error(scheduler, event):
    """
    Handles an error. At this moment only shuts down the scheduler
    on any exception.
    """
    if event.exception:
        print("Exception raised during job, terminating..")
        scheduler.shutdown(wait=False)
        return

def start_scheduler():
    """
    Function for starting the scheduler. Initialises the trading job,
    and adds an error listener.
    """
    access_token = API_CONFIG[cfg['account_type']]['access_token']
    accountID = API_CONFIG[cfg['account_type']]['accountID']

    instrument = cfg["instrument"]

    # print("API config: ", API_CONFIG)

    # predictor = build_predictor(cfg) # build predictor for next timestep
    predictor = Predictor.build_from_cfg(cfg)
    # print(TransactionsTransactionsSinceID(access_token, accountID, params={"id": "20"})[0])

    # Interpreter is called to handle trades. Should it be called the trader?
    # Inter = Interpreter(access_token, accountID, instrument)
    Inter = Interpreter((accountID, access_token), cfg, predictor)
    sched = BackgroundScheduler()
    interval = cfg['period']
    sched.add_job(Inter.perform_trade, args=(), trigger='interval', seconds=interval) 

    # sched.add_job(Inter.perform_trade, args=(access_token, accountID, predictor, Inter), trigger='interval', seconds=interval) 

    error_handler = partial(handle_job_error, sched)
    sched.add_listener(error_handler, EVENT_JOB_ERROR)

    sched.start()
    # TODO: Make this more persistent, automatically re-starting jobs that were running before
    return sched

def main():
    """
    Main trading loop. Starts a job scheduler, which repeatedly makes 
    a trade. Loop ends when scheduler ends or when keyboard interrupt
    has been made.
    """

    sched = start_scheduler()
    try:
        while sched.state != STATE_STOPPED:
            pass
    except KeyboardInterrupt:
        print("Shutting down scheduler.")
        sched.shutdown()

if __name__ == "__main__":
    main()

