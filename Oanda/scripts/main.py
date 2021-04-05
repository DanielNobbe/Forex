'''
Main function for testing
'''
# Modify path to run from root folder:
import sys, os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)

from libs.API import *
from modules.trader.interpreter import Interpreter, VariableSafe
from modules.training.retrieval import gran_to_sec
from modules.trader.predictor import Predictor
from modules.training.models import markov_kernel
import modules.training.retrieval as retrieval
# from libs.API.WorkingFunctions import ReadableOutput

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

# =============================================================================
# Function
# =============================================================================
# def retrieve_current_price(access_token, accountID):
#     value = (float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutAsk'])
#                        + float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutBid']))/2
#     return value

def perform_trade(access_token, accountID, predictor, Inter):
    # retrieve_current_price(access_token, accountID)
    prediction, latest_value = predictor()
    # TODO: Predictor returns latest closing value, rather than real current value. 
    # check whether this works well#
    Inter.trade(prediction, latest_value)

def build_predictor(instrument):
    # TODO: Make this based on model config file
    # This is an example function, should modify arguments here
    dt = [2*gran_to_sec['D'], gran_to_sec['D']]
    hidden_sizes = [8]
    model = markov_kernel.MarkovKernel(2, hidden_sizes, 1, dt_settings=dt, instrument=instrument) # Example
    pt_path = "pre-trained models/markov2n_[8]_M1_i0.pt"
    predictor = Predictor(model, pretrained_path=pt_path, soft_margin=0.2*gran_to_sec['D'])
    # TODO: deal with soft margin in a smarter way

    return predictor

# TODO: Add something that checks whether the whole process has not taken too long
# and perhaps something that checks for errors?
# TODO: Make scheduler stop on error

def handle_job_error(scheduler, event):
    if event.exception:
        print("Exception raised during job, terminating..")
        scheduler.shutdown(wait=False)
        return

def start_scheduler():
    access_token = API_CONFIG['demo']['access_token']
    accountID = API_CONFIG['demo']['accountID']

    instrument = "EUR_USD"

    print("API config: ", API_CONFIG)

    predictor = build_predictor(instrument) # build predictor for next timestep
    
    # print(TransactionsTransactionsSinceID(access_token, accountID, params={"id": "20"})[0])

    # Safe = VariableSafe()
    # Interpreter is called to handle trades. Should it be called the trader?
    Inter = Interpreter(access_token, accountID, instrument)

    sched = BackgroundScheduler()
    sched.add_job(perform_trade, args=(access_token, accountID, predictor, Inter), trigger='interval', seconds=5) 

    error_handler = partial(handle_job_error, sched)
    sched.add_listener(error_handler, EVENT_JOB_ERROR)

    sched.start()
    # TODO: Make this more persistent, automatically re-starting jobs that were running before
    return sched

def main():
    sched = start_scheduler()
    try:
        while sched.state != STATE_STOPPED:
            pass
    except KeyboardInterrupt:
        print("Shutting down scheduler.")
        sched.shutdown()

if __name__ == "__main__":
    main()

