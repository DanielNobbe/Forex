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
from modules.trader.predictor import *
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

from apscheduler.schedulers.blocking import BlockingScheduler # use when the scheduler is the only thing running in your process
from apscheduler.schedulers.background import BackgroundScheduler # use when you’re not using any of the frameworks below, and want the scheduler to run in the background inside your application. Doesn't show in the terminal

# =============================================================================
# Function
# =============================================================================
def retrieve_current_price(access_token, accountID):
    value = (float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutAsk'])
                       + float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutBid']))/2
    return value

def Traden(access_token, accountID, predictor, Inter):
    retrieve_current_price(access_token, accountID)
    prediction = predictor()
    Inter.Trade(Safe.inputt, prediction)

def build_predictor(instrument):
    # TODO: Make this based on model config file
    # This is an example function, should modify arguments here
    dt = [2*gran_to_sec['D'], gran_to_sec['D']]
    hidden_sizes = [8]
    model = markov_kernel.MarkovKernel(2, hidden_sizes, 1, dt_settings=dt, instrument=instrument) # Example
    pt_path = "pre-trained models/markov2n_[8]_M1_i0.pt"
    predictor = Predictor(model, prediction_time=24, pretrained_path=pt_path)

    return predictor


def start_scheduler():
    access_token = API_CONFIG['demo']['access_token']
    accountID = API_CONFIG['demo']['accountID']

    instrument = "EUR_USD"

    print("API config: ", API_CONFIG)

    predictor = build_predictor(instrument)
    
    print(TransactionsTransactionsSinceID(access_token, accountID, params={"id": "20"})[0])

    Safe = VariableSafe()
    Inter = Interpreter(access_token, accountID, 'EUR_USD')

    sched = BackgroundScheduler()
    sched.add_job(Traden, args=(access_token, accountID, predictor, Inter), trigger='interval', seconds=5)  
    sched.start()

    return sched

def main():
    start_scheduler()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down scheduler.")
        sched.shutdown()

if __name__ == "__main__":
    main()

