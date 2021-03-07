'''
Main function for testing
'''
# Modify path to run from root folder:
import sys, os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, path)

from libs.API.Oanda import *
from modules.trader.interpreter import Interpreter, VariableSafe
from libs.API.WorkingFunctions import ReadableOutput

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

def Traden(access_token, accountID, Safe, Inter):
    Safe.RetrievePrice(access_token, accountID)
    Safe.Predic()
    Inter.Trade(Safe.inputt, Safe.prediction)

def main(sched):
    access_token = '378d83764609aa3a4eb262663b7c02ef-482ed5696d2a3cede7fca4aa7ded1c76'
    accountID = '101-004-16661696-001'
    instrument = "EUR_USD"
    
    print(TransactionsTransactionsSinceID(access_token, accountID, params={"id": "20"})[0])

    Safe = VariableSafe()
    Inter = Interpreter(access_token, accountID, 'EUR_USD')
    
    sched.add_job(Traden, args=(access_token, accountID, Safe, Inter), trigger='interval', seconds=5)  
    sched.start()

if __name__ == "__main__":
    try:
        print("Previous scheduler has been shut down")
        sched.shutdown()
    except:
        print("No scheduler running")
        sched = BackgroundScheduler()
        main(sched)

