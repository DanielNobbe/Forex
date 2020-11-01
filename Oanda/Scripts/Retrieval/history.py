from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.definitions.instruments import CandlestickGranularity

import datetime
import matplotlib.pyplot as plt

import re
""" 
Functions for retrieving historical data from OANDA API

This module contains multiple functions for retrieving historical
pricing data from OANDA, and functions for savind the data on disk.

For retrieving data, an instrument name is required.
The available pairs can be found at https://www.oanda.com/rw-en/trading/spreads-margin/
The formatting is as follows: '[base currency]_[quote currency]' (w/o brackets)
"""

# TODO: Create a 'history' class for returning historical data
# TODO: Find a way to load the historical data from disk (allows use of larger datasets)

def to_datetime(t, date_only = False):
    """
    Convert timestamp from OANDA to datetime format.
    Note: datetime has lower accuracy (microsecond) than OANDA.
    """
    if not date_only:
        return datetime.datetime.strptime(t[0:-4], "%Y-%m-%dT%H:%M:%S.%f") 
    else:
        return datetime.datetime.strptime(t, "%Y-%m-%d")

def print_granularities():
    for tuple_ in CandlestickGranularity().definitions.items():
        print(tuple_)

def check_time_format(time, only_date = False):
    """
    Check whether date format is correct, and if so what format it is.

    Based on https://github.com/hootnot/oandapyV20-examples/blob/master/src/candle-data.py
    """
    full_time_format = "[\d]{4}-[\d]{2}-[\d]{2}T[\d]{2}:[\d]{2}:[\d]{2}Z"
    date_format = "[\d]{4}-[\d]{2}-[\d]{2}"
    
    if re.match(full_time_format, time):
        return "full_format"
    elif re.match(date_format, time):
        return "date_only"
    else:
        raise ValueError("Incorrect time format: ", time)

def split_time(start_time, end_time, number):
    """
    Splits time period in [number] smaller chunks.
    """

    times = []

    for time in [start_time, end_time]:
        time_format = check_time_format(time)
        times.append(to_datetime(time, date_only=(time_format=="date_only")))
    
    start_time = times[0]
    end_time = times[1]
    # Now that they are in datetime format, we can split them (hopefully)

        
    




# TODO: Create function that handles granularity (requires converting granularity to count)

def retrieve(instrument, start_time, end_time, count, real_account = False):
    """
    Retrieve instrument values for a time period between
    start_time and end_time, with count intervals.

    Args: 
        instrument (string): instrument name, 
            formatted as '[base currency]_[quote currency]' (w/o brackets)
            see https://www.oanda.com/rw-en/trading/spreads-margin/ 
            for options
        start_time (string): start time in 'YYYY-MM-DDTHH:MM:SSZ' format
            can also be 'YYYY-MM-DD' for dates.
        end_time (string): end time in same format as start_time
        count (int): number of samples to retrieve
    """

    # Maximum count is 5000
    if count > 5000:
        # Split the time up into multiple pieces
        # for now, just set count to 5000
        count = 5000
    
    params = {
        'from': start_time,
        'to': end_time,
        'count': count,
    }

    # Compile request
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)

    # Define API with our access token (for demo acct)
    if not real_account:
        api = API(access_token='378d83764609aa3a4eb262663b7c02ef-482ed5696d2a3cede7fca4aa7ded1c76')
    else:
        raise NotImplementedError("Real account has not been implemented.")

    # Request candles
    rv = api.request(r)



if __name__ == "__main__":

    instrument = "EUR_USD"
    start_time = "2019-01-01"
    end_time = "2020-10-25"
    count = 5000

    retrieve(instrument, start_time, end_time, count)
    