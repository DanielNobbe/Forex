from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.definitions.instruments import CandlestickGranularity

import datetime
import matplotlib.pyplot as plt

import re
import os, sys
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
# TODO: Add __slots__ to these objects to decrease object size
# TODO: Add granularity to instrument series object (and granularity check on extension)
# TODO: convert candlestickvalues and instrumentseries data handling to general method that
#       uses dicts to init the __dict__ of objects
# TODO: Add function that computes end data from granularity and vice versa

MAX_COUNT_PER_REQUEST = 5000 # Dependent on API, so constant.

class CandleStickValues():

    __create_key = object()

    def __init__(self, create_key, open, high, low, close):
        assert (create_key == CandleStickValues.__create_key), "Don't use the normal init function!"
        self.opens = open
        self.highs = high
        self.lows = low
        self.closes = close
    
    @classmethod
    def from_lists(cls, open, high, low, close):
        assert (isinstance(open, list) and isinstance(high, list) 
            and isinstance(low, list) and isinstance(close, list)), "Initilization inputs for CandleStickValues object must all be lists"
        return CandleStickValues(cls.__create_key, open, high, low, close)
    
    @classmethod
    def empty(cls):
        return CandleStickValues(cls.__create_key, [], [], [], [])
    
    def append(self, open, high, low, close):
        # Append values for a single candlestick
        self.opens.append(open)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
    
    def extend(self, opens, highs, lows, closes):
        # Extend with values for multiple candlesticks
        self.opens.extend(opens)
        self.highs.extend(highs)
        self.lows.extend(lows)
        self.closes.extend(closes)
    
        

class InstrumentSeries():
    def __init__(self):
        self.values = None
        self.times = None
        self.volumes = None
        self.completes = None
        raise NotImplementedError #"Don't use base init method!"

    @classmethod
    def from_lists(cls, open, high, low, close, times, volumes, completes):

        assert(
                isinstance(open, list) 
            and isinstance(high, list) 
            and isinstance(low, list) 
            and isinstance(close, list)
            and isinstance(times, list) 
            and isinstance(volumes, list)
            and isinstance(completes, list)
        ), "Initilization inputs for CandleStickValues object must all be lists"


        obj = cls.__new__(cls)
        super(InstrumentSeries, obj).__init__()
        obj.values = CandleStickValues.from_lists(open, high, low, close)
        obj.times = times
        obj.volumes = volumes
        obj.completes = completes
        return obj
    
    @classmethod
    def empty(cls):
        obj = cls.__new__(cls)
        super(InstrumentSeries, obj).__init__()
        obj.values = CandleStickValues.empty()
        obj.times = []
        obj.volumes = []
        obj.completes = []
        return obj

    def append(self, open, high, low, close, time, volume, complete):
        # Append a single candlestick
        self.values.append(open, high, low, close)
        self.times.append(time)
        self.volumes.append(volume)
        self.completes.append(complete)

    def extend(self, opens, highs, lows, closes, times, volumes, completes):
        # Extend with multiple candlesticks
        self.values.extend(opens, highs, lows, closes)
        self.times.extend(times)
        self.volumes.extend(volumes)
        self.completes.extend(completes)


def to_datetime(t, date_only = False):
    """
    Convert timestamp from OANDA to datetime format.
    Note: datetime has lower accuracy (microsecond) than OANDA.
    """
    if not date_only:
        return datetime.datetime.strptime(t[0:-4], "%Y-%m-%dT%H:%M:%S.%f") 
    else:
        return datetime.datetime.strptime(t, "%Y-%m-%d")

def ceildiv(a, b):
    # Divide two numbers and round the result up to the closest integer
    return -(-a // b)

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

    difference = (end_time - start_time) / number

    periods = []

    for i in range(number):
        start = start_time + difference*i
        end = start_time + difference*(i+1)
        periods.append( (start,end) )

    assert end == end_time, "Final period end time is not equal to expected end time"

    return periods




# TODO: Create function that handles granularity (requires converting granularity to count)
        # We receive granularity in the request though, might just use that one
def retrieve(instrument, start_time, granularity, count, real_account = False, 
    series_obj = None, inside=False):
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
        number_of_periods = ceildiv(count, MAX_COUNT_PER_REQUEST)
        series_obj = None
        # count_thus_far = 0
        count_left = count
        while count_left > 0:
            # Make sure that exactly count samples are retrieved
            if count_left > MAX_COUNT_PER_REQUEST:
                next_count = MAX_COUNT_PER_REQUEST
            else:
                next_count = count_left
            print("start time: ", start_time)
            # with open(os.devnull, 'w') as sys.stdout:
            series_obj, end_time = retrieve(instrument, start_time, granularity,
                                        next_count, series_obj=series_obj, inside=True)
            
            # count_thus_far += next_count
            count_left -= next_count
            if end_time == False:
                break
            start_time = end_time.timestamp() # Convert to unix time, which works too
        if len(series_obj.times) < count:
            if not inside:
                print(f"Specified count not reached! Only {len(series_obj.times)}/{count} samples available.")

        return series_obj, end_time

    
    params = {
        'from': start_time,
        'granularity': granularity,
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

    print(rv['granularity'])
    
    closes = [float(candle['mid']['c']) for candle in rv['candles']]
    highs = [float(candle['mid']['h']) for candle in rv['candles']]
    lows = [float(candle['mid']['l']) for candle in rv['candles']]
    opens = [float(candle['mid']['o']) for candle in rv['candles']]

    volumes = [candle['volume'] for candle in rv['candles']] 
    completes = [candle['complete'] for candle in rv['candles']] 
    timestamps = [to_datetime(candle['time']) for candle in rv['candles']] 
    
    
    if series_obj is None:
        # print("A0: ", closes)
        series_obj = InstrumentSeries.from_lists(opens, highs, lows, closes, timestamps,
                                                 volumes, completes)
        if len(series_obj.times) < count:
            if not inside:
                print(f"Specified count not reached! Only {len(series_obj.times)}/{count} samples available.")
            return series_obj, False
        
        return series_obj, timestamps[-1]
    else:
        series_obj.extend(opens, highs, lows, closes, timestamps, volumes, completes)
        return series_obj, timestamps[-1]

def download_history(instrument, start_time, granularity, count):
    max_tries = 3
    for i in range(max_tries):
        try:
            series_obj, _ = retrieve(instrument, start_time, granularity, count)
            return series_obj
        except V20Error as err:
            print("Failed to retrieve data. Error: ", err)
    print("Finished retrieval.")


if __name__ == "__main__":

    instrument = "EUR_USD"
    start_time = "2019-01-01"
    # end_time = "2020-10-25"
    granularity = "H3"
    count = 6000
    
    download_history(instrument, start_time, granularity, count)
    