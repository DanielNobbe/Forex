from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.definitions.instruments import CandlestickGranularity

import datetime
import matplotlib.pyplot as plt
import pickle 

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

gran_to_sec = {
    "S5": 5,
    "S10": 10,
    "S15": 15,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M4": 240,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D": 86400,
    "W": 604800,
    "D30": 2592000, # Months do not always have the same number of seconds
    # "M": "1 month candlesticks, aligned to first day of the month",
}

class HistoryArgs():
    def __init__(self):
        self.instrument = None
        self.start_time = None
        self.granularity = None
        self.max_count = None


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
    def from_lists(cls, opens, highs, lows, closes, times, volumes, completes):

        assert(
                isinstance(opens, list) 
            and isinstance(highs, list) 
            and isinstance(lows, list) 
            and isinstance(closes, list)
            and isinstance(times, list) 
            and isinstance(volumes, list)
            and isinstance(completes, list)
        ), "Initilization inputs for CandleStickValues object must all be lists"


        obj = cls.__new__(cls)
        super(InstrumentSeries, obj).__init__()
        obj.values = CandleStickValues.from_lists(opens, highs, lows, closes)
        obj.times = times
        obj.volumes = volumes
        obj.completes = completes
        obj.timedict = {time: value for time, value in zip(times,[tupl for tupl in zip(opens,highs,lows,closes)])} 
        #TODO: modify this to contain an object for each candlestick 
        return obj
    
    @classmethod
    def empty(cls):
        obj = cls.__new__(cls)
        super(InstrumentSeries, obj).__init__()
        obj.values = CandleStickValues.empty()
        obj.times = []
        obj.volumes = []
        obj.completes = []
        obj.timedict = []
        return obj

    def append(self, open, high, low, close, time, volume, complete):
        # Append a single candlestick
        self.values.append(open, high, low, close)
        self.times.append(time)
        self.volumes.append(volume)
        self.completes.append(complete)
        self.timedict[time] = (open, high, low, close)

    def extend(self, opens, highs, lows, closes, times, volumes, completes):
        # Extend with multiple candlesticks
        self.values.extend(opens, highs, lows, closes)
        self.times.extend(times)
        self.volumes.extend(volumes)
        self.completes.extend(completes)
        self.timedict.update( {time: value for time, value in zip(times,[tupl for tupl in zip(opens,highs,lows,closes)])} )
    
    def __len__(self):
        return len(self.times)


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



# TODO: Check if unix timestamps actually correct with time zones
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
            print(f'Downloading. Samples left {count_left}\r', end="")
            if count_left > MAX_COUNT_PER_REQUEST:
                next_count = MAX_COUNT_PER_REQUEST
            else:
                next_count = int(count_left)
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
    
    closes = [float(candle['mid']['c']) for candle in rv['candles']]
    highs = [float(candle['mid']['h']) for candle in rv['candles']]
    lows = [float(candle['mid']['l']) for candle in rv['candles']]
    opens = [float(candle['mid']['o']) for candle in rv['candles']]

    volumes = [candle['volume'] for candle in rv['candles']] 
    completes = [candle['complete'] for candle in rv['candles']] 
    timestamps = [to_datetime(candle['time']) for candle in rv['candles']] 
    
    
    if series_obj is None:
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
            print("Finished retrieval.")
            return series_obj
        except V20Error as err:
            print("Failed to retrieve data. Error: ", err)
    

# def retrieve_b_e(instrument, start_time, end_time, granularity):
#     """
#     Uses start, end and count to download history
#     """
    # granularity_in_seconds = 
    # First, determine the unix timestamp for start and end
    # then, use the granularity from the dict on top to 
    # determine the total number of samples (count).
    # Then, split it if count > 5000.
    # Then, use the from, to and count words to retrieve candlesticks.
    # We need this to extract extra in-between samples.

    # First, test if this actually works, documentation is vague
    # they say the start, end and count can't be used together

    # Might be easiest to use hour alignment argument..
    # Or we retrieve minutewise data, and make the split ourselves
    # last option is the best.

def unpickle_or_generate(gen_fun, pickle_path, *args):
        if not os.path.isfile(pickle_path):
            obj = gen_fun(*args)
            with open(pickle_path, 'wb') as file:
                pickle.dump(obj, file)
        else:
            with open(pickle_path, 'rb') as file:
                obj = pickle.load(file)
        return obj

def retrieve_cache(args, download=False):
    # we're going to retrieve a large amount of data here,
    # using a small granularity. Then, we can use this data to split
    # into larger-granularity data with certain offsets.

    os.makedirs('Cache', exist_ok=True)

    # Let's begin by retrieving some data with minute granularity.
    # instrument = "EUR_USD"
    # start_time = "2016-01-01"
    # end_time = "2020-10-25"
    # granularity = "M1" # one minute
    # count = 1e9 # 
    # count = 10000
    
    # download_history(instrument, start_time, granularity, count)
    pickle_path = f"Cache/{args.instrument}_s{args.start_time}_{args.granularity}_{args.max_count}"

    if download:
        cache = unpickle_or_generate(download_history, pickle_path,
                    args.instrument, args.start_time, args.granularity, args.max_count)
    else:
        with open(pickle_path, 'rb') as file:
            cache = pickle.load(file)
    return cache
    
# TODO: Check if times are utc
def subtract_time(time, subtraction):
    new_time = int(time.timestamp()) - subtraction
    return datetime.datetime.fromtimestamp(new_time)


def retrieve_training_data(
    args,
    dt = [ 2*gran_to_sec['D'], gran_to_sec['D']  ], # time before target in seconds to return values for
    only_close = True
        ):

        cache = retrieve_cache(args, download=True)

        timedict = cache.timedict

        targets = []
        values = []

        for time in list(timedict.keys()):
            # begins at earliest time
            target = timedict[time]
            if only_close:
                target = target[-1]
            values_i = []
            for delta in dt:
                h_time = subtract_time(time, delta)
                value = timedict.get(h_time, None)
                if value is not None and only_close:
                    value = value[-1] # final value is close value
                values_i.append(value)
            if None in values_i:
                continue
            
            targets.append(target)
            values.append(values_i)

        return values, targets


def test():
    args = HistoryArgs()
    args.instrument = "EUR_USD"
    args.start_time = "2016-01-01"
    args.granularity = "M1"
    args.max_count = 10000

    values, targets = retrieve_training_data(args, only_close=True)
    print(values)
    # we're going to retrieve a large amount of data here,
    # using a small granularity. Then, we can use this data to split
    # into larger-granularity data with certain offsets.

    # os.makedirs('Cache', exist_ok=True)

    # # Let's begin by retrieving some data with minute granularity.
    # instrument = "EUR_USD"
    # start_time = "2016-01-01"
    # # end_time = "2020-10-25"
    # granularity = "M1" # one minute
    # count = 1e9 # 
    
    # # download_history(instrument, start_time, granularity, count)
    # pickle_path = f"Cache/{instrument}_s{start_time}_{granularity}_0"
    # large_history = unpickle_or_generate(download_history, pickle_path,
    #              instrument, start_time, granularity, count)



    # print(len(large_history))


# TODO: Add function that gathers extra data:
"""
Using an offset, we can use the same granularity to gather a lot more data.
This means that we can, for instance in a one-day granularity,
gather a input sample from minute i on day 1 and minute i on day 2,
for all possible i. This means a much larger amount of data to train 
on.
"""

if __name__ == "__main__":

    # instrument = "EUR_USD"
    # start_time = "2019-01-01"
    # # end_time = "2020-10-25"
    # granularity = "H3"
    # count = 6000
    
    # download_history(instrument, start_time, granularity, count)
    test()
    