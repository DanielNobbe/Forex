from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.definitions.instruments import CandlestickGranularity
from .definitions import *
from .tools import *
from .classes import *

# import datetime
import matplotlib.pyplot as plt
# import pickle 

import re
import os, sys
import torch
# from collections import OrderedDict
# from itertools import islice
# from sortedcontainers import SortedDict

from pdb import set_trace
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

    TODO: What is inside for? Is this not captured with series_obj being None?
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
                next_count = int(count_left)
            # with open(os.devnull, 'w') as sys.stdout:
            series_obj, end_time = retrieve(instrument, start_time, granularity,
                                        next_count, series_obj=series_obj, inside=True)
            
            print(f'Downloading. Samples left {count_left}. Timestamp: {unix_to_date(end_time)}\r', end="")
            # count_thus_far += next_count
            
            count_left -= next_count
            if end_time == False:
                break
            start_time = end_time # Already in unix time
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
    timestamps = [to_unix(candle['time']) for candle in rv['candles']] 
    
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
        if len(closes) < count and inside:
            return series_obj, False
        else:
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

def retrieve_cache(args, download=False):
    # we're going to retrieve a large amount of data here,
    # using a small granularity. Then, we can use this data to split
    # into larger-granularity data with certain offsets.
    os.makedirs('cache', exist_ok=True)
    pickle_path = f"cache/{args.instrument}_s{args.start_time}_{args.granularity}_{args.max_count}"

    if download:
        cache = unpickle_or_generate(download_history, pickle_path,
                    args.instrument, args.start_time, args.granularity, args.max_count)
    else:
        with open(pickle_path, 'rb') as file:
            cache = pickle.load(file)
    return cache
    
# TODO: Check if times are utc

def dt_differences(dt):
    dt = sorted(dt, reverse=True) # has to be in right order
    differences = [time - dt[index+1] for index, time in enumerate(dt[:-1])] 
    if differences.count(differences[0]) == len(differences):
        # this means all values are the same
        if all([difference in gran_to_sec.values() for difference in differences]):
            # This means we can use a granularity to retrieve info,
            # but only if the granularity is the same everywhere
            # TODO: Make this compatible with using multiple different gran-
            # ularities to retrieve the data.
            granularity = sec_to_gran[differences[0]]
            return granularity
    return False
    



def retrieve_inference_data(
    instrument,
    dt = [ 2*gran_to_sec['D'], gran_to_sec['D']  ], # time before target in seconds to return values for
    soft_retrieve = True,
    soft_margin = 3000,
    only_close = True,
    realtime = False,
    skip_wknd = True,
    ):
    # Same as for training, except only one sequence (and full one at that)
    # and no target value. Last dt should be now
    now = datetime.datetime.now().timestamp()
    earliest_time = now - (dt[0] - dt[-1]) - 8*gran_to_sec['D'] # Allow for max three weekends to be skipped
    
    args = HistoryArgs()
    args.instrument = instrument
    args.start_time = earliest_time
    granularity = dt_differences(dt)
    args.granularity = granularity if granularity else 'S5' # Shortest option
    args.max_count = len(dt)*4 if granularity else 5e8 # Multiply by two in case we go into a weekend
    print(f"Max count is {args.max_count}")
    # TODO: Handle weekends better here - probably we can predict how many weekends we will encounter
    # or we could already skip the weekends in the dt itself, separately from the loop that adds values
    # to the sequence

    if realtime: # Don't save cache if running in real time
        cache = download_history(args.instrument, args.start_time, args.granularity, args.max_count)
    else:
        cache = retrieve_cache(args, download=True)

    timedict = cache.timedict

    dt.sort(reverse=False)
    values = [0]*len(dt)
    offset = 0
    for idx, delta in enumerate(dt):
        time = now + dt[-1]
        h_time = time - delta - offset# This results in the final h_time being now
        if skip_wknd:
            h_time, offset = skip_weekend(h_time, time, delta, offset)
        if not h_time in timedict and soft_retrieve: # Check if this works correctly
            h_key = sd_closest(timedict, h_time)
            if h_key - h_time < soft_margin:
                value = timedict[h_key]
            else:
                if realtime:
                    # TODO : Move this error handling to a higher level
                    raise ValueError(
                            "Can not run this model right now, "
                            "the required data is not available. "
                            "This can (for instance) be caused by "
                            "running a model that requires data from "
                            "yesterday while no trading "
                            "happened yesterday.")
                else:
                    value = None
                    values[-idx] = value # Having None here will prevent it from being added
                continue
        else:
            value = timedict.get(h_time, None)
        if value is not None and only_close:
            value = value[-1] # final value is close value
        values[-idx] = value
    
    new_now = datetime.datetime.now().timestamp()
    print(f"Retrieval took {new_now-now} seconds.")
    return torch.tensor(values)

def retrieve_training_data(
    args,
    dt = [ 2*gran_to_sec['D'], gran_to_sec['D']  ], # time before target in seconds to return values for
    only_close = True,
    full_sequence = False, # Give full sequence in inputs, including target
    targets = True,
    soft_retrieve = True,
    soft_margin = 3000,
    skip_wknd = True,
        ):
        
        # TODO: Make the offsets 'soft', so it does not have to be exactly dt values
        # TODO: Make sure this works when not using only close values
        cache = retrieve_cache(args, download=True)

        timedict = cache.timedict

        targets = []
        values = []
        empty = 0

        # dt should be in ascending order (i.e. increasing time)
        dt.sort(reverse=False)

        for time in list(timedict.keys()):
            # begins at earliest time
            offset = 0
            target = timedict[time]
            if only_close:
                target = target[-1]
            values_i = [0]*len(dt)
            for idx, delta in enumerate(dt):
                h_time = time - delta - offset
                if skip_wknd:
                    h_time, offset = skip_weekend(h_time, time,
                                                delta, offset)
                if not h_time in timedict and soft_retrieve: # Check if this works correctly
                    h_key = sd_closest(timedict, h_time)
                    if h_key - h_time < soft_margin:
                        value = timedict[h_key]
                    else:
                        value = None
                        values_i[-idx] = value
                        # values_i.append(value) # Having None here will prevent it from being added
                        continue
                    # check if h_key not too far away
                    # timedict[h_time] if h_time in timedict else timedict[min(timedict.keys(), key=lambda k: abs(k-h_time))]
                else:
                    value = timedict.get(h_time, None)
                if value is not None and only_close:
                    value = value[-1] # final value is close value
                values_i[-idx] = value
                # values_i.append(value)
            if full_sequence:
                values_i.append(target)
            if None in values_i:
                empty += 1
                continue
            
            targets.append(target)
            values.append(values_i)
        print(f"Missed {empty} samples")
        if full_sequence:
            return values
        return values, targets

def retrieve_RNN_data(
    args,
    dt = [ 2*gran_to_sec['D'], gran_to_sec['D']  ], # time before target in seconds to return values for
    only_close = True,
    soft_retrieve = True,
    soft_margin = 3600, # Max number of seconds the soft retrieval may deviate from requested time
        ):
        # TODO: Make the offsets 'soft', so it does not have to be exactly dt values
        # TODO: Make sure this works when not using only close values

        # RNN requires an input sequence, and a target sequence, which
        # overlap except for the first and last value.
        # inputs = range(0,20); target = range(1,21)
        cache = retrieve_cache(args, download=True)

        timedict = cache.timedict

        targets = []
        values = []
        empty = 0

        for time in list(timedict.keys()):
            # begins at earliest time
            target = timedict[time]
            if only_close:
                target = target[-1]
            values_i = []
            targets_i = []
            for delta in dt:
                h_time = time - delta
                if not h_time in timedict and soft_retrieve: # Check if this works correctly
                    h_key = sd_closest(timedict, h_time)
                    if h_key - h_time < soft_margin:
                        value = timedict[h_key]
                    else:
                        value = None
                        values_i.append(value) # Having None here will prevent it from being added
                        continue
                    # check if h_key not too far away
                    # timedict[h_time] if h_time in timedict else timedict[min(timedict.keys(), key=lambda k: abs(k-h_time))]
                else:
                    value = timedict.get(h_time, None)
                    
                # print("F:", h_time)
                if value is not None and only_close:
                    value = value[-1] # final value is close value
                values_i.append(value)
                targets_i.append(value)

            if None in values_i:
                empty += 1
                # print("Not available")
                continue
            targets_i.pop(0)
            targets_i.append(target)
            # store indiv. lists in list
            values.append(values_i)
            targets.append(targets_i)

        print(f"Missed {empty} samples")
        print(f"Returning {len(values)} samples")
        return values, targets

def test():
    args = HistoryArgs()
    args.instrument = "EUR_USD"
    args.start_time = "2016-01-01"
    args.granularity = "M1"
    args.max_count = 1e9

    cache = retrieve_cache(args, download=True)

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
    