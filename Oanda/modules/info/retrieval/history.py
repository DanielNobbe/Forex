from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.definitions.instruments import CandlestickGranularity
from libs.API import API_CONFIG
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

This module contains functions for retrieving historical pricing data 
from OANDA, and functions for saving the data on disk.

For retrieving data, an instrument name is required.
The available pairs can be found at https://www.oanda.com/rw-en/trading/spreads-margin/
The formatting is as follows: '[base currency]_[quote currency]' (w/o brackets)
"""

MAX_COUNT_PER_REQUEST = 5000 # Dependent on API, so constant.

def retrieve(
        instrument: str, 
        start_time: str, 
        granularity: str, 
        count: int, 
        real_account: bool = False, 
        series_obj: object = None, 
        inside: bool = False,
    ):
    """
    Retrieve `count` instrument values for a time period after `start_time`,
    for instrument `instrument`.

    Args: 
        instrument: instrument name, 
            formatted as '[base currency]_[quote currency]' (w/o brackets)
            see https://www.oanda.com/rw-en/trading/spreads-margin/ 
            for options
        start_time: start time in 'YYYY-MM-DDTHH:MM:SSZ' format.
            Can also be 'YYYY-MM-DD' for dates.
        granularity: time between retrieved samples.
            Options can be found in ./definitions.py
        count: number of samples to retrieve
        real_account: whether to use real account, if False uses demo
            account. (currently True is not supported)
    Returns: tuple:
        [0]: series_obj (InstrumentSeries object) containing a retrieved 
            time series
        [1]: Latest timestamp of time series (used when retrieving 
            multiple blocks)
    TODO: What is inside for? Is this not captured with series_obj being None?
    TODO: Improve the switching between demo and real accounts. Should
        just pass the account type as an argument when used from train
        or trade methods.
    """
    # Maximum count is 5000
    if count > 5000:
        # Split the time up into multiple pieces
        # for now, just set count to 5000
        number_of_periods = ceildiv(count, MAX_COUNT_PER_REQUEST)
        series_obj = None
        count_left = count
        while count_left > 0:
            # Make sure that exactly count samples are retrieved
            if count_left > MAX_COUNT_PER_REQUEST:
                next_count = MAX_COUNT_PER_REQUEST
            else:
                next_count = int(count_left)
            series_obj, end_time = retrieve(instrument, start_time, granularity,
                                        next_count, series_obj=series_obj, inside=True)
            
            print(f'Downloading. Samples left {count_left}. Timestamp: {unix_to_date(end_time)}\r', end="")
            
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
        access_token = API_CONFIG['demo']['access_token']
        # accountID = API_CONFIG['demo']['accountID']
        api = API(access_token=access_token)
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
    """
    Downloads history, with handling for V20 errors (such as connection
    errors). Tries multiple times if it fails.
    Args: See retrieve function.
    """
    max_tries = 3
    for i in range(max_tries):
        try:
            series_obj, _ = retrieve(instrument, start_time, granularity, count)
            print("Finished retrieval.")
            return series_obj
        except V20Error as err:
            print("Failed to retrieve data. Error: ", err)


def retrieve_cache(args, download=False):
    """
    Opens cached file on disk with arguments args. If the file does not
    exist, it is created by retrieving the data from Oanda.
    Args:
        args: HistoryArgs object, containing arguments for history 
            retrieval.
        download: If True, downloads data if it is not available on disk.
    Returns:
        cache: InstrumentSeries object. Should be same as if calling
            `download_history` with unpacked `args`. (Or similar,
            if cached on earlier date)
    """
    os.makedirs('cache', exist_ok=True)
    pickle_path = f"cache/{args.instrument}_s{args.start_time}_{args.granularity}_{args.max_count}"

    if download:
        cache = unpickle_or_generate(download_history, pickle_path,
                    args.instrument, args.start_time, args.granularity, args.max_count)
    else:
        with open(pickle_path, 'rb') as file:
            cache = pickle.load(file)
    return cache
    

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
    """
    Retrieves data for inference. Loads data until current moment, 
    and converts this into the 1D tensor required to run one inference
    step. The final value corresponds to the current time, resulting in 
    a corresponding (unknown) target value in the future.
    Args:
        instrument: instrument/currency pair. Example: 'EUR_USD'
        dt: Used to define the time offsets corresponding to the required
            datapoints for the predictive model.
            Final element defines how much time in the future the prediction
            is for. (Each element - final element) is the time difference
            between now and the corresponding datapoint.
            `[element - dt[-1] for element in dt] = relative_times`
            When adding a 0 to the end of `dt`, each value corresponds 
            to a candlestick in the required sequence, where the first 
            is only in the `values` list, and the last is in the future, 
            to be predicted by the model.
        soft_retrieve: If True, soft retrieval is allowed. This means that 
            when attempting to find a candlestick corresponging to a 
            time-offset, the closest candlestick is used, as long as the
            time of that candlestick is within `soft_margin` seconds of 
            the requested time.
        soft_margin: Margin in seconds to use for soft retrieval.
        only_close: If True, returns a list with only the `close` values
            for each candlestick.
        realtime: If True, raises error when not all required data is 
            available. 
        skip_wknd: If True, weekend days are skipped over when retrieving
            candlesticks. The retrieval moves to the last Friday before
            the weekend if a weekend day's candlestick is requested.
    Returns:
        torch.tensor: 1D tensor with values corrsponding to the candle-
        sticks needed to make a prediction, as defined in `dt`.
    TODO : Move the ValueError handling to a higher level
    """
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
        args: HistoryArgs,
        dt: list = [ 2*gran_to_sec['D'], gran_to_sec['D']  ], # time before target in seconds to return values for
        only_close: bool = True,
        full_sequence: bool = False, # Give full sequence in inputs, including target
        soft_retrieve: bool = True,
        soft_margin: float = 3000,
        skip_wknd: bool = True,
    ):
    """
    Creates lists of values and targets, where the the values 
    correspond to the offsets wrt each target defined in `dt`. This
    method returns many such pairs, so is useful for creating a dataset
    for training models. Can return only close values, or all values
    (latter is untested).
    Args:
        args: General arguments for retrieval, as defined in HistoryArgs.
        dt: Used to define the time offsets corresponding to the required
            datapoints for the predictive model.
            Each element defines the offset of that candlestick wrt
            the target.
            `[element for element in dt] = target_time - value_i_time`
            When adding a 0 to the end of `dt`, each value corresponds 
            to a candlestick in the required sequence, where the last 
            one is only in the `targets` list.
        only_close: If True, uses only the close values of a candlestick.
        full_sequence: If True, only returns a complete sequence of values
            for each target, where the target is the last value of the
            sequence.
        soft_retrieve: If True, soft retrieval is allowed. This means that 
            when attempting to find a candlestick corresponging to a 
            time-offset, the closest candlestick is used, as long as the
            time of that candlestick is within `soft_margin` seconds of 
            the requested time.
        soft_margin: Margin in seconds to use for soft retrieval.
        skip_wknd: If True, weekend days are skipped over when retrieving
            candlesticks. The retrieval moves to the last Friday before
            the weekend if a weekend day's candlestick is requested.
    Returns:
        if full_sequence:
            list[list]: list of lists of values, where each value corresponds 
            to an entry of `dt`, and the final value to the target 
            (at offset 0). 
            Each list of values corresponds to one timestep in the 
            retrieved interval.
        else:
            tuple:
            [0]: list[list]: list of lists of values, where each value corresponds 
                to an entry of `dt`. 
                Each list of values corresponds to one timestep in the 
                retrieved interval.
            [1]: list: list of targets, where each target corresponds to one 
                value list in [0]. The target is the final value of a 
                sequence, and should be predicted by a model.
    TODO: Make sure this works when not using only close values
    """
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
            if not h_time in timedict and soft_retrieve:
                h_key = sd_closest(timedict, h_time)
                if h_key - h_time < soft_margin:
                    value = timedict[h_key]
                else:
                    value = None
                    values_i[-idx] = value
                    continue
            else:
                value = timedict.get(h_time, None)
            if value is not None and only_close:
                value = value[-1] # final value is close value
            values_i[-idx] = value
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
    """
    Retrieves a dataset for RNNs. An RNN requires a sequence of values
    and a sequence of targets. If predicting a time-series, the values 
    and targets should be from the same series, with an offset between them. 
    E.g. at value t=1, target is from t=2, etc for the complete sequence.
    Besides this, similar to the `retrieve_training_data` fuction.
    Args:
        args: General arguments for retrieval, as defined in HistoryArgs.
        dt: Used to define the time offsets corresponding to the required
            datapoints for the predictive model.
            Each element defines the offset of that candlestick wrt
            the final target value. 
            `[element for element in dt] = final_target_time - value_i_time`
            When adding a 0 to the end of `dt`, each value corresponds to a 
            candlestick in the required sequence, where the first is only 
            in the `values` list, and the last only in the `targets` list.

        only_close: If True, uses only the close values of a candlestick.
        soft_retrieve: If True, soft retrieval is allowed. This means that 
            when attempting to find a candlestick corresponging to a 
            time-offset, the closest candlestick is used, as long as the
            time of that candlestick is within `soft_margin` seconds of 
            the requested time.
        soft_margin: Margin in seconds to use for soft retrieval.
    Returns: tuple:
        [0]: list[list]: list of lists of values, where each value corresponds 
                to an entry of `dt`. 
                Each list of values corresponds to one timestep in the 
                retrieved interval. 
        [1]: list[list]: list of lists of targets, where each target list 
            corresponds to one value list in [0]. The target list is from 
            the same sequence as the corresponding value list, only incremented 
            with 1 dt-step. As such, its first value is the second value of 
            the value list, and its last value the candlestick corresponding 
            to an offset of 0.
            It should be predicted by the RNN model.
    TODO: Make sure this works when not using only close values
    """
    # RNN requires an input sequence, and a target sequence, which
    # overlap except for the first and last value.
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
            if not h_time in timedict and soft_retrieve:
                h_key = sd_closest(timedict, h_time)
                if h_key - h_time < soft_margin:
                    value = timedict[h_key]
                else:
                    value = None
                    values_i.append(value) # Having None here will prevent it from being added
                    continue
            else:
                value = timedict.get(h_time, None)
                
            if value is not None and only_close:
                value = value[-1] # final value is close value
            values_i.append(value)
            targets_i.append(value)

        if None in values_i:
            empty += 1
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
    """
    Function for testing the functions in this file. 
    Runs when running this file.
    """
    args = HistoryArgs()
    args.instrument = "EUR_USD"
    args.start_time = "2016-01-01"
    args.granularity = "M1"
    args.max_count = 1e9

    cache = retrieve_cache(args, download=True)

if __name__ == "__main__":

    test()
    