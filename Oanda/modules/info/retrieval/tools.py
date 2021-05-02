from oandapyV20.definitions.instruments import CandlestickGranularity
import datetime
import os
from .definitions import gran_to_sec
import pickle
import time
from itertools import islice
from sortedcontainers import SortedDict

def sd_closest(sorted_dict, key):
    "Return closest key in `sorted_dict` to given `key`."
    # from https://stackoverflow.com/a/22997000
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))

def ceildiv(a, b):
    # Divide two numbers and round the result up to the closest integer
    return -(-a // b)

def print_granularities():
    for tuple_ in CandlestickGranularity().definitions.items():
        print(tuple_)

def unpickle_or_generate(gen_fun, pickle_path, *args):
        if not os.path.isfile(pickle_path):
            obj = gen_fun(*args)
            with open(pickle_path, 'wb') as file:
                pickle.dump(obj, file)
        else:
            with open(pickle_path, 'rb') as file:
                obj = pickle.load(file)
        return obj

# dt settings
def build_dt(dt_dict):
    # Creates dt_settings list from 
    # cfg specification, for use with retrieve_for_inference
    granularity = gran_to_sec[dt_dict['granularity']]
    no_samples = dt_dict['no_samples']
    dt_settings = [granularity*index for index in range(no_samples, 0, -1)]
    return dt_settings

def skip_weekend(h_time, time, delta, offset):
    wkday = weekday(h_time)
    if on_sunday(wkday):
        offset += gran_to_sec['D'] * 2
        print("Landed on Sunday during retrieval, "
        "adding offset of 2 days.")
        h_time = time - delta - offset
        print(f"Revised weekday from {wkday} to {weekday(h_time)}")
    elif on_saturday(wkday):
        offset = gran_to_sec['D'] * 1
        print("Landed on Saturday during retrieval, "
        "adding offset of 1 day.")
        h_time = time - delta - offset
        print(f"Revised weekday from {wkday} to {weekday(h_time)}")
    return h_time, offset


### Time tools

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

def subtract_time(time, subtraction):
    new_time = int(time.timestamp()) - subtraction
    return datetime.datetime.fromtimestamp(new_time)

def to_datetime(t, date_only = False):
    """
    Convert timestamp from OANDA to datetime format.
    Note: datetime has lower accuracy (microsecond) than OANDA.
    """
    if not date_only:
        return datetime.datetime.strptime(t[0:-4], "%Y-%m-%dT%H:%M:%S.%f") 
    else:
        return datetime.datetime.strptime(t, "%Y-%m-%d")
def to_unix(t):
    time = datetime.datetime.strptime(t[0:-4], "%Y-%m-%dT%H:%M:%S.%f")
    return time.timestamp()

def unix_to_date(t):
    time = datetime.datetime.fromtimestamp(t)
    return time

def weekday(t):
    return time.strftime("%A", time.localtime(t))

def on_saturday(weekday):
    # Checks if timestamp was on a saturday
    return weekday == 'Saturday'

def on_sunday(weekday):
    return weekday == 'Sunday'