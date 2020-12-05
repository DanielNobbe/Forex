""" Contains functions for preprocessing candlesticks for 
    specific loss functions.
    """
from .history import *

def next_values(history: InstrumentSeries, selection="close"):
    # This should give the immediate next value at each timestep
    assert selection=="close", "Other selections than close not yet implemented"
    values = history.values.closes[1:] # Take all values except first one
    return values 

def sequence_of_values(history: InstrumentSeries, selection="close"):
    # Gives the current value of each timestep
    assert selection=="close", "Other selections than close not yet implemented"
    values = history.values.closes
    return values
    
# This should be combined in some way with the 'original' time series,
# where each value is in its normal spot. 
# It should be possible to feed both lists or sequences to the training
# functions, and they should choose which part of the original sequence
# to use, such as the last 10 values.
# Do we discard the timestamps during training?
# Time interval should be a configurable variable, that should be stored
# for each trained model