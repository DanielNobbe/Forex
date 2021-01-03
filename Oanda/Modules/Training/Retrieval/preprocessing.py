""" Contains functions for preprocessing candlesticks for 
    specific loss functions.
    """
from .history import *
from torch.utils.data import Dataset, DataLoader, random_split
import torch.tensor

class CandleStickDataset(Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets), "Inputs and targets not same length!"
        self.inputs = torch.tensor(inputs)
        self.targets = torch.tensor(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
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

def build_dataset(history: InstrumentSeries, selection="close", val_split=0.1, test_split=0.2):
    inputs = sequence_of_values(history)[:-1] 
    # Don't include last entry
    targets = next_values(history)

    dataset = CandleStickDataset(inputs, targets)

    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size # Such an annoying interface
    splits = [train_size, val_size, test_size]

    train, val, test = random_split(dataset, splits)
    return train, val, test


# This should be combined in some way with the 'original' time series,
# where each value is in its normal spot. 
# It should be possible to feed both lists or sequences to the training
# functions, and they should choose which part of the original sequence
# to use, such as the last 10 values.
# Do we discard the timestamps during training?
# Time interval should be a configurable variable, that should be stored
# for each trained model