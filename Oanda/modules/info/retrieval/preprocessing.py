""" Contains functions for preprocessing candlesticks for 
    specific loss functions.
    """
from .history import *
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.tensor
from pdb import set_trace

class CandleStickDataset(Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets), "Inputs and targets not same length!"
        try:
            self.inputs = torch.tensor(inputs)
        except ValueError:
            set_trace()
        self.targets = torch.tensor(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class CandleStickSequenceDataset(Dataset):
    def __init__(self, inputs):
        # does not need super init
        self.inputs = torch.tensor(inputs)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

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

def build_dataset(inputs: list, targets: list, selection="close",
        val_split=0.1, test_split=0.2, rnd_split=True, shuffle=True,
        batch_size=1, num_workers=2,):
    # if sequence_set:
        # dataset = CandleStickSequenceDataset(inputs)
    # else:
    dataset = CandleStickDataset(inputs, targets)

    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size # Such an annoying interface
    splits = [train_size, val_size, test_size]
    print("Splits: ", splits)
    if rnd_split:
        train, val, test = random_split(dataset, splits)
        train_dataloader = DataLoader(train, batch_size=batch_size, # Larger batch size not yet implemented
                        shuffle=shuffle, num_workers=num_workers)
        val_dataloader = DataLoader(val, batch_size=batch_size, # Larger batch size not yet implemented
                            shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test, batch_size=batch_size, # Larger batch size not yet implemented
                            shuffle=shuffle, num_workers=num_workers)
    else:
        if not shuffle:
            # Still need to implement non-random sampling here
            raise NotImplementedError()
        indices = list(range(len(dataset)))
        train_indices = indices[0:train_size+1]
        val_indices = indices[train_size+1:train_size+val_size+1]
        test_indices = indices[train_size+val_size+1:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, # Larger batch size not yet implemented
                        num_workers=num_workers, sampler=train_sampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, # Larger batch size not yet implemented
                            num_workers=num_workers, sampler=val_sampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, # Larger batch size not yet implemented
                            num_workers=num_workers, sampler=test_sampler)

    return train_dataloader, val_dataloader, test_dataloader

def build_simple_dataset(history: InstrumentSeries, selection="close", 
    val_split=0.1, test_split=0.2, random=True, batch_size=1):
    """
    Build dataset that only looks back 1 timestep. 
    """
    inputs = sequence_of_values(history)[:-1] 
    # Don't include last entry
    targets = next_values(history)

    build_dataset(inputs, targets,
        val_split=0.1, test_split=0.2, random=True, batch_size=1)

    dataset = CandleStickDataset(inputs, targets)

    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size # Such an annoying interface
    splits = [train_size, val_size, test_size]

    if random:

        train, val, test = random_split(dataset, splits)
        train_dataloader = DataLoader(train, batch_size=1, # Larger batch size not yet implemented
                        shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val, batch_size=1, # Larger batch size not yet implemented
                            shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test, batch_size=1, # Larger batch size not yet implemented
                            shuffle=True, num_workers=0)
    else:
        indices = list(range(len(dataset)))
        train_indices = indices[0:train_size+1]
        val_indices = indices[train_size+1:train_size+val_size+1]
        test_indices = indices[train_size+val_size+1:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, # Larger batch size not yet implemented
                        num_workers=0, sampler=train_sampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, # Larger batch size not yet implemented
                            num_workers=0, sampler=val_sampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, # Larger batch size not yet implemented
                            num_workers=0, sampler=test_sampler)

    return train_dataloader, val_dataloader, test_dataloader


# This should be combined in some way with the 'original' time series,
# where each value is in its normal spot. 
# It should be possible to feed both lists or sequences to the training
# functions, and they should choose which part of the original sequence
# to use, such as the last 10 values.
# Do we discard the timestamps during training?
# Time interval should be a configurable variable, that should be stored
# for each trained model