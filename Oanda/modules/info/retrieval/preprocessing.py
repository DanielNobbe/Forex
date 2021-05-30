""" Contains functions for preprocessing candlesticks for 
    specific loss functions.
    """
from .history import *
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.tensor
from pdb import set_trace

class CandleStickDataset(Dataset):
    """
    Pytorch Dataset for CandleSticks. Can be initialised with value 
    (`input`) and target values or vectors. When iterated on, returns a 
    new value-target pair.
    """
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
    """
    PyTorch Dataset for a single sequence of values. Can be used for 
    situations where the full sequence is required, such as when working
    with filters or RNNs.
    """
    def __init__(self, inputs):
        # does not need super init
        self.inputs = torch.tensor(inputs)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

def next_values(history: InstrumentSeries, selection="close"):
    """
    Gives all values of an InstrumentSeries object except the first one.
    Can be useful for RNNs.
    """
    assert selection=="close", "Other selections than close not yet implemented"
    values = history.values.closes[1:] # Take all values except first one
    return values 

def sequence_of_values(history: InstrumentSeries, selection="close"):
    """
    Gives the sequence of values in an InstrumentSeries object.
    """
    assert selection=="close", "Other selections than close not yet implemented"
    values = history.values.closes
    return values

def build_dataloader(inputs: list, targets: list,
        val_split=0.1, test_split=0.2, rnd_split=True, shuffle=True,
        batch_size=1, num_workers=2,):
    """
    Converts a list of values (`inputs`) and targets into a triplet
    of dataloaders, for training, validation and testing respectively.
    Args:
        `inputs`: list of values for the (value,target) pairs.
        `targets`: list of targets for the (value,target) pairs.
        `val_split`: fraction of samples to use for validation.
        `test_split`: fraction of samples to use for testing. Any samples
            not used for validation and testing will be used for training.
        `rnd_split`: If True, split dataset randomly but conserve order
        `shuffle`: If True, data is shuffled when dataloader returns it,
            so the order is randomised.
        `batch_size`: Batch size of the batches returned by the dataloaders.
        `num_workers`: Number of workers to use for dataloading. 
            See https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    Returns:
        tuple:
            [0]: dataloader for training
            [1]: dataloader for validation
            [2]: dataloader for testing
    """
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