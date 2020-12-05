""" 
This file contains basic functions for training a model based on a time series
of candles.
"""

import numpy as np
import torch
import torch.nn as nn
from Modules.Training.Models import * 
import Modules.Training.Retrieval as retrieval

# Imports all models, can be changed for efficiency

def train(model, inputs, targets, optimizer, loss_fn, no_epochs):
    losses = []
    final_losses = []
    for epoch in range(1):
        for index, (input, target) in enumerate(zip(inputs, targets)):

            # Temporarily convert floats to tensors here, 
            # should fix this in dataloader
            input = torch.tensor([input])
            target = torch.tensor([target])

            optimizer.zero_grad()

            output = model(input)

            loss = loss_fn(output, target)
            losses.append(loss.item())
            if index>3000:
                final_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if index % 10 == 0:
                print(f"Step {index} - Loss: {loss} - prediction {output.item()}")

    print(f"Max loss is {max(losses)}, mean loss is {np.mean(losses)}")
    print(f"Max final loss is {max(final_losses)}, mean loss is {np.mean(final_losses)}")

def main():
    """ Main function for training.
    Steps to take:
    1. Init model
    2. Load (subset of) historical data and preprocess it
    3. Train model for number of epochs

    What we train for should be configurable.
    Let's start with immediate (next timestep) value and uncertainty
    How do we do uncertainty though?
    """
    print("Hello")
    model = markov_kernel_1n.MarkovKernel([8], 1) # Example

    # TODO: Configuration, should be somewhere else 
    instrument = "EUR_USD"
    start_time = "2019-01-01"
    # end_time = "2020-10-25"
    granularity = "H3"
    count = 6000

    history = retrieval.history.download_history(instrument, 
                                start_time, granularity, count)
    print("Length of history: ", len(history.values.closes))
    inputs = retrieval.preprocessing.sequence_of_values(history)[:-1] 
    # Don't include last entry
    targets = retrieval.preprocessing.next_values(history)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    no_epochs = 20

    train(model, inputs, targets, optimizer, loss_fn, no_epochs)




    
