""" 
This file contains basic functions for training a model based on a time series
of candles.
"""

import numpy as np
import torch
import math
import torch.nn as nn
from Modules.Training.Models import * 
import Modules.Training.Retrieval as retrieval
from torch.utils.data import Dataset, DataLoader

from pdb import set_trace
# Imports all models, can be changed for efficiency


def evaluate(val_dataloader, model, loss_fn, test=False):
    losses = []
    accuracies = []
    # directions = []
    target_directions = []
    for (input, target) in val_dataloader:
        model.eval()
        output = model(input)
        loss = loss_fn(output, target)
        losses.append(loss.item())
        # We also want to check if it always has the right direction
        # The direction it should have is target - input
        # The direction it has is target - output
        # These should be the same
        target_direction = target - input
        target_direction /= torch.abs(target_direction)
        if target == input:
            target_direction = torch.tensor([0])

        direction = target - output
        direction /= torch.abs(direction)
        if target == output:
            direction = torch.tensor([0])

        accuracy = (target_direction == direction)
        accuracies.append(accuracy.item())
        # directions.append(direction.item())
        target_directions.append(target_direction.item())
    # balance = 1-1/(np.sum(directions))
    # print(len(directions), np.sum(directions), directions)
    disbalance = np.sum(target_directions)/len(target_directions)
    # if math.isnan(disbalance):
        # set_trace()
    accuracy = np.mean(accuracies)
    string = "Test" if test else "Validation"
    print(f"\n{string}: average loss {np.mean(losses)}, accuracy {accuracy*100} %, disbalance in eval set {disbalance}\n")
    # Good number for disbalance is close to zero, much less than 1
    return accuracy
    # print([direction == target_direction for direction, target_direction in zip(directions, target_directions)])

def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, no_epochs):
    losses = []
    final_losses = []
    val_accuracy = 0 
    missed_steps = 0 
    for epoch in range(no_epochs):
        if missed_steps > 1 and val_accuracy > 0.9:
            break
        for index, (input, target) in enumerate(train_dataloader):

            # Temporarily convert floats to tensors here, 
            # should fix this in dataloader
            # input = torch.tensor([input])
            # target = torch.tensor([target])
            optimizer.zero_grad()

            output = model(input)

            loss = loss_fn(output, target)
            losses.append(loss.item())
            # if index>800:
            #     final_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if index % 50 == 0:
                print(f"Step {epoch}.{index} - Loss: {loss} - prediction {output.item()} - input {input.item()}, target {target.item()}")
                if index % 100 == 0:
                    new_accuracy = evaluate(val_dataloader, model, loss_fn)
                    if new_accuracy > val_accuracy:
                        val_accuracy = new_accuracy
                    else:
                        missed_steps += 1
                        if missed_steps > 1 and val_accuracy > 0.9:
                            break
    test_acc = evaluate(test_dataloader, model, loss_fn, test=True)
    print(f"Max loss is {max(losses)}, mean loss is {np.mean(losses)}")
    # print(f"Max final loss is {max(final_losses)}, mean loss is {np.mean(final_losses)}")

def main():
    """ Main function for training.
    Steps to take:
    1. Init model
    2. Load (subset of) historical data and preprocess it
        2.a Make dataset/dataloader pytorch object
        2.b Divide into train/val/test splits
    3. Train model for number of epochs

    What we train for should be configurable.
    Let's start with immediate (next timestep) value and uncertainty
    How do we do uncertainty though?
    """
    model = markov_kernel_1n.MarkovKernel([8], 1) # Example
####################################
    # TODO: Configuration, should be somewhere else 
    instrument = "EUR_USD"
    start_time = "2016-01-01"
    # end_time = "2020-10-25"
    # granularity = "H3"
    granularity = "D"
    count = 1500
####################################

    history = retrieval.history.download_history(instrument, 
                                start_time, granularity, count)

    train_set, val_set, test_set = retrieval.build_dataset(history, val_split=0.4, test_split=0.1)

    train_dataloader = DataLoader(train_set, batch_size=1, # Larger batch size not yet implemented
                        shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=1, # Larger batch size not yet implemented
                        shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=1, # Larger batch size not yet implemented
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.MSELoss() # TODO: Move this into model definition?
    loss_fn = nn.L1Loss()
    no_epochs = 200

    train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, no_epochs)
    # TODO: Implement early stopping



    
