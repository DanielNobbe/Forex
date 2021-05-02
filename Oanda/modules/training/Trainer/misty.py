""" 
This file contains basic functions for training a model based on a time series
of candles.
"""

import numpy as np
import torch
import math
import torch.nn as nn
from modules.training.models import * 
import modules.info.retrieval as retrieval
from torch.utils.data import Dataset, DataLoader
import os

from pdb import set_trace
# Imports all models, can be changed for efficiency


def evaluate(val_dataloader, model, loss_fn, test=False):
    """
    NOTE: Last value in input should be most recent
    TODO: Make this work for batched inputs
    """
    losses = []
    accuracies = []
    # directions = []
    target_directions = []
    for (input, target) in val_dataloader:
        model.eval()
        output = model(input)
        loss = loss_fn(output.squeeze(1), target)
        losses.append(loss.item())
        # We also want to check if it always has the right direction
        # The direction it should have is target - input
        # The direction it has is target - output
        # These should be the same
        target_direction = target - input[0,-1]
        target_direction /= torch.abs(target_direction)
        if target == input[0,-1]:
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

def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, no_epochs, min_epochs, save_path):
    losses = []
    final_losses = []
    val_accuracy = 0 
    missed_steps = 0 

    model_notes = input("Please enter a description for the model you are training: ")

    for epoch in range(no_epochs):
        if missed_steps > 1 and val_accuracy > 0.9 and epoch>min_epochs:
            break
        for index, (input_, target) in enumerate(train_dataloader):

            optimizer.zero_grad()

            output = model(input_)

            loss = loss_fn(output.squeeze(1), target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if index % 50 == 0:
                print(f"Step {epoch}.{index} - Loss: {loss} - prediction {output.item()} - input {input_}, target {target.item()}")
                if index % 100 == 0:
                    new_accuracy = evaluate(val_dataloader, model, loss_fn)
                    if new_accuracy > val_accuracy:
                        val_accuracy = new_accuracy
                    else:
                        missed_steps += 1
                        if missed_steps > 1 and val_accuracy > 0.9 and epoch>min_epochs:
                            break
    test_acc = evaluate(test_dataloader, model, loss_fn, test=True)

    # training_notes = input("Please enter some information about training: ")
    

    notes = f"{model_notes} \nTest accuracy: {test_acc}"

    save_dict = {'state_dict': model.state_dict(), 'dt_settings': model.dt_settings, 'notes': notes}

    torch.save(save_dict, save_path)

    print(f"Max loss is {max(losses)}, mean loss is {np.mean(losses)}")
    # print(f"Max final loss is {max(final_losses)}, mean loss is {np.mean(final_losses)}")

# def 

def misty():
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
####################################
    # TODO: Add the value offsets to this
    args = retrieval.HistoryArgs()
    args.instrument = "EUR_USD"
    args.start_time = "2016-01-01"
    # end_time = "2020-10-25"
    # granularity = "H3"
    args.granularity = "M1" # Granularity to retrieve data with
    args.max_count = 1e9
####################################

    # history = retrieval.history.download_history(instrument, 
    #                             start_time, granularity, count)
    dt = [2*retrieval.gran_to_sec['D'], retrieval.gran_to_sec['D']]
    inputs, targets = retrieval.history.retrieve_training_data(args, dt, only_close=True)
    random = False
    train_loader, val_loader, test_loader = retrieval.build_dataset(inputs, targets, val_split=0.4, test_split=0.1, random=random)

    # Define model
    hidden_sizes = [8]
    markov_order = 2
    model = markov_kernel.MarkovKernel(markov_order, hidden_sizes, 1, dt_settings = dt) # Example
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.MSELoss() # TODO: Move this into model definition?
    loss_fn = nn.L1Loss()
    no_epochs = 200
    min_epochs = 1 # For early stopping
    # os.makedirs("")
    for i in range(1000):
        # Sets save_path as the first free slot in the pretrained models folder
        save_path = f"pre-trained models/markov{markov_order}n_{hidden_sizes}_{args.granularity}_i{i}.pt"
        if not os.path.isfile(save_path):
            break

    train(model, train_loader, val_loader, test_loader, optimizer, loss_fn, no_epochs, min_epochs, save_path)



    
