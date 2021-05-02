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
from textwrap import dedent
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
    for (values, targets) in val_dataloader:
        model.eval()
        outputs = model(values.unsqueeze(dim=2))
        loss = loss_fn(outputs.squeeze(2), targets)
        losses.append(loss.item())
        # We also want to check if it always has the right direction
        # The direction it should have is target - input
        # The direction it has is target - output
        # These should be the same
        target_direction = targets.squeeze()[-1] - values.squeeze()[-1] # take only final value in RNN
        target_direction /= torch.abs(target_direction)
        mask = targets[-1] == values[-1]
        target_direction[mask] = torch.tensor([0], dtype=torch.float)

        direction = targets.squeeze()[-1] - outputs.squeeze()[-1] # take only final value in RNN
        direction /= torch.abs(direction)
        mask = targets[-1] == outputs.squeeze()[-1]
        direction[mask] = torch.tensor([0], dtype=torch.float)

        accuracy = (target_direction == direction).to(torch.float).mean()
        accuracies.append(accuracy.item())
        # directions.append(direction.item())

        # target_directions.append(target_direction.item())

    # balance = 1-1/(np.sum(directions))
    # print(len(directions), np.sum(directions), directions)

    #TODO: Reimplement disbalance calculation for batched computations
    # disbalance = np.sum(target_directions)/len(target_directions)

    # if math.isnan(disbalance):
        # set_trace()
    accuracy = np.mean(accuracies)
    string = "Test" if test else "Validation"
    print(f"\n{string}: average loss {np.mean(losses)}, accuracy {accuracy*100} %")

    #, disbalance in eval set {disbalance}\n")
    # Good number for disbalance is close to zero, much less than 1
    return accuracy
    # print([direction == target_direction for direction, target_direction in zip(directions, target_directions)])

def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, no_epochs, min_epochs, 
            save_path, val_interval, print_interval, early_stopping_min_acc = 0.54):
    losses = []
    final_losses = []
    val_accuracy = 0 
    missed_steps = 0 
    interrupted = False
    model_notes = input("Please enter a description for the model you are training: ")

    try:
        for epoch in range(no_epochs):
            if missed_steps > 1 and val_accuracy > 0.9 and epoch>min_epochs:
                break
            for index, (value, target) in enumerate(train_dataloader): 
                # TODO: Generalise this for RNNs (target is not included now)

                optimizer.zero_grad()

                output = model(value.unsqueeze(dim=2))
                # TODO: carry over this hidden state, might allow for
                # longer sequences
                # TODO: Find a way to traverse the time series directly,
                    # without taking only a subset
                # set_trace()
                loss = loss_fn(output.squeeze(2), target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                if index % print_interval == 0:
                    print(
                                f"Step {epoch}.{index} - Loss: {loss} - "
                                f"prediction {output[0].squeeze().tolist()} - "
                                f"input {value[0].squeeze().tolist()} - "
                                f"target {target[0].squeeze().tolist()}"
                    )
                if index % val_interval == 0 and epoch>0:
                    new_accuracy = evaluate(val_dataloader, model, loss_fn)
                    if new_accuracy > val_accuracy:
                        val_accuracy = new_accuracy
                    else:
                        missed_steps += 1
                        if missed_steps > 1 and val_accuracy > early_stopping_min_acc and epoch>min_epochs:
                            break
    except KeyboardInterrupt:
        print(f"training stopped at {epoch} step {index}")
        interrupted = True

    test_acc = evaluate(test_dataloader, model, loss_fn, test=True)

    # training_notes = input("Please enter some information about training: ")
    
    notes = f"{model_notes} \nTest accuracy: {test_acc}"
    if interrupted:
        notes += f"\n Interrupted at iteration {epoch}.{index}"

    save_dict = {'state_dict': model.state_dict(), 'dt_settings': model.dt_settings, 'notes': notes}

    torch.save(save_dict, save_path)

    print(f"Max loss is {max(losses)}, mean loss is {np.mean(losses)}")
    # print(f"Max final loss is {max(final_losses)}, mean loss is {np.mean(final_losses)}")

# def 

def main_rnn():
    """ Main function for training an RNN in a naive way.
    Extracts a limited sequence and applies RNN to it.
    Not very realistic, would be better to implement an RNN method that
    can run through the complete dataset.

    """
####################################
    # TODO: Add the value offsets to this
    ret_args = retrieval.HistoryArgs()
    ret_args.instrument = "EUR_USD"
    ret_args.start_time = "2018-01-01"
    # end_time = "2020-10-25"
    # granularity = "H3"
    ret_args.granularity = "M1" # Granularity to retrieve data with
    ret_args.max_count = 1e9

    model_granularity = retrieval.gran_to_sec['H6'] 
    # TODO: Figure out how to deal with weekend days, for which there is no data
    # (could potentially not use real times, but only numbers of samples?)
    # TODO: Make a database to store info about models so it's easily searchable

####################################

    # history = retrieval.history.download_history(instrument, 
    #                             start_time, granularity, count)
    sequence_length = 10
    dt = range(sequence_length)[::-1]
    dt = [model_granularity*d for d in dt]
    inputs, targets = retrieval.history.retrieve_RNN_data(ret_args, dt, only_close=True, soft_retrieve=True, soft_margin=3600)

    # values = retrieval.history.retrieve_cache(args, download=True).values.closes

    # TODO: Revert all these changes - we need two vectors
    # inputs and targets, which are shifted by one timestep..
    random_split = True
    batch_size = 32
    print("inputs: ", len(inputs))
    train_loader, val_loader, test_loader = retrieval.build_dataset(inputs, targets,
            val_split=0.4, test_split=0.1, rnd_split=random_split, batch_size=batch_size, num_workers=8,)
    print(f"Train/val/test size: {len(train_loader)*batch_size}/{len(val_loader)*batch_size}/{len(test_loader)*batch_size}")
    print("Total data size: ", (len(train_loader)+len(val_loader)+len(test_loader))*batch_size)

    # RNN expects a sequence of inputs. So we should allow the retrieval
    # function to give a single dt value, and return a sequence with that
    # spacing for a number of samples. Try it with a range conversion to
    # list
    # TODO: Not working - the retrieval does not work with so many targets
    #       need to make the input interval more flexible. Perhaps
            # by finding the closest sample (using index and giving a guess?)
    hidden_size = 8 # Simple RNN needs same hidden size as output
    # which will hamper performance quite a bit.
    # model = RNN.OurRNN(dt_settings=dt, input_size=1, hidden_size=hidden_size,
    #              batch_first=True)
    model = RNN.CandleLSTM(instrument=ret_args.instrument, dt_settings=dt, input_size=1, hidden_size=hidden_size,
                 batch_first=True, num_layers=3)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.optim.lr_scheduler.MultiStepLR(optimizer, [4])
    # TODO: Implement learning rate annealing
    # loss_fn = nn.MSELoss() # TODO: Move this into model definition?
    loss_fn = nn.L1Loss()
    no_epochs = 1
    min_epochs = 1 # For early stopping
    val_interval = 1000000
    print_interval = 500
    # os.makedirs("")
    for i in range(1000):
        # Sets save_path as the first free slot in the pretrained models folder
        save_path = f"pre-trained models/RNN_{hidden_size}_{model_granularity}_i{i}.pt"
        if not os.path.isfile(save_path):
            break

    train(model, train_loader, val_loader, test_loader, optimizer, loss_fn, no_epochs, min_epochs, save_path, val_interval, print_interval)

    prediction = model.infer()
    set_trace()
    print("Prediction: ", prediction)



    
