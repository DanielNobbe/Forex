import torch
import torch.nn as nn
from test_scripts.Models.Markov_Kernel_1n import MarkovKernel
import numpy as np

def main():
    
    model = Import()
    kans_input = model.classify(historical_values)
    
    a,b,c,d,e = 1,1,1,1,1 # def optimize a,b,c,d,e
    
    transactiekosten_percentage = 0.02 # transactioncosts
    margin # The margin of the leverage # is this per trade of on the whole portfolio
    portfolio_0 # The leveraged amount, which is tradable
    
    huidige_positie = ["long", "short", "cash"] # everything you go short with is not added to cash, because we want to be able to buy back
    portfolio = huidige_positie.sum()
    
    short_verkopen = 1/(1+a*transactiekosten_percentage) * kans_input[0] * b * huidige_positie[1]
    long = 1/(1+a*transactiekosten_percentage) * kans_input[0] * c * huidige_positie[2]
    long_verkopen = 1/(1+a*transactiekosten_percentage) * kans_input[2] * d * huidige_positie[0]
    short = 1/(1+a*transactiekosten_percentage) * kans_input[2] * e * huidige_positie[2]
    
    # def trade:
    #     if long_verkopen > huidige_positie[0]:
    #         return "verkoop long and go short"
    #     if short_verkopen > huidige_positie[1]:
    #         return "verkoop short and go long"
    
    # # hoeveel_houden blijft gelijk, wanneer houden moet gebaseerd worden op je transactiekosten
    
    # if portfolio_0 - portfolio <= margin: # trades are closed if the margin stored for the leverage has become 0
    #     print("kut hé") # hoe nemen we dit mee in function trade
        
    # def "minimize expected profit of change in portfolio minus transactioncosts"
    
    # def "optimize dt" # nu S30 (30 sec)
    
    # def ""

def Import():
    model = MarkovKernel([16])
    model.load_state_dict(torch.load('Pre-trained Models/markov_kernel_n1.pt'))
    return model

if __name__ == '__main__':
    main()

    