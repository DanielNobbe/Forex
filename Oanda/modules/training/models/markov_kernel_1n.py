import torch.nn as nn
import numpy as np

class MarkovKernel(nn.Module):
    # Simple neural network for markov kernel
    def __init__(self, hidden_size, output_size, regression=True):
        super(MarkovKernel, self).__init__()
        layer_list = [
            nn.Linear(1, hidden_size[0]), 
            nn.Tanh(),]
        for hidden_layer_index in np.arange(1, len( hidden_size ) ):
          layer_list.append( nn.Linear( hidden_size[hidden_layer_index-1], \
            hidden_size[hidden_layer_index]  ) ) # take in_features out_features into account
          layer_list.append( nn.Tanh() ) 
        
        layer_list.append( nn.Linear(hidden_size[-1],  output_size ) )

        if not regression:
          layer_list.append(nn.Softmax())
        
        self.model = nn.Sequential( *layer_list ) #Unpacks list for sequential
        
    def __str__(self):
      return "First order Markov Kernel model"
    
    def forward(self, value):
        classification = self.model(value)
        return classification
    
    def classify(self, history):
        classification = self.model(history[-1])
        return classification
        