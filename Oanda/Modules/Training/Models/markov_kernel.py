import torch.nn as nn
import numpy as np

def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix

class MarkovKernel(nn.Module):
    # Simple neural network for markov kernel
    def __init__(self, input_size, hidden_size, output_size, dt_settings=None, regression=True):
        super(MarkovKernel, self).__init__()
        self.input_size = input_size
        layer_list = [
            nn.Linear(input_size, hidden_size[0]), # input size determines order of markov kernel
            nn.Tanh(),]
        for hidden_layer_index in np.arange(1, len( hidden_size ) ):
          layer_list.append( nn.Linear( hidden_size[hidden_layer_index-1], \
            hidden_size[hidden_layer_index]  ) ) # take in_features out_features into account
          layer_list.append( nn.Tanh() ) 
        
        layer_list.append( nn.Linear(hidden_size[-1],  output_size ) )

        if not regression:
          layer_list.append(nn.Softmax())
        
        self.model = nn.Sequential( *layer_list ) #Unpacks list for sequential
        
        self.dt_settings = dt_settings
        
    def __str__(self):
      return f"{make_ordinal(self.input_size)} order Markov Kernel model"
    
    def forward(self, value):
        assert self.dt_settings is not None, "dt_settings required for using model. Have you loaded the model correctly?"
        output = self.model(value)
        return output
    
    # def classify(self, history):
    #     classification = self.model(history[-1])
    #     return classification
        