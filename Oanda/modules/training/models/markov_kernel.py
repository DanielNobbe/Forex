import torch
import torch.nn as nn
import numpy as np
import modules.training.retrieval as retrieval
from pdb import set_trace

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
    def __init__(self, input_size, hidden_size, output_size, dt_settings=None, regression=True, instrument=None):
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
        self.instrument = instrument
        self.notes = ''
        
    def __str__(self):
      return f"{make_ordinal(self.input_size)} order Markov Kernel model"
    
    def forward(self, value):
        assert self.dt_settings is not None, "dt_settings required for using model. Have you loaded the model correctly?"
        output = self.model(value)
        return output
    
    def retrieve_for_inference(self):
        assert self.dt_settings is not None, "dt_settings required for using model. Have you loaded the model correctly?"
        print("Instrument: ", self.instrument)
        data = retrieval.retrieve_inference_data(
            self.instrument,
            dt = self.dt_settings,
        ).unsqueeze(dim=0)
        # TODO: Add soft margin to this automatically 
        return data

    def infer(self, test_data=None):
        if test_data is None:
            with torch.no_grad():
                data = self.retrieve_for_inference()
                output = self.forward(data)
                return output[-1, -1] # Final value is prediction
        else:
            output = self.forward(test_data)
            return output[-1]

    # def classify(self, history):
    #     classification = self.model(history[-1])
    #     return classification
        