import torch
import torch.nn as nn
import numpy as np
import modules.info.retrieval as retrieval
from pdb import set_trace
"""
Each of our models should, by default, only return a single output, 
which should be a torch.tensor (float)
"""

class OurRNN(nn.RNN):
    # Don't modify yet
    def __init__(self, dt_settings, nonlinearity='relu', *args, **kwargs):
        print("Warning: Standard RNN is not suitable for time series.")
        self.dt_settings = dt_settings
        super(OurRNN, self).__init__(*args, **kwargs)

class CandleLSTM(nn.Module):
    def __init__(self, instrument,  hidden_size, output_size=1, dt_settings=None, *args, **kwargs):
        super().__init__()
        self.dt_settings = dt_settings
        self.instrument = instrument
        self.LSTM = nn.LSTM(*args, hidden_size=hidden_size, **kwargs)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, values):
        assert self.dt_settings is not None, "dt_settings required for using model. Have you loaded the model correctly?"
        output, hidden_state = self.LSTM(values)
        # set_trace()
        return self.output_layer(output)
    
    def retrieve_for_inference(self):
        assert self.dt_settings is not None, "dt_settings required for using model. Have you loaded the model correctly?"
        data = retrieval.retrieve_inference_data(
            self.instrument,
            dt = self.dt_settings,
        ).unsqueeze(dim=0).unsqueeze(dim=2)
        # TODO: Add soft margin to this automatically 
        return data
    
    def infer(self):
        with torch.no_grad():
            data = self.retrieve_for_inference()
            # set_trace()
            output = self.forward(data)
            return output[-1, -1, -1] # Final value is prediction

         