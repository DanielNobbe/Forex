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
    """
    DO NOT USE.
    Simple RNN, based on the torch.nn.RNN built-in model. It has
    not been modified to work with time series yet.
    """
    def __init__(self, dt_settings, nonlinearity='relu', *args, **kwargs):
        print("Warning: Standard RNN is not suitable for time series.")
        self.dt_settings = dt_settings
        super(OurRNN, self).__init__(*args, **kwargs)

class CandleLSTM(nn.Module):
    """
    Model based on the PyTorch built-in LSTM model. This one is suitable
    and tested to be used with time series.
    Attributes:
        `dt_settings`: Input time interval settings.
            See modules/info/history.py for the definition of dt_settings
        `instrument`: The instrument for which this model is built and trained.
        `LSTM`: Underlying LSTM model, based on the PyTorch built in one.
        `output_layer`: Linear layer used to convert hidden size to output size.
    """
    def __init__(self, instrument,  hidden_size, output_size=1, dt_settings=None, soft_margin=2, *args, **kwargs):
        """
        `instrument`: The instrument for which this model is built and trained.
        `hidden_size`: Hidden size to use inside the LSTM
        `output_size`: Output size, usually 1 per timestep for time series.
        `dt_settings`: Input time interval settings. See modules/info/history.py 
            for the definition of dt_settings
        `soft_margin`: Soft_margin settings for retrieval. See
                modules/info/history.py for the definition.
        """
        super().__init__()
        self.dt_settings = dt_settings
        self.soft_margin = soft_margin
        self.instrument = instrument
        self.LSTM = nn.LSTM(*args, hidden_size=hidden_size, **kwargs)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, values):
        """
        Forward pass through the model.
        Args:
            `values`: Input for the model. Should be sequence.
        """
        assert self.dt_settings is not None, "dt_settings required for using model. Have you loaded the model correctly?"
        values = values.unsqueeze(dim=2)
        output, hidden_state = self.LSTM(values)
        return self.output_layer(output).squeeze(dim=2)
    
    def retrieve_for_inference(self):
        """
        Retrieves all samples required for inference, with the last sample
        corresponding to the current time.
        """
        assert self.dt_settings is not None, "dt_settings required for using model. Have you loaded the model correctly?"
        data = retrieval.retrieve_inference_data(
            self.instrument,
            dt = self.dt_settings,
            soft_margin=self.soft_margin
        ).unsqueeze(dim=0).unsqueeze(dim=2)
        # TODO: Add soft margin to this automatically 
        return data
    
    def infer(self):
        """
        Runs inference. First retrieves the required samples to do 
        inference from the current time, then calls the model to 
        perform the inference.
        """
        with torch.no_grad():
            data = self.retrieve_for_inference()
            # set_trace()
            output = self.forward(data)
            return output[-1, -1, -1] # Final value is prediction

         