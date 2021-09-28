from modules.trader.interpreter import Interpreter
from backtesting import Strategy

### Temporary
from libs.API import *
import yaml
import sys, os
from modules.trader.predictor import Predictor
from modules.info.retrieval import retrieve_backtest_input, MarketClosedError, MissingSamplesError
###
from backtesting.test import GOOG
from backtesting import Backtest

import pandas as pd

from pdb import set_trace

# class interpreter_strategy(Interpreter, Strategy):
#     ## We won't use super() for init here, since that 
#     # does not play well with multiple inheritance. 
#     # We could also initialise an interpreter instance into a strategy
#     # instance as a self.interpreter, but this does not allow easy
#     # overriding of existing methods.
#     def __init__(self):
#         Interpreter.__init__()


## Ok let's do this differently:
# Create an interpreter that takes as input a Strategy object, so it
# can use the necessary functions. Otherwise should inherit from 
# Interpreter. We could change this later on to returning 'sell' or
# 'buy' with a number
class BacktestInterpreter(Interpreter):
    def __init__(self, credentials: tuple, cfg: dict, 
    predictor: object, strategy: Strategy, rand_strat=False):

        super().__init__(credentials, cfg, predictor, rand_strat=rand_strat)
        self.strategy = strategy

    def send_trade(self, units):
        print(f"Sending backtesting trade of {units}")
        if units>=1: # otherwise interpreted as a fraction of max buyable
            print("Buying (in backtest)")
            self.strategy.position.close()
            self.strategy.buy(size=units)
        elif units<=-1:
            print("Selling (in backtest)")
            self.strategy.position.close() 
            # This closes the previous trade before opening a new one,
            # which at this stage makes the most sense.
            # We could later implement something where we match a new short
            # with an open long trade, although that might not really give 
            # us any benefit
            self.strategy.sell(size=units)

    def update_position(self, input_):
        self.size = self.strategy.position.size
        self.owned_value = self.size
        self.price = input_

    def perform_trade(self, data):
        """
        Overarching method to perform a trade. Calls the predictor for 
        a prediction, and then the trade method to perform the trade.

        TODO: Predictor returns latest closing value, rather than real current value. 
        NOTE: This version receives data as input, and should retrieve the relevant samples.
        For now, we only use the latest sample and pretend we use a 
        first order markov kernel.
        TODO: Expand this to work with other architectures, requires 
        extracting the relevant samples
        """
        # try:
        current_value = data.Close[-1]
        current_time = data.index[-1]
        dt_settings = self.predictor.model.dt_settings
        ## data.Close is a list with all currently available values, 
        # the final at -1
        ## data.index is the list of all corresponding timestamps
        try:
            input_ = retrieve_backtest_input(data, dt_settings)
        except (MarketClosedError, MissingSamplesError) as e:
            print("Captured a MarketClosed or MissingSamples error. Continuing..")
            # This either means there is a problem with the retrieval,
            # or that we are at the beginning or end of the timeseries
            return
        ## Next step: modify what current_value is
        # For other types of models, we need to extract a different
        # subset of the data. This is defined by dt_settings
        # We can use the dt_settings to extract the right data timesteps
        # but to do that we need to know which time corresponds to each
        # time series point. TODO: CHeck if we have this info
        # The dataframe should be indexed with the datetime

        # TODO: Use the period as specified in the trading settings
        # TODO: Move model-specific extraction to the model definition?
        prediction, latest_value = self.predictor.predict_with_input(input_)
        # except (MissingSamplesError,MarketClosedError) as e:
            # print(f"{type(e).__name__}: {e}\nCancelled this trade.")
            # return
        
        # NOTE: Trades made in backtesting are good-till-cancelled
        
        self.trade(prediction, latest_value)
    
    
    



class InterpreterStrategy(Strategy):
    def init(self):
         ## And then also create this strategy, which contains the backtest 
        # interpreter
        trade_config_file = 'trade.yaml'
        this_path = os.path.relpath(__file__+'/../../')
        trading_cfg_path = '../configs/trading/'
        model_cfg_path = '../configs/models/'
        relative_path = os.path.join(this_path, trading_cfg_path, trade_config_file)
        with open(relative_path) as file:
            cfg = yaml.full_load(file)

        access_token = API_CONFIG[cfg['account_type']]['access_token']
        accountID = API_CONFIG[cfg['account_type']]['accountID']
        credentials = (accountID, access_token)

        predictor = Predictor.build_from_cfg(cfg)
        rand_strat = False
        ## TODO: Create a function to generate a Strategy object
        # from the required combination of predictor, credentials, cfg
        self.interpreter = BacktestInterpreter(credentials, cfg, 
                                predictor, self, rand_strat)
        # TODO: Is there a way to use the interpreter output as 
        # an indicator?
        self.latest_time = 0
        self.period = cfg['period']

    def next(self):
        # This is the core of the Strategy. We can get the latest
        # bit of data through self.data[-1], and insert it into the 
        # interpreter. 
        # Perhaps would be best to use something like a list w times?
        # or something like the timeseries object we already use
        # current_value = self.data.Close[-1]
        # Pass self.data to interpreter to perform a trade
        current_time = self.data.index[-1].timestamp()
        time_difference = current_time - self.latest_time
        if (self.period <= time_difference < 1.2*self.period) or self.latest_time==0:
            self.interpreter.perform_trade(self.data)
            self.latest_time = current_time
        elif time_difference > 1.2*self.period:
            print(f"Trade interval much longer than specified period at {time_difference} (period: {self.period})")
            self.interpreter.perform_trade(self.data)
            self.latest_time = current_time
        else:
            return




def main():
    bt = Backtest(GOOG, InterpreterStrategy, cash=10_000, commission=.002, trade_on_close=True)
    
    stats = bt.run()
    print(stats)
    # strategy = InterpreterStrategy()

