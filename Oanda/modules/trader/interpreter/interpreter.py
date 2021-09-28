"""
The Interpreter handles all trades. It receives predictions from a 
Predictor, and decides whether to, and how much to trade.
"""

from pdb import set_trace
from libs.API.oanda import OrdersOrderCreate, PositionsPositionDetails
from libs.API.orders import MarketOrder, filter_dict
from libs.API.working_functions import readable_output
from modules.info.retrieval.exceptions import *
import numpy as np

import yaml
import sys, os
from copy import deepcopy

# Import safety config
this_path = os.path.relpath(__file__+'/../')
with open(this_path + "/safety.yaml") as file:
    int_cfg = yaml.full_load(file)

allowed_pairs = int_cfg['allowed_pairs']

def extract_last_entry(array):
    try:
        if len(array) > 0:
            array = array[-1]
            return extract_last_entry(array)
    except (TypeError):
        return array

class Interpreter():
    """
    The Interpreter is an object that tracks the current portfolio,
    for a specific instrument, and decides on trades. 
    Its predictor attribute contains a Predictor object, which predicts
    new prices.

    Attributes:
        accountID, access_token: credentials for Oanda
        instrument: the instrument, e.g. a currency pair like EUR_USD
        config: trading configuration dict, usually from a file in 
            configs/trading
        predictor: Predictor object
        upper_bound, lower_bound: max. and min. amount of instrument to 
            have in portfolio. Trading is disabled in one direction when
            respective bound is reached.
        amount: Value of self.instrument to buy/sell per trade.
            (in target currency)
        size: Number of units of self.instrument in portfolio
        owned_value: Current value in base currency in portfolio
            (equal to number of units)
        price: Latest retrieved price of instrument, value of 1 unit
            (base currency) in the target currency
    """
    def __init__(self, credentials: tuple, cfg: dict, predictor: object, rand_strat=False):
        """
        Args:
            credentials: credentials tuple, with first entry accountID
                and second access_token
            cfg: trading config dict, usually from a file in configs/trading/
            predictor: Predictor object, used to predict next candlestick value
        """
        self.accountID, self.access_token = credentials
        self.instrument = cfg['instrument']
        self.config = cfg
        self.predictor = predictor

        self.upper_bound = cfg['limits']['upper_bound']
        self.lower_bound = cfg['limits']['lower_bound']

        amount = cfg['limits']['amount']
        if amount == 'max':
            self.amount = None
        else:
            self.amount = amount

        assert self.instrument in allowed_pairs, (
            f"Currency pair {instrument} not supported. "
            "Add to interpreter.yaml after updating safety rules. "
            "Note that base currencies other than EUR do not work "
            "with default buy/sell limits.")
        self.rand_strat = rand_strat
        if rand_strat:
            print("WARNING: Bypassing model to randomly select buying/selling.")

        self.rng = np.random.default_rng()


    def update_position(self, input_):
        """
        Updates the self.size, self.owned_value and self.price attributes
        with their latest values.
        Args:
            input_: Current value of 1 unit of base currency in target currency
        """
        self.size = (float(
                        PositionsPositionDetails(self.access_token, 
                                                self.accountID,
                                                instrument=self.instrument
                                                )[1]['position']['long']['units'])
                    + float(
                        PositionsPositionDetails(self.access_token, 
                                                self.accountID, 
                                                instrument=self.instrument
                                                )[1]['position']['short']['units'])
                    ) # short values are returned negative
        self.owned_value = self.size # In base currency
        self.price = input_

        
    def check_risk(self, action, amount=None):
        """
        Checks whether we do not take too much risk. Currently only
        limits the amount owned to within the defined upper_ and 
        lower_bound. Can be used to check whether a given trade is 
        allowed (by specifying amount), or to check whether we can trade
        at all and give the max amount to buy/sell (by not specifying 
        amount). 
        Args:
            action: 'buy' or 'sell', to specify what kind of trade to make.
            amount: The amount (in target currency) to trade. Should be negative
                when selling.
        Returns:
            tuple:
                [0]: boolean whether trade is allowed
                [1]: amount to trade
        TODO: Should not need to define 'buy' or 'sell', unless amount
            is not specified.
        """
        if amount is None:
            # amount not specified, so determines max amount to trade
            if action == 'buy':
                amount = int((self.upper_bound-self.owned_value)/self.price) # A unit is 1 dollar here? TODO:
            elif action == 'sell':
                amount = int((self.lower_bound-self.owned_value)/self.price)
            else:
                raise ValueError(f"action should be buy or sell, got {action}")
        if action == 'buy':
            if self.owned_value + amount <= self.upper_bound:
                # Allowed to buy up to upper bound
                return True, amount
            else:
                # Trying to buy too much
                print("Trade not allowed, attempting to increase total amount to more than upper bound.")
                return False, amount
        elif action == 'sell':
            if self.owned_value + amount >= self.lower_bound:
                # Allowed to buy down to lower_bound
                return True, amount
            else:
                print("Trade not allowed, attempting to increase debt to more than lower bound.")
                return False, amount
   
    

    def prepare_trade(self, input_, prediction):
        """
        Function used to prepare trade. Uses prediction and current value
        to determine the action to take, and checks that action with the
        check_risk method. 
        Current strategy is to buy if prediction is larger than current 
        value, and sell if it is lower.
        Could be expanded to perform more checks, retrieve some information,
        or to use a different trading strategy.
        Args:
            input_: Value of target currency per unit of base currency
            prediction: Predicted value of target currency. 
                The time corresponding to this prediction is dependent
                on the model/predictor.
        Returns:
            tuple:
                [0]: 'buy' or 'sell' if the chosen action is allowed,
                    or False if it is not allowed or no action should
                    be taken.
                [1]: amount to trade. Also returned if trade is not made
        TODO: Should implement more complex strategies here
        """
        if self.rand_strat:
            diff = self.rng.choice([-0.1, 0.1])
            prediction = input_ + diff
        if prediction > input_:
            # Price will go up, so we should buy
            amount = self.amount
            allowed, amount_ret = self.check_risk('buy', amount)
            assert amount == amount_ret or amount == 'max', "Mistake in check_risk function"
            if allowed:
                return 'buy', amount_ret
            else:
                return False, amount_ret
        elif prediction < input_:
            # Sell, short or hold?
            amount = -1 * self.amount
            allowed, amount_ret = self.check_risk('buy', amount)
            assert amount == amount_ret, "Mistake in check_risk function"
            if allowed:
                return 'sell', amount_ret
            else:
                return False, amount_ret

    def to_units(self, amount):
        """
        Convert an amount in target currency to units (in base currency).
        A unit should be an integer multiple of the base currency.
        """
        return int(amount / self.price)

    def perform_trade(self):
        """
        Overarching method to perform a trade. Calls the predictor for 
        a prediction, and then the trade method to perform the trade.

        TODO: Predictor returns latest closing value, rather than real current value. 
        """
        try:
            prediction, latest_value = self.predictor()
        except (MissingSamplesError, MarketClosedError) as e:
            print(f"{type(e).__name__}: {e}\nCancelled this trade.")
            return
        latest_value = extract_last_entry(latest_value)
        self.trade(prediction, latest_value)

    def send_trade(self, units):
        
        data = deepcopy(MarketOrder)
        data['order']['units'] = units
        data['order']['instrument'] = self.instrument
        data['order']['timeInForce'] = "FOK"
        
        filter_dict(data)
        
        print(readable_output(data))
        try:
            OrdersOrderCreate(self.access_token, self.accountID, data=data)
            print("Bought ", units, " ", self.instrument, " value of trade: ", units*self.price)
        except Exception as e:
            print("Order was NOT accepted, value of trade: ", units*self.price)
            print("Error: ", e)

    def trade(self, prediction, latest_value):
        """
        Perform a trade, using the prediction given by the predictor.
        Calls prepare_trade method to make decisions, then uses API
        to request trade.
        Args:
            prediction: Predicted value that target currency of instrument
                will have in the future. How much time in the future is defined 
                by predictor.
            latest_value: Latest value of instrument.            
        """
        self.update_position(latest_value)
        buy_or_sell_allowed, amount = self.prepare_trade(latest_value, prediction)
        if buy_or_sell_allowed:
            units = self.to_units(amount)
        else:
            print(f"Can not buy or sell {amount} of {self.instrument}. Returning..")
            return
        self.send_trade(units)

        # data = deepcopy(MarketOrder)
        # data['order']['units'] = units
        # data['order']['instrument'] = self.instrument
        # data['order']['timeInForce'] = "FOK"
        
        # filter_dict(data)
        
        # print(readable_output(data))
        # try:
        #     OrdersOrderCreate(self.access_token, self.accountID, data=data)
        #     print("Bought ", units, " ", self.instrument, " value of trade: ", units*latest_value)
        # except Exception as e:
        #     print("Order was NOT accepted, value of trade: ", units*latest_value)
        #     print("Error: ", e)