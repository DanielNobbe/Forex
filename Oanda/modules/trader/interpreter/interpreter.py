"""
The Interpreter handles all trades. It receives predictions from a 
Predictor, and decides whether to, and how much to trade.
"""
# =============================================================================
# Imports
# =============================================================================

from libs.API.Oanda import OrdersOrderCreate, PositionsPositionDetails
from libs.API.Orders import MarketOrder, filter_dict
from libs.API.WorkingFunctions import readable_output

import yaml
import sys, os

# =============================================================================
# Class
# =============================================================================
this_path = os.path.relpath(__file__+'/../')
with open(this_path + "/safety.yaml") as file:
    int_cfg = yaml.full_load(file)

allowed_pairs = int_cfg['allowed_pairs']

class Interpreter():
    """
    The Interpreter is an object that tracks the current portfolio,
    for a specific instrument, and decides on trades. 
    Its predictor attribute contains a Predictor object, which predicts
    new prices.

    Attributes:
    - accountID, access_token: credentials for Oanda
    - instrument: the instrument, e.g. a currency pair like EUR_USD
    - config: trading configuration dict, usually from a file in 
        configs/trading
    - predictor: Predictor object
    - upper_bound, lower_bound: max. and min. amount of instrument to 
        have in portfolio. Trading is disabled in one direction when
        respective bound is reached.
    - amount: Value of self.instrument to buy/sell per trade.
        (in target currency)
    - size: Number of units of self.instrument in portfolio
    - owned_value: Current value in base currency in portfolio
        (equal to number of units)
    - price: Latest retrieved price of instrument, value of 1 unit
        (base currency) in the target currency
    """
    def __init__(self, credentials: tuple, cfg: dict, predictor: object):
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
    
    def update_position(self, input_):
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
                    )
        self.owned_value = self.size # In base currency
        self.price = input_

        
    def check_risk(self, action, amount=None):
        if amount is None:
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
   
# =============================================================================
#         if prediction > input:
#             if size*price < upper_bound:
#                 self.units = int(round(max_trade/price))
#             else:
#                 self.units = None
#         else:
#             if size*price > -upper_bound:
#                 self.units = -int(round(max_trade/price))
#             else:
#                 self.units = None
# =============================================================================

    def prepare_trade(self, input_, prediction):
        # Determines whether to buy or sell
        # based on prediction, and checks this with risk management
        # TODO: Should implement more complex strategies here
        # TODO: Add max amount to buy here, use config file
        # TODO: Modify to allow for max buy/sell through calculation in check_risk
        if prediction > input_:
            # Price will go up, so we should buy
            # amount = self.amount
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
        # convert to units. 1 of base currency is 1 unit.
        return int(amount / self.price)

    # TODO: Reorganise these function calls (this should be trade_main or so)
    def perform_trade(self):
        # retrieve_current_price(access_token, accountID)
        prediction, latest_value = self.predictor()
        # TODO: Predictor returns latest closing value, rather than real current value. 
        # check whether this works well#
        self.trade(prediction, latest_value)

    def trade(self, prediction, latest_value):
        # input_ is the current value. We don't need it, 
        # if we keep track of it here. Where does it come from
        # though?

        self.update_position(latest_value)
        # self.upper_bound = 20000.
        # self.lower_bound = -20000. # when going short
        # TODO: move history/retrieval from training to info folder, should be 'standalone'
        buy_or_sell_allowed, amount = self.prepare_trade(latest_value, prediction)
        if buy_or_sell_allowed:
            units = self.to_units(amount)
        else:
            print(f"Can not buy or sell {amount} of {self.instrument}. Returning..")
            return

        data = MarketOrder
        data['order']['units'] = units
        data['order']['instrument'] = self.instrument
        data['order']['timeInForce'] = "FOK"
        
        filter_dict(data)
        
        print(readable_output(data))
        try:
            OrdersOrderCreate(self.access_token, self.accountID, data=data)
            print("Bought ", units, " ", self.instrument, " value of trade: ", self.size*latest_value)
        except Exception as e:
            print("Order was NOT accepted, value of trade: ", self.size*latest_value)
            print("Error: ", e)