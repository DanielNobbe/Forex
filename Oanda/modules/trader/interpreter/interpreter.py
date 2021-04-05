''' 
class Interpreter interprets the outcome of the decider and translates that in an action for the trader
'''

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
    
    def __init__(self, access_token, accountID, instrument):
        self.access_token = access_token
        self.accountID = accountID
        assert instrument in allowed_pairs, (
            f"Currency pair {instrument} not supported. "
            "Add to interpreter.yaml after updating safety rules. "
            "Note that base currencies other than EUR do not work "
            "with default buy/sell limits.")
        self.instrument = instrument
        
    # def RiskManager(self):
    #     OrdersOrderList(self, accountID) # add a restriction based on the orders
        
    #     # add a restrictions based on position and cash
    #     cash = AccountSummary(access_token, accountID)['account']['marginAvailable'] # cash is element in account summary
    #     portfoliosize = AccountSummary(access_token, accountID)['account']['balance'] # cash is element in account summary
    #     positions = AccountSummary(access_token, accountID)['account']['openPositionCount']
        
    #     maximum_position = min(cash, 0.25*portfoliosize-positions[instrument])
    #     return maximum_position
    
    def update_position(self, input_):
        # NAV = float(AccountSummary(self.access_token, self.accountID)[1]['account']['NAV'])
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
        self.owned_value = self.size*price # In base currency
        self.price = input_
        # return value
        # max_trade = 0.0005*NAV # ong 10 euro
        # print(size*price)
        
    def check_risk(self, action, amount=None):
        if amount is None:
            if action == 'buy':
                amount = int((self.upperbound-self.owned_value)/self.price) # A unit is 1 dollar here? TODO:
            elif action == 'sell':
                amount = int((self.lowerbound-self.owned_value)/self.price)
            else:
                raise ValueError(f"action should be buy or sell, got {action}")
        if action == 'buy':
            if self.owned_value + amount <= self.upperbound:
                # Allowed to buy up to upper bound
                return True, amount
            else:
                # Trying to buy too much
                print("Trade not allowed, attempting to increase total amount to more than upper bound.")
                return False, amount
        elif action == 'sell':
            if self.owned_value + amount >= self.lowerbound:
                # Allowed to buy down to lowerbound
                return True, amount
            else:
                print("Trade not allowed, attempting to increase debt to more than lower bound.")
                return False, amount
   
# =============================================================================
#         if prediction > input:
#             if size*price < upperbound:
#                 self.units = int(round(max_trade/price))
#             else:
#                 self.units = None
#         else:
#             if size*price > -upperbound:
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
            amount = 10
            allowed, amount_ret = self.check_risk('buy', amount)
            assert amount == amount_ret, "Mistake in check_risk function"
            if allowed:
                return 'buy', amount
            else:
                return False, amount
        elif prediction < input_:
            # Sell, short or hold?
            amount = -10
            allowed, amount_ret = self.check_risk('buy', amount)
            assert amount == amount_ret, "Mistake in check_risk function"
            if allowed:
                return 'sell', amount
            else:
                return False, amount

    def to_units(self, amount):
        # convert to units. 1 of base currency is 1 unit.
        return int(amount / self.price)

    def trade(self, prediction, latest_value):
        # input_ is the current value. We don't need it, 
        # if we keep track of it here. Where does it come from
        # though?

        self.update_position(latest_value)
        self.upperbound = 20000.
        self.lowerbound = -20000. # when going short
        
        buy_or_sell, amount = self.prepare_trade(latest_value, prediction)
        if buy_or_sell:
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
            print("Bought ", units, " ", self.instrument, " value of trade: ", self.size*input_)
        except Exception as e:
            print("Order was NOT accepted, value of trade: ", self.size*input_)
            print("Error: ", e)