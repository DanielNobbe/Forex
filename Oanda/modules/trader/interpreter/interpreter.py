''' 
class Interpreter interprets the outcome of the decider and translates that in an action for the trader
'''

# =============================================================================
# Imports
# =============================================================================

from libs.API.Oanda import OrdersOrderCreate, PositionsPositionDetails
from libs.API.Orders import MarketOrder, FilterDict
from libs.API.WorkingFunctions import ReadableOutput

# =============================================================================
# Class
# =============================================================================

class Interpreter():
    
    def __init__(self, access_token, accountID, instrument):
        self.access_token = access_token
        self.accountID = accountID
        self.instrument = instrument
        
    # def RiskManager(self):
    #     OrdersOrderList(self, accountID) # add a restriction based on the orders
        
    #     # add a restrictions based on position and cash
    #     cash = AccountSummary(access_token, accountID)['account']['marginAvailable'] # cash is element in account summary
    #     portfoliosize = AccountSummary(access_token, accountID)['account']['balance'] # cash is element in account summary
    #     positions = AccountSummary(access_token, accountID)['account']['openPositionCount']
        
    #     maximum_position = min(cash, 0.25*portfoliosize-positions[instrument])
    #     return maximum_position
    
    def update_value(self, input):
        # NAV = float(AccountSummary(self.access_token, self.accountID)[1]['account']['NAV'])
        self.size = float(PositionsPositionDetails(self.access_token, self.accountID, instrument=self.instrument)[1]['position']['long']['units']) + float(PositionsPositionDetails(self.access_token, self.accountID, instrument=self.instrument)[1]['position']['short']['units'])
        self.owned_value = self.size*price
        self.price = input
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
                return False, amount
        elif action == 'sell':
            if self.owned_value + amount >= self.lowerbound:
                # Allowed to buy down to lowerbound
                return True, amount
            else:
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
    def prepare_trade(self, input, prediction):
        # Determines whether to buy or sell
        # based on prediction, and checks this with risk management
        # TODO: Should implement more complex strategies here
        # TODO: Add max amount to buy here, use config file
        if prediction > input:
            # Price will go up, so we should buy
            amount = 10
            allowed, amount_ret = self.check_risk('buy', amount)
            assert amount == amount_ret, "Mistake in check_risk function"
            if allowed:
                return 'buy', amount
            else:
                return False, amount
        elif prediction < input:
            # Sell, short or hold?
            amount = -10
            allowed, amount_ret = self.check_risk('buy', amount)
            assert amount == amount_ret, "Mistake in check_risk function"
            if allowed:
                return 'sell', amount
            else:
                return False, amount

    def to_units(self, amount):
        # TODO: Uses dollars as units?
        return amount / self.price

    def Trade(self, input, prediction):
        
        self.update_value(input)
        self.upperbound = 20000.
        self.lowerbound = -20000. # when going short
        
        buy_or_sell, amount = self.prepare_trade(input, prediction)
        if buy_or_sell:
            units = to_units(amount)
        else:
            print(f"Can not buy or sell {amount} of {self.instrument}. Returning..")
            return

        data = MarketOrder
        data['order']['units'] = units
        data['order']['instrument'] = self.instrument
        data['order']['timeInForce'] = "FOK"
        
        FilterDict(data)
        
        print(ReadableOutput(data))
        try:
            OrdersOrderCreate(self.access_token, self.accountID, data=data)
            print("Bought ", self.units, " ", self.instrument, " value of trade: ", self.size*input)
        except Exception as e:
            print("Order was NOT accepted, value of trade: ", self.size*input)
            print("Error: ", e)