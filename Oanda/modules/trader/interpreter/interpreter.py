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
    
    def Units(self, input, prediction):
        # NAV = float(AccountSummary(self.access_token, self.accountID)[1]['account']['NAV'])
        self.size = float(PositionsPositionDetails(self.access_token, self.accountID, instrument=self.instrument)[1]['position']['long']['units']) + float(PositionsPositionDetails(self.access_token, self.accountID, instrument=self.instrument)[1]['position']['short']['units'])
        price = input
        
        upperbound = 20000 # 1000 euro
        # max_trade = 0.0005*NAV # ong 10 euro
        # print(size*price)
        
        if prediction > input:
            if self.size*price < upperbound:
                self.units = int(round(upperbound/price))
            else:
                self.units = None
        else:
            if self.size*price > -upperbound:
                self.units = -int(round(upperbound/price))
            else:
                self.units = None
        
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
                    
    def Trade(self, input, prediction):
        
        self.Units(input, prediction)
            
        data = MarketOrder
        data['order']['units'] = self.units
        data['order']['instrument'] = self.instrument
        data['order']['timeInForce'] = "FOK"
        
        FilterDict(data)
        
        print(ReadableOutput(data))
        
        try:
            OrdersOrderCreate(self.access_token, self.accountID, data=data)
            print("Bought ", self.units, " ", self.instrument, " value of trade: ", self.size*input)
        except:
            print("Order was NOT accepted, value of trade: ", self.size*input)