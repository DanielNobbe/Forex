''' 
class Interpreter interprets the outcome of the decider and translates that in an action for the trader
'''

from libs.API.Oanda import *

class Interpreter():
    
    def __init__(self, instrument, probabilities, value):
        
    def RiskManager(self):
        OrdersOrderList(self, accountID) # add a restriction based on the orders
        
        # add a restrictions based on position and cash
        cash = AccountSummary(access_token, accountID)['account']['marginAvailable'] # cash is element in account summary
        portfoliosize = AccountSummary(access_token, accountID)['account']['balance'] # cash is element in account summary
        positions = AccountSummary(access_token, accountID)['account']['openPositionCount']
        
        maximum_position = min(cash, 0.25*portfoliosize-positions[instrument])
        return maximum_position
    
    def ToTrade(self, instrument, probabilities, value):
        # Trade instrument based on probabilities an values
        
    
    def 
        
        