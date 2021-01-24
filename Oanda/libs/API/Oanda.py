'''
Module for the import of information from Oanda
'''

'TODO: Check the other functions without #check'
'TODO: Check ForexlabsAutochartist with https://pages.oanda.com/technical-analysis-autochartist.html'

'QUESTION: Change streams into Munches to use them more easily'

'Functions with "check" behind them are known to be working'
'See http://developer.oanda.com/rest-live/orders/ for developement guide'
'(Almost) every functions has a return that you can print with a good visual and a return for adjustments'

# =============================================================================
# Imports
# =============================================================================

from munch import Munch
from json import dumps
from oandapyV20 import API
from oandapyV20.endpoints import accounts
from oandapyV20.endpoints import forexlabs
from oandapyV20.endpoints import instruments
from oandapyV20.endpoints import orders
from oandapyV20.endpoints import positions
from oandapyV20.endpoints import pricing
from oandapyV20.endpoints import trades
from oandapyV20.endpoints import transactions
from oandapyV20.exceptions import StreamTerminated
from libs.API.WorkingFunctions import ReadableOutput

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Account
# =============================================================================

def AccountChanges(access_token, accountID, params=None): # check
    'Endpoint used to poll an Account for its current state and changes since a specified TransactionID.'
    r = accounts.AccountChanges(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
    
def AccountConfiguration(access_token, accountID, data=None): # check
    'Set the client-configurable portions of an Account.'
    r = accounts.AccountConfiguration(accountID=accountID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def AccountDetails(access_token, accountID): # check
    'Get the full details for a single Account that a client has access to. Full pending Order, open Trade and open Position representations are provided.'
    r = accounts.AccountDetails(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def AccountInstruments(access_token, accountID, params=None): # check
    'Get the list of tradable instruments for the given Account. The list of tradeable instruments is dependent on the regulatory division that the Account is located in, thus should be the same for all Accounts owned by a single user.'
    r = accounts.AccountInstruments(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def AccountList(access_token): # check
    'Get a list of all Accounts authorized for the provided token.'
    r = accounts.AccountList()
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def AccountSummary(access_token, accountID): # check
    'Get a summary for a single Account that a client has access to.'
    r = accounts.AccountSummary(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

# =============================================================================
# Forexlabs
# =============================================================================
        
def ForexlabsAutochartist(access_token, params=None): # check
    'Get the ‘autochartist data’.'
    r = forexlabs.Autochartist(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def ForexlabsCalendar(access_token, params=None): # check
    'Get calendar information.'
    r = forexlabs.Calendar(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def ForexlabsCommitmentsOfTraders(access_token, params=None): # check
    'Get the ‘commitments of traders’ information for an instrument.'
    r = forexlabs.CommitmentsOfTraders(params=params) 
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def ForexlabsHistoricalPositionsRatios(access_token, params=None):
    'Error'
    r = forexlabs.HistoricalPositionRatios(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def ForexlabsOrderbookData(access_token, params=None):
    'Error'
    r = forexlabs.OrderbookData(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def ForexlabsSpreads(access_token, params=None):
    'Get the spread information for an instrument.'
    r = forexlabs.Spreads(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

# =============================================================================
# Instruments
# =============================================================================
        
def InstrumentsCandles(access_token, instrument, params): # check
    'Get candle data for a specified Instrument.'
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def InstrumentsOrderBook(access_token, instrument, params): # check
    'Get orderbook data for a specified Instrument.'
    r = instruments.InstrumentsOrderBook(instrument=instrument, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def InstrumentsPositionBook(access_token, instrument, params): # check
    'Get positionbook data for a specified Instrument.'
    r = instruments.InstrumentsPositionBook(instrument=instrument, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

# =============================================================================
# Orders
# =============================================================================

def OrdersOrderCancel(access_token, accountID, orderID):
    'Cancel a pending Order in an Account.'
    r = orders.OrderCancel(accountID=accountID, orderID=orderID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def OrdersOrderClientExtensions(access_token, accountID, orderID, data=None):
    'Update the Client Extensions for an Order in an Account. Warning: Do not set, modify or delete clientExtensions if your account is associated with MT4.'
    r = orders.OrderClientExtensions(accountID, orderID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
    
def OrdersOrderCreate(access_token, accountID, data=None):
    'Create an Order for an Account.'
    r = orders.OrderCreate(accountID=accountID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def OrdersOrderDetails(access_token, accountID, orderID):
    'Get details for a single Order in an Account.'
    r = orders.OrderDetails(accountID=accountID, orderID=orderID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def OrdersOrderList(access_token, accountID):
    'Get a list of orders for an account'
    r = orders.OrderList(accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def OrdersOrderReplace(access_token, accountID, orderID, data=None):
    'Replace an Order in an Account by simultaneously cancelling it and creating a replacement Order.'
    r = orders.OrderReplace(accountID=accountID, orderID=orderID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def OrdersOrdersPending(access_token, accountID):
    'List all pending Orders in an Account.'
    r = orders.OrdersPending(accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

# =============================================================================
# Positions
# =============================================================================
        
def PositionsOpenPosition(access_token, accountID): # check
    'List all open Positions for an Account. An open Position is a Position in an Account that currently has a Trade opened for it.'
    r = positions.OpenPositions(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def PositionsPositionClose(access_token, accountID, instrument, data=None):
    'Closeout the open Position regarding instrument in an Account.'
    r = positions.PositionClose(accountID=accountID, instrument=instrument, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def PositionsPositionDetails(access_token, accountID, instrument):
    'Get the details of a single instrument’s position in an Account. The position may be open or not.'
    r = positions.PositionDetails(accountID=accountID, instrument=instrument)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def PositionsPositionList(access_token, accountID): # check
    'List all Positions for an Account. The Positions returned are for every instrument that has had a position during the lifetime of the Account.'
    r = positions.PositionList(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

# =============================================================================
# Pricing
# =============================================================================
        
def PricingPricingInfo(access_token, accountID, params=None):
    'Get pricing information for a specified list of Instruments within an account.'
    r = pricing.PricingInfo(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
    
def PricingPricingStream(access_token, accountID, params=None): # check
    'Get realtime pricing information for a specified list of Instruments.'
    # terminate(message='') to terminate
    r = pricing.PricingStream(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    maxrecs = 100
    for ticks in r.response:
        print(dumps(ticks, indent = 4, separators=(',', ': ')))
        if maxrecs == 0:
            r.terminate("maxrecs records received")
            
# =============================================================================
# Trades
# =============================================================================
        
def TradesOpenTrades(access_token, accountID): # check
    'Get the list of open Trades for an Account.'
    r = trades.OpenTrades(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

def TradesTradeCRCDO(access_token, accountID, tradeID, data=None):
    'Trade Create Replace Cancel Dependent Orders.'
    r = trades.TradeCRCDO(accountID=accountID, tradeID=tradeID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
    
def TradesTradeClientExtensions(access_token, accountID, tradeID, data=None):
    'Update the Client Extensions for a Trade. Do not add, update or delete the Client Extensions if your account is associated with MT4.'
    r = trades.TradeClientExtensions(accountID=accountID, tradeID=tradeID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
    
def TradesTradeClose(access_token, accountID, tradeID, data=None):
    'Close (partially or fully) a specific open Trade in an Account.'
    r = trades.TradeClose(accountID=accountID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
   
def TradesTradeDetails(access_token, accountID, tradeID):
    'Get the details of a specific Trade in an Account.'
    r = accounts.TradeDetails(accountID=accountID, tradeID=tradeID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
    
def TradesTradesList(access_token, accountID, params=None):
    'Get a list of trades for an Account.'
    r = trades.TradesList(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)

# =============================================================================
# Transactions
# =============================================================================
       
def TransactionsTransactionDetails(access_token, accountID, transactionID):
    'Get the details of a single Account Transaction.'
    r = transactions.TransactionDetails(accountID=accountID, transactionID=transactionID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def TransactionsTransactionIDRange(access_token, accountID, params=None): # check
    'Get a range of Transactions for an Account based on Transaction IDs.'
    r = transactions.TransactionIDRange(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def TransactionsTransactionList(access_token, accountID, params=None): # check
    'Get a list of Transactions pages that satisfy a time-based Transaction query.'
    r = transactions.TransactionList(accountID=accountID, params=params)  
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def TransactionsTransactionsSinceID(access_token, accountID, params=None): # check
    'Get a range of Transactions for an Account starting at (but not including) a provided Transaction ID.'
    r = transactions.TransactionsSinceID(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(Munch(r.response)), Munch(r.response)
        
def TransactionsTransactionsStream(access_token, accountID, params=None): #check
    'Get a stream of Transactions for an Account starting from when the request is made.'
    # terminate(message='') to terminate
    r = transactions.TransactionsStream(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    maxrecs = 5
    try:
        for ticks in r.response:
            print(dumps(ticks, indent = 4, separators=(',', ': ')))
            maxrecs -= 1
            if maxrecs == 0:
                r.terminate("Got them all")
    except StreamTerminated as e:
        print("Finished: {msg}".format(msg=e))        