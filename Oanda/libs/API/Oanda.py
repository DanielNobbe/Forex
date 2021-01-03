'''
Module for the import of information from Oanda
'''

'Get rid of the spaces'
'Readable output put everywhere'

import munch
import json
from oandapyV20 import API
from libs.API.WorkingFunctions import ReadableOutput
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.forexlabs as labs
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.transactions as trans

def AccountChanges(self, access_token, accountID, params=None): # check
    'Endpoint used to poll an Account for its current state and changes since a specified TransactionID.'
    # params = {}
    # params['sinceTransactionID'] = 26
    r = accounts.AccountChanges(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(munch.Munch(r.response))
    
def AccountConfiguration(self, access_token, accountID, data=None): # check, but comment
    'Set the client-configurable portions of an Account.'
    # What can you configure besides margin rate?
    r = accounts.AccountConfiguration(accountID=accountID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(munch.Munch(r.response))
        
def AccountDetails(self, access_token, accountID): # check
    'Get the full details for a single Account that a client has access to. Full pending Order, open Trade and open Position representations are provided.'
    r = accounts.AccountDetails(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(munch.Munch(r.response))
        
def AccountInstruments(self, access_token, accountID, params=None): # check
    'Get the list of tradable instruments for the given Account. The list of tradeable instruments is dependent on the regulatory division that the Account is located in, thus should be the same for all Accounts owned by a single user.'
    r = accounts.AccountInstruments(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(munch.Munch(r.response))
        
def AccountList(self, access_token): # check
    'Get a list of all Accounts authorized for the provided token.'
    r = accounts.AccountList()
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(munch.Munch(r.response))
        
def AccountSummary(access_token, accountID): # check
    'Get a summary for a single Account that a client has access to.'
    r = accounts.AccountSummary(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return ReadableOutput(munch.Munch(r.response))
        
def ForexlabsAutochartist(access_token, params=None): # check, with comment
    'Get the ‘autochartist data’.'
    # https://pages.oanda.com/technical-analysis-autochartist.html
    # params = {
    #     "instrument": "EUR_JPY"
    # }
    r = labs.Autochartist(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def ForexlabsCalendar(access_token, params=None): # check, is volgens mij echt chill
    'Get calendar information.'
    # params = {
    #   "instrument": "EUR_USD",
    #   "period": 86400
    # }
    r = labs.Calendar(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def ForexlabsCommitmentsOfTraders(access_token, params=None): # check
    'Get the ‘commitments of traders’ information for an instrument.'
    # params = {
    #   "instrument": "EUR_USD"
    # }   
    r = labs.CommitmentsOfTraders(params=params) 
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def ForexlabsHistoricalPositionsRatios(access_token, params=None): # error while running, there is an alternative 
    'Error'
    # params = {
    #   "instrument": "EUR_USD",
    #   "period": 86400
    # }
    r = labs.HistoricalPositionRatios(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def ForexlabsOrderbookData(access_token, params=None): # error while running, there is an alternative 
    'Error'
    # params = {
    #   "instrument": "EUR_USD",
    #   "period": 3600
    # }
    r = labs.OrderbookData(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def ForexlabsSpreads(access_token, params=None):
    'Get the spread information for an instrument.'
    # params = {
    #   "instrument": "EUR_USD",
    #   "period": 57600
    # }
    r = labs.Spreads(params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def InstrumentsCandles(access_token, instrument, params): # check
    'Get candle data for a specified Instrument.'
    # instrument = "DE30_EUR"
    # params = {
    #   "count": 5,
    #   "granularity": "M5"
    # }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def InstrumentsOrderBook(access_token, instrument, params): # check
    'Get orderbook data for a specified Instrument.'
    # instrument="EUR_USD"
    # params = {}
    r = instruments.InstrumentsOrderBook(instrument=instrument, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def InstrumentsPositionBook(access_token, instrument, params): # check
    'Get positionbook data for a specified Instrument.'
    r = instruments.InstrumentsPositionBook(instrument=instrument, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def OrdersOrderCancel(access_token, accountID, orderID):
    'Cancel a pending Order in an Account.'
    r = orders.OrderCancel(accountID=accountID, orderID=orderID)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def OrdersOrderClientExtensions(access_token, accountID, orderID, data=None):
    'Update the Client Extensions for an Order in an Account. Warning: Do not set, modify or delete clientExtensions if your account is associated with MT4.'
    r = orders.OrderClientExtensions(accountID, orderID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
    
def OrdersOrderCreate(access_token, accountID, data=None):
    'Create an Order for an Account.'
    r = orders.OrderCreate(accountID=accountID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def OrdersOrderDetails(access_token, accountID, orderID):
    'Get details for a single Order in an Account.'
    r = orders.OrderDetails(accountID=accountID, orderID=orderID)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def OrdersOrderList(access_token, accountID):
    'Get a list of orders for an account'
    r = orders.OrderList(accountID)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def OrdersOrderReplace(access_token, accountID, orderID, data=None):
    'Replace an Order in an Account by simultaneously cancelling it and creating a replacement Order.'
    # data = {
    #   "order": {
    #     "units": "-500000",
    #     "instrument": "EUR_USD",
    #     "price": "1.25000",
    #     "type": "LIMIT"
    #   }
    # }
    r = orders.OrderReplace(accountID=accountID, orderID=orderID, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def OrdersOrdersPending(access_token, accountID):
    'List all pending Orders in an Account.'
    r = orders.OrdersPending(accountID)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def PositionsOpenPosition(access_token, accountID): # check
    'List all open Positions for an Account. An open Position is a Position in an Account that currently has a Trade opened for it.'
    r = positions.OpenPositions(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def PositionsPositionClose(access_token, accountID, instrument, data=None):
    'Closeout the open Position regarding instrument in an Account.'
    # data = {
    #   "longUnits": "ALL"
    # }
    r = positions.PositionClose(accountID=accountID, instrument=instrument, data=data)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def PositionsPositionDetails(access_token, accountID, instrument):
    'Get the details of a single instrument’s position in an Account. The position may be open or not.'
    r = positions.PositionDetails(accountID=accountID, instrument=instrument)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)

def PositionsPositionList(access_token, accountID): # check
    'List all Positions for an Account. The Positions returned are for every instrument that has had a position during the lifetime of the Account.'
    r = positions.PositionList(accountID=accountID)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
        
def PricingPricingInfo(access_token, accountID, params=None):
    'Get pricing information for a specified list of Instruments within an account.'
    params = {
      "instruments": "EUR_USD,EUR_JPY"
    }
    r = pricing.PricingInfo(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    return munch.Munch(r.response)
    
def PricingPricingStream(access_token, accountID, params=None): # check, very nicee
    'Get realtime pricing information for a specified list of Instruments.'
    # params = {
    #   "instruments": "EUR_USD,EUR_JPY"
    # }
    r = pricing.PricingStream(accountID=accountID, params=params)
    client = API(access_token=access_token)
    client.request(r)
    maxrecs = 100
    for ticks in r.response:
        print(json.dumps(ticks, indent = 4, separators=(',', ': ')))
        if maxrecs == 0:
            r.terminate("maxrecs records received")
        
        
    def TradesOpenTrades(access_token, accountID): # check
        'Get the list of open Trades for an Account.'
        r = trades.OpenTrades(accountID=accountID)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)

    def TradesTradeCRCDO(access_token, accountID, tradeID, data=None):
        'Trade Create Replace Cancel Dependent Orders.'
        # data= {
        #     "takeProfit": {
        #         "timeInForce": "GTC",
        #         "price": "1.05"
        #     },
        #     "stopLoss": {
        #         "timeInForce": "GTC",
        #         "price": "1.10"
        #     }
        # }
        r = trades.TradeCRCDO(accountID=accountID, tradeID=tradeID, data=data)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)    
    
    def TradesTradeClientExtensions(access_token, accountID, tradeID, data=None):
        'Update the Client Extensions for a Trade. Do not add, update or delete the Client Extensions if your account is associated with MT4.'
        # data = {
        #   "clientExtensions": {
        #     "comment": "myComment",
        #     "id": "myID2315"
        #   }
        # }
        r = trades.TradeClientExtensions(accountID=accountID, tradeID=tradeID, data=data)
        client = API(access_token=access_token)
        client.request(r)
        return r.response
    
    def TradesTradeClose(access_token, accountID, tradeID, data=None):
        'Close (partially or fully) a specific open Trade in an Account.'
        # data = {
        #   "units": 100
        # }
        r = trades.TradeClose(accountID=accountID, data=data)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)
   
    def TradesTradeDetails(access_token, accountID, tradeID):
        'Get the details of a specific Trade in an Account.'
        r = accounts.TradeDetails(accountID=accountID, tradeID=tradeID)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)
    
    def TradesTradesList(access_token, accountID, params=None):
        'Get a list of trades for an Account.'
        # params = {
        #   "instrument": "DE30_EUR,EUR_USD"
        # }
        r = trades.TradesList(accountID=accountID, params=params)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)
       
    def TransactionsTransactionDetails(access_token, accountID, transactionID):
        'Get the details of a single Account Transaction.'
        r = trans.TransactionDetails(accountID=accountID, transactionID=transactionID)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)
        
    def TransactionsTransactionIDRange(access_token, accountID, params=None): # check
        'Get a range of Transactions for an Account based on Transaction IDs.'
        # params = {
        #   "to": 5,
        #   "from": 1
        # }
        r = trans.TransactionIDRange(accountID=accountID, params=params)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)
        
    def TransactionsTransactionList(access_token, accountID, params=None): # check
        'Get a list of Transactions pages that satisfy a time-based Transaction query.'
        # params = {
        #     "pageSize": 200
        # }
        r = trans.TransactionList(accountID=accountID, params=params)  
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)
        
    def TransactionsTransactionsSinceID(access_token, accountID, params=None): # check
        'Get a range of Transactions for an Account starting at (but not including) a provided Transaction ID.'
        # params = {
        #   "id": 3          
        # }
        r = trans.TransactionsSinceID(accountID=accountID, params=params)
        client = API(access_token=access_token)
        client.request(r)
        return munch.Munch(r.response)
        
    def TransactionsTransactionsStream(access_token, accountID, params=None): #check
        'Get a stream of Transactions for an Account starting from when the request is made.'
        # terminate(message='') will stop the stream
        r = trans.TransactionsStream(accountID=accountID)
        client = API(access_token=access_token)
        client.request(r)
        maxrecs = 5
        try:
            for ticks in r.response:
                print(json.dumps(ticks, indent = 4, separators=(',', ': ')))
                maxrecs -= 1
                if maxrecs == 0:
                    r.terminate("Got them all")
        except StreamTerminated as e:
            print("Finished: {msg}".format(msg=e))        