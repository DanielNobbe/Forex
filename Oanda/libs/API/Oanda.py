import json
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.forexlabs as labs
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.transactions as trans

class Oanda:
    '''This class imports all data through the functions.
    - Params and data should be dictionaries'''
    
    'There are unchecked functions'

    def __init__(self, access_token): # Add accountID, but how? Put in input with practice or not that will decide wich accountID of the two and that one will be loaded
        self.access_token = access_token
        self.client = API(access_token=access_token)
        
    def AccountChanges(self, accountID, params=None): # check
        'Endpoint used to poll an Account for its current state and changes since a specified TransactionID.'
        # params = {}
        # params['sinceTransactionID'] = 26
        r = accounts.AccountChanges(accountID=accountID, params=params)
        self.client.request(r)
        return r.response
    
    def AccountConfiguration(self, accountID, data=None): # check, but comment
        'Set the client-configurable portions of an Account.'
        # What can you configure besides margin rate?
        r = accounts.AccountConfiguration(accountID=accountID, data=data)
        self.client.request(r)
        return r.response
        
    def AccountDetails(self, accountID): # check
        'Get the full details for a single Account that a client has access to. Full pending Order, open Trade and open Position representations are provided.'
        r = accounts.AccountDetails(accountID=accountID)
        self.client.request(r)
        return r.response
        
    def AccountInstruments(self, accountID, params=None): # check
        'Get the list of tradable instruments for the given Account. The list of tradeable instruments is dependent on the regulatory division that the Account is located in, thus should be the same for all Accounts owned by a single user.'
        r = accounts.AccountInstruments(accountID=accountID, params=params)
        self.client.request(r)
        return r.response
        
    def AccountList(self): # check
        'Get a list of all Accounts authorized for the provided token.'
        r = accounts.AccountList()
        self.client.request(r)
        return r.response
        
    def AccountSummary(self, accountID): # check
        'Get a summary for a single Account that a client has access to.'
        r = accounts.AccountSummary(accountID=accountID)
        self.client.request(r)
        return r.response
        
    def ForexlabsAutochartist(self, params=None): # check, with comment
        'Get the ‘autochartist data’.'
        # https://pages.oanda.com/technical-analysis-autochartist.html
        # params = {
        #     "instrument": "EUR_JPY"
        # }
        r = labs.Autochartist(params=params)
        self.client.request(r)
        return r.response
        
        
    def ForexlabsCalendar(self, params=None): # check, is volgens mij echt chill
        'Get calendar information.'
        # params = {
        #   "instrument": "EUR_USD",
        #   "period": 86400
        # }
        r = labs.Calendar(params=params)
        self.client.request(r)
        return r.response
        
    def ForexlabsCommitmentsOfTraders(self, params=None): # check
        'Get the ‘commitments of traders’ information for an instrument.'
        # params = {
        #   "instrument": "EUR_USD"
        # }   
        r = labs.CommitmentsOfTraders(params=params) 
        self.client.request(r)
        return r.response
        
    def ForexlabsHistoricalPositionsRatios(self, params=None): # error while running, there is an alternative 
        'Error'
        # params = {
        #   "instrument": "EUR_USD",
        #   "period": 86400
        # }
        r = labs.HistoricalPositionRatios(params=params)
        self.client.request(r)
        return r.response
        
    def ForexlabsOrderbookData(self, params=None): # error while running, there is an alternative 
        'Error'
        # params = {
        #   "instrument": "EUR_USD",
        #   "period": 3600
        # }
        r = labs.OrderbookData(params=params)
        self.client.request(r)
        return r.response
        
    def ForexlabsSpreads(self, params=None):
        'Get the spread information for an instrument.'
        # params = {
        #   "instrument": "EUR_USD",
        #   "period": 57600
        # }
        r = labs.Spreads(params=params)
        self.client.request(r)
        return r.response
        
    def InstrumentsCandles(self, instrument, params): # check
        'Get candle data for a specified Instrument.'
        # instrument = "DE30_EUR"
        # params = {
        #   "count": 5,
        #   "granularity": "M5"
        # }
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        self.client.request(r)
        return r.response
        
    def InstrumentsOrderBook(self, instrument, params): # check
        'Get orderbook data for a specified Instrument.'
        # instrument="EUR_USD"
        # params = {}
        r = instruments.InstrumentsOrderBook(instrument=instrument, params=params)
        self.client.request(r)
        return r.response

    def InstrumentsPositionBook(self, instrument, params): # check
        'Get positionbook data for a specified Instrument.'
        r = instruments.InstrumentsPositionBook(instrument=instrument, params=params)
        self.client.request(r)
        return r.response

    def OrdersOrderCancel(self, accountID, orderID):
        'Cancel a pending Order in an Account.'
        r = orders.OrderCancel(accountID=accountID, orderID=orderID)
        self.client.request(r)
        return r.response

    def OrdersOrderClientExtensions(self, accountID, orderID, data=None):
        'Update the Client Extensions for an Order in an Account. Warning: Do not set, modify or delete clientExtensions if your account is associated with MT4.'
        r = orders.OrderClientExtensions(accountID, orderID, data=data)
        self.client.request(r)
        return r.response
        
    def OrdersOrderCreate(self, accountID, data=None):
        'Create an Order for an Account.'
        r = orders.OrderCreate(accountID=accountID, data=data)
        self.client.request(r)
        return r.response

    def OrdersOrderDetails(self, accountID, orderID):
        'Get details for a single Order in an Account.'
        r = orders.OrderDetails(accountID=accountID, orderID=orderID)
        self.client.request(r)
        return r.response

    def OrdersOrderList(self, accountID):
        'Get a list of orders for an account'
        r = orders.OrderList(accountID)
        self.client.request(r)
        return r.response

    def OrdersOrderReplace(self, accountID, orderID, data=None):
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
        self.client.request(r)
        return r.response

    def OrdersOrdersPending(self, accountID):
        'List all pending Orders in an Account.'
        r = orders.OrdersPending(accountID)
        self.client.request(r)
        return r.response
        
    def PositionsOpenPosition(self, accountID): # check
        'List all open Positions for an Account. An open Position is a Position in an Account that currently has a Trade opened for it.'
        r = positions.OpenPositions(accountID=accountID)
        self.client.request(r)
        return r.response

    def PositionsPositionClose(self, accountID, instrument, data=None):
        'Closeout the open Position regarding instrument in an Account.'
        r = positions.PositionClose(accountID=accountID, instrument=instrument, data=data)
        self.client.request(r)
        return r.response
        # data = {
        #   "longUnits": "ALL"
        # }
        
    def PositionsPositionDetails(self, accountID, instrument):
        'Get the details of a single instrument’s position in an Account. The position may be open or not.'
        r = positions.PositionDetails(accountID=accountID, instrument=instrument)
        self.client.request(r)
        return r.response

    def PositionsPositionList(self, accountID): # check
        'List all Positions for an Account. The Positions returned are for every instrument that has had a position during the lifetime of the Account.'
        r = positions.PositionList(accountID=accountID)
        self.client.request(r)
        return r.response
        
    def PricingPricingInfo(self, accountID, access_token, params=None):
        'Get pricing information for a specified list of Instruments within an account.'
        # params = {
        #   "instruments": "EUR_USD,EUR_JPY"
        # }
        r = pricing.PricingInfo(accountID=accountID, params=params)
        self.client.request(r)
        return r.response
    
    def PricingPricingStream(self, accountID, params=None): # check, very nicee
        'Get realtime pricing information for a specified list of Instruments.'
        # params = {
        #   "instruments": "EUR_USD,EUR_JPY"
        # }
        r = pricing.PricingStream(accountID=accountID, params=params)
        self.client.request(r)
        maxrecs = 100
        for ticks in r.response:
            print(json.dumps(ticks, indent = 4, separators=(',', ': ')))
            if maxrecs == 0:
                r.terminate("maxrecs records received")
        
        
    def TradesOpenTrades(self, accountID): # check
        'Get the list of open Trades for an Account.'
        r = trades.OpenTrades(accountID=accountID)
        self.client.request(r)
        return r.response

    def TradesTradeCRCDO(self, accountID, tradeID, data=None):
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
        self.client.request(r)
        return r.response        
    
    def TradesTradeClientExtensions(self, accountID, tradeID, data=None):
        'Update the Client Extensions for a Trade. Do not add, update or delete the Client Extensions if your account is associated with MT4.'
        # data = {
        #   "clientExtensions": {
        #     "comment": "myComment",
        #     "id": "myID2315"
        #   }
        # }
        r = trades.TradeClientExtensions(accountID=accountID, tradeID=tradeID, data=data)
        self.client.request(r)
        return r.response
    
    def TradesTradeClose(self, accountID, tradeID, data=None):
        'Close (partially or fully) a specific open Trade in an Account.'
        # data = {
        #   "units": 100
        # }
        r = trades.TradeClose(accountID=accountID, data=data)
        self.client.request(r)
        return r.response
   
    def TradesTradeDetails(self, accountID, tradeID):
        'Get the details of a specific Trade in an Account.'
        r = accounts.TradeDetails(accountID=accountID, tradeID=tradeID)
        self.client.request(r)
        return r.response
    
    def TradesTradesList(self, accountID, params=None):
        'Get a list of trades for an Account.'
        # params = {
        #   "instrument": "DE30_EUR,EUR_USD"
        # }
        r = trades.TradesList(accountID=accountID, params=params)
        self.client.request(r)
        return r.response
       
    def TransactionsTransactionDetails(self, accountID, transactionID):
        'Get the details of a single Account Transaction.'
        r = trans.TransactionDetails(accountID=accountID, transactionID=transactionID)
        self.client.request(r)
        return r.response
        
    def TransactionsTransactionIDRange(self, accountID, params=None): # check
        'Get a range of Transactions for an Account based on Transaction IDs.'
        # params = {
        #   "to": 5,
        #   "from": 1
        # }
        r = trans.TransactionIDRange(accountID=accountID, params=params)
        self.client.request(r)
        return r.response
        
    def TransactionsTransactionList(self, accountID, params=None): # check
        'Get a list of Transactions pages that satisfy a time-based Transaction query.'
        # params = {
        #     "pageSize": 200
        # }
        r = trans.TransactionList(accountID=accountID, params=params)  
        self.client.request(r)
        return r.response
        
    def TransactionsTransactionsSinceID(self, accountID, params=None): # check
        'Get a range of Transactions for an Account starting at (but not including) a provided Transaction ID.'
        # params = {
        #   "id": 3          
        # }
        r = trans.TransactionsSinceID(accountID=accountID, params=params)
        self.client.request(r)
        return r.response
        
    def TransactionsTransactionsStream(self, accountID, params=None): #check
        'Get a stream of Transactions for an Account starting from when the request is made.'
        # terminate(message='') will stop the stream
        r = trans.TransactionsStream(accountID=accountID)
        self.client.request(r)
        maxrecs = 5
        try:
            for ticks in r.response:
                print(json.dumps(ticks, indent = 4, separators=(',', ': ')))
                maxrecs -= 1
                if maxrecs == 0:
                    r.terminate("Got them all")
        except StreamTerminated as e:
            print("Finished: {msg}".format(msg=e))        