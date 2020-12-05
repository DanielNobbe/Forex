' Get the class intergration correct '

class Trader:
    """test"""

    def __init__(self, access_token):
        self.access_token = access_token
        
    def Cancel(self, accountID, orderID):
        'It cancells the order with orderID'
        self.OrdersOrderCancel(accountID, orderID)
        print('Cancel order succesful')
       
    def Create(self, accountID, data):
        'It creates an order based on data'
        self.OrdersOrderCreate(accountID, data)
        print('Create order succesful')
       
    def Replace(self, accountID, orderID, data):
        'It simultaniously cancels order with orderID and creates a new order with data'
        self.OrdersOrderReplace(accountID, orderID, data)
        print('Replace order succesful')
        
    def CRCDO(self, accountID, tradeID, data):
        'It creates, cancels and replaces when data has reached certain value'
        self,TradesTradeCRCDO(accountID, tradeID)
        print('Cancel, create replace (based on data order succesful)')
        
    def Close(self, accountID, tradeID, data=None):
        'Ir closes (partially or fully) a specific open Trade in an Account'
        self.TradesTradeClose(accountID, tradeID, data=None)
        print('Close trade (fully partially based on data) order succesful')