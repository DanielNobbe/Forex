from datetime import datetime, timedelta
from libs.API.Oanda import InstrumentsOrderBook, InstrumentsPositionBook

'Orderbook and Positionbook are published every 20 minutes'
'The formula is used for the most recent'

# TODO: What to do with the info
# TODO: I could dump the try

def Orderbook(access_token, instrument, max_iter):
    Time = InstrumentsOrderBook(access_token, instrument, params={})[1]['orderBook']['time']
    for i in range(0, max_iter+1):
        x = (datetime.strptime(Time, "%Y-%m-%dT%H:%M:%SZ")+i*timedelta(minutes=-20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            InstrumentsOrderBook(access_token, instrument, params={"time": x})[1]
            print("Orderbook for: ", x)
        except:
            print("No orderbook for: ", x)
    return None

def Positionbook(access_token, instrument, max_iter):
    Time = InstrumentsPositionBook(access_token, instrument, params={})[1]['positionBook']['time']
    for i in range(0, max_iter+1):
        x = (datetime.strptime(Time, "%Y-%m-%dT%H:%M:%SZ")+i*timedelta(minutes=-20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            InstrumentsPositionBook(access_token, instrument, params={"time": x})[1]
            print("Positionbook for: ", x)
        except:
            print("No Positionbook for: ", x)
    return None

access_token = '378d83764609aa3a4eb262663b7c02ef-482ed5696d2a3cede7fca4aa7ded1c76'
instrument = "EUR_USD"
Orderbook(access_token, instrument, max_iter=24*3*7)