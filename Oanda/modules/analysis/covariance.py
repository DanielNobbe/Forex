"""
To be updated, not used atm.
Check for delayed correlation between Forex
"""

from datetime import datetime, timedelta
from libs.API.Oanda import InstrumentsCandles, AccountInstruments
from libs.API.WorkingFunctions import ReadableOutput
import numpy as np
import time

# Because of the delay time steps will not coincide
# Maybe change it to returns
# Timesstep are not consistent...

def Dictionary(access_token, accountID, params, instruments=[]):
    dictionary = {}
    
    if instruments == []:
        z = AccountInstruments(access_token, accountID, params=None)[1]['instruments']
        for x in range(len(z)):
            instruments.append(z[x]['name'])

    for i in instruments:
        instruments_dic = {}
        candles = InstrumentsCandles(access_token, i, params=params)[1]['candles']
        
        for j in range(int(params['count'])):
            instruments_dic[candles[j]['time'][:19]+candles[j]['time'][-1]] = str(format((float(candles[j]['mid']['o']) + float(candles[j]['mid']['c']))/2, '.5f'))
            
        dictionary[i] = instruments_dic
    
    print(ReadableOutput(dictionary))
    # print(instruments_dic)
    
    return dictionary

def Covariance(dictionary): # Nee alleen op 2 d
    print(len(dictionary))
    print(dictionary.keys())
    
    x = []
    
    for i in range(len(dictionary.keys()[0])):
        if dictionary[dictionary.keys()[0]][dictionary.keys()]
    #     for j in range(instruments):
    #         if dictionary[instruments][tijd] == dictionary[ander instrument][tijd]:
    #             x[tijd,:] = prijs van beide instrument
    return 1
    

access_token = '378d83764609aa3a4eb262663b7c02ef-482ed5696d2a3cede7fca4aa7ded1c76'
accountID = '101-004-16661696-001'
params = {
    "price": "M",
    "granularity": "S5",
    "count": "5000",
    # "to": now
    # "from": "2021-01-22T21:10:50Z"
        }
# instruments=["XAG_EUR", "CAD_SGD"]

# print(ReadableOutput(InstrumentsCandles(access_token, instruments[0], params=params)[1]['candles']))

dictionary = Dictionary(access_token, accountID, params, instruments=["EUR_USD", "CAD_SGD"]) #, "CAD_SGD"])
cov = Covariance(dictionary)
# cov = Covariance(access_token, accountID, params)

# base = InstrumentsCandles(access_token, instrument="EUR_USD", params=params)[1]['candles'][-1]['time']
# base_1 = base[:19] + base[-1]
# print(base_1)
# x = datetime.strptime(base_1, "%Y-%m-%dT%H:%M:%SZ")
# print(x)
# date_list = [datetime.strptime(base_1, "%Y-%m-%dT%H:%M:%SZ") - timedelta(seconds=-5*x) for x in range(5000)]
# print(date_list[1])




