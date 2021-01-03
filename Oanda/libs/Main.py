import munch
from libs.API.Oanda import *
from libs.API.WorkingFunctions import *
from libs.API.Trader import *

access_token = '378d83764609aa3a4eb262663b7c02ef-482ed5696d2a3cede7fca4aa7ded1c76'
accountID = '101-004-16661696-001'
# accountID = '101-004-16661696-002'

cash = AccountSummary(access_token, accountID)['account']['balance']
print(ReadableOutput(cash))

print(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})['prices']['closeoutAsk'])