'''

'''

'TODO: Check for combining triggers https://apscheduler.readthedocs.io/en/v3.6.3/modules/triggers/combining.html#module-apscheduler.triggers.combining'

# =============================================================================
# Imports
# =============================================================================

from libs.API.Oanda import PricingPricingInfo
from tests.test_predictor import Test

# =============================================================================
# Class
# =============================================================================

class VariableSafe():
    
    def __init__(self):
        self.inputt = None
        self.prediction = None
        
    def RetrievePrice(self, access_token, accountID):
        self.inputt = (float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutAsk'])
                       + float(PricingPricingInfo(access_token, accountID, params={"instruments": "EUR_USD"})[1]['prices'][0]['closeoutBid']))/2
        
    def Predic(self):
        _, prediction = Test(self.inputt)
        self.prediction = prediction[0]
        